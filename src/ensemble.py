"""Multi-architecture ensembling for the deepfake detector.

Phase 2 moves from a single backbone to a *heterogeneous* ensemble: one
checkpoint per architecture, combined at prediction time.  Unlike the Phase 1
model soup (weight-space averaging, which only works between checkpoints of the
same architecture) this combines ResNet-50 / ConvNeXt-T / EfficientNet-B3 /
Swin-T and needs no extra training.

Combination rules
-----------------
``mean_prob`` (default)
    Weighted mean of the member sigmoid probabilities.  Strictly per-image, so
    the submission CSV is reproducible across runs, batch sizes and GPU
    architectures (in FP32).
``mean_logit``
    Weighted mean in logit space, then sigmoid.  Slightly sharper than
    ``mean_prob`` and still strictly per-image.
``rank``
    Weighted mean of the per-model ranks.  Best for AUC when member
    calibrations differ a lot, but the value of one image depends on every
    other prediction, so it is meant for validation-time model selection only.

The submitted recipe uses ``mean_prob`` because it is per-image deterministic.

Example
-------
python src/ensemble.py \\
    --checkpoints checkpoints/runs/p2_convnext_tiny_s42/best.pth \\
                  checkpoints/runs/p2_efficientnet_b3_s42/best.pth \\
    --output checkpoints/ensemble_phase2.json
python src/evaluate.py --weights checkpoints/ensemble_phase2.json --split val --tta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import load_model_from_checkpoint  # noqa: E402
from utils import load_checkpoint, predict_loader, save_json  # noqa: E402

COMBINE_METHODS = ("mean_prob", "mean_logit", "rank")
DEFAULT_COMBINE = "mean_prob"
PROB_EPS = 1e-6
REPO_ROOT = SRC_DIR.parent
MANIFEST_KIND = "ensemble_manifest"


# ---------------------------------------------------------------------------
# Combination rules
# ---------------------------------------------------------------------------
def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` into ``[0, 1]`` (ties share their average rank)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    if n == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and ordered[stop] == ordered[start]:
            stop += 1
        # Ranks are 1-based, so the tie group [start, stop) shares (start+1+stop)/2.
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks / float(n)


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(clipped / (1.0 - clipped))


def combine_probabilities(
    member_probabilities: Sequence[np.ndarray],
    member_weights: Optional[Sequence[float]] = None,
    method: str = DEFAULT_COMBINE,
) -> np.ndarray:
    """Combine per-member probability vectors into one probability vector."""
    if not member_probabilities:
        raise ValueError("at least one member prediction is required")
    if method not in COMBINE_METHODS:
        raise ValueError(f"unknown combine method {method!r}; expected one of {COMBINE_METHODS}")
    arrays = [np.asarray(p, dtype=np.float64).ravel() for p in member_probabilities]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"member predictions have different lengths: {sorted(lengths)}")
    if member_weights is None:
        weights = [1.0 / len(arrays)] * len(arrays)
    else:
        if len(member_weights) != len(arrays):
            raise ValueError("member_weights must match the number of predictions")
        total = float(sum(member_weights))
        if total <= 0:
            raise ValueError("member_weights must sum to a positive value")
        weights = [float(weight) / total for weight in member_weights]

    if method == "mean_prob":
        combined = sum(weight * array for weight, array in zip(weights, arrays))
    elif method == "mean_logit":
        combined_logit = sum(weight * logit(array) for weight, array in zip(weights, arrays))
        combined = 1.0 / (1.0 + np.exp(-combined_logit))
    else:  # rank
        combined = sum(weight * average_ranks(array) for weight, array in zip(weights, arrays))
    return np.clip(combined, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Weight search (fit member weights on out-of-fold validation predictions)
# ---------------------------------------------------------------------------
def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC without a sklearn dependency at import time."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def optimize_weights(
    member_probabilities: Sequence[np.ndarray],
    labels: Sequence[int] | np.ndarray,
    method: str = DEFAULT_COMBINE,
    step: float = 0.05,
    min_weight: float = 0.0,
    max_members: Optional[int] = None,
) -> Dict[str, Any]:
    """Grid-search non-negative member weights that maximise validation AUC.

    Phase 2 ensembles mix architectures with very different raw AUCs.  Equal
    weights are a reasonable default but leave accuracy on the table, so the
    submitted manifest is fitted on **out-of-fold** member predictions and the
    chosen weights are frozen into the manifest (no test-time leakage, because
    the folds never contain test images).

    Returns ``{"weights", "method", "auc", "baseline_equal_auc", "gain"}``.
    """
    arrays = [np.asarray(p, dtype=np.float64).ravel() for p in member_probabilities]
    y = np.asarray(labels, dtype=np.int64).ravel()
    if not arrays:
        raise ValueError("at least one member prediction is required")
    if any(array.size != y.size for array in arrays):
        raise ValueError("every member prediction must have one score per label")
    n = len(arrays)
    if max_members:
        n = min(n, int(max_members))
        arrays = arrays[:n]
    if step <= 0:
        raise ValueError("step must be positive")

    equal = combine_probabilities(arrays, None, method)
    baseline = _auc(y, equal)
    best_weights = [1.0 / n] * n
    best_auc = baseline

    # Simplex grid over the first ``n`` members.  Grid resolution is capped so a
    # 4-member ensemble at step=0.05 stays well under a second (C(24,3)=2024).
    steps = int(round(1.0 / step))
    if steps < 1:
        steps = 1

    def score(weights: List[float]) -> float:
        return _auc(y, combine_probabilities(arrays, weights, method))

    def walk(prefix: List[float], remaining: int, units: int) -> None:
        nonlocal best_auc, best_weights
        if remaining == 1:
            candidate = prefix + [units / steps]
            if min_weight and candidate[-1] < min_weight:
                return
            value = score(candidate)
            if value > best_auc + 1e-12:
                best_auc, best_weights = value, candidate
            return
        for units_left in range(units + 1):
            if min_weight and units_left / steps < min_weight:
                continue
            walk(prefix + [units_left / steps], remaining - 1, units - units_left)

    walk([], n, steps)
    total = float(sum(best_weights))
    normalized = [float(w) / total for w in best_weights]
    return {
        "method": method,
        "weights": normalized,
        "auc": float(best_auc),
        "baseline_equal_auc": float(baseline),
        "gain": float(best_auc - baseline),
        "step": float(step),
        "num_members": n,
    }


# ---------------------------------------------------------------------------
# Manifest handling (paths stay valid when the project directory moves)
# ---------------------------------------------------------------------------
def portable_path(path: Path, base: Path = REPO_ROOT) -> str:
    """Return ``path`` relative to ``base`` when possible (POSIX separators)."""
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:  # different Windows drives -> keep the absolute path
        return path.resolve().as_posix()


def resolve_member_path(entry: str, manifest_path: Path) -> Path:
    """Resolve a manifest entry that may be relative to the repo or the manifest."""
    candidate = Path(entry)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for location in (
        manifest_path.parent / candidate,
        manifest_path.parent.parent / candidate,
        REPO_ROOT / candidate,
        Path.cwd() / candidate,
    ):
        if location.exists():
            return location.resolve()
    return candidate


def is_ensemble_manifest(path: str | Path) -> bool:
    """True when ``path`` is a JSON ensemble manifest (not a ``.pth``)."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("kind") == MANIFEST_KIND


def load_manifest(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError(f"{path} is not an ensemble manifest")
    members = payload.get("members") or []
    if not members:
        raise ValueError(f"{path} lists no ensemble members")
    return payload



# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
class Ensemble:
    """A set of single-logit checkpoints combined at prediction time."""

    def __init__(
        self,
        checkpoints: Sequence[str | Path],
        weights: Optional[Sequence[float]] = None,
        method: str = DEFAULT_COMBINE,
        img_size: int = 224,
        eval_resize: Optional[int] = None,
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ) -> None:
        if not checkpoints:
            raise ValueError("an ensemble needs at least one checkpoint")
        if method not in COMBINE_METHODS:
            raise ValueError(f"unknown combine method {method!r}")
        self.method = method
        self.device = torch.device(device)
        self.checkpoints = [Path(path) for path in checkpoints]
        self.img_size = int(img_size)
        self.eval_resize = int(eval_resize) if eval_resize else None
        self.models: List[torch.nn.Module] = []
        self.metadata: List[Dict[str, Any]] = []

        for path in self.checkpoints:
            model, checkpoint = load_model_from_checkpoint(path, device=self.device)
            self.models.append(model)
            metrics = checkpoint.get("metrics", {}) or {}
            self.metadata.append(
                {
                    "checkpoint": str(path),
                    "arch": checkpoint.get("arch"),
                    "backend": checkpoint.get("backend")
                    or (checkpoint.get("config") or {}).get("backend"),
                    "epoch": checkpoint.get("epoch"),
                    "clean_auc": metrics.get("val_auc") or metrics.get("auc"),
                }
            )
            if verbose:
                print(
                    f"[ensemble] member {len(self.models)}: {path.name} "
                    f"arch={checkpoint.get('arch')}",
                    flush=True,
                )

        if weights is None:
            self.weights: List[float] = [1.0 / len(self.models)] * len(self.models)
        else:
            if len(weights) != len(self.models):
                raise ValueError("weights must match the number of checkpoints")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self.weights = [float(weight) / total for weight in weights]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ) -> "Ensemble":
        """Build an ensemble from a JSON manifest written by ``src/ensemble.py``."""
        manifest_path = Path(manifest_path)
        payload = load_manifest(manifest_path)
        members = payload["members"]
        paths = [resolve_member_path(member["checkpoint"], manifest_path) for member in members]
        weights = [float(member.get("weight", 1.0)) for member in members]
        return cls(
            checkpoints=paths,
            weights=weights,
            method=str(payload.get("method", DEFAULT_COMBINE)),
            img_size=int(payload.get("img_size", 224)),
            eval_resize=payload.get("eval_resize"),
            device=device,
            verbose=verbose,
        )

    @property
    def arches(self) -> List[str]:
        return [str(meta["arch"]) for meta in self.metadata]

    def predict_loader(
        self,
        loader: Any,
        tta: bool = False,
        amp: Optional[bool] = None,
        desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run every member over ``loader`` and return the combined prediction dict.

        Per-member outputs are kept so a validation report can show each
        architecture's own AUC next to the ensemble AUC.
        """
        member_probabilities: List[np.ndarray] = []
        member_logits: List[np.ndarray] = []
        labels: Optional[np.ndarray] = None
        image_ids: Optional[List[str]] = None
        load_failures: List[str] = []
        amp_used = False

        for index, model in enumerate(self.models, start=1):
            label = f"{desc} [{index}/{len(self.models)}] {self.arches[index - 1]}" if desc else None
            outputs = predict_loader(model, loader, self.device, tta=tta, amp=amp, desc=label)
            if labels is None:
                labels = np.asarray(outputs["labels"])
                image_ids = list(outputs["image_ids"])
                load_failures = list(outputs.get("load_failures", []))
            elif list(outputs["image_ids"]) != image_ids:
                raise ValueError("ensemble members returned different image orders")
            member_probabilities.append(np.asarray(outputs["probs"], dtype=np.float64))
            member_logits.append(np.asarray(outputs["logits"], dtype=np.float64))
            amp_used = amp_used or bool(outputs.get("amp"))

        combined = combine_probabilities(member_probabilities, self.weights, self.method)
        return {
            "probs": combined,
            "logits": np.stack(member_logits, axis=0).mean(axis=0),
            "labels": labels if labels is not None else np.empty(0),
            "image_ids": image_ids or [],
            "load_failures": load_failures,
            "num_load_failures": len(load_failures),
            "amp": amp_used,
            "tta": bool(tta),
            "device": str(self.device),
            "member_probs": member_probabilities,
            "member_arches": self.arches,
            "member_weights": list(self.weights),
            "combine_method": self.method,
            "num_members": len(self.models),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "num_members": len(self.models),
            "arches": self.arches,
            "weights": list(self.weights),
            "combine_method": self.method,
            "img_size": self.img_size,
            "eval_resize": self.eval_resize,
            "members": self.metadata,
        }


# ---------------------------------------------------------------------------
# Manifest builder + CLI
# ---------------------------------------------------------------------------
def build_manifest(
    checkpoints: Sequence[str | Path],
    weights: Optional[Sequence[float]] = None,
    method: str = DEFAULT_COMBINE,
    output: Optional[str | Path] = None,
    img_size: int = 224,
    eval_resize: Optional[int] = None,
) -> Dict[str, Any]:
    """Write (and return) an ensemble manifest for ``checkpoints``."""
    paths = [Path(path) for path in checkpoints]
    if not paths:
        raise ValueError("at least one checkpoint is required")
    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("weights must match the number of checkpoints")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized = [float(weight) / total for weight in weights]

    members = []
    for path, weight in zip(paths, normalized):
        checkpoint = load_checkpoint(path, map_location="cpu")
        config = checkpoint.get("config", {}) or {}
        metrics = checkpoint.get("metrics", {}) or {}
        members.append(
            {
                "checkpoint": portable_path(path),
                "arch": checkpoint.get("arch"),
                "backend": checkpoint.get("backend") or config.get("backend"),
                "epoch": checkpoint.get("epoch"),
                "weight": weight,
                "val_auc": metrics.get("val_auc") or metrics.get("auc"),
            }
        )

    manifest: Dict[str, Any] = {
        "kind": MANIFEST_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "img_size": int(img_size),
        "eval_resize": int(eval_resize) if eval_resize else None,
        "members": members,
        "num_members": len(members),
        "arches": [member["arch"] for member in members],
    }
    if output is not None:
        save_json(manifest, output)
    return manifest


def load_predictor(
    path: str | Path,
    device: str | torch.device = "cpu",
    method: Optional[str] = None,
    verbose: bool = True,
) -> Ensemble:
    """Load a single checkpoint **or** an ensemble manifest as one predictor.

    ``evaluate.py``, ``inference.py`` and ``robustness.py`` all go through this
    helper, so a manifest and a plain ``.pth`` follow exactly the same code
    path.  A single checkpoint becomes a one-member ensemble, whose
    ``mean_prob`` combination is numerically identical to the model's own
    sigmoid probability.
    """
    path = Path(path)
    if is_ensemble_manifest(path):
        ensemble = Ensemble.from_manifest(path, device=device, verbose=verbose)
    else:
        checkpoint = load_checkpoint(path, map_location="cpu")
        config = checkpoint.get("config", {}) or {}
        ensemble = Ensemble(
            checkpoints=[path],
            method=DEFAULT_COMBINE,
            img_size=int(config.get("img_size", 224)),
            eval_resize=config.get("eval_resize") or None,
            device=device,
            verbose=verbose,
        )
    if method and method != ensemble.method:
        if method not in COMBINE_METHODS:
            raise ValueError(f"unknown combine method {method!r}")
        print(f"[ensemble] overriding combine method {ensemble.method} -> {method}", flush=True)
        ensemble.method = method
    return ensemble



def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and inspect a multi-architecture ensemble manifest"
    )
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--weights", type=str, default="", help="comma-separated member weights")
    parser.add_argument("--method", type=str, default=DEFAULT_COMBINE, choices=COMBINE_METHODS)
    parser.add_argument("--output", type=str, default=None, help="manifest JSON path to write")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--eval_resize", type=int, default=0)
    parser.add_argument(
        "--fit",
        type=str,
        default=None,
        help="JSON with {'labels': [...], 'member_probs': [[...], ...]} used to "
        "grid-search member weights before writing the manifest",
    )
    parser.add_argument("--fit_step", type=float, default=0.05)
    parser.add_argument(
        "--fit_method",
        type=str,
        default=None,
        choices=list(COMBINE_METHODS),
        help="combination rule to optimise against (default: --method)",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="print an existing manifest instead of writing a new one",
    )
    return parser.parse_args(argv)


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.inspect:
        payload = load_manifest(args.inspect)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    if not args.checkpoints:
        raise SystemExit("--checkpoints is required unless --inspect is used")

    weights: Optional[List[float]] = None
    if args.weights.strip():
        weights = [float(value) for value in args.weights.split(",") if value.strip()]

    fit_report: Optional[Dict[str, Any]] = None
    if args.fit:
        payload = json.loads(Path(args.fit).read_text(encoding="utf-8"))
        member_probs = payload.get("member_probs") or payload.get("probs")
        labels = payload.get("labels") or payload.get("y")
        if member_probs is None or labels is None:
            raise SystemExit("--fit JSON must contain 'labels' and 'member_probs'")
        fit_report = optimize_weights(
            member_probs,
            labels,
            method=args.fit_method or args.method,
            step=float(args.fit_step),
        )
        if weights is not None and len(weights) != len(fit_report["weights"]):
            raise SystemExit(
                f"--weights has {len(weights)} entries but the fit file has "
                f"{len(fit_report['weights'])} members"
            )
        print(
            f"[ensemble] fitted weights {[round(w, 4) for w in fit_report['weights']]} "
            f"AUC {fit_report['auc']:.6f} "
            f"(equal-weight baseline {fit_report['baseline_equal_auc']:.6f}, "
            f"gain {fit_report['gain']:+.6f})",
            flush=True,
        )
        weights = list(fit_report["weights"])
        if args.fit_method and args.fit_method != args.method:
            args.method = args.fit_method

    payload = build_manifest(
        checkpoints=args.checkpoints,
        weights=weights,
        method=args.method,
        output=args.output,
        img_size=int(args.img_size),
        eval_resize=int(args.eval_resize) or None,
    )
    if fit_report is not None:
        payload["weight_fit"] = fit_report
        if args.output:
            save_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        print(f"ensemble manifest -> {args.output}")
    return payload


if __name__ == "__main__":
    main()




