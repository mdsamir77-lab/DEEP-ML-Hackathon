"""Phase 2 orchestration: multi-architecture training, OOF collection, ensembling.

Why this exists
---------------
Phase 1 was three ResNet-50 runs combined into a weight soup, which plateaued at
0.8962 validation AUC.  Phase 2 needs *architecture diversity*, and diversity
cannot be averaged in weight space: ConvNeXt, EfficientNet and Swin have
different parameter layouts.  So Phase 2 combines members at **prediction**
level (``src/ensemble.py``) and this script automates the pipeline:

1. ``train``    -> train each ``arch x fold`` member with the Phase 1 recipe
2. ``oof``      -> collect out-of-fold member predictions + labels
3. ``fit``      -> grid-search member weights on those OOF predictions
4. ``manifest`` -> write the ensemble manifest used by inference

Every stage is resumable and skips work whose output already exists, so the
script can be interrupted (Kaggle session limits) and restarted.

Example
-------
python src/phase2_pipeline.py --stage train \\
    --arches convnext_tiny efficientnet_b3 swin_t resnet50 \\
    --folds 1 2 3 --epochs 12

python src/phase2_pipeline.py --stage oof --arches convnext_tiny efficientnet_b3
python src/phase2_pipeline.py --stage fit --arches convnext_tiny efficientnet_b3
python src/phase2_pipeline.py --stage manifest --arches convnext_tiny efficientnet_b3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ensemble import build_manifest, optimize_weights  # noqa: E402
from utils import save_json  # noqa: E402

REPO_ROOT = SRC_DIR.parent
DEFAULT_DATA_DIR = "hackathon_dataset"
ARTIFACT_DIR = REPO_ROOT / "reports" / "phase2"
OOF_DIR = ARTIFACT_DIR / "oof"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def member_run_name(arch: str, fold: int, seed: int) -> str:
    """Run directory name; ``fold=0`` is the fixed stratified split."""
    if fold > 0:
        return f"p2_{arch}_f{fold}_s{seed}"
    return f"p2_{arch}_s{seed}"


def member_checkpoint(arch: str, fold: int, seed: int, checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "runs" / member_run_name(arch, fold, seed) / "best.pth"


def oof_path(arch: str, fold: int, seed: int) -> Path:
    return OOF_DIR / f"{member_run_name(arch, fold, seed)}.json"


def run(command: Sequence[str], dry_run: bool = False) -> int:
    printable = " ".join(str(part) for part in command)
    print(f"[phase2] $ {printable}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(list(command), cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        print(f"[phase2] command failed with exit code {completed.returncode}", flush=True)
    return completed.returncode


def python_exe() -> str:
    return sys.executable


# ---------------------------------------------------------------------------
# Stage 1: train every member
# ---------------------------------------------------------------------------
def stage_train(args: argparse.Namespace) -> int:
    failures = 0
    for arch in args.arches:
        for fold in args.folds:
            for seed in args.seeds:
                checkpoint = member_checkpoint(arch, fold, seed, args.checkpoint_dir)
                if checkpoint.exists() and not args.force:
                    print(f"[phase2] skip (exists): {checkpoint}", flush=True)
                    continue
                command = [
                    python_exe(), "src/train.py",
                    "--arch", arch,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--seed", str(seed),
                    "--fold", str(fold),
                    "--folds", str(args.num_folds),
                    "--run_name", member_run_name(arch, fold, seed),
                    "--checkpoint_dir", args.checkpoint_dir,
                    "--data_dir", args.data_dir,
                    "--img_size", str(args.img_size),
                    "--num_workers", str(args.num_workers),
                    "--val_tta",
                ]
                if args.robust_aug:
                    command += ["--robust_aug", "--robust_prob", str(args.robust_prob)]
                if args.ema:
                    command += ["--ema", "--ema_decay", str(args.ema_decay)]
                if args.composite_select:
                    command.append("--composite_select")
                if not args.amp:
                    command.append("--no_amp")
                failures += run(command, args.dry_run) != 0
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Stage 2: collect out-of-fold member predictions
# ---------------------------------------------------------------------------
def stage_oof(args: argparse.Namespace) -> int:
    """Write per-member probability vectors over that member's own fold."""
    failures = 0
    for arch in args.arches:
        for fold in args.folds:
            for seed in args.seeds:
                checkpoint = member_checkpoint(arch, fold, seed, args.checkpoint_dir)
                target = oof_path(arch, fold, seed)
                if not checkpoint.exists() and not args.dry_run:
                    print(f"[phase2] missing checkpoint, cannot make OOF: {checkpoint}", flush=True)
                    failures += 1
                    continue
                if target.exists() and not args.force and not args.dry_run:
                    print(f"[phase2] skip (exists): {target}", flush=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    python_exe(), "src/predict_member.py",
                    "--weights", str(checkpoint),
                    "--output", str(target),
                    "--fold", str(fold),
                    "--folds", str(args.num_folds),
                    "--data_dir", args.data_dir,
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                ]
                if args.tta:
                    command.append("--tta")
                if not args.amp:
                    command.append("--no_amp")
                failures += run(command, args.dry_run) != 0
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Stage 3: fit ensemble weights on OOF predictions
# ---------------------------------------------------------------------------
def stage_fit(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("[phase2] dry run: skipping OOF weight fit", flush=True)
        return 0
    members: List[Dict[str, Any]] = []
    for arch in args.arches:
        for fold in args.folds:
            for seed in args.seeds:
                checkpoint = member_checkpoint(arch, fold, seed, args.checkpoint_dir)
                predictions = oof_path(arch, fold, seed)
                if not checkpoint.exists() or not predictions.exists():
                    print(f"[phase2] skipping member without OOF: {arch} f{fold} s{seed}", flush=True)
                    continue
                payload = json.loads(predictions.read_text(encoding="utf-8"))
                members.append(
                    {
                        "arch": arch,
                        "fold": fold,
                        "seed": seed,
                        "checkpoint": str(checkpoint),
                        "probs": payload["probs"],
                        "labels": payload["labels"],
                    }
                )

    if not members:
        print("[phase2] no members with OOF predictions; run --stage oof first", flush=True)
        return 1

    # Members trained on *different* folds have different validation sets, which
    # must not be concatenated: an image can be in one member's val set and in
    # another member's train set.  Weights are therefore fitted only across
    # members that share an identical label vector (the same fold) and then
    # averaged across folds, which keeps the OOF fit honest.
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for member in members:
        key = tuple(member["labels"])
        grouped.setdefault(key, []).append(member)

    print(
        f"[phase2] {len(members)} members across {len(grouped)} distinct validation sets",
        flush=True,
    )

    per_group: List[Dict[str, Any]] = []
    for group in grouped.values():
        arches = [member["arch"] for member in group]
        report = optimize_weights(
            [member["probs"] for member in group],
            group[0]["labels"],
            method=args.method,
            step=args.step,
        )
        per_group.append({"members": arches, "fit": report})
        print(
            f"[phase2] group {arches}: weights {[round(w, 3) for w in report['weights']]} "
            f"AUC {report['auc']:.6f} (equal {report['baseline_equal_auc']:.6f})",
            flush=True,
        )

    # Aggregate per-architecture weight across groups (each member contributes
    # its weight to its architecture, then the architecture total is normalised).
    arch_totals: Dict[str, float] = {}
    for entry in per_group:
        for arch, weight in zip(entry["members"], entry["fit"]["weights"]):
            arch_totals[arch] = arch_totals.get(arch, 0.0) + float(weight)
    grand_total = float(sum(arch_totals.values())) or 1.0
    arch_weights = {arch: value / grand_total for arch, value in arch_totals.items()}

    fitted_auc = float(np.mean([entry["fit"]["auc"] for entry in per_group]))
    equal_auc = float(np.mean([entry["fit"]["baseline_equal_auc"] for entry in per_group]))
    artifact = {
        "method": args.method,
        "step": float(args.step),
        "arch_weights": arch_weights,
        "mean_oof_auc": fitted_auc,
        "mean_equal_weight_auc": equal_auc,
        "gain": fitted_auc - equal_auc,
        "per_group": per_group,
        "num_members": len(members),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.fit_output)
    save_json(artifact, output)
    print(f"[phase2] fitted mean OOF AUC {fitted_auc:.6f} (equal {equal_auc:.6f})", flush=True)
    print(f"[phase2] arch weights {arch_weights}", flush=True)
    print(f"[phase2] wrote {output}", flush=True)
    return 0



# ---------------------------------------------------------------------------
# Stage 4: write the ensemble manifest
# ---------------------------------------------------------------------------
def stage_manifest(args: argparse.Namespace) -> int:
    if args.dry_run:
        print("[phase2] dry run: skipping manifest build", flush=True)
        return 0
    fit_file = Path(args.fit_output)
    if not fit_file.exists():
        print(f"[phase2] missing fit artifact {fit_file}; run --stage fit first", flush=True)
        return 1
    fit = json.loads(fit_file.read_text(encoding="utf-8"))
    arch_weights: Dict[str, float] = fit["arch_weights"]

    # One representative checkpoint per architecture (lowest fold index) so the
    # manifest has one member per architecture rather than every fold duplicate.
    checkpoints: List[str] = []
    weights: List[float] = []
    for arch in args.arches:
        if arch not in arch_weights:
            continue
        candidate: Optional[Path] = None
        for fold in args.folds:
            for seed in args.seeds:
                path = member_checkpoint(arch, fold, seed, args.checkpoint_dir)
                if path.exists():
                    candidate = path
                    break
            if candidate:
                break
        if candidate is None:
            print(f"[phase2] no checkpoint found for arch {arch}", flush=True)
            continue
        checkpoints.append(str(candidate))
        weights.append(float(arch_weights[arch]))

    if not checkpoints:
        print("[phase2] no checkpoints for the manifest", flush=True)
        return 1

    manifest = build_manifest(
        checkpoints,
        weights=weights,
        method=fit.get("method", args.method),
        output=args.manifest_output,
        img_size=args.img_size,
    )
    manifest["phase2_fit"] = {
        "mean_oof_auc": fit.get("mean_oof_auc"),
        "mean_equal_weight_auc": fit.get("mean_equal_weight_auc"),
        "gain": fit.get("gain"),
    }
    save_json(manifest, args.manifest_output)
    print(f"[phase2] wrote manifest {args.manifest_output} with {len(checkpoints)} members", flush=True)
    for checkpoint, weight in zip(checkpoints, weights):
        print(f"  {weight:.3f}  {checkpoint}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 multi-architecture ensemble pipeline")
    parser.add_argument("--stage", required=True, choices=["train", "oof", "fit", "manifest", "all"])
    parser.add_argument(
        "--arches",
        nargs="+",
        default=["convnext_tiny", "efficientnet_b3", "swin_t", "resnet50"],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--method", type=str, default="mean_prob")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--robust_aug", action="store_true", default=True)
    parser.add_argument("--no_robust_aug", dest="robust_aug", action="store_false")
    parser.add_argument("--robust_prob", type=float, default=0.4)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="ema", action="store_false")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--composite_select", action="store_true", default=True)
    parser.add_argument("--no_composite_select", dest="composite_select", action="store_false")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no_tta", dest="tta", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--fit_output", type=str, default=str(ARTIFACT_DIR / "ensemble_fit.json"))
    parser.add_argument("--manifest_output", type=str, default="checkpoints/ensemble_phase2.json")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    args = parse_args(argv)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    stages = ["train", "oof", "fit", "manifest"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n{'=' * 70}\n[phase2] stage: {stage}\n{'=' * 70}", flush=True)
        handler = {
            "train": stage_train,
            "oof": stage_oof,
            "fit": stage_fit,
            "manifest": stage_manifest,
        }[stage]
        code = handler(args)
        if code != 0:
            print(f"[phase2] stage '{stage}' failed (exit {code}); stopping", flush=True)
            return code
    print("\n[phase2] all requested stages completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

