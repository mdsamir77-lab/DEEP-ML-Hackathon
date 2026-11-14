"""Batched inference over an unlabelled test directory.

Submitted recipe: horizontal-flip TTA **on**, **FP32** and deterministic CUDA
settings. FP32 is used because fp16 convolution results can differ between GPU
architectures (RTX 3050 vs the evaluator's T4); FP32 matched AMP on the held-out
split and still fits the 5-minute budget. ``--amp`` opts into fp16 autocast.

Example
-------
python src/inference.py --test_dir hackathon_dataset/test/images \
    --weights checkpoints/best_model.pth --output predictions.csv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

# Must be set before CUDA is initialised (torch is imported by ``model`` below).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_test_dataframe, create_test_loader  # noqa: E402
from ensemble import COMBINE_METHODS, is_ensemble_manifest, load_predictor  # noqa: E402
from utils import (  # noqa: E402
    configure_deterministic_inference,
    environment_info,
    get_device,
    load_checkpoint,
    save_json,
)


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake detector inference")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint (.pth/.pt)")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=0, help="0 = use checkpoint config")
    parser.add_argument("--eval_resize", type=int, default=0)
    parser.add_argument(
        "--tta",
        action="store_true",
        default=True,
        help="horizontal-flip TTA (default: enabled - this is the submitted recipe)",
    )
    parser.add_argument(
        "--no_tta",
        dest="tta",
        action="store_false",
        help="disable horizontal-flip TTA (faster, but NOT the submitted recipe)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="fp16 CUDA autocast (faster; NOT the submitted recipe because fp16 "
        "results can differ across GPU architectures)",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="force FP32 inference (default: the submitted recipe)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--combine",
        type=str,
        default=None,
        choices=list(COMBINE_METHODS),
        help="override the ensemble combination rule (manifests only)",
    )
    parser.add_argument(
        "--sample_submission",
        type=str,
        default=None,
        help="optional CSV used to order rows when its image_id set matches the test images",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args(argv)


def _find_sample_submission(
    test_dir: str | Path, explicit: str | Path | None = None
) -> Optional[Path]:
    """Return the sample-submission file used for row ordering (or ``None``)."""
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        print(f"[inference] sample submission not found: {path}; keeping filename order")
        return None
    test_dir = Path(test_dir)
    for candidate in (
        test_dir.parent.parent / "sample_submission.csv",
        test_dir.parent / "sample_submission.csv",
    ):
        if candidate.is_file():
            return candidate
    return None


def _order_like_sample_submission(
    frame: pd.DataFrame, sample_path: Optional[Path]
) -> tuple[pd.DataFrame, bool]:
    """Order ``frame`` like the sample submission when the ID sets match exactly."""
    if sample_path is None:
        return frame, False
    if frame["image_id"].duplicated().any():
        print("[inference] duplicate image ids in the test directory; keeping filename order")
        return frame, False
    try:
        sample_ids = pd.read_csv(sample_path)["image_id"].astype(str)
    except (KeyError, FileNotFoundError, pd.errors.ParserError) as exc:
        print(f"[inference] could not read {sample_path} ({exc}); keeping filename order")
        return frame, False
    if sample_ids.duplicated().any() or set(sample_ids) != set(frame["image_id"]):
        print(f"[inference] {sample_path} ids do not match the test images; keeping filename order")
        return frame, False
    ordered = frame.set_index("image_id").loc[sample_ids.tolist()].reset_index()
    return ordered, True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_submission_csv(
    path: Path,
    expected_ids: Sequence[str],
    threshold: float,
    sample_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fail loudly if the written CSV violates the submission contract."""
    frame = pd.read_csv(path)
    problems: list[str] = []

    if list(frame.columns) != ["image_id", "label", "confidence"]:
        problems.append(
            f"columns are {list(frame.columns)}, expected ['image_id', 'label', 'confidence']"
        )
    if len(frame) != len(expected_ids):
        problems.append(f"row count is {len(frame)}, expected {len(expected_ids)}")
    if frame["image_id"].duplicated().any():
        problems.append(f"{int(frame['image_id'].duplicated().sum())} duplicate image_id rows")
    frame_ids = frame["image_id"].astype(str)
    expected_set = set(map(str, expected_ids))
    if set(frame_ids) != expected_set:
        missing = len(expected_set - set(frame_ids))
        extra = len(set(frame_ids) - expected_set)
        problems.append(f"image_id set mismatch (missing={missing}, extra={extra})")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    if confidence.isna().any():
        problems.append(f"{int(confidence.isna().sum())} non-numeric confidence values")
    elif ((confidence < 0.0) | (confidence > 1.0)).any():
        problems.append("confidence outside [0, 1]")
    if not frame["label"].isin([0, 1]).all():
        problems.append("label contains values outside {0, 1}")
    elif not frame["label"].astype(int).equals((confidence >= threshold).astype(int)):
        problems.append(f"label != int(confidence >= {threshold})")
    if sample_path is not None and sample_path.is_file():
        sample = pd.read_csv(sample_path)
        sample_ids = sample["image_id"].astype(str)
        if set(sample_ids) != set(frame_ids):
            problems.append("image_id set does not match sample_submission.csv")
        elif list(sample_ids) != list(frame_ids):
            problems.append("row order does not match sample_submission.csv")

    if problems:
        raise RuntimeError("Submission CSV validation failed:\n- " + "\n- ".join(problems))

    return {
        "rows": int(len(frame)),
        "unique_ids": int(frame["image_id"].nunique()),
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "predicted_fake_fraction": float(frame["label"].mean()),
    }


def run_inference(
    test_dir: str | Path,
    weights: str | Path,
    output: str | Path = "predictions.csv",
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 0,
    eval_resize: int = 0,
    tta: bool = True,
    amp: bool = False,
    threshold: float = 0.5,
    force_cpu: bool = False,
    sample_submission: str | Path | None = None,
    seed: int = 42,
    combine: str | None = None,
) -> pd.DataFrame:
    """Run inference and write the submission CSV (image_id, label, confidence)."""
    configure_deterministic_inference(seed=seed)
    device = get_device(prefer_cuda=not force_cpu)
    predictor = load_predictor(weights, device=device, method=combine)
    if is_ensemble_manifest(weights):
        config = {"img_size": predictor.img_size, "eval_resize": predictor.eval_resize or 0}
        arch = "ensemble:" + "+".join(predictor.arches)
    else:
        checkpoint = load_checkpoint(weights, map_location="cpu")
        config = checkpoint.get("config", {}) or {}
        arch = checkpoint.get("arch")
    resolved_img_size = int(img_size or config.get("img_size", 224) or 224)
    resolved_resize = int(eval_resize or config.get("eval_resize", 0) or 0) or None

    test_df = build_test_dataframe(test_dir)
    loader = create_test_loader(
        test_df,
        img_size=resolved_img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_resize=resolved_resize,
    )

    started = time.time()
    outputs = predictor.predict_loader(loader, tta=tta, amp=amp, desc="inference")
    elapsed = time.time() - started
    if len(outputs["image_ids"]) != len(test_df):
        raise RuntimeError(
            f"Model returned {len(outputs['image_ids'])} predictions for {len(test_df)} test images"
        )

    frame = pd.DataFrame(
        {
            "image_id": outputs["image_ids"],
            "label": (outputs["probs"] >= threshold).astype(int),
            "confidence": outputs["probs"].clip(0.0, 1.0),
        }
    )
    sample_path = _find_sample_submission(test_dir, sample_submission)
    frame, ordered = _order_like_sample_submission(frame, sample_path)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    validation = _validate_submission_csv(
        output_path,
        expected_ids=[str(x) for x in test_df["image_id"]],
        threshold=threshold,
        sample_path=sample_path,
    )

    summary = {
        "num_images": int(len(frame)),
        "elapsed_seconds": round(elapsed, 2),
        "images_per_second": round(len(frame) / max(elapsed, 1e-6), 2),
        "device": str(device),
        "tta": bool(outputs["tta"]),
        "amp": bool(outputs["amp"]),
        "precision": "amp (fp16 autocast)" if outputs["amp"] else "fp32",
        "deterministic": True,
        "seed": int(seed),
        "threshold": float(threshold),
        "predicted_fake_fraction": float(frame["label"].mean()) if len(frame) else 0.0,
        "mean_confidence": float(frame["confidence"].mean()) if len(frame) else 0.0,
        "unique_confidences": int(frame["confidence"].nunique()),
        "sample_submission": str(sample_path) if sample_path is not None else None,
        "sample_submission_order": bool(ordered),
        "arch": arch,
        "ensemble": predictor.describe(),
        "checkpoint_sha256": _sha256(Path(weights)),
        "output_sha256": _sha256(output_path),
        "num_load_failures": int(outputs.get("num_load_failures", 0)),
        "load_failures": outputs.get("load_failures", []),
        "validation": validation,
        "environment": environment_info(),
    }
    save_json(summary, output_path.with_suffix(".summary.json"))
    print(f"Wrote {len(frame)} predictions -> {output_path}")
    print(summary)
    return frame


def main(argv: Any = None) -> pd.DataFrame:
    args = parse_args(argv)
    return run_inference(
        test_dir=args.test_dir,
        weights=args.weights,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        eval_resize=args.eval_resize,
        tta=args.tta,
        amp=args.amp,
        threshold=args.threshold,
        force_cpu=args.cpu,
        sample_submission=args.sample_submission,
        seed=args.seed,
        combine=args.combine,
    )


if __name__ == "__main__":
    main()
