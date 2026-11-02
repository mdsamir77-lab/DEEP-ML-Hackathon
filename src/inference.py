"""Batched inference over an unlabelled test directory.

Defaults reproduce the submitted recipe: horizontal-flip TTA **on** and AMP
(fp16 autocast) on CUDA, falling back to FP32 on CPU. Both can be changed
explicitly (``--no_tta``, ``--no_amp``).

Example
-------
python src/inference.py --test_dir hackathon_dataset/test/images \
    --weights checkpoints/best_model.pth --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_test_dataframe, create_test_loader  # noqa: E402
from model import load_model_from_checkpoint  # noqa: E402
from utils import get_device, predict_loader, save_json  # noqa: E402


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
        default=True,
        help="half-precision CUDA autocast (default: the measured best recipe on this GPU)",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="force FP32 inference (full-precision logits, ~2x slower, more unique confidences)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--sample_submission",
        type=str,
        default=None,
        help="optional CSV used to order rows when its image_id set matches the test images",
    )
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


def run_inference(
    test_dir: str | Path,
    weights: str | Path,
    output: str | Path = "predictions.csv",
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 0,
    eval_resize: int = 0,
    tta: bool = True,
    amp: bool = True,
    threshold: float = 0.5,
    force_cpu: bool = False,
    sample_submission: str | Path | None = None,
) -> pd.DataFrame:
    """Run inference and write the submission CSV (image_id, label, confidence)."""
    device = get_device(prefer_cuda=not force_cpu)
    model, checkpoint = load_model_from_checkpoint(weights, device=device)
    config = checkpoint.get("config", {}) or {}
    resolved_img_size = int(img_size or config.get("img_size", 224))
    resolved_resize = int(eval_resize or config.get("eval_resize", 0)) or None

    test_df = build_test_dataframe(test_dir)
    loader = create_test_loader(
        test_df,
        img_size=resolved_img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_resize=resolved_resize,
    )

    started = time.time()
    outputs = predict_loader(model, loader, device, tta=tta, amp=amp, desc="inference")
    elapsed = time.time() - started

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

    summary = {
        "num_images": int(len(frame)),
        "elapsed_seconds": round(elapsed, 2),
        "images_per_second": round(len(frame) / max(elapsed, 1e-6), 2),
        "device": str(device),
        "tta": bool(outputs["tta"]),
        "amp": bool(outputs["amp"]),
        "precision": "amp (fp16 autocast)" if outputs["amp"] else "fp32",
        "threshold": float(threshold),
        "predicted_fake_fraction": float(frame["label"].mean()) if len(frame) else 0.0,
        "mean_confidence": float(frame["confidence"].mean()) if len(frame) else 0.0,
        "unique_confidences": int(frame["confidence"].nunique()),
        "sample_submission": str(sample_path) if sample_path is not None else None,
        "sample_submission_order": bool(ordered),
        "arch": checkpoint.get("arch"),
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
    )


if __name__ == "__main__":
    main()
