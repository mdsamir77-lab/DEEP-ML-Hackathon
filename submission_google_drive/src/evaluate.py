"""Evaluate a trained checkpoint on the held-out validation split.

Writes ``predictions.csv`` (``image_id,label,confidence`` where ``label`` is the
*model's* thresholded prediction, so the file can be submitted as-is) plus
``predictions_detailed.csv`` (``image_id,true_label,predicted_label,confidence``)
for error analysis.

Example
-------
python src/evaluate.py --weights checkpoints/best_model.pth --split val --tta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (  # noqa: E402
    build_eval_transform,
    build_train_dataframe,
    create_dataloaders,
    kfold_split,
    train_val_split,
)
from ensemble import COMBINE_METHODS, is_ensemble_manifest, load_predictor  # noqa: E402
from utils import (  # noqa: E402
    compute_metrics,
    configure_deterministic_inference,
    get_device,
    load_checkpoint,
    plot_confusion_matrix,
    plot_roc_curve,
    save_json,
)


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a deepfake detector checkpoint")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="hackathon_dataset")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"])
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="0 = fixed stratified split; 1..N = that k-fold split (must match training)",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap evaluation samples")
    parser.add_argument("--img_size", type=int, default=0, help="0 = use checkpoint config")
    parser.add_argument("--eval_resize", type=int, default=0)
    parser.add_argument("--tta", action="store_true", default=False)
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
    parser.add_argument("--output", type=str, default=None, help="CSV path for predictions")
    parser.add_argument("--plots_dir", type=str, default=None)
    parser.add_argument(
        "--combine",
        type=str,
        default=None,
        choices=list(COMBINE_METHODS),
        help="override the ensemble combination rule (manifests only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    configure_deterministic_inference(seed=args.seed)
    device = get_device()

    predictor = load_predictor(args.weights, device=device, method=args.combine)
    config = {}
    if not is_ensemble_manifest(args.weights):
        checkpoint = load_checkpoint(args.weights, map_location="cpu")
        config = checkpoint.get("config", {}) or {}
    elif predictor.metadata:
        config = {"img_size": predictor.img_size, "eval_resize": predictor.eval_resize or 0}
    img_size = args.img_size or int(config.get("img_size", 224) or 224)
    eval_resize = args.eval_resize or int(config.get("eval_resize", 0) or 0) or None
    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else data_dir / "train_metadata.csv"
    frame = build_train_dataframe(metadata_csv, data_dir / "train")
    if args.fold > 0:
        train_df, val_df = kfold_split(
            frame, fold=args.fold, folds=args.folds, seed=args.split_seed
        )
    else:
        train_df, val_df = train_val_split(
            frame, val_fraction=args.val_fraction, split_seed=args.split_seed
        )
    selected = {"val": val_df, "train": train_df, "all": frame}[args.split]
    if args.limit and args.limit > 0:
        selected = selected.head(int(args.limit)).reset_index(drop=True)
        print(f"[eval] debug limit: using {len(selected)} samples", flush=True)

    _, loader = create_dataloaders(
        None,
        selected,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_resize=eval_resize,
        eval_transform=build_eval_transform(img_size, eval_resize),
    )
    assert loader is not None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    outputs = predictor.predict_loader(
        loader, tta=args.tta, amp=args.amp, desc=f"eval {args.split}"
    )
    elapsed = time.time() - started
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )

    metrics = compute_metrics(outputs["labels"], outputs["probs"], threshold=args.threshold)
    metrics["n_samples"] = int(len(outputs["labels"]))
    metrics["tta"] = bool(args.tta)
    metrics["amp"] = bool(outputs["amp"])
    metrics["inference_precision"] = "amp (fp16 autocast)" if outputs["amp"] else "fp32"
    metrics["unique_confidences"] = int(np.unique(outputs["probs"]).size)
    metrics["inference_seconds"] = round(elapsed, 3)
    metrics["images_per_second"] = round(len(outputs["labels"]) / max(elapsed, 1e-9), 2)
    metrics["peak_gpu_memory_mb"] = round(peak_gpu_mb, 1) if peak_gpu_mb is not None else None
    metrics["checkpoint"] = str(args.weights)

    # Ensemble provenance: the combined score must be auditable against the
    # individual members, otherwise a multi-architecture result is unverifiable.
    metrics["ensemble"] = predictor.describe()
    if outputs.get("num_members", 1) > 1:
        member_metrics = []
        for arch, weights_, probs in zip(
            outputs["member_arches"], outputs["member_weights"], outputs["member_probs"]
        ):
            member = compute_metrics(outputs["labels"], probs, threshold=args.threshold)
            member_metrics.append(
                {
                    "arch": arch,
                    "weight": float(weights_),
                    "auc": float(member["auc"]),
                    "accuracy": float(member["accuracy"]),
                    "f1": float(member["f1"]),
                }
            )
        metrics["ensemble"]["member_metrics"] = member_metrics
        print("[eval] ensemble members:", flush=True)
        for member in member_metrics:
            print(
                f"  {member['arch']:<22} weight={member['weight']:.3f} "
                f"AUC={member['auc']:.4f} acc={member['accuracy']:.4f}",
                flush=True,
            )

    run_name = Path(args.weights).stem if is_ensemble_manifest(args.weights) else Path(args.weights).parent.name
    plots_dir = Path(args.plots_dir) if args.plots_dir else Path("reports") / f"eval_{run_name}_{args.split}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(
        outputs["labels"],
        outputs["probs"],
        plots_dir / "roc_curve.png",
        title=f"{run_name} | {args.split} | AUC={metrics['auc']:.4f}",
    )
    plot_confusion_matrix(
        outputs["labels"],
        outputs["probs"],
        plots_dir / "confusion_matrix.png",
        threshold=args.threshold,
        title=f"{run_name} | {args.split}",
    )
    save_json(metrics, plots_dir / "metrics.json")

    threshold = float(args.threshold)
    predicted_labels = (outputs["probs"] >= threshold).astype(int)
    output_path = Path(args.output) if args.output else plots_dir / "predictions.csv"
    # A caller-supplied --output may point at a directory that does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Submission-compatible file: ``label`` is the model's *prediction*, never the
    # ground truth (which would make the file look artificially perfect).
    predictions = pd.DataFrame(
        {
            "image_id": outputs["image_ids"],
            "label": predicted_labels,
            "confidence": outputs["probs"].clip(0.0, 1.0),
        }
    )
    predictions.to_csv(output_path, index=False)

    # Separate diagnostic file: keeps ground truth clearly labelled so it can
    # never be mistaken for a prediction.
    detailed_path = output_path.with_name(f"{output_path.stem}_detailed.csv")
    pd.DataFrame(
        {
            "image_id": outputs["image_ids"],
            "true_label": np.asarray(outputs["labels"]).astype(int),
            "predicted_label": predicted_labels,
            "confidence": outputs["probs"].clip(0.0, 1.0),
        }
    ).to_csv(detailed_path, index=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "checkpoint"}, indent=2))
    print(f"predictions (label = predicted @ {threshold:.2f}) -> {output_path}")
    print(f"detailed predictions (true_label, predicted_label, confidence) -> {detailed_path}")
    return metrics


if __name__ == "__main__":
    main()
