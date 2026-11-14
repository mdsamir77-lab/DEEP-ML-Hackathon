"""Write a member's out-of-fold predictions as JSON for ensemble weight fitting.

``src/phase2_pipeline.py --stage oof`` calls this once per trained member.  The
output is the minimal payload ``optimize_weights`` needs:

.. code-block:: json

    {"labels": [0, 1, ...], "probs": [0.02, 0.98, ...],
     "checkpoint": "...", "arch": "convnext_tiny", "fold": 1, "folds": 5,
     "auc": 0.94, "n_samples": 1400, "tta": true, "amp": false}

Example
-------
python src/predict_member.py \
    --weights checkpoints/runs/p2_convnext_tiny_f1_s42/best.pth \
    --output reports/phase2/oof/p2_convnext_tiny_f1_s42.json \
    --fold 1 --folds 5 --tta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

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
from ensemble import is_ensemble_manifest, load_predictor  # noqa: E402
from utils import (  # noqa: E402
    compute_metrics,
    configure_deterministic_inference,
    get_device,
    load_checkpoint,
    save_json,
)


def verify_split_matches_checkpoint(
    checkpoint_config: Dict[str, Any],
    fold: int,
    folds: int,
    split_seed: int,
    val_fraction: float,
) -> None:
    """Refuse to build an 'OOF' file from a checkpoint that saw the same data.

    An out-of-fold prediction is only meaningful if the evaluation split is the
    one the checkpoint *did not* train on.  The checkpoint records its split
    configuration, so the two are compared here and a mismatch is a hard error:
    silently producing an in-fold file would inflate every fitted weight.
    """
    trained_fold = int(checkpoint_config.get("fold", 0) or 0)
    trained_folds = int(checkpoint_config.get("folds", folds) or folds)
    trained_seed = int(checkpoint_config.get("split_seed", split_seed) or split_seed)
    trained_fraction = float(checkpoint_config.get("val_fraction", val_fraction) or val_fraction)

    if trained_fold != fold:
        raise SystemExit(
            f"[oof] split mismatch: checkpoint was trained with --fold {trained_fold} "
            f"but --fold {fold} was requested. Predicting on the checkpoint's own "
            f"training split leaks data; pass --fold {trained_fold} instead."
        )
    if fold > 0 and trained_folds != folds:
        raise SystemExit(
            f"[oof] split mismatch: checkpoint used --folds {trained_folds}, "
            f"requested --folds {folds}"
        )
    if fold == 0 and abs(trained_fraction - val_fraction) > 1e-9:
        raise SystemExit(
            f"[oof] split mismatch: checkpoint used --val_fraction {trained_fraction}, "
            f"requested {val_fraction}"
        )
    if trained_seed != split_seed:
        raise SystemExit(
            f"[oof] split mismatch: checkpoint used --split_seed {trained_seed}, "
            f"requested {split_seed}"
        )
    print(
        f"[oof] split verified: fold={trained_fold}/{trained_folds} "
        f"seed={trained_seed} val_fraction={trained_fraction}",
        flush=True,
    )


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump one member's OOF predictions")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="hackathon_dataset")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"])
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, help="0 = fixed split; 1..N = k-fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=0)
    parser.add_argument("--eval_resize", type=int, default=0)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--combine", type=str, default=None)
    parser.add_argument(
        "--skip_split_check",
        action="store_true",
        default=False,
        help="bypass the train/eval split mismatch guard (only for legacy checkpoints)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    configure_deterministic_inference(seed=args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    predictor = load_predictor(args.weights, device=device, method=args.combine)
    if is_ensemble_manifest(args.weights):
        config = {"img_size": predictor.img_size, "eval_resize": predictor.eval_resize or 0}
        arch = "ensemble:" + "+".join(predictor.arches)
        checkpoint_config: Dict[str, Any] = {}
    else:
        checkpoint = load_checkpoint(args.weights, map_location="cpu")
        config = checkpoint.get("config", {}) or {}
        checkpoint_config = config
        arch = checkpoint.get("arch")
    img_size = int(args.img_size or config.get("img_size", 224) or 224)
    eval_resize = int(args.eval_resize or config.get("eval_resize", 0) or 0) or None

    if checkpoint_config and not args.skip_split_check:
        verify_split_matches_checkpoint(
            checkpoint_config,
            fold=args.fold,
            folds=args.folds,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
        )

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
    if selected.empty:
        raise SystemExit(f"split '{args.split}' is empty")

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

    outputs = predictor.predict_loader(loader, tta=args.tta, amp=args.amp, desc="oof")
    labels = [int(value) for value in outputs["labels"]]
    probs = [float(value) for value in outputs["probs"]]
    metrics = compute_metrics(outputs["labels"], outputs["probs"], threshold=0.5)

    payload = {
        "checkpoint": str(args.weights),
        "arch": arch,
        "ensemble": predictor.describe(),
        "split": args.split,
        "fold": int(args.fold),
        "folds": int(args.folds),
        "split_seed": int(args.split_seed),
        "n_samples": len(labels),
        "tta": bool(args.tta),
        "amp": bool(outputs["amp"]),
        "img_size": img_size,
        "eval_resize": eval_resize,
        "auc": float(metrics["auc"]),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "labels": labels,
        "probs": probs,
    }
    save_json(payload, args.output)
    print(
        f"[oof] {arch} fold={args.fold} n={len(labels)} "
        f"AUC={metrics['auc']:.6f} acc={metrics['accuracy']:.6f} -> {args.output}",
        flush=True,
    )
    # Sanity: probabilities must be finite and inside [0, 1].
    array = np.asarray(probs, dtype=np.float64)
    assert np.isfinite(array).all(), "non-finite probabilities"
    assert array.min() >= 0.0 and array.max() <= 1.0, (array.min(), array.max())
    return payload


if __name__ == "__main__":
    main()
