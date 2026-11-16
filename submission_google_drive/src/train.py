"""Training entry point with MLflow experiment tracking.

Phase 1 example (ResNet-50 + EMA + robust augmentation + composite selection):

python src/train.py --arch resnet50 --backend torchvision --epochs 15 \
    --batch_size 64 --lr 2e-4 --weight_decay 1e-2 --dropout 0.4 \
    --label_smoothing 0.05 --ema --robust_aug --degradation_p 0.4 \
    --robust_start_epoch 3 --selection_metric composite \
    --variant phase1_ema_robust --run_name resnet50_phase1
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# MLflow >= 3.16 blocks the filesystem store by default; the hackathon blueprint
# requires ./mlruns/, so opt out of the maintenance guard explicitly. Both flags
# must be set *before* `import mlflow`, otherwise the hint is emitted anyway.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402  (import order is intentional: env vars must come first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import (  # noqa: E402
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from tqdm import tqdm  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (  # noqa: E402
    build_eval_transform,
    build_robust_train_transform,
    build_train_dataframe,
    build_train_transform,
    class_balance,
    create_dataloaders,
    create_dual_view_loader,
    kfold_split,
    train_val_split,
)
from model import build_model, build_optimizer_param_groups, resolve_backend  # noqa: E402
from robustness import evaluate_variant  # noqa: E402
from utils import (  # noqa: E402
    AverageMeter,
    ModelEMA,
    Timer,
    composite_auc,
    compute_metrics,
    environment_info,
    get_device,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    predict_loader,
    save_checkpoint,
    save_yaml,
    seed_everything,
    update_bn_stats,
)


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deepfake detector")
    # data
    parser.add_argument("--data_dir", type=str, default="hackathon_dataset")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=0, help="debug: cap train samples")
    parser.add_argument("--limit_val", type=int, default=0, help="debug: cap val samples")
    # model
    parser.add_argument("--arch", type=str, default="efficientnet_b0")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "timm", "torchvision"],
        help="auto = torchvision when its ImageNet weights are cached, else timm",
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--drop_path", type=float, default=0.1)
    # optimisation
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--no_aug",
        action="store_true",
        default=False,
        help="ablation: disable augmentation and train on the deterministic transform",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    # image pipeline
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--eval_resize", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    # tracking / output
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Phase 2: 0 = the fixed stratified split; 1..N = that k-fold split",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Phase 2: number of stratified folds used when --fold > 0",
    )
    parser.add_argument("--experiment", type=str, default="Deepfake Detection")
    parser.add_argument("--tracking_uri", type=str, default="mlruns")
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--plots_dir", type=str, default="reports")
    parser.add_argument("--val_tta", action="store_true", default=False)
    # Phase 1: robust augmentation, EMA, composite selection
    parser.add_argument(
        "--robust_aug",
        action="store_true",
        default=False,
        help="add one of JPEG/blur/noise/downscale/brightness degradations per image",
    )
    parser.add_argument("--degradation_p", type=float, default=0.4)
    parser.add_argument(
        "--robust_start_epoch",
        type=int,
        default=3,
        help="epoch at which the robust transform replaces the mild transform (1 = from start)",
    )
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_bn_batches", type=int, default=100)
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="weight of the clean/degraded consistency MSE (0 = disabled)",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="composite",
        choices=["clean_auc", "composite"],
        help="checkpoint/early-stopping criterion",
    )
    parser.add_argument("--selection_clean_weight", type=float, default=0.5)
    parser.add_argument(
        "--robust_eval_every",
        type=int,
        default=1,
        help="compute the degraded monitor every N epochs (reuses the last value otherwise)",
    )
    parser.add_argument("--robust_monitor_samples", type=int, default=200)
    parser.add_argument(
        "--robust_monitor_variants",
        type=str,
        default="jpeg_q50,blur_s1.0,noise_0.03",
    )
    return parser.parse_args(argv)


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    label_smoothing: float = 0.0,
    grad_clip: float = 0.0,
    epoch: int = 0,
    epochs: int = 0,
    ema: Optional[Any] = None,
    consistency_weight: float = 0.0,
) -> float:
    """One supervised pass over the training loader; returns mean loss."""
    model.train()
    meter = AverageMeter()
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=False, dynamic_ncols=True)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
        if label_smoothing > 0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
            if consistency_weight > 0 and "image_degraded" in batch:
                degraded = batch["image_degraded"].to(device, non_blocking=True)
                logits_degraded = model(degraded)
                loss = loss + criterion(logits_degraded, targets)
                consistency = torch.mean(
                    (torch.sigmoid(logits) - torch.sigmoid(logits_degraded)) ** 2
                )
                loss = loss + consistency_weight * consistency

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        meter.update(loss.item(), images.size(0))
        progress.set_postfix(loss=f"{meter.avg:.4f}")

    return meter.avg


def validate(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    tta: bool = False,
    epoch: int = 0,
    epochs: int = 0,
    amp: Optional[bool] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate the model and return (val loss, prediction dict with metrics)."""
    outputs = predict_loader(model, loader, device, tta=tta, amp=amp, desc=f"val {epoch}/{epochs}")
    if outputs["probs"].size and not np.isfinite(outputs["probs"]).all():
        raise ValueError("non-finite probabilities in validation outputs")
    targets = outputs["labels"].astype(np.float32)
    with torch.no_grad():
        logits = torch.from_numpy(outputs["logits"]).float().unsqueeze(1)
        labels = torch.from_numpy(targets).float().unsqueeze(1)
        val_loss = float(criterion(logits, labels).item())
    metrics = compute_metrics(outputs["labels"], outputs["probs"])
    return val_loss, {"metrics": metrics, "outputs": outputs}


def resolve_tracking_uri(tracking_uri: str) -> str:
    """Accept either a URL or a local directory for the MLflow store."""
    if tracking_uri.startswith(("http://", "https://", "file://", "sqlite:", "postgresql:")):
        return tracking_uri
    return Path(tracking_uri).resolve().as_uri()


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Small label-balanced subset used by the per-epoch robustness monitor."""
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=int(n),
        stratify=df["label"].to_numpy(),
        random_state=seed,
    )
    return sampled.reset_index(drop=True)


def _monitor_degraded_aucs(
    model: torch.nn.Module,
    monitor_df: pd.DataFrame,
    variants: list[str],
    device: torch.device,
    img_size: int,
    eval_resize: Optional[int],
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> list[float]:
    """AUC on a few degradations for the composite checkpoint criterion."""
    aucs: list[float] = []
    for variant in variants:
        try:
            metrics = evaluate_variant(
                model=model,
                dataframe=monitor_df,
                name=variant,
                device=device,
                img_size=img_size,
                eval_resize=eval_resize,
                batch_size=batch_size,
                num_workers=num_workers,
                tta=False,
                amp=use_amp,
                verbose=False,
            )
        except ValueError as exc:
            print(f"[train] robustness monitor {variant!r} failed: {exc}", flush=True)
            continue
        aucs.append(float(metrics["auc"]))
    return aucs


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.consistency_weight > 0 and not args.robust_aug:
        print(
            "[train] --consistency_weight requires --robust_aug; disabling the consistency loss",
            flush=True,
        )
        args.consistency_weight = 0.0
    seed_everything(args.seed)
    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")

    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else data_dir / "train_metadata.csv"
    train_dir = data_dir / "train"

    # ------------------------------------------------------------------ data
    dataframe = build_train_dataframe(metadata_csv, train_dir)
    if args.fold > 0:
        train_df, val_df = kfold_split(
            dataframe, fold=args.fold, folds=args.folds, seed=args.split_seed
        )
    else:
        train_df, val_df = train_val_split(
            dataframe, val_fraction=args.val_fraction, split_seed=args.split_seed
        )
    if args.limit_train:
        train_df = train_df.groupby("label", group_keys=False).head(max(1, args.limit_train // 2))
    if args.limit_val:
        val_df = val_df.groupby("label", group_keys=False).head(max(1, args.limit_val // 2))

    eval_transform = build_eval_transform(args.img_size, args.eval_resize or None)
    mild_transform = eval_transform if args.no_aug else build_train_transform(args.img_size)
    robust_transform = (
        build_robust_train_transform(
            img_size=args.img_size,
            degradation_p=args.degradation_p,
        )
        if args.robust_aug
        else mild_transform
    )

    use_dual_view = bool(args.robust_aug and args.consistency_weight > 0)
    if use_dual_view:
        train_loader = create_dual_view_loader(
            train_df,
            clean_transform=mild_transform,
            degraded_transform=robust_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        initial_transform = (
            robust_transform
            if args.robust_aug and args.robust_start_epoch <= 1
            else mild_transform
        )
        train_loader, _ = create_dataloaders(
            train_df,
            None,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=initial_transform,
        )
    _, val_loader = create_dataloaders(
        None,
        val_df,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_resize=args.eval_resize or None,
        eval_transform=eval_transform,
    )
    assert train_loader is not None and val_loader is not None

    bn_loader = None
    if args.ema:
        bn_loader, _ = create_dataloaders(
            train_df,
            None,
            img_size=args.img_size,
            batch_size=min(32, args.batch_size),
            num_workers=args.num_workers,
            train_transform=eval_transform,
            drop_last=False,
        )
        assert bn_loader is not None

    monitor_df = None
    monitor_variants: list[str] = []
    if args.selection_metric == "composite" and args.robust_eval_every > 0:
        monitor_df = _stratified_sample(val_df, args.robust_monitor_samples, seed=args.seed)
        monitor_variants = [
            variant.strip()
            for variant in args.robust_monitor_variants.split(",")
            if variant.strip()
        ]
        if monitor_variants:
            print(
                f"[train] robust monitor: {len(monitor_df)} images, variants={monitor_variants}",
                flush=True,
            )

    # ----------------------------------------------------------------- model
    resolved_backend = resolve_backend(args.arch, args.backend)
    model = build_model(
        arch=args.arch,
        pretrained=args.pretrained,
        num_classes=1,
        dropout=args.dropout,
        drop_path_rate=args.drop_path,
        backend=resolved_backend,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    criterion = nn.BCEWithLogitsLoss()
    decay_params, no_decay_params = build_optimizer_param_groups(model, args.weight_decay)
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    scheduler = None
    if args.scheduler == "cosine":
        warmup_epochs = max(int(round(args.warmup_epochs)), 0)
        cosine_epochs = max(args.epochs - warmup_epochs, 1)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.lr * 0.01)
        if warmup_epochs > 0:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs), cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2, min_lr=1e-6)

    # --------------------------------------------------------------- tracking
    run_name = args.run_name or f"{args.arch}_{args.variant}_{datetime.now():%Y%m%d}"
    run_dir = Path(args.checkpoint_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) / run_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        **vars(args),
        "backend": resolved_backend,
        "augmentation": (
            "none" if args.no_aug else ("robust" if args.robust_aug else "mild")
        ),
        "run_name": run_name,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "device": str(device),
        "amp": use_amp,
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "num_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        **class_balance(train_df),
    }
    config_path = save_yaml(config, run_dir / "config.yaml")

    mlflow.set_tracking_uri(resolve_tracking_uri(args.tracking_uri))
    mlflow.set_experiment(args.experiment)

    best_auc = -1.0
    best_score = -1.0
    best_composite = -1.0
    best_candidate = "raw"
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    best_outputs: Dict[str, Any] = {}
    epochs_without_improvement = 0
    last_degraded_aucs: list[float] = []
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_auc_raw": [],
        "val_auc_ema": [],
        "val_accuracy": [],
        "selection_score": [],
        "monitor_degraded_auc": [],
        "lr": [],
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flat_params(config))
        mlflow.set_tags(
            {
                "arch": args.arch,
                "variant": args.variant,
                "pretrained": str(args.pretrained),
                "scheduler": args.scheduler,
            }
        )
        mlflow.log_dict(environment_info(), "environment.json")
        mlflow.log_artifact(str(config_path))

        timer = Timer()
        for epoch in range(1, args.epochs + 1):
            # Curriculum: switch from mild to robust augmentation once.
            if (
                args.robust_aug
                and not use_dual_view
                and args.robust_start_epoch > 1
                and epoch == args.robust_start_epoch
            ):
                print(
                    f"[train] switching to robust augmentation at epoch {epoch}",
                    flush=True,
                )
                train_loader, _ = create_dataloaders(
                    train_df,
                    None,
                    img_size=args.img_size,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train_transform=robust_transform,
                )
                assert train_loader is not None

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp,
                label_smoothing=args.label_smoothing,
                grad_clip=args.grad_clip,
                epoch=epoch,
                epochs=args.epochs,
                ema=ema,
                consistency_weight=args.consistency_weight,
            )

            val_loss_raw, raw_payload = validate(
                model,
                val_loader,
                criterion,
                device,
                tta=args.val_tta,
                epoch=epoch,
                epochs=args.epochs,
                amp=use_amp,
            )
            raw_auc = float(raw_payload["metrics"]["auc"])

            ema_valid = False
            val_loss_ema = float("nan")
            if ema is not None and bn_loader is not None:
                update_bn_stats(ema.module, bn_loader, device, max_batches=args.ema_bn_batches)
                try:
                    # FP32 matches the submitted inference recipe and avoids fp16
                    # overflow from BatchNorm statistics that were just recalibrated
                    # on a small clean subset.
                    val_loss_ema, ema_payload = validate(
                        ema.module,
                        val_loader,
                        criterion,
                        device,
                        tta=args.val_tta,
                        epoch=epoch,
                        epochs=args.epochs,
                        amp=False,
                    )
                    ema_valid = True
                except ValueError as exc:
                    print(
                        f"[train] EMA validation failed ({exc}); "
                        "using the raw model for this epoch",
                        flush=True,
                    )
            if ema_valid:
                candidate = ema.module
                candidate_name = "ema"
                val_loss, val_payload = val_loss_ema, ema_payload
            else:
                candidate = model
                candidate_name = "raw"
                val_loss, val_payload = val_loss_raw, raw_payload

            metrics = val_payload["metrics"]
            ema_auc = float(metrics["auc"]) if candidate_name == "ema" else None
            current_lr = float(optimizer.param_groups[0]["lr"])

            if (
                monitor_df is not None
                and monitor_variants
                and epoch % max(args.robust_eval_every, 1) == 0
            ):
                last_degraded_aucs = _monitor_degraded_aucs(
                    candidate,
                    monitor_df,
                    monitor_variants,
                    device,
                    args.img_size,
                    args.eval_resize or None,
                    min(64, args.batch_size),
                    0,
                    False,  # FP32 matches the submitted inference recipe
                )

            if args.selection_metric == "composite" and last_degraded_aucs:
                selection_score = composite_auc(
                    metrics["auc"],
                    last_degraded_aucs,
                    clean_weight=args.selection_clean_weight,
                )
            else:
                selection_score = float(metrics["auc"])
            monitor_auc = (
                float(np.mean(last_degraded_aucs)) if last_degraded_aucs else float("nan")
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(metrics["auc"])
            history["val_auc_raw"].append(raw_auc)
            history["val_auc_ema"].append(ema_auc if ema_auc is not None else float("nan"))
            history["val_accuracy"].append(metrics["accuracy"])
            history["selection_score"].append(selection_score)
            history["monitor_degraded_auc"].append(monitor_auc)
            history["lr"].append(current_lr)

            log_payload = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": metrics["auc"],
                "val_auc_raw": raw_auc,
                "val_accuracy": metrics["accuracy"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "selection_score": selection_score,
                "lr": current_lr,
            }
            if ema_auc is not None:
                log_payload["val_auc_ema"] = ema_auc
            if last_degraded_aucs:
                log_payload["monitor_degraded_auc"] = monitor_auc
            mlflow.log_metrics(log_payload, step=epoch)

            print(
                f"[{run_name}] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auc={metrics['auc']:.4f} raw_auc={raw_auc:.4f} "
                f"ema_auc={ema_auc if ema_auc is not None else float('nan'):.4f} "
                f"monitor_auc={monitor_auc:.4f} selection={selection_score:.4f} "
                f"lr={current_lr:.2e}",
                flush=True,
            )

            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(metrics["auc"])
                else:
                    scheduler.step()

            epoch_metrics = {
                **metrics,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "val_auc_raw": raw_auc,
                "selection_score": selection_score,
                "monitor_degraded_auc": monitor_auc,
            }
            if selection_score > best_score:
                best_score = selection_score
                best_auc = float(metrics["auc"])
                best_composite = (
                    selection_score if args.selection_metric == "composite" else -1.0
                )
                best_candidate = candidate_name
                best_epoch = epoch
                best_metrics = {**metrics, "selection_score": selection_score}
                if last_degraded_aucs:
                    best_metrics["monitor_degraded_auc"] = monitor_auc
                best_outputs = val_payload["outputs"]
                epochs_without_improvement = 0
                save_checkpoint(
                    run_dir / "best.pth",
                    candidate,
                    arch=args.arch,
                    config=config,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    extra={
                        "checkpoint_type": candidate_name,
                        "selection_metric": args.selection_metric,
                        "selection_score": selection_score,
                        "clean_auc": float(metrics["auc"]),
                        "degraded_aucs": last_degraded_aucs,
                        "monitor_variants": monitor_variants,
                    },
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                run_dir / "last.pth",
                model,
                arch=args.arch,
                config=config,
                epoch=epoch,
                metrics=epoch_metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                include_optimizer=True,
                extra={"checkpoint_type": "raw"},
            )

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no {args.selection_metric} improvement).",
                    flush=True,
                )
                break

        train_seconds = timer.elapsed

        # ------------------------------------------------------ final artefacts
        curves_path = plot_training_curves(history, plots_dir / "training_curves.png")
        roc_path = plot_roc_curve(
            best_outputs["labels"],
            best_outputs["probs"],
            plots_dir / "roc_curve.png",
            title=f"{run_name} (epoch {best_epoch})",
        )
        cm_path = plot_confusion_matrix(
            best_outputs["labels"],
            best_outputs["probs"],
            plots_dir / "confusion_matrix.png",
            threshold=0.5,
            title=f"{run_name} @ 0.5",
        )
        predictions_path = run_dir / "val_predictions.csv"
        pd.DataFrame(
            {
                "image_id": best_outputs["image_ids"],
                "true_label": np.asarray(best_outputs["labels"]).astype(int),
                "predicted_label": (np.asarray(best_outputs["probs"]) >= 0.5).astype(int),
                "confidence": np.asarray(best_outputs["probs"], dtype=float),
            }
        ).to_csv(predictions_path, index=False)

        final_metrics = {
            "best_val_auc": float(best_auc),
            "best_epoch": int(best_epoch),
            "train_seconds": float(train_seconds),
            "seconds_per_epoch": float(train_seconds / max(len(history["train_loss"]), 1)),
            **{f"best_{k}": float(v) for k, v in best_metrics.items()},
        }
        if best_composite >= 0:
            final_metrics["best_composite_auc"] = float(best_composite)
        mlflow.log_metrics(final_metrics)
        for artifact in (curves_path, roc_path, cm_path, predictions_path, run_dir / "best.pth"):
            mlflow.log_artifact(str(artifact))
        mlflow.set_tag("best_epoch", str(best_epoch))
        mlflow.set_tag("best_candidate", best_candidate)

    results = {
        "run_name": run_name,
        "run_id": run.info.run_id,
        "arch": args.arch,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "best_selection_score": best_score,
        "best_composite_auc": best_composite if best_composite >= 0 else None,
        "best_candidate": best_candidate,
        "best_metrics": best_metrics,
        "checkpoint": str(run_dir / "best.pth"),
        "train_seconds": train_seconds,
        "history": history,
    }
    print(
        f"Best {args.selection_metric}: {best_score:.4f} at epoch {best_epoch} "
        f"(clean AUC {best_auc:.4f}, candidate {best_candidate}) -> {results['checkpoint']}",
        flush=True,
    )
    return results


def _flat_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """MLflow params must be flat primitives; convert everything else to str."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (bool, int, float, str)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            flat[key] = str(list(value))
        else:
            flat[key] = str(value)
    return flat


if __name__ == "__main__":
    main()



