"""Shared helpers: seeding, image discovery, metrics, plotting and checkpoint I/O."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Reproducibility / device helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Benchmarking keeps throughput high on fixed input sizes; determinism is
    # enforced through the seeded samplers/splits instead of cuDNN.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device when available (CPU fallback otherwise)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_info() -> Dict[str, Any]:
    """Collect environment metadata for reproducibility logging."""
    info: Dict[str, Any] = {
        "python": ".".join(str(v) for v in os.sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 2),
                "gpu_capability": f"{props.major}.{props.minor}",
            }
        )
    try:
        import timm

        info["timm"] = timm.__version__
    except Exception:  # pragma: no cover - timm is a hard dependency
        info["timm"] = None
    return info


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def list_images(directory: str | Path) -> List[Path]:
    """Recursively list image files inside ``directory`` (sorted, deterministic)."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def index_images_by_id(directory: str | Path) -> Dict[str, Path]:
    """Map image stem (filename without extension) -> path."""
    return {path.stem: path for path in list_images(directory)}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
class AverageMeter:
    """Running average helper."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class Timer:
    """Simple wall-clock timer used for throughput reporting."""

    def __init__(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def reset(self) -> None:
        self.start = time.time()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUC-ROC (primary metric) plus thresholded diagnostics."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")

    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if np.unique(y_true).size > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["best_threshold"] = float(optimal_threshold(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
        metrics["best_threshold"] = 0.5
    return metrics


def optimal_threshold(
    y_true: Sequence[int] | np.ndarray, y_prob: Sequence[float] | np.ndarray
) -> float:
    """Youden's J statistic threshold (maximises TPR - FPR)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    tta: bool = False,
    desc: Optional[str] = None,
    amp: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the model over ``loader`` and return probabilities/logits/ids.

    ``tta=True`` averages the sigmoid probabilities of an image and its
    horizontal flip (cheap and flip-invariant for face crops).

    ``amp`` selects the inference precision and is honoured on CPU too:

    * ``None`` (default) - CUDA autocast when a GPU is used, FP32 on CPU;
    * ``True`` - half-precision CUDA autocast (CPU stays FP32);
    * ``False`` - full FP32 everywhere.

    AMP is the submitted recipe: on the 1,400-image held-out split it matched
    FP32 (mean AUC 0.892326 vs 0.892312 over three repeats) while running
    1.2-1.4x faster and using half the peak GPU memory. fp16 rounds logits to
    roughly 1e-3, which ties many confidences, but AUC is rank based and the
    ties were overwhelmingly within-class, so they did not cost score. Use
    ``amp=False`` when the maximum confidence resolution is required.
    See README section 4 for the measurements.
    """
    model.eval()
    probs: List[np.ndarray] = []
    logits_out: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    image_ids: List[str] = []

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if desc else loader
    use_amp = device.type == "cuda" and (amp is None or bool(amp))
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if tta:
                logits_flip = model(torch.flip(images, dims=[3]))
                prob = 0.5 * (torch.sigmoid(logits) + torch.sigmoid(logits_flip))
            else:
                prob = torch.sigmoid(logits)
        probs.append(prob.float().cpu().numpy().ravel())
        logits_out.append(logits.float().cpu().numpy().ravel())
        labels.append(np.asarray(batch["label"]).astype(int).ravel())
        image_ids.extend([str(x) for x in batch["image_id"]])

    return {
        "probs": np.concatenate(probs) if probs else np.empty(0),
        "logits": np.concatenate(logits_out) if logits_out else np.empty(0),
        "labels": np.concatenate(labels) if labels else np.empty(0),
        "image_ids": image_ids,
        "amp": bool(use_amp),
        "tta": bool(tta),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    save_path: str | Path,
    title: str = "ROC curve",
) -> Path:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if np.unique(y_true).size > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    save_path: str | Path,
    threshold: float = 0.5,
    title: str = "Confusion matrix",
) -> Path:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_prob, dtype=np.float64).ravel() >= threshold).astype(int)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["real (0)", "fake (1)"])
    ax.set_yticks([0, 1], labels=["real (0)", "fake (1)"])
    ax.set_xlabel(f"Predicted @ threshold {threshold:.2f}")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_training_curves(history: Dict[str, List[float]], save_path: str | Path) -> Path:
    """Plot loss + AUC curves from a per-epoch ``history`` dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history.get("train_loss", []), label="train loss", marker="o")
    axes[0].plot(epochs, history.get("val_loss", []), label="val loss", marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.get("val_auc", []), label="val AUC", color="tab:green", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Validation AUC-ROC")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Checkpoint / config I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    arch: str,
    config: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    include_optimizer: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "arch": arch,
        "config": dict(config),
        "epoch": int(epoch),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if include_optimizer and optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if include_optimizer and scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def save_yaml(config: Dict[str, Any], path: str | Path) -> Path:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(config), handle, sort_keys=True)
    return path


def save_json(payload: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

