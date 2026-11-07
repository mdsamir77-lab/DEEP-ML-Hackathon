"""Robustness evaluation over common image degradations.

The official robustness score is a weighted AUC across undisclosed test-set
variants. This harness applies a broad, deterministic set of plausible
degradations to the held-out validation split and reports the AUC per variant
plus a transparent composite proxy:

    composite = 0.5 * clean_auc + 0.5 * mean(degraded_aucs)

Example
-------
python src/robustness.py --weights checkpoints/best_model.pth \
    --output_dir reports/robustness --tta
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (  # noqa: E402
    DeepfakeDataset,
    build_eval_transform,
    build_train_dataframe,
    train_val_split,
)
from model import load_model_from_checkpoint  # noqa: E402
from utils import (  # noqa: E402
    composite_auc,
    compute_metrics,
    configure_deterministic_inference,
    get_device,
    predict_loader,
    save_json,
)


class JpegVariant:
    """Re-encode as JPEG at a fixed quality (4:2:0 by default)."""

    def __init__(self, quality: int, subsampling: int = 2) -> None:
        self.quality = int(quality)
        self.subsampling = int(subsampling)

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.convert("RGB").save(
            buffer, format="JPEG", quality=self.quality, subsampling=self.subsampling
        )
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class BlurVariant:
    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=self.sigma))


class NoiseVariant:
    """Gaussian noise in [0, 1] pixel space with a content-derived seed.

    The seed depends only on the image pixels, so results are identical
    regardless of DataLoader worker scheduling.
    """

    def __init__(self, std: float) -> None:
        self.std = float(std)

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        seed = zlib.crc32(array.tobytes()[:4096]) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.std, size=array.shape).astype(np.float32)
        array = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


class DownscaleVariant:
    """Resize down and back up (resolution loss, common in re-uploads)."""

    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        small = image.resize(
            (max(1, int(width * self.scale)), max(1, int(height * self.scale))),
            Image.LANCZOS,
        )
        return small.resize((width, height), Image.BICUBIC)


class BrightnessVariant:
    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image.convert("RGB")).enhance(self.factor)


class ContrastVariant:
    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image.convert("RGB")).enhance(self.factor)


class GrayscaleVariant:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("L").convert("RGB")


VARIANT_FACTORIES: Dict[str, Callable[[], Any]] = {
    "jpeg_q90": lambda: JpegVariant(90),
    "jpeg_q70": lambda: JpegVariant(70),
    "jpeg_q50": lambda: JpegVariant(50),
    "jpeg_q30": lambda: JpegVariant(30),
    "blur_s0.5": lambda: BlurVariant(0.5),
    "blur_s1.0": lambda: BlurVariant(1.0),
    "blur_s2.0": lambda: BlurVariant(2.0),
    "noise_0.01": lambda: NoiseVariant(0.01),
    "noise_0.03": lambda: NoiseVariant(0.03),
    "downscale_0.75": lambda: DownscaleVariant(0.75),
    "downscale_0.50": lambda: DownscaleVariant(0.50),
    "brightness_0.8": lambda: BrightnessVariant(0.8),
    "brightness_1.2": lambda: BrightnessVariant(1.2),
    "contrast_0.8": lambda: ContrastVariant(0.8),
    "contrast_1.2": lambda: ContrastVariant(1.2),
    "grayscale": lambda: GrayscaleVariant(),
}


def build_variant_transform(name: str, img_size: int, eval_resize: Optional[int]) -> Any:
    base = build_eval_transform(img_size, eval_resize)
    if name == "clean":
        return base
    if name not in VARIANT_FACTORIES:
        raise ValueError(f"Unknown variant '{name}'. Choose from: {sorted(VARIANT_FACTORIES)}")
    return transforms.Compose([VARIANT_FACTORIES[name](), base])


def evaluate_variant(
    model: Any,
    dataframe: pd.DataFrame,
    name: str,
    device: Any,
    img_size: int,
    eval_resize: Optional[int],
    batch_size: int,
    num_workers: int,
    tta: bool,
    amp: bool,
) -> Dict[str, Any]:
    loader = DataLoader(
        DeepfakeDataset(dataframe, build_variant_transform(name, img_size, eval_resize)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    outputs = predict_loader(model, loader, device, tta=tta, amp=amp, desc=f"robust {name}")
    metrics = compute_metrics(outputs["labels"], outputs["probs"], threshold=0.5)
    metrics.update(
        {
            "variant": name,
            "n_samples": int(len(outputs["labels"])),
            "num_load_failures": int(outputs.get("num_load_failures", 0)),
        }
    )
    return metrics


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness under image degradations")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="hackathon_dataset")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=0, help="0 = use checkpoint config")
    parser.add_argument("--eval_resize", type=int, default=0)
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no_tta", dest="tta", action="store_false")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="reports/robustness")
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="comma-separated subset (e.g. clean,jpeg_q50,blur_s1.0); default = all",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: cap validation samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    configure_deterministic_inference(seed=args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    model, checkpoint = load_model_from_checkpoint(args.weights, device=device)
    config = checkpoint.get("config", {}) or {}
    img_size = int(args.img_size or config.get("img_size", 224))
    eval_resize = int(args.eval_resize or config.get("eval_resize", 0)) or None

    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else data_dir / "train_metadata.csv"
    frame = build_train_dataframe(metadata_csv, data_dir / "train")
    _, val_df = train_val_split(frame, val_fraction=args.val_fraction, split_seed=args.split_seed)
    if args.limit and args.limit > 0:
        val_df = val_df.head(int(args.limit)).reset_index(drop=True)
        print(f"[robustness] debug limit: using {len(val_df)} validation images", flush=True)

    variant_names = ["clean"]
    if args.variants.strip():
        requested = [v.strip() for v in args.variants.split(",") if v.strip()]
        variant_names += [v for v in requested if v != "clean"]
    else:
        variant_names += list(VARIANT_FACTORIES)

    metrics: Dict[str, Dict[str, Any]] = {}
    for name in variant_names:
        print(f"[robustness] evaluating {name} ...", flush=True)
        metrics[name] = evaluate_variant(
            model=model,
            dataframe=val_df,
            name=name,
            device=device,
            img_size=img_size,
            eval_resize=eval_resize,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tta=args.tta,
            amp=args.amp,
        )
        print(
            f"[robustness] {name}: AUC={metrics[name]['auc']:.4f} "
            f"acc={metrics[name]['accuracy']:.4f}",
            flush=True,
        )

    clean_auc = float(metrics["clean"]["auc"])
    degraded = [float(m["auc"]) for name, m in metrics.items() if name != "clean"]
    composite = composite_auc(clean_auc, degraded)
    summary = {
        "checkpoint": str(args.weights),
        "n_samples": int(metrics["clean"]["n_samples"]),
        "tta": bool(args.tta),
        "amp": bool(args.amp),
        "img_size": img_size,
        "eval_resize": eval_resize,
        "clean_auc": clean_auc,
        "mean_degraded_auc": float(np.mean(degraded)) if degraded else None,
        "worst_degraded_auc": float(np.min(degraded)) if degraded else None,
        "best_degraded_auc": float(np.max(degraded)) if degraded else None,
        "composite_auc": float(composite),
        "robustness_score_proxy": float(20.0 * max(0.0, (composite - 0.5) / 0.5)),
        "accuracy_pts_proxy": float(50.0 * max(0.0, (clean_auc - 0.5) / 0.5)),
        "variants": metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "metrics.json")

    table = pd.DataFrame(
        [
            {
                "variant": name,
                "auc": m["auc"],
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "n_samples": m["n_samples"],
            }
            for name, m in metrics.items()
        ]
    )
    table.to_csv(output_dir / "robustness_table.csv", index=False)
    header = "| Variant | AUC | Accuracy | F1 | Precision | Recall | N |\n"
    header += "|---|---:|---:|---:|---:|---:|---:|\n"
    rows = [
        f"| {row.variant} | {row.auc:.4f} | {row.accuracy:.4f} | {row.f1:.4f} | "
        f"{row.precision:.4f} | {row.recall:.4f} | {row.n_samples} |"
        for row in table.itertuples(index=False)
    ]
    (output_dir / "robustness_table.md").write_text(
        header + "\n".join(rows) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = list(metrics)
    aucs = [float(metrics[name]["auc"]) for name in names]
    colors = ["tab:green" if name == "clean" else "tab:orange" for name in names]
    ax.bar(range(len(names)), aucs, color=colors)
    ax.axhline(
        clean_auc, color="tab:green", linestyle="--", linewidth=1, label=f"clean {clean_auc:.4f}"
    )
    ax.axhline(
        composite, color="tab:red", linestyle=":", linewidth=1, label=f"composite {composite:.4f}"
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Robustness under image degradations")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_auc.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "variants"}, indent=2))
    print(table.to_string(index=False))
    print(f"Saved robustness report -> {output_dir}")
    return summary


if __name__ == "__main__":
    main()

