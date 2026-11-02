"""Dataset, transforms and dataloader helpers for the deepfake detection task."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import IMAGENET_MEAN, IMAGENET_STD, index_images_by_id, list_images  # noqa: E402


# ---------------------------------------------------------------------------
# Dataframe builders
# ---------------------------------------------------------------------------
def build_train_dataframe(
    metadata_csv: str | Path,
    image_dir: str | Path,
    label_col: str = "label",
) -> pd.DataFrame:
    """Join ``train_metadata.csv`` with the on-disk image paths.

    Returns a dataframe with ``image_id``, ``label`` and ``path`` columns.
    """
    metadata_csv = Path(metadata_csv)
    df = pd.read_csv(metadata_csv)
    if "image_id" not in df.columns:
        raise ValueError(f"'image_id' column missing from {metadata_csv}")
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column missing from {metadata_csv}")

    index = index_images_by_id(image_dir)
    missing = [img_id for img_id in df["image_id"].astype(str) if img_id not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) from {metadata_csv} were not found under {image_dir}. "
            f"First missing id: {missing[0]}"
        )

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["path"] = df["image_id"].map(index)
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    df["label"] = df["label"].astype(int)
    return df[["image_id", "label", "path"]].reset_index(drop=True)


def build_test_dataframe(image_dir: str | Path) -> pd.DataFrame:
    """Build a dataframe for an unlabelled test directory (label = -1)."""
    paths = list_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return pd.DataFrame(
        {
            "image_id": [path.stem for path in paths],
            "label": [-1] * len(paths),
            "path": paths,
        }
    )


def _stratified_group_split(
    df: pd.DataFrame,
    val_fraction: float,
    split_seed: int,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split that keeps groups intact **and** preserves the label distribution.

    ``GroupShuffleSplit`` only guarantees group separation, so once subject /
    video / source ids exist the validation set can end up badly imbalanced.
    ``StratifiedGroupKFold`` holds out ``1 / n_splits`` of the *whole groups*
    while keeping the real/fake ratio as close to the full dataset as possible.
    """
    groups = df[group_col].astype(str).to_numpy()
    labels = df["label"].to_numpy()
    n_splits = max(2, int(round(1.0 / val_fraction)))

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        train_idx, val_idx = next(splitter.split(df, labels, groups))
    except Exception as exc:  # e.g. a group with a single class, too few groups
        print(
            f"[dataset] stratified group split unavailable ({exc}); falling back to "
            "GroupShuffleSplit (group-preserving but NOT stratified)",
            flush=True,
        )
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=split_seed)
        train_idx, val_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    achieved = len(val_df) / max(len(df), 1)
    if abs(achieved - val_fraction) > 0.02:
        print(
            f"[dataset] note: group split holds out {achieved:.1%} of the data "
            f"(target {val_fraction:.1%}) because whole groups are never split",
            flush=True,
        )
    train_positive = float(train_df["label"].mean())
    val_positive = float(val_df["label"].mean())
    if abs(train_positive - val_positive) > 0.05:
        print(
            f"[dataset] warning: class balance drift after grouping "
            f"train={train_positive:.3f} val={val_positive:.3f}",
            flush=True,
        )
    return train_df, val_df


def train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Label-stratified train/val split.

    When ``group_col`` exists in ``df`` the split becomes group-aware *and*
    stratified (``StratifiedGroupKFold``), so no group leaks across the split
    while the real/fake ratio is preserved. The provided dataset exposes only
    UUID file names (verified free of duplicate image hashes), so by default a
    plain label-stratified ``train_test_split`` is used.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    if group_col and group_col in df.columns:
        return _stratified_group_split(df, val_fraction, split_seed, group_col)

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=split_seed,
        shuffle=True,
        stratify=df["label"].to_numpy(),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB, applying EXIF orientation when present."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


class DeepfakeDataset(Dataset):
    """Face-crop dataset returning ``{"image", "label", "image_id"}`` dicts."""

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Any] = None) -> None:
        required = {"image_id", "path"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Dataframe is missing columns: {sorted(missing)}")
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.has_labels = "label" in self.dataframe.columns

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        image = load_image(row["path"])
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"]) if self.has_labels else -1
        return {"image": image, "label": label, "image_id": str(row["image_id"])}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_train_transform(
    img_size: int = 224,
    crop_scale: Sequence[float] = (0.80, 1.0),
    rotation_degrees: float = 10.0,
    jitter: float = 0.2,
    hflip: float = 0.5,
) -> transforms.Compose:
    """Mild augmentation that preserves forensic / frequency artefacts."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=tuple(crop_scale),
                ratio=(0.9, 1.111),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomApply([transforms.RandomRotation(rotation_degrees)], p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=jitter, contrast=jitter, saturation=jitter * 0.5, hue=0.02
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int = 224, resize_size: Optional[int] = None) -> transforms.Compose:
    """Deterministic resize + center crop + ImageNet normalisation."""
    if resize_size is None:
        # Standard 1.14x resize followed by center crop (256 -> 224).
        resize_size = int(round(img_size * 1.14 / 32.0)) * 32
    return transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
def create_dataloaders(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    eval_resize: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Create (train_loader, val_loader); either may be ``None``."""
    train_loader = None
    val_loader = None

    if train_df is not None and len(train_df):
        train_loader = DataLoader(
            DeepfakeDataset(train_df, train_transform or build_train_transform(img_size)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    if val_df is not None and len(val_df):
        val_loader = DataLoader(
            DeepfakeDataset(val_df, eval_transform or build_eval_transform(img_size, eval_resize)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return train_loader, val_loader


def create_test_loader(
    test_df: pd.DataFrame,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    eval_resize: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Inference loader for an unlabelled test dataframe."""
    return DataLoader(
        DeepfakeDataset(test_df, build_eval_transform(img_size, eval_resize)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def class_balance(df: pd.DataFrame) -> Dict[str, float]:
    """Convenience helper used for logging the split composition."""
    counts = df["label"].value_counts().to_dict()
    total = max(int(sum(counts.values())), 1)
    return {f"class_{int(k)}_fraction": float(v) / total for k, v in counts.items()}


__all__ = [
    "DeepfakeDataset",
    "build_train_dataframe",
    "build_test_dataframe",
    "train_val_split",
    "build_train_transform",
    "build_eval_transform",
    "create_dataloaders",
    "create_test_loader",
    "class_balance",
    "load_image",
]


if __name__ == "__main__":  # tiny manual sanity check
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the dataset pipeline")
    parser.add_argument("--data_dir", default="hackathon_dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.data_dir)
    frame = build_train_dataframe(root / "train_metadata.csv", root / "train")
    train_part, val_part = train_val_split(frame)
    loader, _ = create_dataloaders(
        train_part, val_part, batch_size=args.batch_size, num_workers=args.num_workers
    )
    batch = next(iter(loader))
    print("train:", len(train_part), "val:", len(val_part))
    print("batch image:", tuple(batch["image"].shape), batch["image"].dtype)
    print("batch label:", batch["label"].tolist())
    print("batch ids:", batch["image_id"][:3])

