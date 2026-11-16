"""Shortcut and leakage audit for the deepfake dataset.

Two independent checks:

1. **Near-duplicate leakage** - perceptual hashes (pHash) cluster visually
   identical frames/crops. The script reports clusters, cross-class clusters
   (possible label noise), train/validation leakage under the official split,
   and optional test-train near-duplicates.
2. **Shortcut audit** - JPEG quantization tables, EXIF tags, file size,
   sharpness, saturation and border statistics by class. Large class-wise
   differences can be learned as shortcuts that do not generalise to degraded
   test variants.

Example
-------
python src/audit.py --data_dir hackathon_dataset --output_dir reports/audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.fftpack import dct  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (  # noqa: E402
    build_test_dataframe,
    build_train_dataframe,
    train_val_split,
)
from utils import save_json  # noqa: E402


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """DCT-based perceptual hash (64-bit, DC term excluded -> 63 bits)."""
    size = hash_size * highfreq_factor
    gray = image.convert("L").resize((size, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size].flatten()[1:]
    return (low > np.median(low)).astype(np.uint8)


def bits_to_uint64(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


_POPCOUNT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def hamming_pairs(
    hashes: np.ndarray,
    threshold: int,
    max_pairs: int = 200_000,
    chunk_size: int = 512,
) -> List[Tuple[int, int, int]]:
    """Return ``(i, j, distance)`` for pairs with Hamming distance <= threshold."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    pairs: List[Tuple[int, int, int]] = []
    for start in range(0, n, chunk_size):
        block = hashes[start : start + chunk_size]
        xor = np.bitwise_xor(block[:, None], hashes[None, :])
        distances = np.zeros(xor.shape, dtype=np.uint8)
        for shift in (0, 16, 32, 48):
            distances += _POPCOUNT[((xor >> shift) & 0xFFFF).astype(np.uint32)]
        rows, cols = np.where(distances <= threshold)
        for row, col in zip(rows, cols):
            i = start + int(row)
            j = int(col)
            if i < j:
                pairs.append((i, j, int(distances[row, col])))
                if max_pairs and len(pairs) >= max_pairs:
                    return pairs
    return pairs


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _exif_value(exif: Any, tag: int) -> str:
    try:
        value = exif.get(tag, "")
    except Exception:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00 ")
        except Exception:
            return repr(value)
    return str(value).strip("\x00 ")


def extract_image_record(path: str | Path, label: int, split: str) -> Dict[str, Any]:
    """Read one image and return size/JPEG/EXIF/pixel-statistics features."""
    path = Path(path)
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
        mode = image.mode
        quantization = image.quantization or {}
        if quantization:
            payload = json.dumps(
                {str(key): list(value) for key, value in sorted(quantization.items())},
                sort_keys=True,
            ).encode("utf-8")
            quant_hash = hashlib.sha1(payload).hexdigest()[:16]
        else:
            quant_hash = "none"
        try:
            exif = image.getexif()
            has_exif = len(exif) > 0
            exif_make = _exif_value(exif, 271)
            exif_model = _exif_value(exif, 272)
            exif_software = _exif_value(exif, 305)
        except Exception:
            has_exif, exif_make, exif_model, exif_software = False, "", "", ""
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    channel_mean = pixels.mean(axis=(0, 1))
    channel_std = pixels.std(axis=(0, 1))
    gray = pixels.mean(axis=2)
    sharpness = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    border = np.concatenate(
        [
            pixels[:4].reshape(-1, 3),
            pixels[-4:].reshape(-1, 3),
            pixels[:, :4].reshape(-1, 3),
            pixels[:, -4:].reshape(-1, 3),
        ],
        axis=0,
    )
    return {
        "path": str(path),
        "label": int(label),
        "split": split,
        "width": int(width),
        "height": int(height),
        "format": image_format,
        "mode": mode,
        "file_size": int(path.stat().st_size),
        "quant_hash": quant_hash,
        "has_exif": bool(has_exif),
        "exif_make": exif_make,
        "exif_model": exif_model,
        "exif_software": exif_software,
        "mean_r": float(channel_mean[0]),
        "mean_g": float(channel_mean[1]),
        "mean_b": float(channel_mean[2]),
        "std_r": float(channel_std[0]),
        "std_g": float(channel_std[1]),
        "std_b": float(channel_std[2]),
        "saturation": float((pixels.max(axis=2) - pixels.min(axis=2)).mean()),
        "sharpness": sharpness,
        "border_mean": float(border.mean()),
        "border_std": float(border.std()),
    }


def audit_metadata(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Compare JPEG/EXIF/pixel statistics between the real and fake classes."""
    records = []
    for position, row in enumerate(df.itertuples(index=False), start=1):
        records.append(extract_image_record(row.path, int(row.label), str(row.split)))
        if position % 500 == 0 or position == len(df):
            print(f"[audit] metadata {position}/{len(df)}", flush=True)
    metadata = pd.DataFrame(records)
    metadata.to_csv(output_dir / "image_metadata.csv", index=False)

    numeric = [
        "file_size",
        "sharpness",
        "saturation",
        "border_mean",
        "border_std",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
    ]
    comparisons = []
    for column in numeric:
        real = metadata.loc[metadata["label"] == 0, column].to_numpy(dtype=float)
        fake = metadata.loc[metadata["label"] == 1, column].to_numpy(dtype=float)
        statistic = ks_2samp(real, fake)
        pooled = float(np.sqrt((real.var(ddof=1) + fake.var(ddof=1)) / 2.0)) or 1.0
        comparisons.append(
            {
                "feature": column,
                "real_mean": float(real.mean()),
                "fake_mean": float(fake.mean()),
                "real_std": float(real.std(ddof=1)),
                "fake_std": float(fake.std(ddof=1)),
                "effect_size": float(abs(fake.mean() - real.mean()) / pooled),
                "ks_statistic": float(statistic.statistic),
                "ks_pvalue": float(statistic.pvalue),
            }
        )
    comparison_df = pd.DataFrame(comparisons).sort_values("effect_size", ascending=False)
    comparison_df.to_csv(output_dir / "class_feature_comparison.csv", index=False)

    quant_counts = (
        metadata.groupby(["quant_hash", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_values(by=1, ascending=False)
    )
    quant_counts.to_csv(output_dir / "quantization_tables_by_class.csv")

    exif_summary = {
        "has_exif_fraction_real": float(metadata.loc[metadata["label"] == 0, "has_exif"].mean()),
        "has_exif_fraction_fake": float(metadata.loc[metadata["label"] == 1, "has_exif"].mean()),
        "top_make_real": metadata.loc[metadata["label"] == 0, "exif_make"]
        .value_counts()
        .head(5)
        .to_dict(),
        "top_make_fake": metadata.loc[metadata["label"] == 1, "exif_make"]
        .value_counts()
        .head(5)
        .to_dict(),
        "top_software_real": metadata.loc[metadata["label"] == 0, "exif_software"]
        .value_counts()
        .head(5)
        .to_dict(),
        "top_software_fake": metadata.loc[metadata["label"] == 1, "exif_software"]
        .value_counts()
        .head(5)
        .to_dict(),
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plot_columns = ["file_size", "sharpness", "saturation", "border_mean", "border_std", "std_r"]
    for ax, column in zip(axes.ravel(), plot_columns):
        ax.hist(
            metadata.loc[metadata["label"] == 0, column],
            bins=50,
            alpha=0.6,
            label="real",
            density=True,
        )
        ax.hist(
            metadata.loc[metadata["label"] == 1, column],
            bins=50,
            alpha=0.6,
            label="fake",
            density=True,
        )
        ax.set_title(column, fontsize=9)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "shortcut_distributions.png", dpi=150)
    plt.close(fig)

    return {
        "n_images": int(len(metadata)),
        "feature_comparison": comparisons,
        "exif_summary": exif_summary,
    }




def hamming_cross_pairs(
    left: np.ndarray,
    right: np.ndarray,
    threshold: int,
    max_pairs: int = 200_000,
    chunk_size: int = 512,
) -> List[Tuple[int, int, int]]:
    """Return ``(left_index, right_index, distance)`` pairs across two hash sets."""
    left = np.asarray(left, dtype=np.uint64)
    right = np.asarray(right, dtype=np.uint64)
    pairs: List[Tuple[int, int, int]] = []
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        xor = np.bitwise_xor(block[:, None], right[None, :])
        distances = np.zeros(xor.shape, dtype=np.uint8)
        for shift in (0, 16, 32, 48):
            distances += _POPCOUNT[((xor >> shift) & 0xFFFF).astype(np.uint32)]
        rows, cols = np.where(distances <= threshold)
        for row, col in zip(rows, cols):
            pairs.append((start + int(row), int(col), int(distances[row, col])))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs


def _compute_hashes(df: pd.DataFrame, label: str) -> np.ndarray:
    hashes: List[int] = []
    for position, row in enumerate(df.itertuples(index=False), start=1):
        with Image.open(row.path) as image:
            hashes.append(bits_to_uint64(phash(image)))
        if position % 500 == 0 or position == len(df):
            print(f"[audit] phash {label} {position}/{len(df)}", flush=True)
    return np.asarray(hashes, dtype=np.uint64)


def audit_phash(
    combined: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    output_dir: Path,
    threshold: int = 6,
    max_pairs: int = 200_000,
) -> Dict[str, Any]:
    """Cluster near-duplicates and report cross-class and split leakage."""
    hashes = _compute_hashes(combined, "train")
    image_ids = combined["image_id"].astype(str).tolist()
    labels = combined["label"].astype(int).tolist()
    splits = combined["split"].astype(str).tolist()

    hash_frame = pd.DataFrame(
        {
            "image_id": image_ids,
            "label": labels,
            "split": splits,
            "phash_hex": [f"{int(value):016x}" for value in hashes],
        }
    )
    hash_frame.to_csv(output_dir / "phash.csv", index=False)

    pairs = hamming_pairs(hashes, threshold, max_pairs=max_pairs)
    capped = bool(max_pairs and len(pairs) >= max_pairs)
    pair_rows = [
        {
            "image_id_1": image_ids[i],
            "image_id_2": image_ids[j],
            "distance": distance,
            "label_1": labels[i],
            "label_2": labels[j],
            "split_1": splits[i],
            "split_2": splits[j],
        }
        for i, j, distance in pairs
    ]
    pair_frame = pd.DataFrame(
        pair_rows,
        columns=[
            "image_id_1",
            "image_id_2",
            "distance",
            "label_1",
            "label_2",
            "split_1",
            "split_2",
        ],
    )
    pair_frame.to_csv(output_dir / "phash_pairs.csv", index=False)

    uf = UnionFind(len(combined))
    for i, j, _ in pairs:
        uf.union(i, j)
    clusters: Dict[int, List[int]] = {}
    for index in range(len(combined)):
        clusters.setdefault(uf.find(index), []).append(index)
    cluster_rows = []
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        member_labels = [labels[index] for index in members]
        member_splits = [splits[index] for index in members]
        cluster_rows.append(
            {
                "cluster_id": root,
                "size": len(members),
                "n_real": member_labels.count(0),
                "n_fake": member_labels.count(1),
                "n_val": member_splits.count("val"),
                "image_ids": ";".join(image_ids[index] for index in members[:10]),
            }
        )
    cluster_frame = pd.DataFrame(
        cluster_rows,
        columns=["cluster_id", "size", "n_real", "n_fake", "n_val", "image_ids"],
    )
    if len(cluster_frame):
        cluster_frame = cluster_frame.sort_values("size", ascending=False)
    cluster_frame.to_csv(output_dir / "phash_clusters.csv", index=False)

    cross_class_pairs = (
        int((pair_frame["label_1"] != pair_frame["label_2"]).sum()) if len(pair_frame) else 0
    )
    leakage_frame = (
        pair_frame[pair_frame["split_1"] != pair_frame["split_2"]]
        if len(pair_frame)
        else pair_frame
    )
    leaking_val_ids = set()
    if len(leakage_frame):
        for row in leakage_frame.itertuples(index=False):
            leaking_val_ids.add(row.image_id_1 if row.split_1 == "val" else row.image_id_2)
    n_val = int((combined["split"] == "val").sum())


    summary: Dict[str, Any] = {
        "n_images": int(len(combined)),
        "n_val_images": n_val,
        "threshold": int(threshold),
        "max_pairs_cap": int(max_pairs),
        "pairs_capped": capped,
        "n_pairs": int(len(pair_frame)),
        "n_clusters": int(len(cluster_frame)),
        "max_cluster_size": int(cluster_frame["size"].max()) if len(cluster_frame) else 1,
        "n_cross_class_pairs": cross_class_pairs,
        "n_cross_class_clusters": int(
            ((cluster_frame["n_real"] > 0) & (cluster_frame["n_fake"] > 0)).sum()
        )
        if len(cluster_frame)
        else 0,
        "n_train_val_leakage_pairs": int(len(leakage_frame)),
        "n_leaking_val_images": int(len(leaking_val_ids)),
        "leakage_fraction_val": float(len(leaking_val_ids) / max(n_val, 1)),
    }

    if test_df is not None and len(test_df):
        test_hashes = _compute_hashes(test_df, "test")
        test_ids = test_df["image_id"].astype(str).tolist()
        cross_pairs = hamming_cross_pairs(test_hashes, hashes, threshold, max_pairs=max_pairs)
        cross_frame = pd.DataFrame(
            [
                {
                    "test_image_id": test_ids[i],
                    "train_image_id": image_ids[j],
                    "distance": distance,
                    "train_label": labels[j],
                }
                for i, j, distance in cross_pairs
            ],
            columns=["test_image_id", "train_image_id", "distance", "train_label"],
        )
        cross_frame.to_csv(output_dir / "test_train_pairs.csv", index=False)
        matched_test = set(cross_frame["test_image_id"]) if len(cross_frame) else set()
        summary.update(
            {
                "n_test_images": int(len(test_df)),
                "n_test_train_pairs": int(len(cross_frame)),
                "n_test_images_with_train_match": int(len(matched_test)),
                "test_train_match_fraction": float(len(matched_test) / max(len(test_df), 1)),
            }
        )

    return summary



def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shortcut and near-duplicate audit")
    parser.add_argument("--data_dir", type=str, default="hackathon_dataset")
    parser.add_argument("--output_dir", type=str, default="reports/audit")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--phash_threshold", type=int, default=6)
    parser.add_argument("--max_pairs", type=int, default=200_000)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap images")
    parser.add_argument("--include_test", action="store_true", default=True)
    parser.add_argument("--no_test", dest="include_test", action="store_false")
    return parser.parse_args(argv)


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    frame = build_train_dataframe(data_dir / "train_metadata.csv", data_dir / "train")
    train_part, val_part = train_val_split(
        frame, val_fraction=args.val_fraction, split_seed=args.split_seed
    )
    train_part = train_part.copy()
    train_part["split"] = "train"
    val_part = val_part.copy()
    val_part["split"] = "val"
    combined = pd.concat([train_part, val_part], ignore_index=True)
    if args.limit and args.limit > 0:
        combined = combined.head(int(args.limit)).reset_index(drop=True)
        print(f"[audit] debug limit: using {len(combined)} training images", flush=True)

    test_df = None
    test_dir = data_dir / "test" / "images"
    if args.include_test and test_dir.exists():
        test_df = build_test_dataframe(test_dir)
        if args.limit and args.limit > 0:
            test_df = test_df.head(int(args.limit)).reset_index(drop=True)
        print(f"[audit] test images: {len(test_df)}", flush=True)

    print("[audit] metadata / shortcut audit ...", flush=True)
    metadata_summary = audit_metadata(combined, output_dir)
    print("[audit] perceptual-hash leakage audit ...", flush=True)
    phash_summary = audit_phash(
        combined,
        test_df,
        output_dir,
        threshold=args.phash_threshold,
        max_pairs=args.max_pairs,
    )

    summary = {
        "data_dir": str(data_dir),
        "val_fraction": float(args.val_fraction),
        "split_seed": int(args.split_seed),
        "metadata": metadata_summary,
        "phash": phash_summary,
    }
    save_json(summary, output_dir / "summary.json")

    lines = [
        "# Audit summary",
        "",
        f"- Training images audited: {metadata_summary['n_images']}",
        f"- Near-duplicate pairs (pHash <= {phash_summary['threshold']}): {phash_summary['n_pairs']}",
        f"- Near-duplicate clusters: {phash_summary['n_clusters']} "
        f"(max size {phash_summary['max_cluster_size']})",
        f"- Cross-class near-duplicate pairs: {phash_summary['n_cross_class_pairs']}",
        f"- Cross-class clusters: {phash_summary['n_cross_class_clusters']}",
        f"- Train/val leakage pairs: {phash_summary['n_train_val_leakage_pairs']}",
        f"- Validation images in a leaking cluster: {phash_summary['n_leaking_val_images']} "
        f"({phash_summary['leakage_fraction_val']:.2%})",
    ]
    if "n_test_images" in phash_summary:
        lines.append(
            f"- Test images with a train near-duplicate: "
            f"{phash_summary['n_test_images_with_train_match']} "
            f"({phash_summary['test_train_match_fraction']:.2%})"
        )
    lines += [
        "",
        "## Largest class-wise feature differences",
        "",
        "| Feature | Real mean | Fake mean | Effect size | KS p-value |",
        "|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        metadata_summary["feature_comparison"], key=lambda item: item["effect_size"], reverse=True
    )
    for row in ranked[:8]:
        lines.append(
            f"| {row['feature']} | {row['real_mean']:.4f} | {row['fake_mean']:.4f} | "
            f"{row['effect_size']:.3f} | {row['ks_pvalue']:.2e} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"phash": phash_summary}, indent=2))
    print(f"Saved audit report -> {output_dir}")
    return summary


if __name__ == "__main__":
    main()

