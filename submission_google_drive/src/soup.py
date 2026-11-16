"""Average compatible checkpoints into a model soup.

Example
-------
python src/soup.py \
    --checkpoints checkpoints/runs/p1_seed42/best.pth checkpoints/runs/p1_seed43/best.pth \
    --output checkpoints/soup_phase1.pth \
    --report reports/soup_phase1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (  # noqa: E402
    average_state_dicts,
    load_checkpoint,
    save_json,
    state_dict_sha256,
)


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average checkpoints into a model soup")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="comma-separated parent weights (default: equal weights)",
    )
    parser.add_argument("--arch", type=str, default=None, help="optional architecture override")
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="optional JSON object of soup metrics, e.g. '{\"auc\": 0.93}'",
    )
    parser.add_argument("--report", type=str, default=None, help="optional JSON report path")
    return parser.parse_args(argv)


def main(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    paths = [Path(path) for path in args.checkpoints]
    checkpoints = [load_checkpoint(path, map_location="cpu") for path in paths]

    arches = {checkpoint.get("arch") for checkpoint in checkpoints}
    if len(arches) != 1:
        raise ValueError(f"checkpoints have different architectures: {sorted(arches)}")
    arch = args.arch or str(next(iter(arches)))

    reference = checkpoints[0]["model_state"]
    for checkpoint in checkpoints[1:]:
        if set(checkpoint["model_state"]) != set(reference):
            raise ValueError("checkpoints have different state-dict keys")

    weights: Optional[List[float]] = None
    if args.weights.strip():
        weights = [float(value) for value in args.weights.split(",") if value.strip()]
        if len(weights) != len(checkpoints):
            raise ValueError("--weights must match the number of checkpoints")

    averaged = average_state_dicts([checkpoint["model_state"] for checkpoint in checkpoints], weights)
    config = dict(checkpoints[0].get("config", {}) or {})
    config.update(
        {
            "checkpoint_type": "soup",
            "soup_parents": [str(path) for path in paths],
            "soup_weights": weights,
        }
    )
    metrics = dict(checkpoints[0].get("metrics", {}) or {})
    if args.metrics.strip():
        metrics.update(json.loads(args.metrics))

    parent_hashes = [state_dict_sha256(checkpoint["model_state"]) for checkpoint in checkpoints]
    soup_hash = state_dict_sha256(averaged)
    state = {
        "checkpoint_type": "soup",
        "arch": arch,
        "backend": config.get("backend", "torchvision"),
        "config": config,
        "epoch": None,
        "metrics": metrics,
        "model_state": averaged,
        "soup": {
            "parents": [str(path) for path in paths],
            "weights": weights,
            "parent_metrics": [checkpoint.get("metrics", {}) for checkpoint in checkpoints],
            "parent_sha256": parent_hashes,
            "soup_sha256": soup_hash,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)

    report = {
        "output": str(output),
        "arch": arch,
        "parents": [str(path) for path in paths],
        "weights": weights,
        "parent_sha256": parent_hashes,
        "soup_sha256": soup_hash,
        "metrics": metrics,
    }
    if args.report:
        save_json(report, args.report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
