"""Hackathon-compliant inference entry point.

The submitted recipe is horizontal-flip TTA enabled, **FP32**, and deterministic
CUDA settings. FP32 avoids cross-GPU fp16 differences (RTX 3050 -> T4 canary
check) and matched AMP on the held-out split while staying inside the 5-minute
budget. ``--amp`` opts into fp16 autocast; ``--no_tta`` disables flip TTA.

Usage
-----
python inference_script.py \
    --test_dir ./hackathon_dataset/test/images/ \
    --weights ./checkpoints/best_model.pth \
    --output predictions.csv

Output CSV columns: image_id,label,confidence
  - image_id   : file name without extension
  - label      : 0 (real) / 1 (fake)
  - confidence : P(fake) in [0, 1]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Deterministic cuBLAS requires this before CUDA initialisation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import main  # noqa: E402

if __name__ == "__main__":
    main()
