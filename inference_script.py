"""Hackathon-compliant inference entry point.

Defaults reproduce the submitted recipe: horizontal-flip TTA enabled and AMP
(fp16 autocast) on CUDA, falling back to FP32 on CPU. ``--no_tta`` / ``--no_amp``
are explicit opt-outs for faster or full-precision runs.

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

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import main  # noqa: E402

if __name__ == "__main__":
    main()
