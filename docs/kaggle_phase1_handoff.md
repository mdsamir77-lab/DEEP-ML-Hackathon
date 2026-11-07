# Phase 1 Kaggle handoff

Phase 0 is verified locally. Phase 1 training will run on Kaggle (single T4, not 2×T4
automatically). This document lists exactly what to upload, run, and bring back.

## 1. What to upload

**Code** (small, upload as a private Kaggle Dataset or a notebook attachment):

- `src/` (all `.py` files)
- `inference_script.py`
- `requirements.txt`
- `requirements-inference.txt`
- `README.md`
- `evaluation_metric.md`
- `docs/` (optional but useful)

**Dataset** (private Kaggle Dataset):

- `hackathon_dataset/train/real/`
- `hackathon_dataset/train/fake/`
- `hackathon_dataset/train_metadata.csv`
- `hackathon_dataset/test/images/`
- `hackathon_dataset/sample_submission.csv`

Do **not** upload `checkpoints/*.pth` for Phase 1 training; the run starts from ImageNet
weights. Do **not** upload `logs/`, `mlruns/`, or `reports/` (not needed to train).

## 2. Expected Kaggle dataset layout

```
/kaggle/input/deepfake-hackathon/hackathon_dataset/
├── train/
│   ├── real/*.jpg
│   └── fake/*.jpg
├── train_metadata.csv
├── test/images/*.jpg
└── sample_submission.csv
```

`src/dataset.py` expects `train_metadata.csv` with `image_id,label` and finds images by
stem under `train/`. If your Kaggle dataset slug differs, change `--data_dir` only.

## 3. Notebook setup

1. Create a Kaggle notebook, **Accelerator = GPU T4 ×2**, **Internet = On**.
2. Use only GPU 0. Do not add `DataParallel`/`DDP` in Phase 1; the code is single-GPU.
3. Copy the code into the working directory:

```bash
mkdir -p /kaggle/working/deepfake
cp -r /kaggle/input/<code-dataset>/src /kaggle/working/deepfake/
cp /kaggle/input/<code-dataset>/inference_script.py /kaggle/working/deepfake/
cp /kaggle/input/<code-dataset>/requirements*.txt /kaggle/working/deepfake/
cp /kaggle/input/<code-dataset>/README.md /kaggle/working/deepfake/
cd /kaggle/working/deepfake
```

4. Install dependencies. Kaggle already ships PyTorch/torchvision; check first:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
# If compatible, only install the missing packages:
pip install -q timm scikit-learn matplotlib tqdm pyyaml
# Otherwise:
# pip install -q -r requirements-inference.txt
```

5. Set environment variables:

```bash
export PYTHONPATH=/kaggle/working/deepfake
export MLFLOW_ALLOW_FILE_STORE=true
export MLFLOW_DISABLE_AGENT_HINT=1
# Optional if huggingface.co is slow:
# export HF_MIRROR=https://hf-mirror.com
```

6. Confirm the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Phase 1 training commands (exact)

All commands use the Phase 1 code delivered in this phase. Use a **single T4** and
`CUDA_VISIBLE_DEVICES=0`. `--num_workers 4` is a good default on Kaggle.

### P1-A — EMA + mild augmentation (clean-AUC control)

```bash
python src/train.py \
  --data_dir /kaggle/input/deepfake-hackathon/hackathon_dataset \
  --arch resnet50 --backend torchvision \
  --epochs 15 --batch_size 64 --lr 2e-4 --weight_decay 1e-2 \
  --dropout 0.4 --label_smoothing 0.05 \
  --img_size 224 --eval_resize 224 --num_workers 4 \
  --scheduler cosine --warmup_epochs 1 --early_stopping_patience 5 \
  --ema --ema_decay 0.999 --ema_bn_batches 100 \
  --selection_metric clean_auc \
  --checkpoint_dir /kaggle/working/deepfake/checkpoints \
  --plots_dir /kaggle/working/deepfake/reports \
  --tracking_uri /kaggle/working/deepfake/mlruns \
  --variant phase1_ema --run_name p1a_ema_<date>
```

### P1-B — EMA + robust augmentation + composite selection (main candidate)

```bash
python src/train.py \
  --data_dir /kaggle/input/deepfake-hackathon/hackathon_dataset \
  --arch resnet50 --backend torchvision \
  --epochs 15 --batch_size 64 --lr 2e-4 --weight_decay 1e-2 \
  --dropout 0.4 --label_smoothing 0.05 \
  --img_size 224 --eval_resize 224 --num_workers 4 \
  --scheduler cosine --warmup_epochs 1 --early_stopping_patience 5 \
  --ema --ema_decay 0.999 --ema_bn_batches 100 \
  --robust_aug --degradation_p 0.4 --robust_start_epoch 3 \
  --selection_metric composite --selection_clean_weight 0.5 \
  --robust_eval_every 1 --robust_monitor_samples 200 \
  --robust_monitor_variants jpeg_q50,blur_s1.0,noise_0.03 \
  --checkpoint_dir /kaggle/working/deepfake/checkpoints \
  --plots_dir /kaggle/working/deepfake/reports \
  --tracking_uri /kaggle/working/deepfake/mlruns \
  --variant phase1_ema_robust --run_name p1b_ema_robust_<date>
```

### P1-C — optional consistency loss (only if P1-B’s worst variant is below ~0.87)

```bash
python src/train.py \
  --data_dir /kaggle/input/deepfake-hackathon/hackathon_dataset \
  --arch resnet50 --backend torchvision \
  --epochs 15 --batch_size 64 --lr 2e-4 --weight_decay 1e-2 \
  --dropout 0.4 --label_smoothing 0.05 \
  --img_size 224 --eval_resize 224 --num_workers 4 \
  --scheduler cosine --warmup_epochs 1 --early_stopping_patience 5 \
  --ema --ema_decay 0.999 --ema_bn_batches 100 \
  --robust_aug --degradation_p 0.4 --consistency_weight 0.05 \
  --selection_metric composite \
  --robust_eval_every 1 --robust_monitor_samples 200 \
  --robust_monitor_variants jpeg_q50,blur_s1.0,noise_0.03 \
  --checkpoint_dir /kaggle/working/deepfake/checkpoints \
  --plots_dir /kaggle/working/deepfake/reports \
  --tracking_uri /kaggle/working/deepfake/mlruns \
  --variant phase1_consistency --run_name p1c_consistency_<date>
```

### P1-D — soup the best runs

```bash
python src/soup.py \
  --checkpoints \
    checkpoints/runs/p1b_ema_robust_<date>/best.pth \
    checkpoints/runs/p1b_ema_robust_<date2>/best.pth \
    checkpoints/runs/p1a_ema_<date>/best.pth \
  --output checkpoints/soup_phase1.pth \
  --report reports/soup_phase1.json
```

### Evaluate the candidate and the soup

```bash
python src/evaluate.py --weights checkpoints/soup_phase1.pth --split val --tta --no_amp
python src/robustness.py --weights checkpoints/soup_phase1.pth \
  --output_dir reports/soup_robustness --tta --no_amp
```

### Validation precision and EMA safety

- The **EMA candidate is validated in FP32**, and the per-epoch robustness monitor
  also runs in FP32, matching the submitted inference recipe. This avoids rare fp16
  overflow from BatchNorm statistics that were just recalibrated on a clean subset;
  the raw model still uses AMP during validation for speed.
- If EMA validation ever yields non-finite probabilities, training logs a warning and
  falls back to the raw model for that epoch instead of crashing.
- `best.pth` records `checkpoint_type` (`ema`/`raw`), `selection_metric`,
  `selection_score`, `clean_auc`, `degraded_aucs` and `monitor_variants`, so the
  selected checkpoint is auditable.
- Smoke-tested locally with ResNet-50, 8 train / 8 val, 1 epoch, both
  `--ema_bn_batches 1` (worst case) and the production `--ema_bn_batches 100`:
  EMA validation, composite selection, the robustness monitor and checkpoint save
  all complete without NaN.

Expected runtime: ~20–30 min per run on a T4 (ResNet-50, 15 epochs, 5,600 images,
plus EMA validation and the 3-variant monitor). Phase 1 budget: 3–5 runs,
**~2–3 GPU hours** (leave margin for Kaggle restarts).

## 5. Outputs to download

After each run, download:

- `checkpoints/runs/<run_name>/best.pth` (required)
- `checkpoints/runs/<run_name>/config.yaml`
- `checkpoints/runs/<run_name>/val_predictions.csv`
- `reports/<run_name>/training_curves.png`, `roc_curve.png`, `confusion_matrix.png`
- `logs/train_<run_name>.out.log` and `.err.log`
- `mlruns/` (optional; a metrics CSV is enough)
- A `run_summary.json` with environment info and best metrics

`last.pth` includes optimizer state (~282 MB); download it only if you need to resume.

## 6. Bring the results back locally

Place the downloaded files under the same relative paths in the local project:

```
checkpoints/runs/<run_name>/best.pth
checkpoints/runs/<run_name>/config.yaml
checkpoints/runs/<run_name>/val_predictions.csv
reports/<run_name>/*.png
logs/train_<run_name>.out.log
```

Then tell me the run name and paths. I will:

1. Verify the checkpoint loads with the local `src/model.py`.
2. Recompute validation AUC/F1/accuracy on the same split.
3. Compare against the Phase 0 baseline (AUC 0.8923, F1 0.8636).
4. Check the config matches the intended recipe.
5. Recommend keep/reject and update the score estimate.

## 7. Rules and budget

- **Single GPU only** in Phase 1; do not introduce multi-GPU complexity.
- **No pseudo-labeling** and **no external training data**; ImageNet weights only.
- Kaggle sessions can time out: save a checkpoint every epoch and keep the notebook
  output small.
- Do not run expensive training locally; local use is limited to smoke tests.
- Phase 1 GPU budget: ~2–3 hours of the ~30 hours/week; keep the rest for Phase 2.
