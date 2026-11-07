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

## 4. Phase 1 training command (preview)

The exact Phase 1 code (EMA, robust augmentation, composite model selection) will be
delivered after Phase 0 approval. The command shape will be:

```bash
python src/train.py \
  --data_dir /kaggle/input/deepfake-hackathon/hackathon_dataset \
  --arch resnet50 --backend torchvision \
  --epochs 15 --batch_size 64 --lr 2e-4 --weight_decay 1e-2 \
  --dropout 0.4 --label_smoothing 0.05 \
  --img_size 224 --eval_resize 224 --num_workers 4 \
  --scheduler cosine --warmup_epochs 1 --early_stopping_patience 4 \
  --checkpoint_dir /kaggle/working/deepfake/checkpoints \
  --plots_dir /kaggle/working/deepfake/reports \
  --tracking_uri /kaggle/working/deepfake/mlruns \
  --variant phase1_ema_robust --run_name resnet50_phase1_<date>
```

Expected runtime: ~15–25 min per run on a T4 (ResNet-50, 15 epochs, 5,600 images).
Phase 1 budget: 3–5 runs, ~2–3 GPU hours (leave margin for Kaggle restarts).

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
