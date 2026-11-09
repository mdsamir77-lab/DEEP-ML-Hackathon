# Deepfake Detection — Hackathon Submission

Binary face-image classifier that labels a crop as **real (0)** or **deepfake (1)** and is scored by
**AUC-ROC** on the predicted P(fake).

| | |
|---|---|
| **Backbone** | ResNet-50 (ImageNet-1k V2 weights, `torchvision`) + `Dropout → Linear(2048, 1)`; submitted weights = **3-way soup of Phase 1 EMA fine-tunes** (Phase 0 soup retained as `best_model_phase0.pth`) |
| **Input** | 224 × 224 RGB, ImageNet normalisation |
| **Loss / optimiser** | `BCEWithLogitsLoss` (label smoothing 0.05 / 0.1) + `AdamW` + cosine schedule with 1-epoch warm-up |
| **Primary metric** | Validation ROC-AUC (see [`evaluation_metric.md`](evaluation_metric.md)) |
| **Tracking** | MLflow, experiment `Deepfake Detection`, store `./mlruns/` |
| **Submitted inference** | flip TTA **on** + **FP32** + deterministic CUDA settings — the defaults of `inference_script.py`; `--amp` / `--no_tta` are explicit opt-ins |
| **Inference budget** | 3,000 test images in **52 s on a T4** (57.7 img/s, FP32 + flip TTA) and **73–84 s on the RTX 3050** — far below the 5-minute limit. AMP (`--amp`) remains available for speed |

## 1. Problem and constraints

* 7,000 labelled training crops (3,478 real / 3,522 fake) and 3,000 unlabelled test crops.
* Only the provided dataset may be used for training; **standard ImageNet pre-trained backbones are
  allowed** and are the only external weights used here.
* Scores: 50 pts AUC-ROC, 20 pts inference speed, 20 pts robustness/generalisation, 10 pts
  documentation.

## 2. Dataset findings (EDA, `check_data.ipynb`)

| property | value |
|----------|-------|
| training images | 7,000 (`train/real` 3,478, `train/fake` 3,522 → 49.7 / 50.3 %) |
| count integrity | 7,000 images under `train/` = 7,000 `train_metadata.csv` rows; per-class on-disk counts match the labels; 0 missing IDs |
| test images | 3,000, UUID file names, unlabelled; stems match `sample_submission.csv` exactly |
| resolution | every real and fake image is exactly **224 × 224** |
| format | JPEG, RGB |
| duplicates | 0 duplicate MD5 groups across the **whole** training tree (0 cross-class) |
| metadata | `train_metadata.csv` = `image_id,label`; `sample_submission.csv` = `image_id,label,confidence` |

Because every crop is already a pre-cropped, square 224 × 224 face, **no face detector / MTCNN stage
is required** (the blueprint's "face validation" step resolves to "images are pre-cropped").

## 3. Pipeline

```
train_metadata.csv ──► build_train_dataframe ──► stratified 80/20 split (seed 42)
                                                   │
                    ┌──────────────────────────────┴───────────────────────────┐
                    ▼                                                          ▼
      mild augmentation (train)                                deterministic (val/test)
      RandomResizedCrop(224, scale 0.8–1.0)                    Resize(224) → CenterCrop(224)
      HFlip p=0.5, ±10° rotation p=0.5                         ToTensor
      ColorJitter(0.2/0.2/0.1/0.02) p=0.5                      Normalize(ImageNet)
      ToTensor, Normalize(ImageNet)
```

Augmentation is deliberately **mild**: blur, heavy noise or strong geometric warps destroy the
blending-boundary and frequency cues that separate real from generated faces.

Training details (`src/train.py`):

* `BCEWithLogitsLoss` with label smoothing `0.05` (prevents over-confident logits → better ranking).
* `AdamW`, `lr=2e-4`, `weight_decay=1e-4`, no weight decay on norms/biases
  (`build_optimizer_param_groups`).
* Cosine annealing with a 1-epoch linear warm-up, gradient clipping at 1.0, AMP (`torch.amp`).
* Batch size 32, 8 epochs max, early stopping (patience 3) on validation AUC.
* Model selection is **strictly by validation AUC**, never by accuracy or loss.

## 4. Results

<!-- RESULTS_START -->
Held-out validation split: **1,400 images** (20 %, stratified, seed 42). Phase 0 numbers come from
`src/evaluate.py`; Phase 1 rows are the Kaggle training validation on the same split, and the soup
row was reproduced exactly by the local `evaluate.py` (AUC 0.8962353872779518). `TTA` =
horizontal-flip test-time augmentation.

| # | recipe | val AUC | acc @0.5 | precision | recall | F1 | best thr. |
|---|--------|---------|----------|-----------|--------|----|-----------|
| 1 | baseline: lr 2e-4, dropout 0.4, wd 1e-4, 8 epochs (best = epoch 7) | 0.8806 | 0.8471 | 0.8441 | 0.8537 | 0.8489 | — |
| 2 | baseline + flip TTA | 0.8846 | 0.8464 | 0.8410 | 0.8565 | 0.8487 | 0.413 |
| 3 | regularised: lr 1e-4, dropout 0.5, wd 1e-2, LS 0.1, 12 epochs, val-TTA (best = epoch 4) | 0.8848 | 0.8221 | 0.7785 | 0.9034 | 0.8363 | 0.675 |
| 4 | Phase 0 baseline: checkpoint soup of #1 + #3, flip TTA | 0.8923 | 0.8586 | 0.8382 | 0.8906 | 0.8636 | 0.507 |
| 5 | ablation — same recipe as #3 but `--no_aug` (deterministic transform), 6 epochs (best = epoch 2) | 0.8797 | 0.8200 | 0.7943 | 0.8665 | 0.8288 | — |
| 6 | Phase 1 P1-A: ResNet-50, EMA + mild aug, 15 epochs, clean-AUC selection | 0.8905 | 0.8493 | 0.8438 | 0.8594 | 0.8515 | 0.512 |
| 7 | Phase 1 P1-B: EMA + robust aug (p=0.4), composite selection | 0.8903 | 0.8479 | 0.8453 | 0.8537 | 0.8495 | 0.631 |
| 8 | Phase 1 P1-B seed 3: EMA + robust aug | 0.8893 | 0.8521 | 0.8535 | 0.8523 | 0.8529 | 0.654 |
| 9 | **Phase 1 3-way soup of #6 + #7 + #8, flip TTA → submitted model** | **0.8962** | **0.8564** | 0.8567 | 0.8580 | **0.8573** | 0.398 |

* **Ablation (augmentation, #3 vs #5):** with an otherwise identical recipe, disabling augmentation
  costs **−0.51 AUC** (0.8848 → 0.8797) and peaks 2 epochs earlier (it memorises the training set
  faster). Mild augmentation is therefore kept — but it stays *mild*, because heavy blur/noise would
  remove the forensic cues entirely.
* **Phase 0 rejected variant:** a 3-way soup (adding the no-aug model) scored 0.8913 AUC / 0.8471
  acc — slightly worse than the 2-way soup, so it was not shipped (`checkpoints/soup3.pth`).
* **Phase 1 submitted model:** `checkpoints/best_model.pth` = 3-way soup of the three Phase 1 EMA
  runs (#6 + #7 + #8). The soup gains **+0.57 AUC points over its best parent** (0.8905 → 0.8962)
  and **+0.39 over the Phase 0 soup** (0.8923 → 0.8962). The individual parents are below the Phase 0
  soup; the soup is the Phase 1 win. The Phase 0 soup is retained as `checkpoints/best_model_phase0.pth`
  for fallback. The Phase 1 soup file SHA-256 is `73770803…`; its state-dict hash `93e5a10b…` matches
  the recomputed hash. Predictions SHA-256 `c81a551c…`.
* **Phase 1 robustness (Kaggle, 17 variants):** composite **0.8950**, proxy **15.80/20**; clean
  0.8962, worst `blur_s2.0` 0.8833, best `downscale_0.75` 0.8983. Every variant is within 0.013 AUC
  of clean. Full table and analysis: [`docs/phase1_report.md`](docs/phase1_report.md).
* **Phase 1 limitation:** clean AUC 0.8962 is still below the Phase 1 target (0.91–0.93). ResNet-50
  has plateaued; Phase 2 (ConvNeXt-T / EfficientNet-B3 / Swin-T + ensemble) is required.
* **Plots / diagnostics:** Phase 0 in `reports/eval_soup/`; Phase 1 in
  `kagggle_out_phase_01/reports/eval_checkpoints_val/` and `.../soup_robustness/`.
* **Test set (3,000 images, flip TTA + FP32, T4):** **52.0 s** total → **57.7 images/s**, 49.8 %
  predicted fake, mean confidence 0.491, **2,999 unique confidence values**, 0 load failures,
  SHA-256 `c81a551c…`. The local RTX 3050 FP32 run is 73–84 s. The Phase 0 FP32/AMP comparison
  (2,998/3,000 labels, Spearman 0.999994) remains in `docs/phase0_report.md`.
  The 51.6 % predicted-fake share is close to the training prior; it is *not* an estimate of the
  sealed test prior.
* **Training cost:** 8 epochs ≈ 6.8 min wall clock on the RTX 3050 (≈50 s/epoch incl. validation),
  AMP enabled, batch 32, 224 × 224. Under sustained GPU load the laptop GPU throttles and epochs can
  take 3–4× longer (run #3 was measured at ~226 s/epoch for that reason); the reported AUC is
  unaffected.
* **Experiment tracking:** both parents and the ablation are stored in `mlruns/` (experiment
  `Deepfake Detection`) with parameters, per-epoch metrics, configs and artifacts.

### Phase 0 model — inference precision: FP32 submitted, AMP optional

The Phase 0 CSV was generated in **FP32** (2,997 distinct confidences for 3,000 images). AMP's fp16
rounding was audited as a possible AUC leak (ties receive half credit in ROC-AUC) and, more
importantly, as a **cross-GPU reproducibility risk**: the CSV was generated on an RTX 3050 while
the evaluator runs a T4. Three repeat runs per precision on the same 1,400-image held-out split:

| precision | val AUC (3 runs) | mean AUC | unique confidences / 1,400 | runtime / 1,400 | peak GPU memory |
|---|---|---|---|---|---|
| AMP (fp16 autocast, `--amp`) | 0.892324 / 0.892322 / 0.892332 | 0.892326 | 889 / 891 / 895 | 34.2–46.5 s | 571 MB |
| **FP32 (submitted)** | 0.892305 / 0.892325 / 0.892307 | **0.892312** | 1,399 / 1,400 / 1,400 | 50.8–62.5 s | 1,157 MB |

* The mean AUC difference is **+1.4e-5**, far inside run-to-run noise (cuDNN benchmark autotuning
  alone moves repeated AUCs by ~2e-5; the standard error of AUC on 1,400 samples is ~0.008). The
  hypothesis that fp16 ties cost AUC is **not supported** on this split: tied confidences are
  overwhelmingly within the same class.
* FP32 is now the submitted recipe because it removes the cross-architecture fp16 difference
  (2/3,000 labels differed between the RTX 3050 AMP and FP32 CSVs; Spearman 0.999994) at **no
  meaningful AUC cost**. FP32 runs 3,000 images in **73–84 s**, still far inside the 5-minute T4
  budget. AMP remains available with `--amp` when speed is preferred.
* The comparison is a **measured validation result**; no hidden-test score is claimed.

Reproduce the final numbers:

```sh
# Phase 0 model (FP32 + flip TTA)
python src/evaluate.py --weights checkpoints/best_model_phase0.pth --split val --tta

# Phase 1 submitted soup (FP32 + flip TTA)
python src/evaluate.py --weights checkpoints/best_model.pth --split val --tta

# AMP comparison (Phase 0 model)
python src/evaluate.py --weights checkpoints/best_model_phase0.pth --split val --tta --amp
```

**Phase 1 update:** the submitted soup was generated on a **T4** with FP32 + flip TTA: 3,000 images
in **52.0 s**, 2,999 unique confidences, predictions SHA-256 `c81a551c…`. The local `evaluate.py`
reproduced the Kaggle validation AUC exactly (**0.8962353872779518**), and the local robustness
subset matched the Kaggle table exactly (clean 0.896235, jpeg_q50 0.894174, blur_s2.0 0.883331,
grayscale 0.890786). The FP32/AMP table above is the Phase 0 model comparison and is retained for
the canary-risk rationale.
<!-- RESULTS_END -->

## 5. Design justification / ablation notes

| decision | alternative | rationale |
|----------|-------------|-----------|
| ResNet-50 (25 M params, ImageNet V2) | EfficientNet-B0 timm (`efficientnet_b0.ra_in1k`) | Both are permitted ImageNet backbones. The timm mirror was only reachable at ~25 kB/s in this environment while the torchvision ResNet-50 weights were already cached, so ResNet-50 was the reproducible choice. The `--backend timm` path remains fully implemented. |
| Single logit + sigmoid | 2-way softmax | AUC-ROC consumes a single P(fake); a scalar head is the most direct formulation and keeps the submission contract trivial. |
| Mild augmentation | heavy augmentation / mixup | Measured ablation (`--no_aug`, same recipe): 0.8797 vs 0.8848 AUC → augmentation helps, and forensic artefacts are high-frequency so it must stay mild. |
| Label smoothing 0.05 | hard 0/1 targets | Keeps the probability ranking informative instead of saturating logits. |
| Dropout 0.4 head + `weight_decay` | no regularisation | 5,600 training crops is small for 25 M parameters; both are cheap over-fitting guards. |
| Threshold 0.5 for `label`, raw probabilities for `confidence` | tuned threshold | AUC is threshold-free, so the label column cannot change the score; 0.5 keeps the output interpretable. |
| Horizontal-flip TTA at inference (default; `--no_tta` to disable) | single view | Cheap variance reduction, ~2× inference cost (still far inside the 5-minute budget). The submitted CSV was generated with TTA, so it is the default. |
| **FP32 submitted** (deterministic CUDA); `--amp` for fp16 autocast | always AMP | AMP matched FP32 on val (0.892326 vs 0.892312), but fp16 results can differ across GPU architectures: 2/3,000 test labels changed between the RTX 3050 AMP and FP32 CSVs. FP32 removes the canary-check risk at 73–80 s / 3,000 images (20/20 efficiency). |

## 6. Repository layout

```
.
├─ hackathon_dataset/            # provided data (train/real, train/fake, test/images, csvs)
├─ src/
│  ├─ dataset.py                 # dataframes, stratified split, transforms, dataloaders
│  ├─ model.py                   # timm + torchvision factories, ImageNet weight loading, checkpoint loader
│  ├─ train.py                   # training loop, MLflow logging, checkpoints, plots
│  ├─ evaluate.py                # held-out evaluation harness (val/train/all, optional TTA)
│  ├─ inference.py               # batched inference → submission CSV
│  ├─ robustness.py              # degradation harness (JPEG/blur/noise/downscale/…)
│  ├─ audit.py                   # pHash leakage + JPEG/EXIF shortcut audit
│  ├─ soup.py                    # weighted state-dict averaging (model soup)
│  └─ utils.py                   # seeding, metrics, plotting, EMA, checkpoint/config I/O
├─ inference_script.py           # hackathon CLI entry point
├─ checkpoints/
│  ├─ best_model.pth             # Phase 1 3-way soup (submitted; = best_model_phase1.pth)
│  ├─ best_model_phase1.pth      # same bytes as best_model.pth
│  ├─ best_model_phase0.pth      # Phase 0 2-way soup (fallback)
│  ├─ soup.pth                   # Phase 0 soup (historical)
│  ├─ soup3.pth                  # Phase 0 rejected 3-way soup
│  └─ runs/<run_name>/           # parent best.pth, last.pth, config.yaml, val_predictions.csv
├─ reports/<run_name>/           # training_curves.png, roc_curve.png, confusion_matrix.png
├─ reports/phase0/               # baseline verification, robustness, audit evidence
├─ reports/phase1_kaggle_verify/ # local reproduction of the Kaggle soup metrics
├─ docs/                         # metric notes, Phase 0/1 reports, Kaggle handoff
├─ kagggle_out_phase_01/         # Kaggle Phase 1 outputs (checkpoint, predictions, reports)
├─ mlruns/                       # MLflow tracking store
├─ predictions.csv               # Phase 1 T4-generated submission for the 3,000 test images
├─ check_data.ipynb              # EDA
├─ claude.md                     # project blueprint (not part of the submission archive)
├─ evaluation_metric.md          # official metric guide (IIIT Bangalore)
├─ requirements.txt              # exact training environment
├─ requirements-inference.txt    # minimal inference environment
└─ .gitignore
```


## 7. Setup

```sh
conda create -n dl_env python=3.11 -y && conda activate dl_env
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Notes for this machine:

* The interpreter that has the stack installed is `D:\anaconda\envs\dl_env\python.exe`
  (the Anaconda `base` env does not contain `timm`).
* If you hit `OMP: Error #15 ... libiomp5md.dll already initialized`, set
  `KMP_DUPLICATE_LIB_OK=TRUE`.
* `torchvision` weight downloads can fail with `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]` when the
  Windows certificate store is corrupt; `src/model.py` therefore streams weights with `requests`
  (certifi bundle) and reads the local torch-hub cache first.
* The same broken cert store makes the `mlflow` **CLI** (`mlflow ui`, `mlflow experiments search`)
  fail on import of `aiohttp`. Training/tracking through the Python API is unaffected; to browse runs
  either repair the cert store or query them programmatically:

  ```python
  from mlflow.tracking import MlflowClient
  client = MlflowClient("file:///C:/Users/mdsam/Desktop/deep_ml_hakathon/mlruns")
  for run in client.search_runs(experiment_ids=["975800176316826289"]):
      print(run.info.run_name, run.data.metrics.get("best_val_auc"))
  ```

## 8. Reproducing the run

```sh
# 1) train (writes mlruns/, checkpoints/runs/<run>, reports/<run>)
python src/train.py --arch resnet50 --backend torchvision --epochs 8 --batch_size 32 \
    --lr 2e-4 --weight_decay 1e-4 --dropout 0.4 --label_smoothing 0.05 \
    --img_size 224 --eval_resize 224 --num_workers 4 --scheduler cosine --warmup_epochs 1 \
    --early_stopping_patience 3 --variant imagenet_pretrained \
    --run_name resnet50_imagenet_pretrained_20261031 --experiment "Deepfake Detection"

# 2) evaluate the submitted Phase 1 soup on the held-out split (FP32 + flip TTA)
python src/evaluate.py --weights checkpoints/best_model.pth --split val --tta
#    expected AUC ~0.8962 (Kaggle and local reproduced exactly)

#    optional AMP comparison on the Phase 0 model
python src/evaluate.py --weights checkpoints/best_model_phase0.pth --split val --tta --amp

# 3) submission - FP32 + flip TTA + deterministic settings are the defaults
python inference_script.py --test_dir ./hackathon_dataset/test/images/ \
    --weights ./checkpoints/best_model.pth --output predictions.csv

# 4) robustness baseline (16 degradations x 1,400 val images; ~6-10 min)
python src/robustness.py --weights checkpoints/best_model.pth \
    --output_dir reports/robustness --tta

# 5) shortcut / near-duplicate audit (~6-10 min)
python src/audit.py --data_dir hackathon_dataset --output_dir reports/audit

# 6) inspect experiments
mlflow ui --backend-store-uri ./mlruns
```

Sanity checks that ship with the code:

```sh
python src/dataset.py --num_workers 0             # prints split sizes and one batch
python src/model.py --arch resnet50 --pretrained  # prints output shape, params, head
```

## 9. Inference contract

`inference_script.py` implements exactly the required CLI:

```sh
python inference_script.py \
  --test_dir ./test/images/ \
  --weights ./model_weights.pt \
  --output predictions.csv
```

* Loads the checkpoint (`state_dict` + `arch` + `backend` + `config`), calls `model.eval()` and runs
  inside `torch.inference_mode()`.
* **Defaults reproduce the submitted recipe:** horizontal-flip TTA is **on**, CUDA inference is
  **FP32**, and deterministic cuDNN/cuBLAS settings are enabled (`cudnn.deterministic=True`,
  `benchmark=False`, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). `--amp` and `--no_tta` are
  explicit opt-ins. The README's documented command therefore needs no extra flags.
* Batched (default 64) and byte-reproducible: two full 3,000-image runs produced the same
  SHA-256 (`c46225fd…`).
* A corrupt/unreadable image gets a gray fallback and still receives a row; `num_load_failures`
  and the failing `image_id`s are recorded in `predictions.summary.json`.
* The written CSV is re-read and validated (columns, row count, unique IDs, ID set vs the test
  directory, confidence range, `label == int(confidence >= threshold)`, sample-submission order)
  before the script exits.
* Writes `image_id,label,confidence` with `confidence = P(fake) ∈ [0, 1]` and `label ∈ {0, 1}`
  (`label = int(confidence >= threshold)`, threshold 0.5 by default).
* If `sample_submission.csv` is found next to the test directory (or passed via `--sample_submission`),
  rows are reordered to match its `image_id` order when the ID sets are identical; otherwise the
  deterministic filename order is kept.
* Also writes `predictions.summary.json` (count, seconds, images/s, device, TTA, precision,
  deterministic flag, seed, unique confidences, mean confidence, row-order flag, checkpoint/output
  SHA-256, load failures, environment).
* CPU fallback via `--cpu`; `--img_size` / `--eval_resize` default to the values stored in the
  checkpoint, so evaluation and inference transforms always match.

## 10. MLflow tracking

* Store `./mlruns/` (`--tracking_uri` accepts a URL, sqlite URI or local directory).
  MLflow ≥ 3.16 blocks the file store by default, so `src/train.py` sets
  `MLFLOW_ALLOW_FILE_STORE=true` before touching the tracking API.
* Experiment `Deepfake Detection`; run name `{architecture}_{variant}_{date}`.
* Logged per epoch: `train_loss`, `val_loss`, `val_auc`, `val_accuracy`, `val_precision`,
  `val_recall`, `val_f1`, `lr` (with `step=epoch`).
* Logged params/tags: architecture, backend, pretrained flag, lr, batch size, epochs, optimiser,
  scheduler, image size, weight decay, dropout, augmentation, seeds, sample counts, param counts.
* Artifacts: `best.pth`, `config.yaml`, `environment.json`, `training_curves.png`, `roc_curve.png`,
  `confusion_matrix.png`, `val_predictions.csv`.

## 11. Robustness / generalisation notes

### 11.1 Measured degradation baseline (Phase 0)

`src/robustness.py` applies 16 deterministic degradations to the 1,400-image held-out split with the
submitted FP32 + flip-TTA recipe. Full results: `reports/phase0/robustness/`.

| Variant | AUC | Δ vs clean |
|---|---:|---:|
| clean | 0.8923 | — |
| jpeg_q90 | 0.8915 | −0.0008 |
| jpeg_q70 | 0.8890 | −0.0033 |
| jpeg_q50 | 0.8911 | −0.0012 |
| jpeg_q30 | 0.8891 | −0.0032 |
| blur_s0.5 | 0.8927 | +0.0004 |
| blur_s1.0 | 0.8877 | −0.0047 |
| blur_s2.0 | 0.8728 | −0.0195 |
| noise_0.01 | 0.8909 | −0.0014 |
| noise_0.03 | 0.8889 | −0.0034 |
| downscale_0.75 | 0.8950 | +0.0027 |
| downscale_0.50 | 0.8900 | −0.0023 |
| brightness_0.8 | 0.8917 | −0.0006 |
| brightness_1.2 | 0.8886 | −0.0037 |
| contrast_0.8 | 0.8916 | −0.0007 |
| contrast_1.2 | 0.8924 | +0.0001 |
| grayscale | 0.8808 | −0.0115 |

* Mean degraded AUC **0.8890**; equal-weight composite **0.8906**; proxy robustness score
  **15.63/20** under the assumption `20 × (AUC − 0.5)/0.5` (the official variant weights are
  undisclosed).
* The soup is **more robust than expected**: JPEG q30, noise, 0.5× downscale and ±20 % lighting
  stay within ~0.004 AUC of clean. The only material drops are **strong blur (σ=2, −0.0195)** and
  **grayscale (−0.0115)**. Phase 1 added blur/JPEG/noise augmentation; see 11.1b.

### 11.1b Phase 1 soup (Kaggle, 17 variants)

Composite **0.8950**, proxy **15.80/20**; clean 0.8962, worst `blur_s2.0` 0.8833, best
`downscale_0.75` 0.8983. Every variant is within **0.013 AUC** of clean. The local robustness subset
reproduced the Kaggle values exactly (clean 0.896235, jpeg_q50 0.894174, blur_s2.0 0.883331,
grayscale 0.890786). Full table: [`docs/phase1_report.md`](docs/phase1_report.md).

### 11.2 Near-duplicate and shortcut audit (Phase 0)

`src/audit.py` (pHash Hamming ≤ 6, all 7,000 train + 3,000 test images):

| Finding | Value | Interpretation |
|---|---:|---|
| Near-duplicate pairs | 295 | sparse; max cluster size 3 |
| Cross-class pairs / clusters | 55 / 53 | possible label noise or genuinely hard real/fake pairs |
| Train/val leakage pairs | 87 | 85 val images (6.07 %) have a train near-duplicate |
| Test images with a train near-duplicate | 238 (7.93 %) | the hidden test may be slightly easier than a clean holdout |
| JPEG quantization tables | 1 table for all 7,000 images | no JPEG-header shortcut |
| EXIF present | 0 % real, 0 % fake | no metadata shortcut |
| Largest class-wise pixel effect | std_r, effect size 0.177 | small; no dominant colour/border shortcut |

The 6.07 % train/val near-duplicate leakage means the local 0.8923 AUC is **mildly optimistic**;
the honest estimate is likely a few tenths of a point lower. The 7.93 % test-train match means the
hidden test contains a similar fraction of near-duplicates, so the *relative* model ranking should
still transfer. Phase 1 will report both the random-split and a pHash-grouped validation number.

### 11.3 Design choices that support robustness

* The stratified split preserves the **training-label prior** (3,478 real / 3,522 fake) in both
  subsets. The test labels are sealed, so no claim is made about the test class prior.
* Mild, artefact-preserving augmentation (small crops, flips, ±10° rotation, subtle colour jitter)
  instead of blur/noise that would teach shortcut features. Phase 1 adds **mild degradation
  augmentation** on top, because the hidden robustness variants include such degradations.
* Early stopping and model selection on **AUC** (a ranking metric) rather than accuracy, so the
  submission keeps a good ordering even if the hidden test prior shifts.
* Horizontal-flip TTA averages two views, reducing prediction variance on unseen variants; it is on
  by default in `inference_script.py` (the submitted recipe) and documented with `--tta` for
  evaluation.
* The deterministic eval transform (`Resize → CenterCrop → Normalize`) is shared by validation,
  evaluation and inference, so there is no train/test transform drift.

## 12. Submission checklist

### 12.1 Required files in the archive

- [ ] `checkpoints/best_model.pth` — the submitted soup checkpoint (~94 MB). **Required.**
- [ ] `inference_script.py`, `src/`, `README.md`, `evaluation_metric.md`, `requirements.txt`.
- [ ] `hackathon_dataset/` is provided by the organisers; include it only if the rules ask for a
      self-contained archive.
- [ ] Optional: `checkpoints/soup.pth` (byte-identical copy), `predictions.csv`,
      `predictions.summary.json`, `reports/`, `mlruns/`.

> **Packaging warning.** `.gitignore` excludes `checkpoints/` and `*.pth`, so a plain `git archive`
> or a clone of the repository will **not** contain the model. Do not remove the ignore rules; copy
> the file into the archive explicitly:
>
> ```powershell
> Compress-Archive -Path inference_script.py, src, checkpoints\best_model.pth, README.md, `
>     evaluation_metric.md, requirements.txt -DestinationPath deepfake_submission.zip -Force
> ```
>
> or force-add it if the submission is a git repository: `git add -f checkpoints/best_model.pth`.
> Verify the archive contains the checkpoint before uploading.

### 12.2 Final generation command

```sh
python inference_script.py \
  --test_dir ./hackathon_dataset/test/images/ \
  --weights ./checkpoints/best_model.pth \
  --output predictions.csv
```

Do **not** add `--no_tta` or `--no_amp`: the defaults are the submitted recipe (flip TTA + AMP on
CUDA, FP32 on CPU). The command also writes `predictions.summary.json`.

### 12.3 CSV validation

- [ ] exactly 3,000 data rows (+1 header);
- [ ] columns exactly `image_id,label,confidence`;
- [ ] `image_id` set (and, with the provided sample submission, row order) equals
      `hackathon_dataset/sample_submission.csv`;
- [ ] no duplicate `image_id`;
- [ ] `confidence` numeric and inside [0, 1];
- [ ] `label` ∈ {0, 1} and equals `int(confidence >= 0.5)`.

```python
import pandas as pd

pred = pd.read_csv("predictions.csv")
sample = pd.read_csv("hackathon_dataset/sample_submission.csv")
assert list(pred.columns) == ["image_id", "label", "confidence"]
assert len(pred) == 3000
assert pred["image_id"].is_unique
assert set(pred["image_id"]) == set(sample["image_id"])
assert pred["confidence"].between(0, 1).all()
assert (pred["label"] == (pred["confidence"] >= 0.5).astype(int)).all()
print("predictions.csv: OK")
```

### 12.4 Reproducibility verification

- [ ] run the generation command twice into different files and compare SHA-256:

```sh
python inference_script.py --test_dir ./hackathon_dataset/test/images/ \
    --weights ./checkpoints/best_model.pth --output run1.csv
python inference_script.py --test_dir ./hackathon_dataset/test/images/ \
    --weights ./checkpoints/best_model.pth --output run2.csv
python -c "import hashlib; a=hashlib.sha256(open('run1.csv','rb').read()).hexdigest(); b=hashlib.sha256(open('run2.csv','rb').read()).hexdigest(); print(a, b, a == b); assert a == b"
```

- [ ] `predictions.summary.json` reports `"tta": true`, `"amp": false`, `"deterministic": true`,
      `"num_images": 3000`, `"num_load_failures": 0` and an `output_sha256` equal to the file hash.
- [ ] `python src/evaluate.py --weights checkpoints/best_model.pth --split val --tta` reproduces the
      reported ~0.8962 validation AUC and writes both `predictions.csv` and
      `predictions_detailed.csv`.

### 12.5 Environment sanity

- [ ] `python -m compileall -q src inference_script.py` exits 0.
- [ ] `ruff check inference_script.py src` passes.
- [ ] `python src/dataset.py --num_workers 0` prints 5,600 train / 1,400 val and one batch.
- [ ] If you see `OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` (see §7).

## 13. Limitations

* Phase 1 is a 3-way soup of ResNet-50 EMA runs; still a single architecture family. No
  diverse-architecture ensemble yet (Phase 2).
* Clean AUC 0.8962 is below the Phase 1 target (0.91–0.93); ResNet-50 has plateaued. Phase 2
  stronger backbones (ConvNeXt-T / EfficientNet-B3 / Swin-T) are required.
* Training data is pre-cropped 224 × 224 faces, so the model expects that framing; full frames would
  need a face detector in front of it.
* No frequency-domain branch (e.g. FFT/DCT) — a known strong signal for GAN/diffusion artefacts,
  left as future work.
* Calibration is not enforced: `confidence` is a ranking score, not a calibrated probability.

