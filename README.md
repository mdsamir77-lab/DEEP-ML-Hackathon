# Deepfake Detection — Hackathon Submission

Binary face-image classifier that labels a crop as **real (0)** or **deepfake (1)** and is scored by
**AUC-ROC** on the predicted P(fake).

| | |
|---|---|
| **Submitted model** | Phase 2 weighted probability ensemble: ResNet-50 (0.40) + ConvNeXt-Tiny (0.35) + EfficientNet-B3 (0.25); Swin-T was evaluated and has weight 0.00. Manifest: `checkpoints/ensemble_phase2.json`. Tuned holdout AUC **0.901004**: the weights were fitted and scored on the same 1,400 validation examples, so this may overstate generalization and is not an independent OOF estimate. Phase 1 3-way ResNet-50 soup (`checkpoints/best_model.pth`, AUC 0.896235) is a fallback. |
| **Input** | 224 × 224 RGB, ImageNet normalisation |
| **Phase 2 training** | `BCEWithLogitsLoss` (no label smoothing), `AdamW`, lr `3e-4`, weight decay `1e-4`, batch 32, max 12 epochs, EMA decay 0.999, robust augmentation. |
| **Validation protocol** | One stratified 80/20 holdout (5,600/1,400), seed 42; the saved Phase 2 outputs contain fold 0 only. Member predictions are held out from their own training, but the ensemble weights were fitted and scored on those same 1,400 examples. Not five-fold CV or an independent OOF ensemble estimate. |
| **Primary metric** | Validation ROC-AUC (see [`evaluation_metric.md`](evaluation_metric.md)) |
| **Tracking** | MLflow, experiment `Deepfake Detection`, store `./mlruns/` |
| **Submitted inference** | Horizontal-flip TTA **on**, **FP32**, deterministic CUDA settings. `--amp` and `--no_tta` are explicit opt-ins; default inference is FP32, not AMP. |
| **Inference budget** | Recorded Phase 2 run: 3,000 test images in **146.61 s** (**20.46 img/s**, FP32 + flip TTA; `predictions.summary.json`). A local RTX 3050 replay took **204 s**. Both are under the 5-minute efficiency limit. Phase 1 fallback: **52 s** (historical T4 run). |

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

Historical Phase 0 training details (`src/train.py`):

* `BCEWithLogitsLoss` with label smoothing `0.05`.
* `AdamW`, `lr=2e-4`, `weight_decay=1e-4`, no weight decay on norms/biases.
* Cosine annealing with a 1-epoch linear warm-up, gradient clipping at 1.0, AMP.

Phase 2 used a different recipe: lr `3e-4`, weight decay `1e-4`, label smoothing `0.0`, batch 32, up to 12 epochs, early stopping patience 6, EMA (`decay=0.999`), robust augmentation with degradation probability 0.4 from epoch 3, and five-variant/600-image robustness monitoring for checkpoint selection. Best checkpoints were selected by a composite score (`selection_clean_weight=0.5`), not clean AUC alone. Exact configs are under `kaggle_output_phase2/checkpoints/runs/`.

## 4. Results

<!-- RESULTS_START -->
Held-out validation split: **1,400 images** (20 %, stratified, seed 42). The Phase 0/1/2 rows use
this same split, but the Phase 2 ensemble weights were selected on it too; its score is therefore a
tuned single-split estimate, not an independent comparison or cross-validation result. Phase 0
numbers come from `src/evaluate.py`; Phase 1 rows are Kaggle validation results, and the soup row was
reproduced locally (AUC 0.8962353872779518). Phase 2 member rows use the saved candidate reports in
`kaggle_output_phase2/reports/phase2/oof/`; the ensemble's accuracy/precision/recall/F1 are computed
from the stored weighted probabilities at threshold 0.5. `TTA` = horizontal-flip test-time
augmentation (on for every row from #2 onward).

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
| 9 | Phase 1 3-way soup of #6 + #7 + #8, flip TTA (single-model fallback) | 0.8962 | 0.8564 | 0.8567 | 0.8580 | 0.8573 | 0.398 |
| 10 | Phase 2 ResNet-50, EMA + robust aug, 12 ep (best = epoch 5) | 0.8969 | 0.8636 | 0.8691 | 0.8580 | 0.8635 | 0.413 |
| 11 | Phase 2 ConvNeXt-Tiny, same recipe, 12 ep (best = epoch 4) | 0.8970 | 0.8507 | 0.8825 | 0.8111 | 0.8453 | 0.328 |
| 12 | Phase 2 EfficientNet-B3, same recipe, 12 ep (best = epoch 4) | 0.8905 | 0.8443 | 0.8481 | 0.8409 | 0.8445 | 0.567 |
| 13 | Phase 2 Swin-T, same recipe, 12 ep (best = epoch 9) | 0.8936 | 0.8357 | 0.8680 | 0.7940 | 0.8294 | 0.296 |
| 14 | **Phase 2 weighted probability ensemble of #10–#13 (0.40 / 0.35 / 0.25 / 0.00), flip TTA → submitted model** | **0.9010** | **0.8643** | **0.8791** | 0.8466 | **0.8625** | 0.5 |

* **Historical Phase 0 augmentation ablation (#3 vs #5):** under that Phase 0 recipe, disabling
  augmentation reduced AUC from 0.8848 to 0.8797. This does not establish an ablation result for the
  distinct Phase 2 robust-augmentation recipe.
* **Phase 0 rejected variant:** a 3-way soup (adding the no-aug model) scored 0.8913 AUC / 0.8471
  acc — slightly worse than the 2-way soup, so it was not shipped (`checkpoints/soup3.pth`).
* **Historical Phase 1 fallback:** `checkpoints/best_model.pth` is the 3-way soup of the three Phase 1
  EMA runs (#6 + #7 + #8), with AUC 0.896235 on the shared random split. It improved by 0.0039 over
  the Phase 0 soup (0.8923). Its predictions SHA-256 was `c81a551c…`; it is not the current submitted
  ensemble. The Phase 0 checkpoint remains at `checkpoints/best_model_phase0.pth`.
* **Phase 1 robustness (Kaggle, 17 variants):** composite **0.8950**, proxy **15.80/20**; clean
  0.8962, worst `blur_s2.0` 0.8833, best `downscale_0.75` 0.8983. Every variant is within 0.013 AUC
  of clean. Full table and analysis: [`docs/phase1_report.md`](docs/phase1_report.md).
* **Phase 2 submitted ensemble (`kaggle_output_phase2/`):** four `torchvision` backbones used a
  shared recipe (224px, batch 32, lr `3e-4`, weight decay `1e-4`, up to 12 epochs, EMA 0.999, AMP,
  robust augmentation with `degradation_p=0.4` from epoch 3, composite checkpoint selection, flip TTA).
  Saved validation AUCs: ResNet-50 **0.896947**, ConvNeXt-Tiny **0.896972**, EfficientNet-B3
  **0.890503**, Swin-T **0.893630**. Fitting weights in steps of 0.05 gave **0.40 / 0.35 / 0.25 /
  0.00**; Swin-T is an evaluated candidate with zero final weight. On these same predictions, the
  weighted probability ensemble scored **0.901004** vs **0.900790** for equal weights (+0.000214).
  This comparison is tuned and scored on the same holdout, so it does not show an independently
  validated benefit from weight fitting. It is +0.004032 vs the best single member on this split.
  See [`docs/phase2_report.md`](docs/phase2_report.md) for the artifact-by-artifact account.
* **Phase 2 validation provenance:** each saved report has `split=val`, `fold=0`, `folds=5`,
  `split_seed=42`, and 1,400 examples; the four reports use the same labels. Only fold 0 is present.
  Although files live under `reports/phase2/oof/` and the fit JSON calls its result `mean_oof_auc`,
  this is not a five-fold pooled score. Member predictions are held out from their own training,
  but the ensemble weights were fitted and evaluated on these same 1,400 examples. Therefore 0.901004
  is a tuned single-split result, not an independent/generalization estimate. The Phase 1 soup's
  0.896235 is also a single-split result, although its weights were not fitted as an ensemble here.
* **Plots / diagnostics:** Phase 0 in `reports/eval_soup/`; Phase 1 in
  `kagggle_out_phase_01/reports/eval_checkpoints_val/` and `.../soup_robustness/`; Phase 2 fit and
  fold-0 validation JSONs in `kaggle_output_phase2/reports/phase2/`. A concise Phase 2 artifact-based
  report is [`docs/phase2_report.md`](docs/phase2_report.md).
* **Phase 1 test set (3,000 images, flip TTA + FP32, T4) — fallback model:** **52.0 s** total →
  **57.7 images/s**, 49.8 % predicted fake, mean confidence 0.491, **2,999 unique confidence
  values**, 0 load failures, SHA-256 `c81a551c…`. This is the *single-model fallback* CSV; it is
  **not** the submitted `predictions.csv`.
* **Phase 2 submitted test set (`kaggle_output_phase2/predictions_phase2.csv`):** 3,000 rows,
  0 load failures, SHA-256 **`e7f3fc55…`** — byte-identical to the submitted `predictions.csv`.
  Recorded run: **146.6 s** (20.5 img/s), 47.6 % predicted fake, mean confidence 0.475,
  **2,824 unique confidence values**. `predictions.summary.json` records
  `arch = ensemble:resnet50+convnext_tiny+efficientnet_b3+swin_t`, `tta = true`,
  `amp = false`, `deterministic = true`, `num_load_failures = 0`.
* **Cross-hardware reproducibility (Phase 2 ensemble):** a local FP32 replay on the RTX 3050
  reproduced **2,999 / 3,000** labels (99.967 % agreement) with max
  **|Δconfidence| = 2.48e-03** and took **204 s** (14.7 img/s); exactly **one** image near the
  0.5 boundary flips (`0.4999` vs `0.5004`). Per-row float noise from cuDNN reduction order
  means the CSV is *not* bit-identical across GPU architectures — see §13.
  The Phase 0 FP32/AMP comparison (2,998/3,000 labels, Spearman 0.999994) remains in
  `docs/phase0_report.md`.
* **Phase 2 training runs:** all four candidate architectures used seed/split seed 42, `fold=0`,
  batch 32, up to 12 epochs, AMP training, EMA, robust augmentation, and composite checkpoint
  selection. Best epochs were ResNet-50 5, ConvNeXt-Tiny 4, EfficientNet-B3 4, and Swin-T 9.
  The recorded inference runtime below is separate from training time.
* **Experiment tracking:** Phase 2 runs are recorded in `mlruns/` (experiment `Deepfake Detection`)
  with parameters, per-epoch metrics, configs and artifacts; Phase 0/1 results are historical runs.

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

**Historical Phase 1 fallback:** the 3-way soup was generated on a T4 with FP32 + flip TTA: 3,000
images in **52.0 s** (SHA-256 prefix `c81a551c…`). Its local held-out validation AUC reproduced as
**0.8962353873**; a four-variant robustness subset also matched the Phase 1 report.

**Phase 2 submitted predictions:** `predictions.csv` is the weighted probability ensemble (ResNet-50
0.40 + ConvNeXt-Tiny 0.35 + EfficientNet-B3 0.25; Swin-T evaluated at 0.00). The recorded 3,000-image
run took **146.61 s**, FP32 with flip TTA and deterministic settings; its output SHA-256 is
`e7f3fc5564bfa0bbde7d590b85b8013000da0fd2d9f0bb8d0463901446e21ad3`. The local held-out validation
AUC is **0.9010039103**, but it is a tuned, single-split estimate (see the validation caveat below).
The FP32/AMP table above remains a Phase 0 experiment, not a Phase 2 result.
<!-- RESULTS_END -->

## 5. Design justification / ablation notes

The choices below summarize Phase 0/1 experiments unless explicitly identified as the current Phase 2
submission. Do not read the historical ResNet-only rows as the Phase 2 recipe.

| decision | alternative | rationale / phase |
|----------|-------------|------------------|
| Phase 2 ensemble: ResNet-50 + ConvNeXt-Tiny + EfficientNet-B3 | single backbone | Complementary models improved held-out AUC from 0.89697 (best individual) to 0.901004; weights were fitted on the same split, so this gain is exploratory and likely optimistic. |
| Single logit + sigmoid | 2-way softmax | AUC-ROC consumes the single P(fake) confidence; the scalar head directly matches the submission contract. |
| Robust Phase 2 augmentation (degradation probability 0.4 from epoch 3) | no augmentation | The Phase 2 configs used robust augmentation and five-variant monitoring; Phase 0's separate no-augmentation ablation scored 0.8797 vs 0.8848 under that Phase 0 recipe. |
| Phase 2 BCE with hard 0/1 targets (`label_smoothing=0`) | label smoothing | This is what the Phase 2 run configs record; smoothing 0.05 belongs to earlier training experiments. |
| `AdamW`, dropout 0.4, weight decay `1e-4` | no regularisation | Exact Phase 2 run configs; 5,600 training images are limited relative to backbone capacity. |
| Threshold 0.5 for `label`, raw probabilities for `confidence` | tuned submission threshold | AUC is threshold-free; 0.5 is the documented output convention, not an AUC optimization. |
| Horizontal-flip TTA at inference | single view | The submitted CSV used TTA. It approximately doubles forward passes but remained within the measured 5-minute inference budget. |
| **FP32 inference**; AMP training | AMP inference | The submitted Phase 2 run used FP32 inference (146.61 s, 3,000 images). The Phase 0 AMP/FP32 comparison is historical and does not establish Phase 2 cross-GPU identity. |

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
│  ├─ best_model.pth             # Phase 1 3-way soup (single-model fallback; = best_model_phase1.pth)
│  ├─ best_model_phase1.pth      # same bytes as best_model.pth
│  ├─ best_model_phase0.pth      # Phase 0 2-way soup (fallback)
│  ├─ soup.pth                   # Phase 0 soup (historical)
│  ├─ soup3.pth                  # Phase 0 rejected 3-way soup
│  ├─ ensemble_phase2.json       # Phase 2 ensemble manifest (submitted weights, 4 members)
│  └─ runs/<run_name>/           # parent best.pth, last.pth, config.yaml, val_predictions.csv
├─ reports/<run_name>/           # training_curves.png, roc_curve.png, confusion_matrix.png
├─ reports/phase0/               # baseline verification, robustness, audit evidence
├─ reports/phase1_kaggle_verify/ # local reproduction of the Kaggle soup metrics
├─ reports/phase2/               # Phase 2 ensemble fit (mirror of the Kaggle output)
├─ docs/                         # metric notes, Phase 0/1/2 reports, historical Kaggle handoff
├─ kagggle_out_phase_01/         # Kaggle Phase 1 outputs (checkpoint, predictions, reports)
├─ kaggle_output_phase2/         # Kaggle Phase 2 outputs: 4 member checkpoints, fold-0 validation
│                                #   reports (in an `oof/` folder), manifest, predictions CSV
├─ mlruns/                       # MLflow tracking store
├─ predictions.csv               # Phase 2 ensemble T4-generated submission (3,000 test images)
├─ predictions.summary.json      # run metadata for predictions.csv (hash, timing, members)
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
# Phase 2 Submitted Recipe (Multi-Architecture Ensemble, held-out val AUC = 0.9010):
python inference_script.py --test_dir ./hackathon_dataset/test/images/ \
    --weights ./checkpoints/ensemble_phase2.json --output predictions.csv

# Single-model fallback (Phase 1 3-way ResNet soup, AUC = 0.8962):
# python inference_script.py --test_dir ./hackathon_dataset/test/images/ \
#     --weights ./checkpoints/best_model.pth --output predictions.csv

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
  --test_dir ./hackathon_dataset/test/images/ \
  --weights ./checkpoints/ensemble_phase2.json \
  --output predictions.csv
```

* Loads the checkpoint (`state_dict` + `arch` + `backend` + `config`), calls `model.eval()` and runs
  inside `torch.inference_mode()`.
* **Defaults reproduce the submitted recipe:** horizontal-flip TTA is **on**, CUDA inference is
  **FP32**, and deterministic cuDNN/cuBLAS settings are enabled (`cudnn.deterministic=True`,
  `benchmark=False`, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). `--amp` and `--no_tta` are
  explicit opt-ins. The README's documented command therefore needs no extra flags.
* Batched (default 64). Deterministic settings support repeatable runs on a fixed software/hardware
  stack; they do **not** guarantee byte-identical predictions across GPU architectures. The recorded
  Phase 2 CSV hash is `e7f3fc55…`; the T4-to-RTX 3050 replay differed on one thresholded label.
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

The 6.07 % train/validation near-duplicate leakage may make random-split estimates optimistic.
No pHash-grouped validation number is available for Phase 1 or Phase 2, so the size of this effect
and transfer of relative model rankings are unknown. The 7.93 % test-train match is descriptive only;
test labels are sealed.

### 11.3 Design choices that support robustness

* The stratified split preserves the **training-label prior** (3,478 real / 3,522 fake) in both
  subsets. The test labels are sealed, so no claim is made about the test class prior.
* Phase 2 used robust augmentation (including degradation injection from epoch 3) and a five-variant,
  600-image robustness monitor as part of checkpoint selection. The Phase 2 configs record composite
  selection (`selection_clean_weight=0.5`), not selection by clean AUC alone.
* The final ensemble weights were additionally fitted on the same 1,400-image validation split used to
  report AUC. This tuning reuse is a limitation; the result should not be interpreted as an unbiased
  estimate of future or hidden-test performance.
* Horizontal-flip TTA averages two views, reducing prediction variance on unseen variants; it is on
  by default in `inference_script.py` (the submitted recipe) and documented with `--tta` for
  evaluation.
* The deterministic eval transform (`Resize → CenterCrop → Normalize`) is shared by validation,
  evaluation and inference, so there is no train/test transform drift.

## 12. Submission checklist

### 12.1 Required files in the archive

- [ ] `checkpoints/ensemble_phase2.json` and its four referenced `checkpoints/runs/p2_*_s42/best.pth` files — **359,308,819 bytes total** (342.5 MiB; includes the zero-weight Swin-T manifest member). Required to run the manifest.
- [ ] `checkpoints/best_model.pth` — Phase 1 3-way ResNet soup checkpoint (~94 MB, single-model fallback).
- [ ] `inference_script.py`, `src/`, `README.md`, `evaluation_metric.md`, `technical_report.md`, `requirements.txt`, and `requirements-inference.txt`.
- [ ] `predictions.csv` and `predictions.summary.json`.
- [ ] `reports/gradcam/gradcam_summary.png`, `reports/gradcam/gradcam_findings.md`, and `reports/gradcam/training_curves.png` — existing Grad-CAM/plot artifacts (bonus awards are at organizer discretion).

> **Packaging warning.** `.gitignore` excludes `checkpoints/` and `*.pth`, so a plain `git archive`
> or a clone will **not** contain the models. Before delivery, assemble and verify an archive that
> includes the Phase 2 manifest and every referenced checkpoint; a manifest without its member weights
> is not a runnable submission.

### 12.2 Final generation command

```sh
# Submitted Phase 2 ensemble (AUC = 0.9010):
python inference_script.py \
  --test_dir ./hackathon_dataset/test/images/ \
  --weights ./checkpoints/ensemble_phase2.json \
  --output predictions.csv

# Single-model fallback (Phase 1 3-way soup, AUC = 0.8962):
# python inference_script.py \
#   --test_dir ./hackathon_dataset/test/images/ \
#   --weights ./checkpoints/best_model.pth \
#   --output predictions.csv
```

The submitted recipe is flip TTA + FP32 inference on CUDA (`--amp` is an opt-in). Do not add
`--no_tta` if reproducing the submitted CSV. The command also writes `predictions.summary.json`.

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
    --weights ./checkpoints/ensemble_phase2.json --output run1.csv
python inference_script.py --test_dir ./hackathon_dataset/test/images/ \
    --weights ./checkpoints/ensemble_phase2.json --output run2.csv
python -c "import hashlib; a=hashlib.sha256(open('run1.csv','rb').read()).hexdigest(); b=hashlib.sha256(open('run2.csv','rb').read()).hexdigest(); print(a, b, a == b); assert a == b"
```

- [ ] `predictions.summary.json` reports `"tta": true`, `"amp": false`, `"deterministic": true`,
      `"num_images": 3000`, `"num_load_failures": 0` and an `output_sha256` equal to the file hash.
- [ ] `python src/evaluate.py --weights checkpoints/ensemble_phase2.json --split val --tta --no_amp`
      evaluates the submitted Phase 2 ensemble and writes prediction/metric artifacts. The reported
      0.901004 AUC is a tuned single-holdout value: ensemble weights were fitted on these same
      validation examples, so this command does not provide an independent generalization estimate.
- [ ] Optional fallback check: replace the manifest with `checkpoints/best_model.pth`; that Phase 1
      soup historically scored about 0.8962 AUC on this split.

### 12.5 Environment sanity

- [ ] `python -m compileall -q src inference_script.py` exits 0.
- [ ] `ruff check inference_script.py src` passes.
- [ ] `python src/dataset.py --num_workers 0` prints 5,600 train / 1,400 val and one batch.
- [ ] If you see `OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` (see §7).

## 13. Limitations and Notes

* **Phase 2 result:** The submitted probability ensemble scored **0.901004 AUC** on the one shared
  1,400-image holdout. Weights were fitted on this same holdout; this is not a cross-validated or
  unbiased generalization estimate. The full 17-variant robustness table is for the Phase 1 soup,
  not the Phase 2 ensemble.
* **Cross-hardware reproducibility:** The saved Phase 2 CSV was generated on a T4. In the recorded
  RTX 3050 replay, one of 3,000 thresholded labels changed. Deterministic inference settings support
  reproducibility on a fixed stack, not byte-identical output across GPU architectures.
* **Framing Assumption:** Training data is pre-cropped 224 × 224 faces, so the model expects that framing; in-the-wild full frames would require a face detector (e.g., MTCNN or RetinaFace) in front of the classifier.
* **Spatial vs Frequency Domain:** The architecture operates purely in the spatial domain; adding an explicit frequency-domain branch (e.g. 2D-DCT or FFT) could provide additional discriminative signal for subtle GAN/diffusion checkerboard artifacts.
* **Calibration:** Probabilities are used as ranking scores for AUC-ROC, not claimed to be calibrated
  empirical posteriors. Phase 2 used no label smoothing; Phase 0/1 smoothing settings are historical.

