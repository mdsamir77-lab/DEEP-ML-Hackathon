# Phase 2 report — multi-architecture probability ensemble

**Status:** Phase 2 runs and ensemble generation are complete. This report is based on the saved outputs in [`kaggle_output_phase2/`](../kaggle_output_phase2/). The current inference recipe, submission details, and cross-phase context are summarized in the [README](../README.md) and [technical report](../technical_report.md).

## 1. Result at a glance

Four ImageNet-pretrained `torchvision` models were trained with the same Phase 2 recipe. Their saved validation predictions were combined by a weighted mean of probabilities (`mean_prob`). The final manifest assigns Swin-T zero weight; the three nonzero contributors are ResNet-50, ConvNeXt-Tiny, and EfficientNet-B3.

| Candidate | Best epoch | Validation AUC | Accuracy @ 0.5 | F1 @ 0.5 | Ensemble weight |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 5 | 0.896947 | 0.863571 | 0.863474 | 0.40 |
| ConvNeXt-Tiny | 4 | 0.896972 | 0.850714 | 0.845300 | 0.35 |
| EfficientNet-B3 | 4 | 0.890503 | 0.844286 | 0.844508 | 0.25 |
| Swin-T | 9 | 0.893630 | 0.835714 | 0.829377 | 0.00 |
| **Weighted probability ensemble** | — | **0.901004** | **0.864286** | **0.862510** | **1.00 total** |

The ensemble AUC is reconstructed from the four saved probability arrays and the weights in `reports/phase2/ensemble_fit.json`; it matches that artifact's `mean_oof_auc` value exactly. The equal-weight baseline is 0.900790. The fitted weights therefore improve AUC by 0.000214 over equal weights **on this validation split**. This is a small, tuned-on-the-scoring-split gain, not an independently validated benefit of weight tuning.

## 2. Training and validation protocol

All four `config.yaml` files record the same principal recipe: 224 × 224 inputs; pretrained `torchvision` backbones; batch size 32; learning rate `3e-4`; weight decay `1e-4`; up to 12 epochs with early-stopping patience 6; cosine scheduling with a one-epoch warm-up; EMA decay 0.999; AMP during training/validation; robust augmentation; gradient clipping at 1.0; and composite checkpoint selection with clean weight 0.5. Degradation augmentation uses probability 0.4 from epoch 3. Checkpoint selection monitors five variants on 600 validation images. Validation uses horizontal-flip TTA.

Each run used seed 42 and records `train_samples: 5600`, `val_samples: 1400`, `val_fraction: 0.2`, `split_seed: 42`, `fold: 0`, and `folds: 5`. The four validation reports have the same 1,400 labels and all identify `split: val`. Only fold 0 is present in this output; the files do **not** show a completed five-fold evaluation.

The report files live in a directory named `reports/phase2/oof/` and contain a field named `mean_oof_auc`. Those names do not make the ensemble score an independent OOF estimate: the ensemble weights were fitted and scored using these same single-split validation predictions. Thus 0.901004 is a tuned single-holdout result and may be optimistic. It is not five-fold cross-validation, nor an independent estimate of ensemble generalization.

## 3. Artifact inventory and reproducibility

The Phase 2 output folder contains:

- `checkpoints/ensemble_phase2.json` — member paths, epochs, weights, and fit summary.
- `checkpoints/runs/p2_<arch>_s42/config.yaml` — one training configuration per architecture.
- `checkpoints/runs/p2_<arch>_s42/best.pth` and `last.pth` — selected and final training checkpoints.
- `checkpoints/runs/p2_<arch>_s42/val_predictions.csv` — 1,400 validation rows per model.
- `reports/phase2/ensemble_fit.json` and `reports/phase2/oof/p2_<arch>_s42.json` — fit metadata and per-model validation labels/probabilities/metrics.
- `predictions_phase2.csv` — 3,000 test predictions.

The four manifest-referenced `best.pth` files total **359,308,819 bytes** (359.3 MB decimal; 342.5 MiB). This includes the Swin-T file even though its weight is zero, because it remains a member in the manifest. The three positive-weight checkpoints total 248,950,467 bytes (249.0 MB). The larger `last.pth` files are for training/resume provenance and are not required for the inference manifest.

`predictions_phase2.csv` has 3,000 rows, 3,000 unique IDs, confidence values in [0, 1], and labels consistent with the 0.5 threshold. Its SHA-256 is `e7f3fc5564bfa0bbde7d590b85b8013000da0fd2d9f0bb8d0463901446e21ad3`; it is byte-identical to the repository-root `predictions.csv`. The separate root [`predictions.summary.json`](../predictions.summary.json) records the inference run: 146.61 seconds, 20.46 images/second, CUDA, FP32, horizontal-flip TTA, deterministic mode, and zero load failures. That timing metadata is outside `kaggle_output_phase2/` itself.

## 4. Scope and limitations

The Phase 2 output folder does not contain a full corruption-robustness evaluation, Grad-CAM images, training-curve plots, or a prediction summary/timing file. The five-variant/600-image monitor is recorded in training configs, but it is checkpoint-selection evidence—not a full final-ensemble robustness table. The repository's 17-variant robustness table belongs to the historical Phase 1 soup and must not be presented as a Phase 2 ensemble result.

The single-holdout tuning reuse limits what can be inferred from 0.901004. A stronger estimate would require fitting weights without using the examples on which the ensemble is scored (for example, nested or independently held-out validation). No such independent Phase 2 ensemble evaluation is present in these artifacts.

---

*Artifact source: [`kaggle_output_phase2/`](../kaggle_output_phase2/). The folder records Phase 2 model and prediction outputs; historical Phase 0/1 results remain in their respective reports.*
