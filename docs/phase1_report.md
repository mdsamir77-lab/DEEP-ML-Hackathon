# Phase 1 report — ResNet-50 improvements (Kaggle)

**Historical Phase 1 report — completed.** The Phase 1 clean-AUC target (0.91–0.93) was not met; Phase 2 was subsequently completed. The Phase 1 soup is now a single-model fallback, not the current ensemble. See the repository [`README.md`](../README.md) and [`technical_report.md`](../technical_report.md) for Phase 2 results.

Phase 1 added EMA, robust augmentation, composite selection, a robustness monitor and a model-soup
utility. Its Kaggle soup was a **real but small improvement** over the Phase 0 baseline. The Phase 1
ResNet-50 plateau around 0.89–0.896 motivated the later Phase 2 architecture comparison; it is a
historical observation, not a statement that Phase 2 remains future work. See [`phase2_report.md`](phase2_report.md)
for the subsequent runs and their validation limitation.

---

## 1. Pre-phase baseline (Phase 0, measured)

| Metric | Value |
|---|---:|
| Validation AUC | **0.892324** |
| Accuracy @ 0.5 | 0.858571 |
| F1 @ 0.5 | 0.863636 |
| Robustness composite | 0.890645 |
| Robustness proxy (20 pts) | 15.63 |
| Inference (3,000, FP32, RTX 3050) | 73–84 s |
| Estimated total score | ~83.9–84.9 |

The Phase 0 submission was a 2-way soup of two ResNet-50 fine-tunes (AUC 0.8923).

---

## 2. Kaggle runs and results

All runs used ResNet-50 / torchvision, 224×224, 15 epochs, EMA, and the Phase 1 code. The soup is
an equal-weight average of three EMA checkpoints.

### 2.1 Individual runs

| Run | Selection | Clean AUC | Accuracy | F1 | Selection score | 200-image monitor AUC |
|---|---|---:|---:|---:|---:|---:|
| P1-A `p1a_ema` (EMA + mild aug) | clean AUC | **0.89051** | 0.84929 | 0.85151 | 0.89051 | — (not run) |
| P1-B `p1b_ema_robust` (EMA + robust aug) | composite | **0.89031** | 0.84786 | 0.84947 | 0.90870 | 0.92709 |
| P1-B seed 3 `p1b_ema_robust_3` | composite | **0.88927** | 0.85214 | 0.85288 | 0.90663 | 0.92399 |

**Every individual parent is below the Phase 0 soup (0.89232).** The EMA/robust recipe did not
beat the original 2-way soup on clean AUC.

### 2.2 Submitted soup (3-way equal average)

| Metric | Value |
|---|---:|
| Validation AUC | **0.896235** |
| Accuracy @ 0.5 | 0.856429 |
| F1 @ 0.5 | 0.857346 |
| Precision / Recall | 0.8567 / 0.8580 |
| Robustness composite | **0.895023** |
| Mean degraded AUC | 0.893810 |
| Worst degraded AUC | 0.883331 (`blur_s2.0`) |
| Best degraded AUC | 0.898315 (`downscale_0.75`) |
| Robustness proxy (20 pts) | **15.80** |
| Accuracy proxy (50 pts) | **39.62** |
| Test inference (T4, 3,000 images, FP32 + TTA) | **52.0 s** (57.7 img/s) |
| Test predictions | 3,000 rows, 0 load failures, SHA-256 `c81a551c…` |

The soup gains **+0.0057 AUC over its best parent** and **+0.0039 over the Phase 0 baseline**.
The accuracy/F1 are slightly lower than Phase 0, but AUC is the scored metric.

### 2.3 Full robustness table (Kaggle, 1,400 held-out images)

| Variant | AUC | Δ vs clean |
|---|---:|---:|
| clean | 0.8962 | — |
| jpeg_q90 | 0.8952 | −0.0010 |
| jpeg_q70 | 0.8930 | −0.0032 |
| jpeg_q50 | 0.8942 | −0.0020 |
| jpeg_q30 | 0.8922 | −0.0040 |
| blur_s0.5 | 0.8977 | +0.0014 |
| blur_s1.0 | 0.8969 | +0.0006 |
| blur_s2.0 | 0.8833 | −0.0129 |
| noise_0.01 | 0.8938 | −0.0024 |
| noise_0.03 | 0.8904 | −0.0058 |
| downscale_0.75 | 0.8983 | +0.0021 |
| downscale_0.50 | 0.8957 | −0.0005 |
| brightness_0.8 | 0.8964 | +0.0002 |
| brightness_1.2 | 0.8923 | −0.0039 |
| contrast_0.8 | 0.8958 | −0.0004 |
| contrast_1.2 | 0.8948 | −0.0014 |
| grayscale | 0.8908 | −0.0054 |

The model is **uniformly robust**: every variant is within 0.013 AUC of clean, and the worst case
is strong blur (0.8833). Robustness is no longer the main weakness; clean AUC is.

---

## 3. Phase 0 vs Phase 1 comparison

| Metric | Phase 0 | Phase 1 soup | Change |
|---|---:|---:|---:|
| Validation AUC | 0.892324 | **0.896235** | **+0.003911** |
| Accuracy @ 0.5 | 0.858571 | 0.856429 | −0.002142 |
| F1 @ 0.5 | 0.863636 | 0.857346 | −0.006290 |
| Robustness composite | 0.890645 | **0.895023** | **+0.004378** |
| Robustness proxy | 15.63 | **15.80** | **+0.17** |
| Inference (3,000) | 73–84 s (RTX 3050 FP32) | 52 s (T4 FP32) | faster on T4 |
| Accuracy sub-score | 39.23 | **39.62** | **+0.39** |
| Estimated total score | ~83.9–84.9 | **~84.4–85.4** | **+0.5–0.6** |

---

## 4. Findings

1. **Every individual ResNet-50 parent was below the Phase 0 soup.** P1-A 0.8905, P1-B 0.8903,
   P1-B seed 3 0.8893 vs 0.8923 baseline. The EMA/robust recipe did not beat the original 2-way
   soup on clean AUC.
2. **The 3-way soup is a real but small gain.** 0.8962, +0.0057 over the best parent and +0.0039
   over the Phase 0 baseline. The soup is the Phase 1 win.
3. **The 200-image Phase 1 robustness monitor was optimistic.** P1-B's monitor AUC was 0.927, but
   the Phase 1 full 17-variant mean is 0.8938 and the composite is 0.8950. Phase 2 later used a
   five-variant, 600-image monitor for checkpoint selection; that monitor is not a full final-ensemble
   robustness evaluation.
4. **Phase 1 robustness was relatively stable on its tested variants.** Each Phase 1 variant was
   within 0.013 AUC of that soup's clean score; the worst was strong blur (0.8833). This describes
   only the measured Phase 1 set, not general robustness or the Phase 2 ensemble.
5. **The Phase 1 clean-AUC target was not met.** Its ResNet-50 soup scored 0.8962. The planned
   stronger-backbone comparison and ensemble were subsequently carried out in Phase 2; see
   [`phase2_report.md`](phase2_report.md) for the actual results and single-holdout caveat.
6. **Efficiency is excellent.** 3,000 images in 52 s on a T4 (57.7 img/s), FP32 deterministic,
   20/20. The Kaggle-generated predictions are T4-native and canary-friendly.
7. **Predictions are valid.** 3,000 rows, unique IDs, sample-submission order, confidence in
   [0.024, 0.991], labels consistent, SHA-256 `c81a551c…`.

---

## 5. Score assessment

| Component | Phase 0 | Phase 1 soup | Change |
|---|---:|---:|---:|
| Accuracy (50) | 39.23 | **39.62** | +0.39 |
| Efficiency (20) | 20 | **20** | 0 |
| Robustness (20) | 15.63 | **15.80** | +0.17 |
| Documentation (10) | 9–10 | **9–10** | 0 |
| Bonus (5) | 0 | **0** | 0 |
| **Estimated total** | **83.9–84.9** | **84.4–85.4** | **+0.5–0.6** |

**Historical Phase 1 score estimate only:** at AUC 0.8962, the Phase 1 soup's estimated accuracy
component was 39.62/50. The ceiling arithmetic below was calculated for that Phase 1 result; it is
not an estimate of the later Phase 2 model's score, and no hidden-test score is inferred here.
Even with perfect robustness (20) and documentation (10) and no bonus, the Phase 1-only ceiling was
**89.62**; with all bonuses (+5), **94.62**. Phase 2 has since been run; its validation result and
limitations are documented in [`phase2_report.md`](phase2_report.md).

---

## 6. Decision

- **Historical Phase 1 outcome:** the 3-way soup improved AUC by 0.0039 and the robustness proxy by 0.17 over Phase 0; its T4 inference time was 52 s. It is retained as a fallback, not the current ensemble.
- The Phase 1 target (clean AUC 0.91–0.93) was not met. ResNet-50 plateaued, which motivated the subsequent Phase 2 architecture comparison.
- The Phase 2 plan below was subsequently carried out; its single-split validation limitation is documented in the root README and technical report.
- Phase 0 artifacts remain useful as historical baselines.

---

## 7. Local verification

- Checkpoint loads locally with torch 2.5.1: `resnet50` / `torchvision`, 320 state keys, all finite,
  23,510,081 params.
- Soup provenance: `soup_sha256 = 93e5a10b…` matches the recomputed state-dict hash; parent hashes
  match the Kaggle report.
- Predictions: 3,000 rows, columns `image_id,label,confidence`, unique IDs, sample-submission order,
  confidence [0.0241, 0.9906], labels consistent, SHA-256 `c81a551c…`.
- Local `evaluate.py` reproduction: **exact match** — AUC 0.8962353872779518, accuracy 0.8564285714,
  F1 0.8573456352 (1,400 samples, 57.8 s on RTX 3050 FP32 + TTA).
- Local robustness subset (clean, jpeg_q50, blur_s2.0, grayscale): **exact match** — clean
  0.896235, jpeg_q50 0.894174, blur_s2.0 0.883331, grayscale 0.890786 (`ROBUST_EXITCODE=0`).

---

## 8. Artifacts

- `kagggle_out_phase_01/checkpoints/soup_phase1.pth` — submitted soup.
- `kagggle_out_phase_01/predictions_phase1_soup.csv` + `.summary.json` — T4 predictions.
- `kagggle_out_phase_01/reports/eval_checkpoints_val/` — validation metrics/plots.
- `kagggle_out_phase_01/reports/soup_robustness/` — full 17-variant robustness table.
- `kagggle_out_phase_01/reports/soup_phase1.json` — soup provenance.

---

## 9. Historical Phase 2 proposal (superseded)

This was a pre-run proposal, not a description of the final Phase 2 protocol. The implemented shared-recipe backbone comparison, probability ensemble, and five-variant/600-image checkpoint-selection monitor are documented in [`phase2_report.md`](phase2_report.md). The saved Phase 2 outputs contain only fold 0 of one shared validation split: they do **not** provide five-fold results, mean ± standard deviation across folds, or an independent ensemble evaluation. A full 17-variant evaluation of the final Phase 2 ensemble is also not present in the available artifacts. Grad-CAM materials were produced later and are described in the technical report.

---

## 10. Historical Phase 1 status

**Phase 1 is complete.** Its clean-AUC target was not met. The Phase 1 soup is a fallback; the current submitted model is the Phase 2 weighted probability ensemble. Phase 2 results and validation caveats are in the root [`README.md`](../README.md) and [`technical_report.md`](../technical_report.md).

