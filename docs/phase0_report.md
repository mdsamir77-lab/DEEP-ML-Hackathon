# Phase 0 report — Verification, safety, and measurement

**Date:** 2026-11-07  
**Environment:** local Windows, RTX 3050 Laptop 6 GB, PyTorch 2.5.1+cu121, Python 3.10  
**Status:** COMPLETED locally; Kaggle was **not** used in Phase 0. This is a historical Phase 0 snapshot; its “next phase” section records the plan at that time. Phase 1 and Phase 2 have since completed (see the root README and technical report).

---

## A. Phase summary

Phase 0 had four objectives:

1. **Verify the baseline** against the actual artifacts (not the earlier audit's summary).
2. **Remove the canary/reproducibility risk** by making submission inference FP32 and
   deterministic, adding corrupt-image fallback and post-write CSV validation.
3. **Measure the real robustness baseline** with a new degradation harness.
4. **Audit shortcuts and leakage** with perceptual hashes and JPEG/EXIF/pixel statistics,
   and prepare the Kaggle handoff for Phase 1.

No model training was performed in Phase 0. The model weights are unchanged.

---

## B. Files changed

| File | Change | Why |
|---|---|---|
| `src/utils.py` | Added `configure_deterministic_inference`, `composite_auc`; switched `predict_loader` to `torch.inference_mode()`; returns `load_failures` | Deterministic FP32 inference + safer loading |
| `src/dataset.py` | `DeepfakeDataset` gains `fallback_on_error`/`fallback_size` and a `load_ok` flag; test loader enables the fallback | One corrupt image cannot abort a submission |
| `src/inference.py` | FP32 default, deterministic seed/flags, corrupt fallback reporting, post-write CSV validation, SHA-256/environment summary | Removes canary risk, enforces the submission contract |
| `src/evaluate.py` | Uses `configure_deterministic_inference` | Reproducible validation evaluation |
| `inference_script.py` | Sets `CUBLAS_WORKSPACE_CONFIG` before CUDA init; updated docstring | Deterministic cuBLAS |
| `src/robustness.py` | **New** 16-variant degradation harness + composite proxy | Measures the hidden robustness sub-score locally |
| `src/audit.py` | **New** pHash near-duplicate + JPEG/EXIF/shortcut audit | Honest validation + shortcut detection |
| `requirements-inference.txt` | **New** minimal inference environment | Safer T4 setup |
| `evaluation_metric.md` | Replaced with the official IIIT Bangalore guide | Rule compliance |
| `docs/metric_notes.md` | Local metric notes moved here | Keep the official file authoritative |
| `docs/kaggle_phase1_handoff.md` | **New** Kaggle upload/run/download instructions | Phase 1 readiness |
| `README.md` | FP32/deterministic recipe, new timings, robustness/audit commands, layout | Documentation accuracy |
| `predictions.csv` / `predictions.summary.json` | Regenerated with FP32 deterministic inference | Submitted recipe |

The original AMP predictions were preserved at
`reports/phase0/predictions_amp_original.csv` for comparison.

---

## C. Technical explanation (beginner-friendly)

- **Why FP32?** The previous CSV was generated with fp16 autocast on an RTX 3050. The evaluator
  runs a T4. fp16 maths can round differently on different GPUs, so the same script might not
  reproduce the submitted labels exactly. FP32 is the safer common denominator. It is slower,
  but 3,000 images still take only 73–80 s, far below the 5-minute limit.
- **What does "deterministic" mean?** We fixed the random seeds and told cuDNN/cuBLAS to use
  deterministic algorithms, disable TF32, and disable the benchmark autotuner. Two runs of the
  exact command then produced the same file hash.
- **What is the corrupt-image fallback?** If a test file cannot be decoded, the dataset returns a
  gray image instead of crashing, records the `image_id`, and still writes a prediction row.
  Missing rows are scored as incorrect, so this protects the submission.
- **What is the robustness harness?** It takes the held-out validation images and applies 16
  plausible degradations (JPEG q90/70/50/30, Gaussian blur σ0.5/1/2, Gaussian noise σ0.01/0.03,
  downscale 0.75/0.50, brightness/contrast ±20 %, grayscale). It reports AUC per variant and an
  equal-weight composite. The official variant weights are undisclosed, so the composite is a
  transparent local proxy, not the official score.
- **What is the audit?** Perceptual hashes (pHash) find visually near-identical images that MD5
  misses. JPEG quantization tables, EXIF tags, file size, sharpness, saturation and border
  statistics reveal class-wise shortcuts that a model could memorise instead of learning
  deepfake artifacts.

---

## D. Testing results

| Test | Command | Result |
|---|---|---|
| Syntax/import check | `python -m compileall -q src inference_script.py` | **PASS** |
| Inference CLI | `python inference_script.py --help` | **PASS** |
| Robustness CLI | `python src/robustness.py --help` | **PASS** |
| Audit CLI | `python src/audit.py --help` | **PASS** |
| Corrupt-image fallback | smoke dir with 8 valid + 1 corrupt JPEG | **PASS**: 9 rows, `num_load_failures=1`, `load_failures=['corrupt_image']` |
| Full-test determinism | two runs of the exact submission command | **PASS**: identical SHA-256 `c46225fd…` |
| Submission contract | 3,000 rows, columns, ID set/order, confidence range, label consistency | **PASS** |
| Validation AUC through new path | `python src/evaluate.py --weights checkpoints/best_model.pth --split val --tta --no_amp` | **PASS**: AUC 0.892305 (previous 0.892325; 2e-5 difference) |

The full robustness and audit runs are recorded under `reports/phase0/`.

---

## E. Experiment results

### E.1 Baseline verification (confirmed from artifacts)

| Metric | Value | Source |
|---|---:|---|
| Validation AUC | **0.892324** | `reports/eval_amp_tta/metrics.json`; recomputed from `predictions_detailed.csv` |
| Accuracy @ 0.5 | **0.858571** | same |
| F1 @ 0.5 | **0.863636** | same |
| Precision / Recall | 0.8382 / 0.8906 | same |
| Inference, 3,000 test images, AMP (old recipe) | **37.82 s** (79.3 img/s) | original `predictions.summary.json` |
| Inference, 3,000 test images, FP32 (submitted) | **73.3–83.6 s** (35.9–40.9 img/s) | new `predictions_fp32*.summary.json` |
| FP32 vs AMP label agreement | 2,998 / 3,000; Spearman 0.999994 | Phase 0 comparison |
| Unique confidences | 2,997 (FP32) vs 1,591 (AMP) | new vs old summary |

### E.2 Robustness baseline (1,400 held-out images, FP32 + flip TTA)

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

* Mean degraded AUC **0.8890**; worst **0.8728**; best **0.8950**.
* Equal-weight composite **0.8906**.
* Robustness proxy **15.63 / 20** (assumes `20 × (AUC − 0.5)/0.5`; the official weights are
  undisclosed).
* The current soup is **more robust than the pre-phase estimate (12–15)**. The only material
  weaknesses are strong blur and grayscale.

### E.3 Near-duplicate / shortcut audit

| Finding | Value | Interpretation |
|---|---:|---|
| Near-duplicate pairs (pHash ≤ 6) | 295 | sparse; max cluster size 3 |
| Cross-class pairs / clusters | 55 / 53 | possible label noise or genuinely hard pairs |
| Train/val leakage pairs | 87 | 85 val images (6.07 %) have a train near-duplicate |
| Test images with a train near-duplicate | 238 (7.93 %) | hidden test may be slightly easier than a clean holdout |
| JPEG quantization tables | 1 table for all 7,000 images | **no JPEG-header shortcut** |
| EXIF present | 0 % real / 0 % fake | **no metadata shortcut** |
| Largest class-wise pixel effect | std_r, effect size 0.177 | small; no dominant colour/border shortcut |

The 6.07 % train/validation near-duplicate leakage may make the random-split AUC optimistic. No pHash-grouped validation number was subsequently documented for Phase 1 or Phase 2, so the size of the effect and transfer of relative model rankings remain unknown. The 7.93 % test-train match is descriptive only; test labels are sealed.

---

## F. Pre-phase vs post-phase score assessment

Phase 0 does not change the model, so AUC/F1/accuracy are unchanged. The **measured** robustness
and the **removed canary risk** change the expected score:

| Metric | Before Phase 0 | After Phase 0 | Change |
|---|---:|---:|---:|
| Validation AUC | 0.8923 (reported) | **0.8923 (verified)** | 0.0000 |
| Accuracy @ 0.5 | 0.8586 | **0.8586** | 0.0000 |
| F1 @ 0.5 | 0.8636 | **0.8636** | 0.0000 |
| Inference speed (3,000) | 37.8 s (AMP) | **73–84 s (FP32 deterministic)** | +35–46 s (still 20/20) |
| Robustness | **assumed** 12–15 / 20 | **measured 15.63 / 20** | evidence-based |
| Estimated total score | **79–84** | **~82–85** | +1–3 (canary risk reduced; robustness measured at the upper end) |

**Why not higher?** The accuracy sub-score is still `50 × (0.8923 − 0.5)/0.5 = 39.23/50`.
With efficiency 20, robustness 15.63, documentation ~9, and no bonus, the total is
`39.23 + 20 + 15.63 + 9 = 83.86`. Documentation can reach 10 with the Phase 0 additions
(→ 84.86), but the model itself is unchanged. The hard ceiling with the current AUC remains
**89.2** even with perfect robustness/documentation and no bonus.

**Historical schedule note (2026-11-07):** the original Phase 0 plan estimated about seven days remaining in an eight-day window and listed Phase 1 as the next priority. That schedule is past; Phases 1 and 2 have since completed.


---

## G. Open questions recorded at the end of Phase 0 (historical)

The items below are the Phase 0 assessment as of 2026-11-07, not the current project backlog. Phases 1 and 2 have since completed; their measured outcomes are in [`phase1_report.md`](phase1_report.md), [`phase2_report.md`](phase2_report.md), and the root technical report.

1. **AUC gap at Phase 0.** The baseline measured 0.8923; raising clean AUC was a Phase 1 goal, not a current requirement.
2. **Blur / grayscale.** The listed drops describe the Phase 0 baseline and motivated the later robust-augmentation work. They do not describe Phase 2 ensemble robustness.
3. **Near-duplicate leakage.** The 6.07% estimate is from the Phase 0 audit. No pHash-grouped Phase 2 estimate is present in the saved Phase 2 artifacts.
4. **Cross-class near-duplicates.** The audit flagged 55 pairs for inspection; labels were not changed. This is historical audit evidence, not a Phase 1 action item.
5. **Hidden test / robustness weights.** The local composite was a proxy, not the official score; the official hidden-test result is not inferred from it.
6. **Documentation / interpretability.** The architecture, ensemble, and Grad-CAM materials were subsequently produced. TensorFlow export is not documented as a completed deliverable here.
7. **Submission size limit.** The Phase 0 report noted uncertainty at the time. The Phase 2 best-checkpoint payload is now measured in [`phase2_report.md`](phase2_report.md); that figure is not the size of a complete archive.

---

## H. Historical Kaggle handoff

[`kaggle_phase1_handoff.md`](kaggle_phase1_handoff.md) preserves the Phase 1 upload layout, T4 setup, training command examples, and artifact-download workflow. It is a historical handoff, not current training guidance: Phase 1 and Phase 2 have completed. Current Phase 2 artifacts are under `kaggle_output_phase2/`; see [`phase2_report.md`](phase2_report.md).

---

## I. Phase 0 verification checklist (historical snapshot)

These checks and expected values document the Phase 0 verification state. They are not an approval checklist for the completed Phase 1/2 submission:

```powershell
# 1. Submission contract and hash
python -c "import hashlib, pandas as pd; p=pd.read_csv('predictions.csv'); s=pd.read_csv('hackathon_dataset/sample_submission.csv'); assert len(p)==3000; assert list(p.columns)==['image_id','label','confidence']; assert list(p.image_id.astype(str))==list(s.image_id.astype(str)); assert p.confidence.between(0,1).all(); assert p.label.astype(int).equals(p.confidence.ge(0.5).astype(int)); print('OK', hashlib.sha256(open('predictions.csv','rb').read()).hexdigest())"

# 2. Summary fields
python -c "import json; d=json.load(open('predictions.summary.json')); print(d['amp'], d['deterministic'], d['num_load_failures'], d['output_sha256'])"

# 3. Robustness report
python src/robustness.py --weights checkpoints/best_model.pth --output_dir reports/robustness --tta --no_amp

# 4. Audit report
python src/audit.py --data_dir hackathon_dataset --output_dir reports/audit

# 5. Code quality
python -m compileall -q src inference_script.py
ruff check src inference_script.py
```

Expected: `predictions.csv` SHA-256 `c46225fd…`; summary `amp=False`, `deterministic=True`,
`num_load_failures=0`; robustness composite ~0.8906; audit summary with 295 pHash pairs and 6.07 %
val leakage; compileall/ruff clean.

---

## J. Historical Phase 1 plan (superseded)

This was the proposed Phase 1 work at the time of the Phase 0 report. Phase 1 and Phase 2 have since been completed; do not use this section as the current run plan.

- Add EMA of weights + `update_bn`.
- Add snapshot/soup across 3–5 seeds and 2 recipes.
- Add robust augmentation: JPEG q30–90, blur σ0.1–1.5, noise σ0.005–0.03, downscale 0.5–0.9,
  brightness/contrast ±0.2, with a clean→degraded curriculum.
- Optional clean/degraded consistency loss.
- Tune TTA (flip + multi-scale/crop).
- Select checkpoints on the **composite** clean/degraded AUC, not clean AUC alone.
- Report both random-split and pHash-grouped validation AUC.

Historical Phase 1 projection (written before those runs): clean AUC **0.91–0.93**, degraded AUC
**0.88–0.90**, composite **0.89–0.91**, robustness proxy **15.6–16.5/20**. This was a forecast,
not the measured Phase 1 outcome; the actual results are recorded in [`phase1_report.md`](phase1_report.md).

**Not planned for Phase 1:** new architectures, Grad-CAM, TF export, or pseudo-labeling (forbidden).
Grad-CAM was later generated; see the technical report. The forecast and scope above are historical,
not current phase status.

---

## K. Phase status

**Historical Phase 0 snapshot (completed).** Its local checks and Kaggle handoff describe the state at that phase. Phase 1 and Phase 2 have since completed; the current submission is the Phase 2 ensemble. Deterministic inference settings improve repeatability on a fixed stack but do not guarantee byte-identical output across GPU architectures. See the root [`README.md`](../README.md) and [`technical_report.md`](../technical_report.md) for current results.

