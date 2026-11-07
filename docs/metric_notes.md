# Metric notes (local interpretation)

> The authoritative hackathon metric guide is the root [`evaluation_metric.md`](../evaluation_metric.md)
> supplied by the IIIT Bangalore organizing team. This file keeps the team's
> original working notes for reference only; it does not override the official rules.

## 1. Task framing

Binary classification of a face crop:

| label | meaning |
|-------|---------|
| `0` | real |
| `1` | deepfake (fake) |

The model outputs a single logit; `sigmoid(logit)` is interpreted as **`confidence` = P(fake)**.
The leaderboard metric is **AUC-ROC computed on those `confidence` values**, therefore the ranking of
probabilities matters and the hard `label` column does not influence the score.

## 2. Definition

For a set of `n_pos` fake images and `n_neg` real images:

```
AUC = (1 / (n_pos * n_neg)) * Σ_{i ∈ fake} Σ_{j ∈ real} 1[ p_i > p_j ]  (+ 0.5 · 1[ p_i = p_j ])
```

Equivalently, it is the area under the curve traced by the True Positive Rate against the False
Positive Rate while sweeping every possible decision threshold:

```
TPR(τ) = TP(τ) / (TP(τ) + FN(τ))        FPR(τ) = FP(τ) / (FP(τ) + TN(τ))
AUC    = ∫₀¹ TPR d(FPR)
```

Key properties:

* **Threshold-free** — a single score summarises the whole ranking quality, so the metric is
  invariant to any monotone transformation of the probabilities (calibration does not matter).
* **Class-balance robust** — insensitive to the ~50/50 prior shift between validation and hidden
  test variants.
* **Chance level = 0.5**, perfect ranking = 1.0.
* **Interpretation** — probability that a randomly drawn fake image receives a higher fake-score
  than a randomly drawn real image.

## 3. Implementation

* `src/utils.py :: compute_metrics` calls `sklearn.metrics.roc_auc_score(y_true, y_prob)` on the
  float `confidence` values (`y_prob`), never on the thresholded labels.
* The same function is used by `src/train.py` (per-epoch validation, the **model-selection
  criterion**), `src/evaluate.py` (held-out reporting) and by the local harness that mimics the
  private leaderboard.
* `src/utils.py :: optimal_threshold` additionally returns the Youden-J threshold
  (`argmax(TPR − FPR)`) so we can report the best achievable operating point; the shipped
  submission keeps the conventional `0.5` threshold for the `label` column while `confidence`
  stays untouched (AUC does not depend on it).
* Ties are handled by `roc_auc_score`'s rank averaging (equivalent to the `0.5` credit above).

## 4. Secondary diagnostics (reported, not optimised)

| metric | why it is tracked |
|--------|-------------------|
| Accuracy @ 0.5 | sanity check that the threshold is sensible |
| Precision / Recall / F1 @ 0.5 | catches class collapse or threshold bias |
| Confusion matrix | shows the FP (real flagged as fake) vs FN (fake missed) trade-off |
| Best threshold (Youden-J) | indicates how far the operating point is from 0.5 |
| Loss curves | detects over/under-fitting that AUC alone can hide |

## 5. Submission contract

`predictions.csv` must contain exactly:

```
image_id,label,confidence
<uuid>,0|1,<float in [0,1]>
```

* `image_id` — file name without extension.
* `confidence` — P(fake); this is what the AUC-ROC scorer consumes.
* `label` — `int(confidence >= threshold)`, provided for completeness.
