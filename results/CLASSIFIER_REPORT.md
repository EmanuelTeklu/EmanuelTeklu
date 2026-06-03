# Selective Sparse Attention — Sparse-Safe Classifier Report

**Thesis:** apply sparse attention *selectively*. Predict which (head, query, context) cases are sparse-safe from cheap features, route only those through a cheap sparse router (sink+recent+block-route at a 10% read budget), and fall back to full attention otherwise.

> **Scale-up update (see [`SCALE_UP_REPORT.md`](SCALE_UP_REPORT.md)):** repeated on
> 0.5B at 2k–32k and 1.5B at 4k–16k with a memory-safe last-token SDPA capture.
> Sparse-safe structure **strengthens with context length** (0.5B base rate
> 0.60→0.78; held-out-regime coverage peaks 0.60 @16k; read reduction 2.16× @16k)
> and **context-length transfer holds** (precision ~0.94 from ≤8k→16k/32k). But it
> does **not** strengthen with **model size** (1.5B slightly harder), and
> **model-size transfer is weak** (a 0.5B-trained threshold drops to 0.68 precision
> on 1.5B — retrain per model). Verdict remains **KEEP** (read reduction capped
> ~2.2× by the fixed 10% router budget; oracle ceiling ≥100×).

## Verdict: **KEEP**

- **Beats simple thresholding (KILL-rule check):** single-feature entropy/top-mass thresholds reach **~0% coverage at 95% precision** (AUC≈0.69); the multivariate classifier reaches **40%** (cheap, random) / **32%** (cheap, held-out regime). Decisively better. ✓
- **Precision/coverage (cheap, deployable features):** at the 95%-precision operating point, coverage is 40% (random), 32% (held-out regime), 38% (held-out head), 26% (held-out layer).
- **Long-context deployment (the relevant population, T≥512):** held-out-regime selective policy delivers **2.0× overall KV read reduction** at 57% coverage and 5.0% routed failure (cheap features).
- **Mixed-set caveat:** on the FULL dataset (incl. 43–161-token prompts) the overall reduction is only ~1.3–1.6×, because a 10% budget barely helps a 44-token context — short prompts are not a sparse-attention use case and a real system would not route them.
- **Transfer:** holds to needle/longdoc/length-shift (read reduction 4.0–4.2× on held-out long regimes); **weak for the agent/repeated-prefix regime** (precision-coverage tradeoff degrades — see transfer table).

Base rate (router-safe @10% reads, rel-L2≤0.10): 0.498. Pre-registered: KEEP if cheap held-out-regime coverage≥30%, long-context read-reduction≥2×, beats thresholding, and selective beats universal routing (universal fails 50% of cases vs 5.0% here).

## 1. Coverage @ 95% precision — primary label (router-safe @10%, rel-L2≤0.10)

Threshold-free: the largest fraction of cases that can be routed sparse while the sparse decisions stay ≥95% precise. Out-of-fold; grouped where noted.

**Cheap (deployable) features — best model per split:**

| split       | model   |   auc |   cov_at_95prec |   realized_prec |
|:------------|:--------|------:|----------------:|----------------:|
| random      | gboost  | 0.954 |           0.399 |           0.951 |
| loho_head   | rforest | 0.944 |           0.377 |           0.95  |
| loro_regime | gboost  | 0.929 |           0.325 |           0.95  |
| lolo_layer  | rforest | 0.918 |           0.258 |           0.95  |

**All features (cheap + oracle attention stats; upper bound) — best per split:**

| split       | model   |   auc |   cov_at_95prec |   realized_prec |
|:------------|:--------|------:|----------------:|----------------:|
| random      | rforest | 0.981 |           0.473 |           0.951 |
| loho_head   | gboost  | 0.976 |           0.454 |           0.95  |
| loro_regime | gboost  | 0.953 |           0.443 |           0.95  |
| lolo_layer  | gboost  | 0.963 |           0.423 |           0.951 |

**Single-feature threshold baselines (the KILL-rule comparator):**

| split       | features          |   auc |   cov_at_95prec |
|:------------|:------------------|------:|----------------:|
| random      | entropy_only      | 0.661 |           0     |
| random      | mass_only         | 0.692 |           0     |
| random      | cent_entropy_only | 0.654 |           0     |
| loro_regime | entropy_only      | 0.547 |           0     |
| loro_regime | mass_only         | 0.556 |           0     |
| loro_regime | cent_entropy_only | 0.592 |           0     |
| lolo_layer  | entropy_only      | 0.651 |           0.001 |
| lolo_layer  | mass_only         | 0.687 |           0     |
| lolo_layer  | cent_entropy_only | 0.632 |           0     |
| loho_head   | entropy_only      | 0.658 |           0.005 |
| loho_head   | mass_only         | 0.693 |           0     |
| loho_head   | cent_entropy_only | 0.653 |           0     |

## 2. Selective policy savings (full mixed dataset, OOF @95% precision)

`universal_sparse` routes everything; `full_only` routes nothing; selective rows use the classifier. Failure = routed case with realized rel-L2>0.10.

| policy           |   coverage |   fail_rate_routed |   fail_rate_overall |   kv_read_reduction |
|:-----------------|-----------:|-------------------:|--------------------:|--------------------:|
| universal_sparse |          1 |              0.502 |               0.502 |               7.165 |
| full_only        |          0 |              0     |               0     |               1     |

| split       | model   |   coverage |   fail_rate_routed |   fail_rate_overall |   kv_read_reduction |
|:------------|:--------|-----------:|-------------------:|--------------------:|--------------------:|
| random      | gboost  |      0.399 |              0.049 |               0.02  |               1.549 |
| loho_head   | rforest |      0.377 |              0.05  |               0.019 |               1.507 |
| loro_regime | gboost  |      0.325 |              0.05  |               0.016 |               1.407 |
| lolo_layer  | rforest |      0.258 |              0.05  |               0.013 |               1.299 |

## 3. Long-context selective policy (T≥512, held-out regime)

The deployment-relevant population. Read reduction here is not diluted by tiny prompts.

| features   |   base_rate |   auc |   cov_at_95prec |   coverage |   fail_rate_routed |   kv_read_reduction |
|:-----------|------------:|------:|----------------:|-----------:|-------------------:|--------------------:|
| cheap      |       0.734 | 0.911 |           0.567 |      0.567 |               0.05 |               2.023 |
| all        |       0.734 | 0.97  |           0.736 |      0.736 |               0.05 |               2.912 |

**Amdahl end-to-end** at this long-context read reduction (2.0×), by attention/KV-read share of decode runtime:

|   reduction |   frac=0.5 |   frac=0.7 |   frac=0.9 |   frac=0.95 |
|------------:|-----------:|-----------:|-----------:|------------:|
|       2.023 |      1.338 |      1.548 |      1.835 |       1.925 |

## 4. Transfer (train on some regimes/lengths, test on held-out)

Train-chosen threshold (95% train precision) applied to the held-out test set — the honest generalization number. `train_thr_test_prec` shows whether the 95% target holds under shift.

| transfer             | features   | model   |   test_base_rate |   auc |   train_thr_test_prec |   train_thr_test_cov |   kv_read_reduction |   fail_rate_routed |
|:---------------------|:-----------|:--------|-----------------:|------:|----------------------:|---------------------:|--------------------:|-------------------:|
| short_med->long      | cheap      | logreg  |            0.8   | 0.861 |                 1     |                0.212 |               1.234 |              0     |
| short_med->long      | cheap      | rforest |            0.8   | 0.851 |                 0.943 |                0.559 |               1.996 |              0.057 |
| short_med->long      | cheap      | gboost  |            0.8   | 0.857 |                 0.926 |                0.628 |               2.279 |              0.074 |
| short_med->long      | all        | logreg  |            0.8   | 0.888 |                 1     |                0.006 |               1.005 |              0     |
| short_med->long      | all        | rforest |            0.8   | 0.961 |                 0.984 |                0.668 |               2.48  |              0.016 |
| short_med->long      | all        | gboost  |            0.8   | 0.947 |                 0.938 |                0.811 |               3.632 |              0.062 |
| non_needle->needle   | cheap      | logreg  |            0.795 | 0.857 |                 0.935 |                0.552 |               1.972 |              0.065 |
| non_needle->needle   | cheap      | rforest |            0.795 | 0.918 |                 0.911 |                0.838 |               3.976 |              0.089 |
| non_needle->needle   | cheap      | gboost  |            0.795 | 0.907 |                 0.908 |                0.842 |               4.04  |              0.092 |
| non_needle->needle   | all        | logreg  |            0.795 | 0.938 |                 0.965 |                0.679 |               2.539 |              0.035 |
| non_needle->needle   | all        | rforest |            0.795 | 0.977 |                 0.938 |                0.814 |               3.666 |              0.062 |
| non_needle->needle   | all        | gboost  |            0.795 | 0.978 |                 0.933 |                0.826 |               3.815 |              0.067 |
| non_agent->agent     | cheap      | logreg  |            0.539 | 0.808 |                 0.912 |                0.17  |               1.176 |              0.088 |
| non_agent->agent     | cheap      | rforest |            0.539 | 0.81  |                 0.63  |                0.812 |               3.552 |              0.37  |
| non_agent->agent     | cheap      | gboost  |            0.539 | 0.892 |                 0.662 |                0.783 |               3.25  |              0.338 |
| non_agent->agent     | all        | logreg  |            0.539 | 0.924 |                 0.905 |                0.47  |               1.712 |              0.095 |
| non_agent->agent     | all        | rforest |            0.539 | 0.932 |                 0.78  |                0.676 |               2.486 |              0.22  |
| non_agent->agent     | all        | gboost  |            0.539 | 0.942 |                 0.809 |                0.64  |               2.305 |              0.191 |
| non_longdoc->longdoc | cheap      | logreg  |            0.81  | 0.888 |                 0.982 |                0.5   |               1.811 |              0.018 |
| non_longdoc->longdoc | cheap      | rforest |            0.81  | 0.941 |                 0.93  |                0.848 |               4.16  |              0.07  |
| non_longdoc->longdoc | cheap      | gboost  |            0.81  | 0.944 |                 0.93  |                0.851 |               4.207 |              0.07  |
| non_longdoc->longdoc | all        | logreg  |            0.81  | 0.963 |                 0.976 |                0.753 |               3.071 |              0.024 |
| non_longdoc->longdoc | all        | rforest |            0.81  | 0.989 |                 0.957 |                0.83  |               3.9   |              0.043 |
| non_longdoc->longdoc | all        | gboost  |            0.81  | 0.989 |                 0.953 |                0.83  |               3.9   |              0.047 |

Reading: tree ensembles buy coverage but precision can dip to ~0.91–0.93 under regime shift; logistic regression holds precision (~0.94–1.0) at lower coverage. The agent/repeated-prefix regime is the hard transfer target.

## 5. Stricter / weaker labels & oracle ceiling (gboost, random OOF)

| label        | features   |   base_rate |   auc |   cov_at_95prec |   realized_prec |
|:-------------|:-----------|------------:|------:|----------------:|----------------:|
| router10_t10 | cheap      |       0.498 | 0.954 |           0.399 |           0.951 |
| router10_t10 | all        |       0.498 | 0.981 |           0.471 |           0.95  |
| router10_t05 | cheap      |       0.418 | 0.965 |           0.349 |           0.95  |
| router10_t05 | all        |       0.418 | 0.987 |           0.394 |           0.95  |
| router10_t20 | cheap      |       0.604 | 0.946 |           0.507 |           0.951 |
| router10_t20 | all        |       0.604 | 0.971 |           0.543 |           0.951 |
| router15_t10 | cheap      |       0.567 | 0.951 |           0.485 |           0.95  |
| router15_t10 | all        |       0.567 | 0.981 |           0.537 |           0.95  |
| oracle10_t10 | cheap      |       0.678 | 0.948 |           0.598 |           0.95  |
| oracle10_t10 | all        |       0.678 | 0.988 |           0.691 |           0.95  |

## 6. Feature importances (random forest)

Top cheap features:

| features   | feature          |   importance |
|:-----------|:-----------------|-------------:|
| cheap      | log_T            |        0.175 |
| cheap      | cent_recent_frac |        0.125 |
| cheap      | q_norm           |        0.099 |
| cheap      | cent_margin12    |        0.091 |
| cheap      | k_norm_mean      |        0.076 |
| cheap      | v_norm_std       |        0.068 |
| cheap      | value_spectral   |        0.066 |
| cheap      | cent_top1        |        0.064 |

## 7. Keep / Park / Kill

- **Decision: KEEP.**
- KEEP criteria met: ≥95% precision operating point (by construction); held-out-regime cheap coverage 32% (≥30% ✓); long-context read reduction 2.0× (≥2× ✓); beats entropy/mass thresholding (✓); selective slashes failures vs universal routing (50%→5.0%).
- Not STRONG KEEP because: on cheap features the strict combination (coverage≥50% AND ≥4× AND ≤2% fail AND clean transfer everywhere) is met on needle/longdoc but not on the agent regime, and held-out-layer cheap coverage (26%) is below 30%.
- The 'all-features' variant (using true attention stats) reaches 44% coverage under held-out regime — an upper bound showing better cheap proxies for entropy/top-mass would raise deployable coverage.

## 8. Honest caveats

- One 0.5B model, CPU, ≤3.2k tokens, decode last-token queries. Read reductions are component-level, not measured wall-clock.
- 'Cheap' features still include a per-block centroid router pass over K; this is O(T·d/block) and cacheable but not free.
- Labels use a specific router (sink+recent+block-route @10%). A stronger router would raise the safe base rate and coverage.
- Precision target is held in-fold; under regime shift realized precision is ~0.91–0.98, not always ≥0.95 — size the budget accordingly.
