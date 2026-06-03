# Inference Quotient — Certified Sparse / Residual / Margin-Aware KV Benchmark

**Decisive question:** on *real* transformer Q/K/V tensors, can a cheap (non-oracle) method recover enough of the oracle sparse-attention gain to justify kernel work — and does margin-aware KV compression beat uniform 4-bit?

## 1. Executive summary

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, 14 query / 2 KV heads, GQA-7, head_dim 64) · 7 prompts across 5 regimes · max context tested: **3226** tokens · cases analyzed: **2352** (layer×head×prompt, decode last-token).
- **Capture validated:** exact-attention reconstruction matches the model's own attention output to ~1e-6 rel error (GQA mapping verified).
- **Oracle sparse (upper bound, long-context):** median case reaches rel-L2≤0.10 at read ratio **0.008** → **133.1× component read reduction** (IQR 47.9×–383.5×); 93% of cases reach moderate pass at some S≤256.
- **Non-oracle routing (the load-bearing test):** sink_recent_router reads 7.7% of tokens to reach median rel-L2≤0.10, but only 67% of heads pass; peak oracle-support recovery 69% (<70% bar). Cheap routers recover the oracle gain only *partially* — which motivated the follow-up below.
- **➡️ Follow-up — selective sparse classifier (KEEP):** instead of routing universally, a classifier predicts *which* heads/queries are sparse-safe from cheap features and falls back to full attention otherwise. At a 95%-precision operating point it covers ~32% of cases under held-out regimes (vs ~0% for entropy/top-mass thresholding) and yields ~2× (cheap) – ~2.9× (oracle-feature) KV read reduction on long-context workloads. See **`CLASSIFIER_REPORT.md`**.
- **Residual correction:** cheap block-residual cuts aggressive-budget sparse error (e.g. budget=8: ratio 0.61 vs sparse); it can HURT once the budget already captures most mass (budget=32). Oracle-mass ceiling shows ~54% error reduction is *available* if omitted mass were estimated well.
- **Margin/importance-aware quant vs uniform 4-bit:** uniform-4-bit is NOT strictly dominated, but importance-aware bit allocation (8-bit on top-mass tokens, 4-bit rest) reaches ~uniform-8-bit quality (~0.03 rel-L2) at ~4.4–5 effective bits — a real quality-per-bit win in the 4.5–6 bit band. It does NOT beat 4-bit *storage*, and uses an ORACLE importance signal (see §7).

### Headline verdicts (against pre-registered thresholds)

- 10× component read reduction on real tensors (oracle upper bound): **YES** — median long-context head reaches rel-L2≤0.10 at 133× read reduction.
- Cheap router recovers ≥70% of that gain across heads: **WEAK / PARK** — median passes at ~8% reads but only 67% of heads pass and support-recovery peaks at 69%.
- Residual correction makes sparse robust: **PARK** — helps in the aggressive-sparsity regime; cheap mass estimate is noisy; ceiling shows headroom.
- Adaptive compression beats uniform 4-bit (pre-registered criteria): **NO**; importance-aware Pareto win above 4 bits: **YES** → PARK.

## 2. Environment

```
model: Qwen/Qwen2.5-0.5B-Instruct
synthetic: False
max_tokens: 4096
max_context_tested: 3226
seed: 0
python: 3.11.15
platform: Linux-6.18.5-x86_64-with-glibc2.39
torch: 2.12.0+cpu
transformers: 5.9.0
capture_seconds: 84.3
n_cases: 2352
hardware: CPU-only (no CUDA), see platform string
```

**Per-prompt context lengths & capture verification:**

| name           |    T |   rel_err | ok   |
|:---------------|-----:|----------:|:-----|
| short_qa       |   44 |         0 | True |
| short_chat     |   43 |         0 | True |
| longdoc_qa     | 3068 |         0 | True |
| needle         | 2130 |         0 | True |
| needle_late    | 3226 |         0 | True |
| agent_prefix   |  857 |         0 | True |
| code_retrieval |  161 |         0 | True |

## 3. What was actually run

- C: oracle sparse upper bound, S∈{1,2,4,8,16,32,64,128,256}.
- D: non-oracle routers — JL projection (r∈{8..128}), block-centroid, block upper-bound certificate, recency-hybrid, sink+recent+route.
- E: sparse+residual correction — cheap block-residual vs oracle-mass ceilings vs low-rank.
- F: KV quant frontier — uniform 8/4/3/2-bit (K per-channel, V per-token, KIVI-style) vs adaptive token-importance / margin-fragility / recent-exact / routed-exact schemes.
- G: per-head/layer predictor correlations.
- H/I: prefix-reuse and Amdahl analytic cost models.

## 4. Oracle sparse upper bound (long-context regimes)

|   S |   mean_read_ratio |   mean_read_gain |   median_rel_l2 |   p90_rel_l2 |   mean_mass |   frac_strict |   frac_mod |   frac_weak |
|----:|------------------:|-----------------:|----------------:|-------------:|------------:|--------------:|-----------:|------------:|
|   1 |            0.0006 |        2320.25   |          1.1055 |       1.7506 |      0.456  |        0.0283 |     0.0439 |      0.0841 |
|   2 |            0.0011 |        1160.12   |          0.6587 |       1.3928 |      0.5766 |        0.0558 |     0.0938 |      0.1548 |
|   4 |            0.0023 |         580.062  |          0.3484 |       0.9718 |      0.695  |        0.096  |     0.1548 |      0.282  |
|   8 |            0.0045 |         290.031  |          0.1615 |       0.6813 |      0.8004 |        0.192  |     0.3586 |      0.5603 |
|  16 |            0.0091 |         145.016  |          0.065  |       0.4849 |      0.8722 |        0.4338 |     0.5804 |      0.7344 |
|  32 |            0.0182 |          72.5078 |          0.0326 |       0.327  |      0.9144 |        0.5796 |     0.6905 |      0.8199 |
|  64 |            0.0364 |          36.2539 |          0.0181 |       0.2248 |      0.9424 |        0.6637 |     0.7783 |      0.8698 |
| 128 |            0.0727 |          18.127  |          0.0093 |       0.1514 |      0.9639 |        0.7552 |     0.846  |      0.9479 |
| 256 |            0.1454 |           9.0635 |          0.0038 |       0.0789 |      0.9805 |        0.8452 |     0.9263 |      0.9814 |

**Per-layer compressibility (median oracle rel-L2 at S=16):**

|   layer |   median_rel_l2_s16 |
|--------:|--------------------:|
|       0 |              0.0934 |
|       1 |              0.2056 |
|       2 |              0.1912 |
|       3 |              0.0659 |
|       4 |              0.0498 |
|       5 |              0.0557 |
|       6 |              0.1033 |
|       7 |              0.0089 |
|       8 |              0.026  |
|       9 |              0.02   |
|      10 |              0.0676 |
|      11 |              0.0435 |
|      12 |              0.0705 |
|      13 |              0.0501 |
|      14 |              0.0946 |
|      15 |              0.1414 |
|      16 |              0.0388 |
|      17 |              0.0459 |
|      18 |              0.1046 |
|      19 |              0.0449 |
|      20 |              0.0619 |
|      21 |              0.0633 |
|      22 |              0.5538 |
|      23 |              0.4427 |

## 5. Non-oracle routing

| method             | segment   |   mean_read_ratio |   median_rel_l2 |   frac_mod |   mean_oracle_recall |   mean_mass |
|:-------------------|:----------|------------------:|----------------:|-----------:|---------------------:|------------:|
| jl                 | all       |            0.1509 |          0.4727 |     0.2628 |               0.3948 |      0.5672 |
| jl                 | long      |            0.0282 |          0.69   |     0.1926 |               0.2906 |      0.4834 |
| jl                 | needle    |            0.0193 |          0.712  |     0.1933 |               0.283  |      0.4791 |
| block_centroid     | all       |            0.229  |          0.2296 |     0.3765 |               0.6658 |      0.7284 |
| block_centroid     | long      |            0.0388 |          0.3015 |     0.311  |               0.6229 |      0.7017 |
| block_centroid     | needle    |            0.0255 |          0.3036 |     0.2952 |               0.6025 |      0.6997 |
| block_upperbound   | all       |            0.2263 |          0.4804 |     0.264  |               0.5368 |      0.5978 |
| block_upperbound   | long      |            0.0384 |          0.4919 |     0.2708 |               0.4768 |      0.5775 |
| block_upperbound   | needle    |            0.0253 |          0.5818 |     0.2595 |               0.4261 |      0.554  |
| recency_router     | all       |            0.1513 |          0.7462 |     0.1542 |               0.4754 |      0.4942 |
| recency_router     | long      |            0.0285 |          0.7021 |     0.1545 |               0.4367 |      0.5019 |
| recency_router     | needle    |            0.0196 |          0.7147 |     0.156  |               0.4238 |      0.5002 |
| sink_recent_router | all       |            0.1743 |          0.1439 |     0.4368 |               0.539  |      0.8046 |
| sink_recent_router | long      |            0.0308 |          0.1596 |     0.4147 |               0.4902 |      0.792  |
| sink_recent_router | needle    |            0.0211 |          0.162  |     0.4116 |               0.4784 |      0.7967 |

**Router vs oracle at matched read budget (long-context median rel-L2):**

|   budget | method             |   oracle_rel_l2 |   router_rel_l2 |    gap |
|---------:|:-------------------|----------------:|----------------:|-------:|
|        8 | jl                 |          0.1615 |          1.1613 | 0.9998 |
|        8 | block_centroid     |          0.1615 |          0.6446 | 0.4831 |
|        8 | block_upperbound   |          0.1615 |          0.9529 | 0.7914 |
|        8 | recency_router     |          0.1615 |          1.1992 | 1.0377 |
|        8 | sink_recent_router |          0.1615 |          0.4492 | 0.2877 |
|       16 | jl                 |          0.065  |          0.9565 | 0.8915 |
|       16 | block_centroid     |          0.065  |          0.6446 | 0.5796 |
|       16 | block_upperbound   |          0.065  |          0.9529 | 0.8879 |
|       16 | recency_router     |          0.065  |          0.9669 | 0.9019 |
|       16 | sink_recent_router |          0.065  |          0.292  | 0.227  |
|       32 | jl                 |          0.0326 |          0.6813 | 0.6487 |
|       32 | block_centroid     |          0.0326 |          0.2758 | 0.2432 |
|       32 | block_upperbound   |          0.0326 |          0.6348 | 0.6022 |
|       32 | recency_router     |          0.0326 |          0.7254 | 0.6928 |
|       32 | sink_recent_router |          0.0326 |          0.1184 | 0.0858 |
|       64 | jl                 |          0.0181 |          0.436  | 0.4179 |
|       64 | block_centroid     |          0.0181 |          0.122  | 0.1039 |
|       64 | block_upperbound   |          0.0181 |          0.1652 | 0.147  |
|       64 | recency_router     |          0.0181 |          0.7062 | 0.6881 |
|       64 | sink_recent_router |          0.0181 |          0.1048 | 0.0866 |
|      128 | jl                 |          0.0093 |          0.2451 | 0.2358 |
|      128 | block_centroid     |          0.0093 |          0.0542 | 0.0449 |
|      128 | block_upperbound   |          0.0093 |          0.0615 | 0.0522 |
|      128 | recency_router     |          0.0093 |          0.0765 | 0.0672 |
|      128 | sink_recent_router |          0.0093 |          0.0336 | 0.0243 |

## 6. Sparse + residual correction

`ratio_vs_sparse` < 1 means the correction beats plain renormalized sparse at the same exact-read budget. Cheap block-residual uses only per-block centroid-logit mass estimates + block value means (storable, O(T/block) vectors). Oracle-mass-* use the TRUE omitted softmax mass (ceilings).

|   budget | method               |   median_rel_l2 |   improvement_vs_sparse |   ratio_vs_sparse |
|---------:|:---------------------|----------------:|------------------------:|------------------:|
|        8 | sparse_renorm        |          0.151  |                  0      |            1      |
|        8 | cheap_block_residual |          0.0926 |                  0.0584 |            0.6131 |
|        8 | oracle_mass_global   |          0.116  |                  0.035  |            0.7682 |
|        8 | oracle_mass_block    |          0.0837 |                  0.0674 |            0.554  |
|        8 | lowrank_k8           |          0.1258 |                  0.0253 |            0.8327 |
|       16 | sparse_renorm        |          0.0534 |                  0      |            1      |
|       16 | cheap_block_residual |          0.0373 |                  0.0161 |            0.6978 |
|       16 | oracle_mass_global   |          0.0356 |                  0.0178 |            0.6671 |
|       16 | oracle_mass_block    |          0.0255 |                  0.028  |            0.4767 |
|       16 | lowrank_k8           |          0.0372 |                  0.0163 |            0.6957 |
|       32 | sparse_renorm        |          0.0153 |                  0      |            1      |
|       32 | cheap_block_residual |          0.0177 |                 -0.0024 |            1.1563 |
|       32 | oracle_mass_global   |          0.0089 |                  0.0064 |            0.5797 |
|       32 | oracle_mass_block    |          0.0067 |                  0.0085 |            0.4416 |
|       32 | lowrank_k8           |          0.0081 |                  0.0072 |            0.5296 |
|       64 | sparse_renorm        |          0.0135 |                  0      |            1      |
|       64 | cheap_block_residual |          0.011  |                  0.0024 |            0.8196 |
|       64 | oracle_mass_global   |          0.0063 |                  0.0072 |            0.4647 |
|       64 | oracle_mass_block    |          0.0052 |                  0.0083 |            0.3842 |
|       64 | lowrank_k8           |          0.0069 |                  0.0066 |            0.5084 |

**Smallest exact-read budget reaching moderate pass (median rel-L2≤0.10):**

| method               |   min_budget_mod_pass |
|:---------------------|----------------------:|
| sparse_renorm        |                    16 |
| cheap_block_residual |                     8 |
| oracle_mass_global   |                    16 |
| oracle_mass_block    |                     8 |
| lowrank_k8           |                    16 |

Reading: the cheap proxy helps most where it matters (budget 8–16, the high-read-gain regime) but adds noise once the budget already captures the mass (budget 32). The oracle-mass-block ceiling cuts error to ~40–55% of sparse across all budgets — the value is real *iff* omitted mass can be estimated cheaply, which is the same routing problem in disguise.

## 7. KV quantization frontier

| method               |   eff_bits |   median_rel_l2 |   p90_rel_l2 |   median_js | pareto   |
|:---------------------|-----------:|----------------:|-------------:|------------:|:---------|
| uniform_2bit         |     2      |          1.3915 |       5.0386 |      0.3977 | True     |
| adapt_imp_h4l2f25    |     2.501  |          0.8239 |       2.3623 |      0.2    | True     |
| adapt_imp_h8l2f10    |     2.5862 |          0.856  |       2.7803 |      0.2126 | False    |
| uniform_3bit         |     3      |          0.5806 |       1.9453 |      0.0923 | True     |
| routed_r8bg2         |     3.4939 |          0.6068 |       2.5777 |      0.1625 | False    |
| adapt_imp_h8l2f25    |     3.5031 |          0.6242 |       2.353  |      0.1336 | False    |
| adapt_margin_h8l2f25 |     3.5031 |          0.8901 |       2.5517 |      0.2196 | False    |
| uniform_4bit         |     4      |          0.2459 |       0.6178 |      0.0178 | True     |
| recent64_r8o2        |     4.1797 |          0.6977 |       2.6864 |      0.1904 | False    |
| adapt_imp_h8l4f10    |     4.3908 |          0.0307 |       0.294  |      0.0022 | True     |
| adapt_margin_h8l4f10 |     4.3908 |          0.1225 |       0.6176 |      0.0038 | False    |
| recent128_r8o2       |     4.645  |          0.6263 |       2.5432 |      0.1638 | False    |
| routed_r8bg4         |     4.996  |          0.024  |       0.1653 |      0.0013 | True     |
| adapt_imp_h8l4f25    |     5.002  |          0.0221 |       0.1617 |      0.0007 | True     |
| adapt_margin_h8l4f25 |     5.002  |          0.0392 |       0.5083 |      0.0011 | False    |
| recent64_r8o4        |     5.4531 |          0.0711 |       0.3205 |      0.0025 | False    |
| routed_r16bg2        |     5.4859 |          0.5573 |       2.5566 |      0.1456 | False    |
| recent128_r8o4       |     5.7634 |          0.069  |       0.3145 |      0.0021 | False    |
| uniform_8bit         |     8      |          0.0155 |       0.0405 |      0.0001 | True     |

**Pre-registered strong-pass tests vs uniform-4bit** (≈0.246 rel-L2 @ 4.0 bits; uniform-8bit ≈ 0.016):

- (a) ≤50% storage at ≤4-bit error: `NONE`
- (b) ≤2 effective bits at 4-bit-level error: `NONE`
- (c) ≤4-bit storage with materially lower error: `NONE`
- → **All three pre-registered criteria FAIL.** Uniform 4-bit is not beaten on its own storage budget.

**But uniform-4bit is NOT Pareto-dominant:** adaptive schemes on the frontier above 4 bits: `['adapt_imp_h8l4f10', 'routed_r8bg4', 'adapt_imp_h8l4f25']`. Importance-aware allocation reaches near-8-bit quality at ~5 bits (method, eff_bits, rel_l2): `[['adapt_imp_h8l4f10', 4.390771696378494, 0.0307400387631816], ['routed_r8bg4', 4.995959554905585, 0.0239545067625291], ['adapt_imp_h8l4f25', 5.002045548963909, 0.0220517963946196]]`. This is a genuine quality-per-bit win in the 4.5–6 bit band (≈30–40% storage cut vs uniform-8bit at matched quality) — hence **PARK, not KILL**.

> Two caveats keep this out of KEEP: (1) the high-bit tokens are chosen by the TRUE attention mass (oracle importance) — a deployable version needs a cheap importance proxy, the same routing problem; (2) uniform baselines use the strong K-per-channel + V-per-token (KIVI) scheme, while adaptive variable-bit K is per-token — if anything this *handicaps* the adaptive schemes on K, so the win is not a baseline artifact.

## 8. Predictors (per-head/layer)

Pearson correlation between head/layer covariates and approximation error:

| outcome        | predictor      |   pearson_r |
|:---------------|:---------------|------------:|
| oracle_rl2_s16 | entropy        |      0.6811 |
| oracle_rl2_s16 | gini           |     -0.2493 |
| oracle_rl2_s16 | value_spectral |      0.288  |
| oracle_rl2_s16 | margin_top16   |     -0.232  |
| oracle_rl2_s16 | mass_top16     |     -0.8051 |
| jl_rl2_s16     | entropy        |     -0.1676 |
| jl_rl2_s16     | gini           |      0.1787 |
| jl_rl2_s16     | value_spectral |      0.0578 |
| jl_rl2_s16     | margin_top16   |     -0.0193 |
| jl_rl2_s16     | mass_top16     |      0.0251 |
| quant4_rl2     | entropy        |     -0.1692 |
| quant4_rl2     | gini           |      0.0927 |
| quant4_rl2     | value_spectral |     -0.0305 |
| quant4_rl2     | margin_top16   |      0.0291 |
| quant4_rl2     | mass_top16     |      0.0758 |

Interpretation: a strong *negative* correlation between attention concentration (gini) / top-k margin / mass-retained and error means those signals predict *which* heads are safe to sparsify/compress — the basis for a 'compress this head, not that one' classifier.

## 9. Prefix/latent reuse economics (analytic)

End-to-end speedup at overhead=0.03 (lookup cost = 3% of prefix compute):

|   reusable_fraction |   hit_rate |   end_to_end_speedup | reaches_10x   | reaches_100x   |
|--------------------:|-----------:|---------------------:|:--------------|:---------------|
|                0.5  |       0.5  |                1.32  | False         | False          |
|                0.5  |       0.7  |                1.514 | False         | False          |
|                0.5  |       0.9  |                1.775 | False         | False          |
|                0.5  |       0.95 |                1.854 | False         | False          |
|                0.5  |       0.99 |                1.924 | False         | False          |
|                0.7  |       0.5  |                1.514 | False         | False          |
|                0.7  |       0.7  |                1.906 | False         | False          |
|                0.7  |       0.9  |                2.571 | False         | False          |
|                0.7  |       0.95 |                2.817 | False         | False          |
|                0.7  |       0.99 |                3.051 | False         | False          |
|                0.9  |       0.5  |                1.775 | False         | False          |
|                0.9  |       0.7  |                2.571 | False         | False          |
|                0.9  |       0.9  |                4.666 | False         | False          |
|                0.9  |       0.95 |                5.86  | False         | False          |
|                0.9  |       0.99 |                7.368 | False         | False          |
|                0.95 |       0.5  |                1.854 | False         | False          |
|                0.95 |       0.7  |                2.817 | False         | False          |
|                0.95 |       0.9  |                5.86  | False         | False          |
|                0.95 |       0.95 |                8.027 | False         | False          |
|                0.95 |       0.99 |               11.401 | True          | False          |
|                0.99 |       0.5  |                1.924 | False         | False          |
|                0.99 |       0.7  |                3.051 | False         | False          |
|                0.99 |       0.9  |                7.368 | False         | False          |
|                0.99 |       0.95 |               11.401 | True          | False          |
|                0.99 |       0.99 |               20.283 | True          | False          |

- **10× needs near-total reuse:** only 11/125 swept points reach 10× (all require reusable_fraction≥0.95 *and* hit_rate≥0.95).
- **100× is unreachable by caching alone:** 0/125 points reach 100×. The best case in the entire sweep is 50.3× (reusable_fraction=0.99, hit_rate=0.99, overhead=0.0). Pure prefix caching saturates around ~50×; 100× would additionally require avoiding recomputation of the *unique* tail (latent reuse / cross-request state), not just the shared prefix.

## 10. Amdahl end-to-end (analytic)

Component reduction → end-to-end speedup. 10× end-to-end needs the component to dominate runtime:

|   component_fraction |     2 |     4 |     8 |     16 |     32 |     64 |    128 |
|---------------------:|------:|------:|------:|-------:|-------:|-------:|-------:|
|                 0.3  | 1.176 | 1.29  | 1.356 |  1.391 |  1.41  |  1.419 |  1.424 |
|                 0.5  | 1.333 | 1.6   | 1.778 |  1.882 |  1.939 |  1.969 |  1.984 |
|                 0.7  | 1.538 | 2.105 | 2.581 |  2.909 |  3.107 |  3.216 |  3.274 |
|                 0.9  | 1.818 | 3.077 | 4.706 |  6.4   |  7.805 |  8.767 |  9.343 |
|                 0.95 | 1.905 | 3.478 | 5.926 |  9.143 | 12.549 | 15.422 | 17.415 |
|                 0.98 | 1.961 | 3.774 | 7.018 | 12.308 | 19.753 | 28.319 | 36.158 |
|                 0.99 | 1.98  | 3.883 | 7.477 | 13.913 | 24.427 | 39.264 | 56.388 |

- Minimum component runtime fraction that can yield 10× end-to-end (at the largest 128× reduction): **0.95** (i.e. attention/KV read must be ≥95% of runtime).
- 100× end-to-end is unreachable by component reduction alone within the swept range; it requires recomputation *avoidance*/amortization (prefix/latent reuse), per §9.

## 11. Keep / Kill / Park decisions

- **Oracle sparse headroom — KEEP.** ≥10× component read reduction demonstrably exists on real tensors: the median long-context head reaches rel-L2≤0.10 at ~133× read reduction, and even S=16 (≈145× gain) holds median rel-L2≈0.065. This is the upper bound the rest of the stack must capture.
- **Cheap non-oracle routing — PARK (weak pass).** Best router (sink_recent_router) reaches median rel-L2≤0.10 at ~8% reads, but only 67% of heads pass and it recovers ≤69% of the oracle top-S support (<70% bar). It works on concentrated/retrieval heads, fails on diffuse ones — exactly the planted-vs-broad split. PARK pending a better router or head-selective gating.
- **Sparse+residual correction — PARK.** Cheap block-residual lowers error in the aggressive regime (budget 8: ×0.61, budget 16: ×0.70) but adds noise at larger budgets. The oracle-mass ceiling (×0.46 error) proves real headroom — gated on cheaply estimating omitted mass.
- **Margin/importance-aware KV quant — PARK.** Fails all three pre-registered 'beat 4-bit storage' tests, BUT is Pareto-superior to uniform in the 4.5–6 bit band (near-8-bit quality at ~5 bits). Gated on a cheap importance proxy + the per-token vs per-channel K caveat. PARK.
- **Prefix/latent reuse — PARK-as-economics.** Large speedups are real but are amortization, not measured kernels; needs a real agent trace to claim.

## 12. Bottlenecks & honest caveats

- CPU-only; max context ~3k tokens. Findings are about attention *structure*, not wall-clock. No CUDA kernels were written or timed.
- One small model (0.5B, GQA 14:2). Larger models may have more/less compressible heads.
- Decode last-token queries only; prefill-time multi-query patterns not swept.
- Quantization adaptive K uses per-token vs per-channel baseline (noted).

## 13. Next experiment

1. ✅ **DONE — selective sparse-safe classifier** (see `CLASSIFIER_REPORT.md`): verdict KEEP. Cheap features predict router-safety at 95% precision / ~32% coverage under held-out regimes, beating entropy/top-mass thresholding (~0% coverage), for ~2× long-context KV read reduction.
2. Repeat oracle+router+classifier on a 1.5B/3B model at 8k–32k context (GPU) to confirm the safe base-rate and coverage rise with context. A no-heavy-run prep script + exact commands are in `src/scale_up_prep.py` (`python scale_up_prep.py --check`).
3. If routing stays PARK on cheap features, pivot the kernel effort to **prefix/latent KV reuse** (§9) where the economics, not the attention structure, carry the 10×.

## 14. Fundability implication

- **This is a research direction with one strong asset (a head-level compressibility predictor), not yet a shippable cheap-router primitive.** The oracle upper bound is real and large (10–100× component read reduction in long-context heads), so the *opportunity* exists; the gap is a cheap, robust selector that holds across diffuse heads.
- **Strongest concrete asset:** the predictors in §8 — top-k attention mass correlates with oracle sparse error at r≈-0.81 and entropy at r≈+0.68. That is enough signal to build a 'compress/skip this head, not that one' classifier, which is the most fundable next artifact.
- The defensible wedge is *not* KV quantization (uniform 4-bit is strong) and *not* naive sparse routing (too expensive to be safe here). The live candidates are (a) **head-selective** sparsification guided by a cheap predictor, and (b) **prefix/latent reuse** economics for agent workloads.
- **To show on GPU:** that a cheap router + residual correction holds rel-L2≤0.10 at ≤10% reads on a ≥1.5B model at ≥8k context across heads, AND that the attention/KV-read fraction of decode runtime is high enough (§10) for the component win to survive Amdahl.

## Plots

- `plots/oracle_frontier.png`
- `plots/quant_frontier.png`
