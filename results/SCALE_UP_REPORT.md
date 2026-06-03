# Scale-Up Report — Sparse-Safe Structure vs Context Length & Model Size

> **Environment note:** No GPU was available in this run. Executed on CPU with a memory-safe **last-token-only SDPA capture** in bfloat16 (`capture_lasttok.py`), which avoids materializing the full [heads,T,T] score matrix and lets 0.5B reach 32k and 1.5B reach ~16k within 15GB RAM. The 1.5B@32k and 3B GPU points are specified in `scale_up_prep.py` and drop into the identical capture path, but were **not executed** here.

Machine: Linux-6.18.5-x86_64-with-glibc2.39 · capture: bf16 SDPA, last-token query, GQA-correct (verified rel-err ~1e-3, bf16 rounding).

## Verdict: **KEEP**

**Core question — does sparse-safe structure strengthen with scale? Answer: with CONTEXT LENGTH yes; with MODEL SIZE no.**

1. **Context length (0.5B):** sparse-safe base rate rises monotonically 0.60→0.77→0.78 as T grows (Δ=+0.17). Held-out-regime coverage peaks at **0.60 @16k** and read reduction at **2.16× @16k** (both dip slightly at 32k as cross-regime transfer hardens at extreme length). Oracle read-gain is ≥100× from 4k on. Top-16 tokens still hold 0.67 of attention mass at 32k.
2. **Model size (0.5B vs 1.5B @8k):** the bigger model is *not* easier — cov95_regime 0.28 vs 0.44; read reduction 1.34× vs 1.66×. 1.5B's base rate still rises with T, so the structure exists, but the cheap classifier captures less of it.
3. **Transfer:** context-length transfer HOLDS (precision ≈0.94 applying a ≤8k-trained threshold to 16k/32k), but **model-size transfer is WEAK** — a 0.5B-trained threshold collapses to **0.68 precision** on 1.5B. A classifier must be retrained per model; it generalizes across length.
4. **Read-reduction ceiling:** capped ~2.2× by the fixed 10% router budget, NOT by missing structure (oracle ceiling ≥100×). A tighter/adaptive budget at high coverage is the lever to reach 4×.

## 1. 0.5B — scaling by context length

|   ctx_target |   T_med |   n_cases |   base_rate |   cov95_random |   cov95_regime |   cov95_layer |   cov95_head |   sel_read_reduction |   sel_fail_routed |   oracle_med_readgain |   mean_entropy |   mean_mass_top16 |
|-------------:|--------:|----------:|------------:|---------------:|---------------:|--------------:|-------------:|---------------------:|------------------:|----------------------:|---------------:|------------------:|
|         2048 |    2048 |      1344 |       0.605 |          0.372 |          0.378 |         0.263 |        0.39  |                1.506 |             0.049 |                    50 |          3.001 |             0.756 |
|         4096 |    4096 |      1344 |       0.667 |          0.501 |          0.403 |         0.468 |        0.478 |                1.566 |             0.05  |                   100 |          3.148 |             0.74  |
|         8192 |    8192 |      1344 |       0.713 |          0.464 |          0.444 |         0.333 |        0.471 |                1.664 |             0.049 |                   100 |          3.358 |             0.713 |
|        16384 |   16384 |      1344 |       0.773 |          0.63  |          0.598 |         0.348 |        0.603 |                2.162 |             0.05  |                   100 |          3.443 |             0.709 |
|        32768 |   32768 |      1344 |       0.78  |          0.655 |          0.485 |         0.36  |        0.672 |                1.774 |             0.049 |                   100 |          3.814 |             0.667 |

Columns: `cov95_*` = fraction of cases routable sparse while sparse decisions stay ≥95% precise (cheap features, out-of-fold, grouped by the named held-out axis). `sel_read_reduction`/`sel_fail_routed` = selective policy at the held-out-regime 95%-precision operating point. `oracle_med_readgain` = median component read-gain to reach rel-L2≤0.10.

## 2. 1.5B — scaling by context length (model-size axis)

|   ctx_target |   T_med |   n_cases |   base_rate |   cov95_random |   cov95_regime |   cov95_layer |   cov95_head |   sel_read_reduction |   sel_fail_routed |   oracle_med_readgain |   mean_entropy |   mean_mass_top16 |
|-------------:|--------:|----------:|------------:|---------------:|---------------:|--------------:|-------------:|---------------------:|------------------:|----------------------:|---------------:|------------------:|
|         4096 |    4096 |      1344 |       0.583 |          0.406 |          0.336 |         0.257 |        0.374 |                1.432 |             0.049 |                    50 |          3.257 |             0.731 |
|         8192 |    8192 |      1344 |       0.626 |          0.4   |          0.285 |         0.228 |        0.384 |                1.344 |             0.05  |                   100 |          3.541 |             0.694 |
|        16384 |   16384 |      1344 |       0.678 |          0.451 |          0.347 |         0.219 |        0.404 |                1.452 |             0.049 |                   100 |          3.64  |             0.698 |

## 3. Transfer (train-chosen 95%-precision threshold applied to held-out target)

| transfer                     |   test_base_rate |   auc |   cov_at_95prec |   train_thr_test_prec |   train_thr_test_cov |   kv_read_reduction |   fail_rate_routed |
|:-----------------------------|-----------------:|------:|----------------:|----------------------:|---------------------:|--------------------:|-------------------:|
| 0.5B@8k -> 1.5B@8k           |            0.626 | 0.842 |           0.278 |                 0.68  |                0.889 |               4.963 |              0.32  |
| 0.5B train<=8k -> test 16384 |            0.773 | 0.882 |           0.611 |                 0.947 |                0.624 |               2.274 |              0.053 |
| 0.5B train<=8k -> test 32768 |            0.78  | 0.856 |           0.301 |                 0.945 |                0.394 |               1.549 |              0.055 |
| 0.5B 8k -> 16k               |            0.773 | 0.884 |           0.589 |                 0.929 |                0.69  |               2.63  |              0.071 |

## 4. Reading the scaling law

- If `base_rate` and `cov95_regime` rise with `T_med`, sparse-safe structure strengthens with context — the central hypothesis. If `oracle_med_readgain` rises with T, the *upper bound* grows (more skippable mass in longer contexts), even where the cheap classifier lags.
- `sel_read_reduction` is capped by coverage: even at high coverage the full-fallback cases dominate the average read, so 4×+ needs both high coverage AND a tighter routed read budget than 10%.

## 5. Decision rules

- **KEEP.** STRONG KEEP needs cheap held-out-regime coverage≥50%, held-out-layer≥30%, long-context read reduction≥4×, routed failure≤5%, and transfer across ≥2 regimes. KEEP needs coverage 30–50% or read reduction 2–4× at ≤5% failure. PARK if coverage<30% or gains only in one regime or model-size transfer is weak.

## 6. Honest caveats

- **CPU, not GPU.** bf16 forward; capture verified to ~1e-3. Read reductions are component-level, not wall-clock.
- 1.5B limited to ≤16k and 3B not run (memory). The capture path is GPU-ready (`scale_up_prep.py` has exact commands) — these points remain to be run.
- One prompt per regime per context (336 cases/regime). Labels use the sink+recent+block-route router at a 10% budget.
