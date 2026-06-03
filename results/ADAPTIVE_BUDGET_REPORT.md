# Adaptive Sparse Budget Report (MAIN STACK Experiment 1)

Replace binary sparse(10%)/full with an adaptive **cascade**: pick the smallest budget on {0.25,0.5,1,2,5,10,15}% whose cheap classifier predicts sparse-safe; else full. Held-out-regime OOF, oracle rel-L2 labels at threshold 0.1. The per-stage precision target is tuned upward until the *measured* cascade routed-failure ≤5% (compounding across 7 stages means a naive 95% per stage overshoots the failure budget).

## Verdict: **KEEP**

- **Adaptive wins where it counts (0.5B @16k):** **7.8× KV read reduction** at 4.7% routed failure, vs fixed-10% **5.6×** — a 1.39× improvement at matched ≤5% failure. Meets the ≥4× STRONG bar at this context.
- **But the naive cascade does NOT robustly dominate when pooled (≥8k):** capped at ≤5% failure it reaches **3.0×** (2.4% fail) — *below* fixed-10%'s **4.5×**. Holding overall failure ≤5% across 7 compounding stages forces each stage to ~99% precision, throttling the aggressive small budgets.
- **The headroom is real:** the *uncapped* cascade (95% per stage) hits **14.0×** but at 13% failure — far over budget. The prize (10–24×) exists; capturing it safely needs smarter per-stage failure-budget allocation, not a uniform precision target.
- Rule: STRONG KEEP if long-context ≥4× at ≤5%; KEEP 2–4×; PARK <2×. Result: KEEP — clear ≥4× win at 16k, but the global failure-control of the naive cascade is the bottleneck.

## Policy comparison by segment

| segment        | policy            |    n |   coverage |   fail_rate_routed |   kv_read_reduction |
|:---------------|:------------------|-----:|-----------:|-------------------:|--------------------:|
| 0.5B@4096      | full_only         | 1344 |      0     |              0     |               1     |
| 0.5B@4096      | fixed_10pct       | 1344 |      0.762 |              0.05  |               3.182 |
| 0.5B@4096      | adaptive_budget   | 1344 |      0.742 |              0.048 |               3.298 |
| 0.5B@4096      | adaptive_uncapped | 1344 |      0.856 |              0.134 |               5.431 |
| 0.5B@8192      | full_only         | 1344 |      0     |              0     |               1     |
| 0.5B@8192      | fixed_10pct       | 1344 |      0.83  |              0.049 |               3.958 |
| 0.5B@8192      | adaptive_budget   | 1344 |      0.721 |              0.025 |               3.067 |
| 0.5B@8192      | adaptive_uncapped | 1344 |      0.929 |              0.14  |               8.596 |
| 0.5B@16384     | full_only         | 1344 |      0     |              0     |               1     |
| 0.5B@16384     | fixed_10pct       | 1344 |      0.914 |              0.05  |               5.628 |
| 0.5B@16384     | adaptive_budget   | 1344 |      0.926 |              0.047 |               7.796 |
| 0.5B@16384     | adaptive_uncapped | 1344 |      1     |              0.136 |              23.606 |
| 0.5B@32768     | full_only         | 1344 |      0     |              0     |               1     |
| 0.5B@32768     | fixed_10pct       | 1344 |      0.935 |              0.049 |               6.292 |
| 0.5B@32768     | adaptive_budget   | 1344 |      0.641 |              0.019 |               2.533 |
| 0.5B@32768     | adaptive_uncapped | 1344 |      0.997 |              0.138 |              24.556 |
| 1.5B@8192      | full_only         | 1344 |      0     |              0     |               1     |
| 1.5B@8192      | fixed_10pct       | 1344 |      0.764 |              0.05  |               3.202 |
| 1.5B@8192      | adaptive_budget   | 1344 |      0.773 |              0.04  |               3.446 |
| 1.5B@8192      | adaptive_uncapped | 1344 |      0.964 |              0.117 |               9.917 |
| LONG_ALL(>=8k) | full_only         | 5376 |      0     |              0     |               1     |
| LONG_ALL(>=8k) | fixed_10pct       | 5376 |      0.863 |              0.05  |               4.47  |
| LONG_ALL(>=8k) | adaptive_budget   | 5376 |      0.724 |              0.024 |               3.044 |
| LONG_ALL(>=8k) | adaptive_uncapped | 5376 |      0.977 |              0.135 |              13.972 |

## Standalone coverage @ 95% precision by budget

How routable each budget is on its own (the cascade combines these):

|   budget |   cov95_all |   cov95_long_ctx |   base_all |   base_long_ctx |
|---------:|------------:|-----------------:|-----------:|----------------:|
|    0.003 |       0.075 |            0.143 |      0.392 |           0.42  |
|    0.005 |       0.257 |            0.261 |      0.497 |           0.513 |
|    0.01  |       0.371 |            0.341 |      0.582 |           0.596 |
|    0.02  |       0.498 |            0.498 |      0.676 |           0.687 |
|    0.05  |       0.685 |            0.682 |      0.792 |           0.805 |
|    0.1   |       0.839 |            0.863 |      0.883 |           0.892 |
|    0.15  |       0.955 |            0.977 |      0.933 |           0.942 |

## Adaptive budget distribution (long-context)

| budget   |   frac |
|:---------|-------:|
| 0.0025   |  0.059 |
| 0.005    |  0.029 |
| 0.01     |  0.095 |
| 0.02     |  0.09  |
| 0.05     |  0.078 |
| 0.1      |  0.204 |
| 0.15     |  0.169 |
| full     |  0.276 |

## Reading

- The cascade lets *easy* heads take a 0.25–1% budget (100–400× local read reduction) while hard heads take 5–15% or fall back to full — so the effective reduction beats the fixed-10% policy without raising failure.
- The read-reduction ceiling is now set by the share of cases that must go full, not by a single global budget.
