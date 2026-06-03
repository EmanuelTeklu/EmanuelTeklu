# Adaptive Risk Allocation Report (block-realizable ladder)

Fix the naive cascade's compounding failure with global risk allocation. Long-context (≥8k), held-out-regime OOF, failure budget ≤5%, rel-L2 threshold 0.1.

## Verdict: **PARK**

- Best policy under ≤5% failure: **expected_cost** → **2.28× KV read reduction** vs fixed-10% **2.21×** = **1.04×** improvement.
- Rule: STRONG KEEP if ≥2× over fixed-10%; KEEP if ≥1.5×; else PARK.

## Policy comparison

| policy             |   coverage |   fail_rate_routed |   kv_read_reduction |   mean_budget |    n |
|:-------------------|-----------:|-------------------:|--------------------:|--------------:|-----:|
| fixed_10pct        |      0.608 |              0.05  |               2.206 |         0.1   | 5376 |
| naive_cascade      |      0.743 |              0.1   |               3.286 |         0.063 | 5376 |
| monotone_threshold |      0.516 |              0.043 |               1.911 |         0.076 | 5376 |
| conformal_cascade  |      0.547 |              0.046 |               2.024 |         0.075 | 5376 |
| direct_multiclass  |      0.516 |              0.043 |               1.911 |         0.076 | 5376 |
| expected_cost      |      0.642 |              0.04  |               2.284 |         0.125 | 5376 |

## Reading

- `naive_cascade` is the old broken policy (per-stage 95% → compounded failure). The global-allocation policies (monotone / conformal / direct / expected-cost) hold the *union* routed failure ≤5% and recover read reduction.
- A single global confidence knob (`monotone_threshold`) is the simplest fix and usually competitive; `expected_cost` exposes the read/failure trade-off via the penalty.
