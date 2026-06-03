# Adaptive Risk Allocation Report (token ladder)

Fix the naive cascade's compounding failure with global risk allocation. Long-context (≥8k), held-out-regime OOF, failure budget ≤5%, rel-L2 threshold 0.1.

## Verdict: **PARK**

- Best policy under ≤5% failure: **expected_cost** → **6.37× KV read reduction** vs fixed-10% **4.47×** = **1.42×** improvement.
- Rule: STRONG KEEP if ≥2× over fixed-10%; KEEP if ≥1.5×; else PARK.

## Policy comparison

| policy             |   coverage |   fail_rate_routed |   kv_read_reduction |   mean_budget |    n |
|:-------------------|-----------:|-------------------:|--------------------:|--------------:|-----:|
| fixed_10pct        |      0.863 |              0.05  |               4.47  |         0.1   | 5376 |
| naive_cascade      |      0.977 |              0.135 |              13.972 |         0.05  | 5376 |
| monotone_threshold |      0.85  |              0.046 |               4.821 |         0.067 | 5376 |
| conformal_cascade  |      0.823 |              0.043 |               4.339 |         0.066 | 5376 |
| direct_multiclass  |      0.85  |              0.046 |               4.821 |         0.067 | 5376 |
| expected_cost      |      0.943 |              0.043 |               6.368 |         0.106 | 5376 |

## Reading

- `naive_cascade` is the old broken policy (per-stage 95% → compounded failure). The global-allocation policies (monotone / conformal / direct / expected-cost) hold the *union* routed failure ≤5% and recover read reduction.
- A single global confidence knob (`monotone_threshold`) is the simplest fix and usually competitive; `expected_cost` exposes the read/failure trade-off via the penalty.
