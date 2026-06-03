# Locality-Regularized Routing (MAIN NEXT Experiment 2)

Cheap NON-oracle routers that select blocks (centroid, centroid+radius upper-bound, max-sketch, recent/sink+routed). Long-context (≥8k). Best operating point = smallest read ratio with median rel-L2≤0.10.

## Verdict: **KEEP**

- Best block router: **recent_sink_routed @ block 32** → **51.6× read reduction** at median rel-L2 0.094, mass 0.76, oracle-support 0.39.
- Rule: KEEP if a non-oracle block router hits rel-L2≤0.10 at ≥4× read reduction with block utilization≥0.5 / over-read≤2×.

## Block-router operating points (median pass)

| policy             |   block_size |   read_ratio |   read_reduction |   rl2_med |   pass10 |   mass |   support |
|:-------------------|-------------:|-------------:|-----------------:|----------:|---------:|-------:|----------:|
| centroid           |           16 |        0.1   |           10.042 |     0.075 |    0.545 |  0.809 |     0.552 |
| centroid           |           32 |        0.101 |            9.949 |     0.083 |    0.523 |  0.79  |     0.516 |
| centroid           |           64 |        0.15  |            6.687 |     0.079 |    0.603 |  0.833 |     0.52  |
| centroid           |          128 |        0.151 |            6.619 |     0.092 |    0.562 |  0.814 |     0.499 |
| centroid_ub        |           16 |        0.1   |           10.032 |     0.063 |    0.567 |  0.782 |     0.423 |
| centroid_ub        |           32 |        0.101 |            9.947 |     0.068 |    0.569 |  0.781 |     0.425 |
| centroid_ub        |           64 |        0.101 |            9.902 |     0.08  |    0.561 |  0.78  |     0.418 |
| centroid_ub        |          128 |        0.098 |           10.176 |     0.078 |    0.561 |  0.78  |     0.401 |
| centroid_ub        |          256 |        0.093 |           10.735 |     0.086 |    0.538 |  0.772 |     0.371 |
| recent_sink_routed |           16 |        0.02  |           51.216 |     0.083 |    0.531 |  0.776 |     0.435 |
| recent_sink_routed |           32 |        0.019 |           51.598 |     0.094 |    0.511 |  0.756 |     0.391 |
| recent_sink_routed |           64 |        0.049 |           20.272 |     0.053 |    0.607 |  0.815 |     0.417 |
| recent_sink_routed |          128 |        0.048 |           20.938 |     0.07  |    0.583 |  0.8   |     0.37  |
| recent_sink_routed |          256 |        0.056 |           17.828 |     0.083 |    0.57  |  0.797 |     0.354 |

## Reading

- Block routers select *contiguous* blocks, so by construction utilization is 1.0 within a selected block and over-read ≈1 — the GPU-hostile scatter problem disappears IF accuracy holds. The question is purely the rel-L2 / read-reduction trade-off vs the block-oracle ceiling (Exp 1).
- `centroid_ub` adds a radius term so a block is only skipped when its best possible score is below the cut — a cheap certificate. `maxsketch` approximates the per-block max logit via a random projection.
- The selective (classifier-gated) version of the best block router, with global risk allocation, is evaluated in ADAPTIVE_RISK_REPORT_BLOCK.md.
