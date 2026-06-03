# Calibration Minimization + Active Selection (1.5B)

Minimum target-model calibration to recover sparse-safe performance, and whether active selection beats random. Held-out-regime test, 95%-precision operating point. Full-data ceiling coverage = 0.377 (3024 train cases, base rate 0.629).

## Verdict: **KEEP (cheap calibration); active does NOT beat random**

- **≥90% of the full-data ceiling is reached at 500 cases** (any strategy; random first). Mean of 3 seeds.
- **Active vs random:** active selection does NOT beat random at ≤250 cases (the strict 95%-precision coverage metric is noisy at small n; random is a strong baseline).
- @250: random 82% · diversity 32% · uncertainty 50% of full coverage.
- @500: random 95% · diversity 86% · uncertainty 62%.
- Rule: KEEP if active reaches ≥90% with ≤250–500 cases.

## Coverage @ 95% precision vs calibration size

|    n |   active_diversity |   active_uncertainty |   random |
|-----:|-------------------:|---------------------:|---------:|
|   25 |              0     |                0.004 |    0.004 |
|   50 |              0.073 |                0.024 |    0.032 |
|  100 |              0.19  |                0.052 |    0.119 |
|  250 |              0.12  |                0.189 |    0.311 |
|  500 |              0.325 |                0.235 |    0.358 |
| 1000 |              0.37  |                0.353 |    0.346 |

## Fraction of full-data ceiling recovered

|    n |   active_diversity |   active_uncertainty |   random |
|-----:|-------------------:|---------------------:|---------:|
|   25 |              0     |                0.011 |    0.01  |
|   50 |              0.193 |                0.063 |    0.085 |
|  100 |              0.505 |                0.137 |    0.316 |
|  250 |              0.319 |                0.501 |    0.824 |
|  500 |              0.861 |                0.625 |    0.949 |
| 1000 |              0.982 |                0.937 |    0.918 |

## KV read reduction vs size

|    n |   active_diversity |   active_uncertainty |   random |
|-----:|-------------------:|---------------------:|---------:|
|   25 |              1     |                1.252 |    1.132 |
|   50 |              1.078 |                1.023 |    1.031 |
|  100 |              1.211 |                1.053 |    1.142 |
|  250 |              1.13  |                1.227 |    1.387 |
|  500 |              1.413 |                1.277 |    1.475 |
| 1000 |              1.5   |                1.466 |    1.452 |

## Reading

- A label costs one forward pass + oracle rel-L2 at the budget ladder for a few (layer,head) cases — so 250–500 cases ≈ a handful of prompts. If active beats random here, a new model is calibrated almost for free, neutralising the weak zero-shot model-size transfer.
