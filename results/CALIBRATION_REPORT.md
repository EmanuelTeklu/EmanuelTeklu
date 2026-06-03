# Per-Model Calibration Report (1.5B)

**Question:** model-size transfer is weak — how many *target-model* calibration cases are needed to recover most sparse-safe performance? Cheap features, gradient-boosted classifier trained from scratch, evaluated at the 95%-precision operating point on held-out regime / layer / head / random.

## Verdict: **KEEP (cheap calibration)**

- At **1000** calibration cases, the classifier recovers **95%** of the full-data coverage (mean over splits); at **250** cases it already recovers **51%**.
- Pool: 4032 1.5B cases across contexts; base rate 0.629.
- ≤1000 cases recover most performance → cheap to calibrate a new model.

## Coverage @ 95% precision vs calibration size

|   target_n |   heldout_head |   heldout_layer |   heldout_regime |   random |
|-----------:|---------------:|----------------:|-----------------:|---------:|
|         50 |          0.212 |           0.021 |            0.08  |    0.01  |
|        100 |          0.079 |           0.208 |            0.041 |    0.288 |
|        250 |          0.126 |           0.265 |            0.198 |    0.237 |
|        500 |          0.337 |           0.344 |            0.372 |    0.286 |
|       1000 |          0.349 |           0.362 |            0.368 |    0.469 |
|       2000 |          0.38  |           0.39  |            0.359 |    0.443 |
|       2830 |                |                 |                  |    0.443 |
|       3024 |          0.386 |           0.419 |            0.377 |          |

## Fraction of full-data coverage recovered

|   target_n |   heldout_head |   heldout_layer |   heldout_regime |   random |
|-----------:|---------------:|----------------:|-----------------:|---------:|
|         50 |          0.55  |           0.05  |            0.213 |    0.023 |
|        100 |          0.206 |           0.498 |            0.108 |    0.65  |
|        250 |          0.326 |           0.633 |            0.526 |    0.536 |
|        500 |          0.874 |           0.822 |            0.987 |    0.647 |
|       1000 |          0.905 |           0.865 |            0.976 |    1.06  |
|       2000 |          0.985 |           0.931 |            0.953 |    1.002 |
|       2830 |                |                 |                  |    1     |
|       3024 |          1     |           1     |            1     |          |

## KV read reduction vs calibration size (held-out regime)

|   target_n |   cov_at_95prec |   kv_read_reduction |   fail_rate_routed |   auc |
|-----------:|----------------:|--------------------:|-------------------:|------:|
|         50 |           0.08  |               1.078 |              0.049 | 0.722 |
|        100 |           0.041 |               1.038 |              0.049 | 0.797 |
|        250 |           0.198 |               1.217 |              0.05  | 0.842 |
|        500 |           0.372 |               1.502 |              0.048 | 0.888 |
|       1000 |           0.368 |               1.494 |              0.049 | 0.904 |
|       2000 |           0.359 |               1.476 |              0.05  | 0.903 |
|       3024 |           0.377 |               1.512 |              0.05  | 0.915 |

## Reading

- If coverage saturates by ~250–1000 cases, a new model can be calibrated with a tiny labelled set (each label = one forward + oracle rel-L2 at the budget ladder), making the weak zero-shot model-transfer a non-issue.
- Held-out-layer/head splits test whether a few layers/heads of calibration generalize to the rest of the same model.
