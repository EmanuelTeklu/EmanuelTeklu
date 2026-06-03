# Prefix / Latent Reuse — Canonicalization Ablation (Probe A)

> Simulated token-level agent traces; analytic speedup (economics, not wall-clock). Each session carries several independent prefix-noise sources; each canonicalization transform neutralises one.

## Verdict: **KEEP**

- Mean reusable-prefix fraction **0.98**.
- Exact-match hit rate **0.01** → full-canonicalization hit rate **0.88**.
- Modeled end-to-end speedup (overhead 3%): exact **1.0×** → canonicalized **5.8×**.
- Under 10% cache invalidation + 3% overhead, still **3.8×**.
- Rule: KEEP if canonicalization ≥5× modeled speedup; STRONG KEEP ≥10×.

## Canonicalization ablation (cumulative)

| stage                   |   hit_rate |   speedup_ov0 |   speedup_ov03 |
|:------------------------|-----------:|--------------:|---------------:|
| exact (no canon)        |      0.015 |         1.014 |          1.014 |
| +strip_volatile_ids     |      0.073 |         1.077 |          1.074 |
| +versioned_memory       |      0.175 |         1.206 |          1.199 |
| +normalize_tool_schemas |      0.277 |         1.371 |          1.356 |
| +sort_tool_definitions  |      0.467 |         1.836 |          1.791 |
| +stable_doc_hashes      |      0.875 |         6.848 |          5.826 |

## Invalidation sensitivity (full canonicalization)

|   invalidation |   overhead |   hit_rate |   speedup |
|---------------:|-----------:|-----------:|----------:|
|           0    |       0    |      0.875 |     6.848 |
|           0    |       0.03 |      0.875 |     5.826 |
|           0    |       0.1  |      0.875 |     4.321 |
|           0.05 |       0    |      0.835 |     5.415 |
|           0.05 |       0.03 |      0.835 |     4.782 |
|           0.05 |       0.1  |      0.835 |     3.757 |
|           0.1  |       0    |      0.775 |     4.105 |
|           0.1  |       0.03 |      0.775 |     3.755 |
|           0.1  |       0.1  |      0.775 |     3.132 |
|           0.2  |       0    |      0.677 |     2.948 |
|           0.2  |       0.03 |      0.677 |     2.785 |
|           0.2  |       0.1  |      0.677 |     2.467 |
|           0.3  |       0    |      0.642 |     2.676 |
|           0.3  |       0.03 |      0.642 |     2.547 |
|           0.3  |       0.1  |      0.642 |     2.292 |

## Reading

- The biggest hit-rate recoveries come from `strip_volatile_ids` and `versioned_memory` (the highest-probability noise sources). Schema normalization/sorting add the rest.
- Speedup is bounded by the *reusable fraction* (≈0.98); even a perfect cache cannot beat 1/(1-r). 100× needs reuse of the *unique tail* too (cross-request latent reuse), not just the prefix.
- This is the recommended lever for the agent / repeated-prefix regime, which transferred worst for sparse routing.
