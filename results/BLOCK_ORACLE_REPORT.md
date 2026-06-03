# Block-Oracle Upper Bound (MAIN NEXT Experiment 1)

Does GPU-friendly *block* selection retain the token-oracle headroom? Long-context (≥8k). Pass = median rel-L2≤0.1. Read = smallest read ratio reaching the pass point; gain = 1/read.

## Verdict: **PARK**

- Best block size ≤64: **32** retains **26%** of the token-oracle read-gain at rel-L2≤0.10.
- Rule: KEEP if block-oracle keeps ≥50% of token-oracle gain; PARK if only at large budgets; KILL if it collapses.

## Block-oracle vs token-oracle (pass point)

|   block_size |   token_oracle_read |   block_oracle_read |   gain_retention |   block_gain |   token_gain |
|-------------:|--------------------:|--------------------:|-----------------:|-------------:|-------------:|
|           16 |               0.005 |               0.02  |            0.257 |       51.205 |      199.607 |
|           32 |               0.005 |               0.019 |            0.258 |       51.587 |      199.607 |
|           64 |               0.005 |               0.022 |            0.231 |       46.193 |      199.607 |
|          128 |               0.005 |               0.048 |            0.105 |       20.935 |      199.607 |
|          256 |               0.005 |               0.051 |            0.098 |       19.587 |      199.607 |

## Interval-cover over-read (gathering the scattered token-oracle as blocks)

This is the cost of the *old* token-router selection if forced onto blocks (the GPU-hostile number). Block-oracle above avoids it by choosing blocks directly.

|   block_size |   over_read |   rl2_med |
|-------------:|------------:|----------:|
|           16 |       5.788 |     0.013 |
|           32 |       8.473 |     0.009 |
|           64 |      12.004 |     0.006 |
|          128 |      15.466 |     0.003 |
|          256 |      18.502 |     0.002 |

## Reading

- If small blocks (16–32) retain ≥50% gain, the GPU-hostile finding is *escapable*: select whole blocks directly (utilization 1.0 within selected blocks, contiguous) instead of scattered tokens — over-read ≈1, not 6–19×.
- Large blocks (128–256) collapsing means the attention mass is finer-grained than those blocks; 16–32 is the realizable sweet spot.
