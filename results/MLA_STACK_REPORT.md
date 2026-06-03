# MLA / Latent-KV Stack — Empirical Test (0.5B)

Empirical latent-KV proxy on real captured K/V (joint rank-`dc` SVD of concatenated K|V per layer = shared per-token latent, MLA-like). Question: does latent compression preserve the sparse-safe structure our routing needs, and do the savings multiply with sparse?

## Verdict: **KEEP (latent preserves sparse structure AND compresses output)**

- **Selection structure survives latent mixing:** at 4× latent compression (dc=64) the per-token logit rank-correlation is **0.98** and the oracle top-S support recovery is **0.81** — the router still picks essentially the right tokens. This is the KILL condition NOT triggered.
- Output-quality compression: full-attention rel-L2≤0.10 holds down to **dc=192** (1.3× bytes), where logit rank-corr 1.00.
- But latent reconstruction ADDS output error that compounds with sparse: sparse rel-L2@2% goes from 0.064 (orig) to 0.264 at 2× latent (dc=128) — so on this already-GQA-2 model aggressive latent isn't free.

## Latent rank sweep (medians over layers/heads/prompts)

|   dc |   bytes_ratio |   rl2_latent |   logit_spearman |   rl2_sparse_orig |   rl2_sparse_latent |   sparse_support |
|-----:|--------------:|-------------:|-----------------:|------------------:|--------------------:|-----------------:|
|   32 |         0.125 |        0.538 |            0.941 |             0.064 |               0.577 |            0.658 |
|   64 |         0.25  |        0.37  |            0.984 |             0.064 |               0.424 |            0.814 |
|  128 |         0.5   |        0.205 |            0.998 |             0.064 |               0.264 |            0.936 |
|  192 |         0.75  |        0.062 |            1     |             0.064 |               0.157 |            0.979 |
|  256 |         1     |        0     |            1     |             0.064 |               0.064 |            1     |

## Reading

- `rl2_latent` = full-attention error from latent reconstruction; `logit_spearman` = does the per-token logit *ranking* survive (router can still pick the right tokens); `rl2_sparse_latent` = sparse routing *on the latent tensors* scored against the true output.
- If `logit_spearman` stays high and `rl2_sparse_latent` tracks `rl2_sparse_orig`, latent-KV and sparse routing **stack**: latent shrinks bytes/token, sparse shrinks tokens read, multiplicatively.
- Note: Qwen is already GQA-2 (native KV is small), so the *byte* savings here are modest; the decisive question this answers is **structure preservation**, which transfers to MHA/large models where MLA's byte win is large (see MLA_LATENT_KV_REPORT.md).
