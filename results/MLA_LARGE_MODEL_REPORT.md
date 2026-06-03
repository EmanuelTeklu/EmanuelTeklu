# MLA / Latent-KV on a Real MHA Model — GPT-2 (Track B)

GPT-2 is MHA (12 heads, no GQA): native KV cache = 2·n_heads·d = **1536 elems/token/layer** — the regime MLA targets. Empirical latent proxy = joint rank-`dc` SVD of concatenated K|V per layer.

## Verdict: **KEEP (large byte win on MHA + preserves selection/reuse structure)**

- **Byte reduction:** MHA native 1536/token/layer; an MLA-equivalent latent (dc≈512) is **3.0× smaller**, with logit rank-corr 1.00, support 1.00.
- **Selection structure survives:** at ~4× compression (dc=384) rank-corr **1.00**, oracle-support **1.00** — routing/reuse selection is preserved.
- **Output-quality:** full-attention rel-L2≤0.10 holds to dc=64 (**24.0× smaller**).
- Rule: KEEP if significant byte reduction AND structure preserved; KILL if it destroys attention/routing structure.

## Latent rank sweep (GPT-2 MHA; medians)

|   dc |   bytes_ratio |   rl2_latent |   logit_spearman |   rl2_sparse_orig |   rl2_sparse_latent |   sparse_support |
|-----:|--------------:|-------------:|-----------------:|------------------:|--------------------:|-----------------:|
|   64 |         0.042 |        0.036 |                1 |             0.139 |               0.295 |            0.971 |
|  128 |         0.083 |        0.011 |                1 |             0.139 |               0.151 |            0.983 |
|  256 |         0.167 |        0.003 |                1 |             0.139 |               0.139 |            0.995 |
|  384 |         0.25  |        0.001 |                1 |             0.139 |               0.139 |            0.995 |
|  512 |         0.333 |        0     |                1 |             0.139 |               0.139 |            0.998 |
|  768 |         0.5   |        0     |                1 |             0.139 |               0.139 |            0.999 |
| 1536 |         1     |        0     |                1 |             0.139 |               0.139 |            1     |

## MHA vs GQA-2 contrast (structure preservation by compression factor)

native KV elems/token/layer: **MHA (GPT-2) 1536** vs **GQA-2 (Qwen-0.5B) 256** vs **MLA latent 576**. So MLA is ~2.7× smaller than this MHA but *larger* than GQA-2 — confirming latent-KV is a big win on MHA/high-KV models and ~neutral on already-GQA-heavy small models.

GPT-2 (MHA) structure vs compression:

|   compression |   bytes_ratio |   logit_spearman |   sparse_support |   rl2_latent |
|--------------:|--------------:|-----------------:|-----------------:|-------------:|
|            24 |         0.042 |                1 |            0.971 |        0.036 |
|            12 |         0.083 |                1 |            0.983 |        0.011 |
|             6 |         0.167 |                1 |            0.995 |        0.003 |
|             4 |         0.25  |                1 |            0.995 |        0.001 |
|             3 |         0.333 |                1 |            0.998 |        0     |
|             2 |         0.5   |                1 |            0.999 |        0     |
|             1 |         1     |                1 |            1     |        0     |

Qwen-0.5B (GQA-2) structure vs compression (from mla_stack.csv):

|   compression |   logit_spearman |   sparse_support |
|--------------:|-----------------:|-----------------:|
|         8     |            0.941 |            0.658 |
|         4     |            0.984 |            0.814 |
|         2     |            0.998 |            0.936 |
|         1.333 |            1     |            0.979 |
|         1     |            1     |            1     |

## Reading

- On a real MHA model the latent win is large (~2.7×+ bytes/token) and the **selection structure that sparse routing / prefix reuse rely on survives** the latent mixing (high logit rank-corr, support) — so latent-KV and persistent-state reuse **stack**.
- This is the model class to pursue latent-KV on (and via MHA→MLA conversion to avoid pretraining), NOT tiny GQA models where the byte win is absent.
