# MLA / Latent-KV vs Sparse-Safe Routing (Probe)

## Background (from literature)

- **MLA (Multi-head Latent Attention, DeepSeek-V2/V3).** Projects keys and values into a shared low-rank latent `c_KV` (rank ≈512) plus a small decoupled RoPE key (≈64 dims). The KV *cache stores only this latent*, so the per-token footprint is **independent of head count** — a large memory / bandwidth win for many-head models, with quality on par with MHA.
- **MHA→MLA conversion (e.g. MHA2MLA).** Post-hoc, data-light procedures that low-rank-distill an existing MHA/GQA checkpoint into MLA, recovering most quality while shrinking the KV cache — i.e. MLA without pretraining from scratch.
- Both reduce **bytes per cached token**; neither reduces the **number of tokens attended**. Sparse-safe routing does the opposite.

## KV footprint comparison (elements per token per layer)

Combined column uses our measured selective sparse read fraction ≈ **0.46** (0.5B @16k held-out-regime, ≈2.2× token reduction).

| model              | native   |   per_tok_MHA |   per_tok_GQA |   per_tok_MLA |   MLA_vs_native |   sparse_vs_native |   MLA_plus_sparse_vs_native |
|:-------------------|:---------|--------------:|--------------:|--------------:|----------------:|-------------------:|----------------------------:|
| Qwen2.5-0.5B (GQA) | GQA      |          1792 |           256 |           576 |            0.44 |               2.17 |                        0.97 |
| Qwen2.5-1.5B (GQA) | GQA      |          3072 |           512 |           576 |            0.89 |               2.17 |                        1.93 |
| Llama-3-8B (GQA)   | GQA      |          8192 |          2048 |           576 |            3.56 |               2.17 |                        7.73 |
| MHA-70B-like (MHA) | MHA      |         16384 |         16384 |           576 |           28.44 |               2.17 |                       61.84 |

## Reading

- **MLA dominates for MHA / many-KV-head models.** For the MHA-70B-like exemplar MLA is ~28.4× smaller per token; stacked with sparse routing the KV-read reduction is ~61.8×.
- **MLA does NOT help already-GQA-heavy small models.** Qwen2.5-0.5B GQA-2 stores only 256 elems/token — *less* than MLA's 576. There the latent is larger than the already-aggressive GQA cache, so MLA is neutral-to-negative; **sparse routing is the only KV lever**.
- The two are **orthogonal and multiplicative**: latent-KV shrinks the `d` axis, sparse-safe routing shrinks the `T` axis. On a large MHA model both apply and compound.

## Should latent-KV become a serious parallel branch?

**Yes — but scoped to large / MHA / many-head models, where our own sparse work is least sufficient on its own.** Recommendation:

1. Keep sparse-safe routing as the primary branch for GQA models (where MLA adds nothing).
2. Open a *parallel* latent-KV branch targeting MHA/large models, ideally via MHA→MLA conversion (no pretraining). Measure whether sparse-safe routing still finds ≥2× token reduction *on top of* the latent cache (the latent mixes heads, so per-head sparsity structure may change — this must be re-measured on a converted model, not assumed).
3. Do not build a kernel for either until (a) adaptive-budget + block locality pass on the sparse side, and (b) the latent+sparse stacking is confirmed empirically on one converted checkpoint.

> Caveat: MLA params here are DeepSeek-V2 defaults; exact latent rank varies by model. The qualitative regimes (MLA wins on MHA, neutral on GQA-2) are robust to the precise rank.
