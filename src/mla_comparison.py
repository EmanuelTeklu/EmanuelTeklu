"""PARALLEL PROBE 2 — MLA / latent-KV vs sparse-safe routing.

Question: does architecture-level KV reduction (DeepSeek Multi-head Latent
Attention, or MHA->MLA conversion) stack with, or dominate, sparse-safe routing?

Key idea: the two attack ORTHOGONAL dimensions of KV-read cost.
    KV read cost  ∝  (#tokens read)  ×  (bytes per token per layer)
  * sparse-safe routing      shrinks  #tokens read          (the T axis)
  * MLA / latent-KV          shrinks  bytes per token        (the d axis)
So they multiply. This probe computes per-token KV footprints for MHA / GQA /
MLA and the combined effect with our measured sparse read fractions.

KV cache elements per token per layer:
  * MHA :  2 * n_heads     * head_dim
  * GQA :  2 * n_kv_heads  * head_dim
  * MLA :  kv_lora_rank + qk_rope_head_dim   (shared across heads; the only
           thing DeepSeek-V2/V3 store in cache)

Writes results/MLA_LATENT_KV_REPORT.md + results/tables/mla_kv_sizes.csv.
"""
from __future__ import annotations
import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES = os.path.join(RESULTS, "tables")

# DeepSeek-V2 MLA cache params (kv_lora_rank=512, decoupled rope key dim=64).
MLA_LORA = 512
MLA_ROPE = 64

# Model exemplars: (n_layers, n_heads, n_kv_heads, head_dim)
MODELS = {
    "Qwen2.5-0.5B (GQA)":  dict(L=24, H=14, KV=2,  d=64),
    "Qwen2.5-1.5B (GQA)":  dict(L=28, H=12, KV=2,  d=128),
    "Llama-3-8B (GQA)":    dict(L=32, H=32, KV=8,  d=128),
    "MHA-70B-like (MHA)":  dict(L=80, H=64, KV=64, d=128),
}


def per_token_elems(m):
    mha = 2 * m["H"] * m["d"]
    gqa = 2 * m["KV"] * m["d"]
    mla = MLA_LORA + MLA_ROPE
    return mha, gqa, mla


def run(sparse_read_fraction=0.46, bytes_per_elem=2):
    """sparse_read_fraction: effective fraction of tokens read under the selective
    policy (≈ measured 0.5B@16k held-out-regime avg read ≈ 0.46 -> 2.16x)."""
    rows = []
    for name, m in MODELS.items():
        mha, gqa, mla = per_token_elems(m)
        native = gqa if m["KV"] < m["H"] else mha
        native_kind = "GQA" if m["KV"] < m["H"] else "MHA"
        rows.append(dict(model=name, native=native_kind,
            per_tok_MHA=mha, per_tok_GQA=gqa, per_tok_MLA=mla,
            MLA_vs_native=native / mla,
            sparse_vs_native=1.0 / sparse_read_fraction,
            MLA_plus_sparse_vs_native=(native / mla) * (1.0 / sparse_read_fraction)))
    df = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, "mla_kv_sizes.csv"), index=False)
    write_report(df, sparse_read_fraction)
    return df


def _md(df, fmt="{:.2f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(df, srf):
    L = []; A = L.append
    A("# MLA / Latent-KV vs Sparse-Safe Routing (Probe)")
    A("")
    A("## Background (from literature)")
    A("")
    A("- **MLA (Multi-head Latent Attention, DeepSeek-V2/V3).** Projects keys and "
      "values into a shared low-rank latent `c_KV` (rank ≈512) plus a small "
      "decoupled RoPE key (≈64 dims). The KV *cache stores only this latent*, so "
      "the per-token footprint is **independent of head count** — a large memory / "
      "bandwidth win for many-head models, with quality on par with MHA.")
    A("- **MHA→MLA conversion (e.g. MHA2MLA).** Post-hoc, data-light procedures "
      "that low-rank-distill an existing MHA/GQA checkpoint into MLA, recovering "
      "most quality while shrinking the KV cache — i.e. MLA without pretraining "
      "from scratch.")
    A("- Both reduce **bytes per cached token**; neither reduces the **number of "
      "tokens attended**. Sparse-safe routing does the opposite.")
    A("")
    A(f"## KV footprint comparison (elements per token per layer)")
    A("")
    A("Combined column uses our measured selective sparse read fraction "
      f"≈ **{srf:.2f}** (0.5B @16k held-out-regime, ≈{1/srf:.1f}× token reduction).")
    A("")
    A(_md(df))
    A("")
    A("## Reading")
    A("")
    A("- **MLA dominates for MHA / many-KV-head models.** For the MHA-70B-like "
      f"exemplar MLA is ~{df[df.model.str.contains('70B')].MLA_vs_native.iloc[0]:.1f}× "
      "smaller per token; stacked with sparse routing the KV-read reduction is "
      f"~{df[df.model.str.contains('70B')].MLA_plus_sparse_vs_native.iloc[0]:.1f}×.")
    A("- **MLA does NOT help already-GQA-heavy small models.** Qwen2.5-0.5B GQA-2 "
      f"stores only {df[df.model.str.contains('0.5B')].per_tok_GQA.iloc[0]:.0f} "
      f"elems/token — *less* than MLA's {MLA_LORA+MLA_ROPE}. There the latent is "
      "larger than the already-aggressive GQA cache, so MLA is neutral-to-negative; "
      "**sparse routing is the only KV lever**.")
    A("- The two are **orthogonal and multiplicative**: latent-KV shrinks the `d` "
      "axis, sparse-safe routing shrinks the `T` axis. On a large MHA model both "
      "apply and compound.")
    A("")
    A("## Should latent-KV become a serious parallel branch?")
    A("")
    A("**Yes — but scoped to large / MHA / many-head models, where our own sparse "
      "work is least sufficient on its own.** Recommendation:")
    A("")
    A("1. Keep sparse-safe routing as the primary branch for GQA models (where MLA "
      "adds nothing).")
    A("2. Open a *parallel* latent-KV branch targeting MHA/large models, ideally via "
      "MHA→MLA conversion (no pretraining). Measure whether sparse-safe routing "
      "still finds ≥2× token reduction *on top of* the latent cache (the latent "
      "mixes heads, so per-head sparsity structure may change — this must be "
      "re-measured on a converted model, not assumed).")
    A("3. Do not build a kernel for either until (a) adaptive-budget + block "
      "locality pass on the sparse side, and (b) the latent+sparse stacking is "
      "confirmed empirically on one converted checkpoint.")
    A("")
    A("> Caveat: MLA params here are DeepSeek-V2 defaults; exact latent rank varies "
      "by model. The qualitative regimes (MLA wins on MHA, neutral on GQA-2) are "
      "robust to the precise rank.")
    A("")
    p = os.path.join(RESULTS, "MLA_LATENT_KV_REPORT.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p)


if __name__ == "__main__":
    run()
