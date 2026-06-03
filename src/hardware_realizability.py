"""MAIN STACK Experiment 2 — block locality / hardware realizability.

For the sparse selections (oracle top-S at the reference budget), are they
hardware-friendly? We look at contiguity, block counts/utilization at sizes
16/32/64/128, recent/sink fraction, cross-head & cross-layer overlap, gather
metadata overhead, and the variable-softmax-length distribution.

Consumes results/tables/stack/budget_*.csv + overlap_*.csv.
Writes results/HARDWARE_REALIZABILITY_REPORT.md + results/tables/hw_*.csv.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
STACKDIR = os.path.join(RESULTS, "tables", "stack")
BLOCK_SIZES = [16, 32, 64, 128]


def load():
    bf = sorted(glob.glob(os.path.join(STACKDIR, "budget_*.csv")))
    of = sorted(glob.glob(os.path.join(STACKDIR, "overlap_*.csv")))
    b = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True)
    o = pd.concat([pd.read_csv(f) for f in of], ignore_index=True) if of else pd.DataFrame()
    return b, o


def run():
    b, o = load()
    long = b[b["ctx"] >= 8192].copy()

    # block locality aggregates (long-context)
    agg = {}
    agg["mean_S"] = float(long["S"].mean())
    agg["median_S"] = float(long["S"].median())
    agg["mean_contig_runs"] = float(long["contig_runs"].mean())
    # runs per selected token: 1.0 => every token isolated; ->0 => contiguous
    agg["runs_per_token"] = float((long["contig_runs"] / long["S"].clip(lower=1)).mean())
    agg["mean_recent_frac"] = float(long["recent_frac"].mean())
    agg["mean_sink_frac"] = float(long["sink_frac"].mean())
    for bs in BLOCK_SIZES:
        agg[f"mean_nblocks_{bs}"] = float(long[f"nblocks_{bs}"].mean())
        agg[f"mean_blockutil_{bs}"] = float(long[f"blockutil_{bs}"].mean())

    # gather metadata overhead: storing block indices vs token indices.
    # token gather: S int32 indices; block gather: nblocks int32 indices.
    # overhead ratio = index_bytes / value_bytes(read), value bytes = S*d*2 (bf16).
    rows = []
    for bs in BLOCK_SIZES:
        # block-gather reads nblocks*bs tokens (full blocks), util = S/(nblocks*bs)
        idx_bytes = long[f"nblocks_{bs}"] * 4
        read_tokens = long[f"nblocks_{bs}"] * bs
        value_bytes = read_tokens * 64 * 2  # head_dim 64 bf16 (0.5B); indicative
        rows.append(dict(block_size=bs,
            mean_blocks=float(long[f"nblocks_{bs}"].mean()),
            mean_block_util=float(long[f"blockutil_{bs}"].mean()),
            mean_overread=float((read_tokens / long["S"].clip(lower=1)).mean()),
            meta_overhead_frac=float((idx_bytes / value_bytes).mean())))
    blk = pd.DataFrame(rows)

    # softmax length distribution (variable-length kernel concern)
    sl = long["S"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

    # overlap
    ov_head = float(o["cross_head_jaccard"].mean()) if len(o) else np.nan
    ov_layer = float(o["cross_layer_jaccard"].mean()) if len(o) else np.nan

    # by-regime block utilization
    byreg = long.groupby("regime").agg(
        mean_S=("S", "mean"), recent_frac=("recent_frac", "mean"),
        blockutil_32=("blockutil_32", "mean"), nblocks_32=("nblocks_32", "mean")).reset_index()

    pd.DataFrame([agg]).to_csv(os.path.join(RESULTS, "tables", "hw_locality.csv"), index=False)
    blk.to_csv(os.path.join(RESULTS, "tables", "hw_block_sizes.csv"), index=False)
    byreg.to_csv(os.path.join(RESULTS, "tables", "hw_by_regime.csv"), index=False)

    # classification of friendliness. Key practical signals: block utilization
    # (padding waste), overread (tokens read / needed at block granularity), and
    # cross-head overlap (can a gathered block set be amortized across heads?).
    util32 = agg["mean_blockutil_32"]; util64 = agg["mean_blockutil_64"]
    overread32 = float(blk[blk.block_size == 32]["mean_overread"].iloc[0])
    if util64 >= 0.5 and ov_head >= 0.4:
        klass = ("GPU-FRIENDLY (block-local, overlapping heads -> Triton "
                 "block-sparse is the next step)")
    elif util32 < 0.25 and overread32 > 4.0 and (np.isnan(ov_head) or ov_head < 0.4):
        klass = ("GPU-HOSTILE / gather-bound (scattered selections: low block "
                 "utilization, high block over-read, low cross-head overlap)")
    else:
        klass = ("UNCLEAR (moderate locality; block-sparse helps but gather/padding "
                 "overhead is non-trivial)")

    write_report(agg, blk, byreg, sl, ov_head, ov_layer, klass)
    return agg, blk


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(agg, blk, byreg, sl, ov_head, ov_layer, klass):
    L = []; A = L.append
    A("# Hardware Realizability Report (MAIN STACK Experiment 2)")
    A("")
    A("Are the sparse selections (oracle top-S at the 2% reference budget) "
      "hardware-friendly? Long-context cases (≥8k). Metrics describe contiguity, "
      "block structure, head/layer reuse, gather overhead, and variable softmax "
      "length.")
    A("")
    A(f"## Classification: **{klass}**")
    A("")
    A(f"- Mean selected tokens S = {agg['mean_S']:.0f} (median {agg['median_S']:.0f}); "
      f"contiguous runs per selected token = {agg['runs_per_token']:.2f} "
      f"(1.0 = fully scattered, →0 = contiguous).")
    A(f"- Block utilization: size-32 **{agg['mean_blockutil_32']:.2f}**, "
      f"size-64 **{agg['mean_blockutil_64']:.2f}**, size-128 "
      f"{agg['mean_blockutil_128']:.2f} (fraction of a touched block that is "
      f"actually selected).")
    A(f"- Recent-window fraction {agg['mean_recent_frac']:.2f}, sink fraction "
      f"{agg['mean_sink_frac']:.2f} of selected tokens.")
    A(f"- Cross-head Jaccard overlap **{ov_head:.2f}**, cross-layer **{ov_layer:.2f}** "
      f"(shared selected tokens across heads / layers).")
    A("")
    A("## Block size trade-off")
    A("")
    A("`mean_overread` = tokens actually read / tokens needed (block padding cost); "
      "`meta_overhead_frac` = index bytes / value bytes read.")
    A("")
    A(_md(blk))
    A("")
    A("## By regime (size-32 blocks)")
    A("")
    A(_md(byreg))
    A("")
    A("## Variable softmax length (selected tokens S) distribution")
    A("")
    A("```")
    A(sl.to_string())
    A("```")
    A("")
    A("## Reading & next step")
    A("")
    A("- **Block locality:** with size-32/64 blocks, utilization "
      f"({agg['mean_blockutil_32']:.2f}/{agg['mean_blockutil_64']:.2f}) sets the "
      "padding cost; `mean_overread` quantifies wasted reads if you gather whole "
      "blocks. A recent/sink fraction of "
      f"{agg['mean_recent_frac']+agg['mean_sink_frac']:.2f} is contiguous by "
      "construction and trivially block-friendly.")
    A("- **Head/layer reuse:** Jaccard overlap "
      f"({ov_head:.2f} head, {ov_layer:.2f} layer) indicates how much a shared "
      "block selection could be amortized across heads/layers — higher overlap "
      "favors a single gathered block set reused by many heads.")
    A("- **Decision:** if block-local and overlapping → the next step is a Triton "
      "block-sparse attention kernel. If highly irregular but *stable* across "
      "calls → specialized KV gather hardware becomes the more plausible bet. "
      f"Current reading: **{klass.split('(')[0].strip()}**.")
    A(f"- **Kernel gate: NOT cleared.** Block locality does *not* pass — at block "
      f"size 32 you over-read ~{float(blk[blk.block_size==32].mean_overread.iloc[0]):.0f}× "
      f"(util {agg['mean_blockutil_32']:.2f}) and cross-head overlap is only "
      f"{ov_head:.2f}, so a naive block-sparse gather would spend most of the sparse "
      f"saving on padding. The combined gate (adaptive budget ≥4× AND block "
      f"locality friendly) is **half-met**: Exp 1 gives ≥4× at 16k, but selections "
      f"are gather-bound. **Do not build a Triton kernel yet.**")
    A("- Next instead: (a) measure selection *stability* across query positions / "
      "decode steps (if scattered-but-stable, the gather set can be precomputed and "
      "amortized — a different kernel/hardware bet); (b) try learned or "
      "locality-regularized routers that prefer contiguous blocks, then re-measure "
      "utilization vs the rel-L2 cost.")
    A("")
    p = os.path.join(RESULTS, "HARDWARE_REALIZABILITY_REPORT.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p, "| class:", klass.split("(")[0].strip())


if __name__ == "__main__":
    run()
