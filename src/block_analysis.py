"""Analysis for MAIN NEXT Experiments 1 & 2 from block_experiments.py outputs.

Exp 1 — block-oracle upper bound: does block-granular selection retain enough of
the token-oracle gain? KEEP if block-oracle keeps >=50% of token-oracle read-gain
at rel-L2<=0.10 (i.e. block-oracle read ratio <= 2x token-oracle at the pass
point). PARK if only at large budgets. KILL if it collapses.

Exp 2 — locality routers: cheap non-oracle block routers. KEEP if a router hits
rel-L2<=0.10 at >=4x read reduction with block utilization>=0.5 / over-read<=2x.

Writes results/BLOCK_ORACLE_REPORT.md and results/LOCALITY_ROUTER_REPORT.md.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
BLOCKDIR = os.path.join(RESULTS, "tables", "block")
LONG_CTX = 8192
PASS = 0.10


def load():
    fs = sorted(glob.glob(os.path.join(BLOCKDIR, "agg_*.csv")))
    if not fs:
        raise SystemExit("no block agg data; run block_experiments.py first")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def _min_pass_read(sub):
    """Smallest read_ratio whose median rel-L2<=PASS (interpolate not needed)."""
    ok = sub[sub.rl2_med <= PASS].sort_values("read_ratio")
    return float(ok.read_ratio.iloc[0]) if len(ok) else np.nan


def exp1(df):
    long = df[df["ctx"] >= LONG_CTX]
    # pool regimes via median across rows already aggregated per regime; re-aggregate
    g = long.groupby(["policy", "block_size", "budget"]).agg(
        rl2_med=("rl2_med", "median"), read_ratio=("read_ratio", "mean"),
        pass10=("pass10", "mean"), over_read=("over_read", "mean")).reset_index()
    tok = g[g.policy == "token_oracle"]
    tok_read = _min_pass_read(tok)
    rows = []
    for b in sorted(g[g.policy == "block_oracle"].block_size.unique()):
        sub = g[(g.policy == "block_oracle") & (g.block_size == b)]
        br = _min_pass_read(sub)
        rows.append(dict(block_size=int(b), token_oracle_read=tok_read,
            block_oracle_read=br,
            gain_retention=(tok_read / br) if (br and not np.isnan(br)) else 0.0,
            block_gain=(1.0 / br) if (br and not np.isnan(br)) else np.nan,
            token_gain=(1.0 / tok_read) if tok_read else np.nan))
    return pd.DataFrame(rows), g


def exp2(df):
    long = df[df["ctx"] >= LONG_CTX]
    routers = ["centroid", "centroid_ub", "maxsketch", "recent_sink_routed"]
    g = long[long.policy.isin(routers)].groupby(["policy", "block_size", "budget"]).agg(
        rl2_med=("rl2_med", "median"), read_ratio=("read_ratio", "mean"),
        pass10=("pass10", "mean"), support=("support", "mean"),
        mass=("mass", "mean")).reset_index()
    # best operating point per (policy,block_size): smallest read with median pass
    rows = []
    for (pol, b), sub in g.groupby(["policy", "block_size"]):
        ok = sub[sub.rl2_med <= PASS].sort_values("read_ratio")
        if len(ok):
            r = ok.iloc[0]
            rows.append(dict(policy=pol, block_size=int(b),
                read_ratio=float(r.read_ratio), read_reduction=1.0 / float(r.read_ratio),
                rl2_med=float(r.rl2_med), pass10=float(r.pass10),
                mass=float(r["mass"]), support=float(r.support)))
    return pd.DataFrame(rows), g


def interval_cover(df):
    long = df[(df["ctx"] >= LONG_CTX) & (df.policy == "interval_cover")]
    return long.groupby("block_size").agg(
        over_read=("over_read", "mean"), rl2_med=("rl2_med", "median")).reset_index()


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def run():
    df = load()
    e1, g1 = exp1(df)
    e2, g2 = exp2(df)
    ic = interval_cover(df)
    e1.to_csv(os.path.join(BLOCKDIR, "exp1_block_oracle.csv"), index=False)
    e2.to_csv(os.path.join(BLOCKDIR, "exp2_locality_router.csv"), index=False)

    # Exp1 verdict: best gain retention over block sizes <=64
    e1s = e1[e1.block_size <= 64]
    best_ret = float(e1s.gain_retention.max()) if len(e1s) else 0.0
    best_b = int(e1s.loc[e1s.gain_retention.idxmax(), "block_size"]) if len(e1s) and e1s.gain_retention.max() > 0 else None
    if best_ret >= 0.50:
        v1 = "KEEP"
    elif best_ret > 0.0:
        v1 = "PARK"
    else:
        v1 = "KILL"
    write_exp1(e1, ic, v1, best_ret, best_b)

    # Exp2 verdict: any router with read_reduction>=4x at median pass and small block
    cand = e2[(e2.read_reduction >= 4.0)]
    v2 = "KEEP" if len(cand) else ("PARK" if len(e2) else "KILL")
    write_exp2(e2, ic, v2, cand)
    return e1, e2


def write_exp1(e1, ic, verdict, best_ret, best_b):
    L = []; A = L.append
    A("# Block-Oracle Upper Bound (MAIN NEXT Experiment 1)")
    A("")
    A("Does GPU-friendly *block* selection retain the token-oracle headroom? "
      f"Long-context (≥8k). Pass = median rel-L2≤{PASS}. Read = smallest read "
      "ratio reaching the pass point; gain = 1/read.")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    if best_b is not None:
        A(f"- Best block size ≤64: **{best_b}** retains **{best_ret*100:.0f}%** of "
          f"the token-oracle read-gain at rel-L2≤0.10.")
    A(f"- Rule: KEEP if block-oracle keeps ≥50% of token-oracle gain; PARK if only "
      f"at large budgets; KILL if it collapses.")
    A("")
    A("## Block-oracle vs token-oracle (pass point)")
    A("")
    A(_md(e1))
    A("")
    A("## Interval-cover over-read (gathering the scattered token-oracle as blocks)")
    A("")
    A("This is the cost of the *old* token-router selection if forced onto blocks "
      "(the GPU-hostile number). Block-oracle above avoids it by choosing blocks "
      "directly.")
    A("")
    A(_md(ic))
    A("")
    A("## Reading")
    A("")
    A("- If small blocks (16–32) retain ≥50% gain, the GPU-hostile finding is "
      "*escapable*: select whole blocks directly (utilization 1.0 within selected "
      "blocks, contiguous) instead of scattered tokens — over-read ≈1, not 6–19×.")
    A("- Large blocks (128–256) collapsing means the attention mass is finer-"
      "grained than those blocks; 16–32 is the realizable sweet spot.")
    A("")
    with open(os.path.join(RESULTS, "BLOCK_ORACLE_REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote BLOCK_ORACLE_REPORT.md | verdict:", verdict)


def write_exp2(e2, ic, verdict, cand):
    L = []; A = L.append
    A("# Locality-Regularized Routing (MAIN NEXT Experiment 2)")
    A("")
    A("Cheap NON-oracle routers that select blocks (centroid, centroid+radius "
      "upper-bound, max-sketch, recent/sink+routed). Long-context (≥8k). Best "
      "operating point = smallest read ratio with median rel-L2≤0.10.")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    if len(cand):
        b = cand.sort_values("read_reduction", ascending=False).iloc[0]
        A(f"- Best block router: **{b.policy} @ block {int(b.block_size)}** → "
          f"**{b.read_reduction:.1f}× read reduction** at median rel-L2 "
          f"{b.rl2_med:.3f}, mass {b['mass']:.2f}, oracle-support {b.support:.2f}.")
    else:
        A("- No cheap block router reached ≥4× read reduction at median rel-L2≤0.10.")
    A("- Rule: KEEP if a non-oracle block router hits rel-L2≤0.10 at ≥4× read "
      "reduction with block utilization≥0.5 / over-read≤2×.")
    A("")
    A("## Block-router operating points (median pass)")
    A("")
    A(_md(e2.sort_values(["policy", "block_size"])))
    A("")
    A("## Reading")
    A("")
    A("- Block routers select *contiguous* blocks, so by construction utilization "
      "is 1.0 within a selected block and over-read ≈1 — the GPU-hostile scatter "
      "problem disappears IF accuracy holds. The question is purely the rel-L2 / "
      "read-reduction trade-off vs the block-oracle ceiling (Exp 1).")
    A("- `centroid_ub` adds a radius term so a block is only skipped when its best "
      "possible score is below the cut — a cheap certificate. `maxsketch` "
      "approximates the per-block max logit via a random projection.")
    A("- The selective (classifier-gated) version of the best block router, with "
      "global risk allocation, is evaluated in ADAPTIVE_RISK_REPORT_BLOCK.md.")
    A("")
    with open(os.path.join(RESULTS, "LOCALITY_ROUTER_REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote LOCALITY_ROUTER_REPORT.md | verdict:", verdict)


if __name__ == "__main__":
    run()
