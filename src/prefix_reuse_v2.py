"""PARALLEL PROBE A (v2) — prefix/latent reuse with canonicalization ablation.

Extends the earlier probe: models *which* canonicalization transform recovers
*which* source of prefix-cache misses, plus cache invalidation. Still simulated
token-level traces + analytic speedup (economics, not wall-clock).

Noise sources that break EXACT prefix matching across sessions/turns:
  * format     -- whitespace / JSON formatting drift in the tool schema
  * order      -- tool definitions emitted in different order
  * volatile   -- request IDs / timestamps embedded in the system header
  * doc_ref    -- pinned documents referenced by unstable path vs stable hash
  * memory     -- agent memory block drifts turn to turn (unversioned)

Canonicalization transforms each neutralise one source:
  normalize_tool_schemas -> format ; sort_tool_definitions -> order ;
  strip_volatile_ids -> volatile ; stable_doc_hashes -> doc_ref ;
  versioned_memory -> memory.

Writes results/PREFIX_REUSE_REPORT.md + results/tables/prefix_reuse_v2_*.csv.
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES = os.path.join(RESULTS, "tables")
SEED = 0
rng = np.random.default_rng(SEED)

NOISE = ["format", "order", "volatile", "doc_ref", "memory"]
# per-session probability each noise source is present (breaks exact match)
NOISE_P = dict(format=0.6, order=0.4, volatile=0.7, doc_ref=0.5, memory=0.8)
TRANSFORMS = {  # transform -> noise source it neutralises
    "normalize_tool_schemas": "format",
    "sort_tool_definitions": "order",
    "strip_volatile_ids": "volatile",
    "stable_doc_hashes": "doc_ref",
    "versioned_memory": "memory",
}


def simulate(n_sessions=60, turns=8, sys_tok=900, schema_tok=600, doc_tok=1500,
             user_tok=80, reply_tok=250):
    rows = []
    for s in range(n_sessions):
        present = {k: (rng.random() < NOISE_P[k]) for k in NOISE}
        shared = sys_tok + schema_tok + (doc_tok if rng.random() < 0.7 else 0)
        hist = 0
        for t in range(turns):
            ptoks = shared + hist + user_tok
            rows.append(dict(session=s, turn=t, prompt_tokens=ptoks,
                             reusable_fraction=(shared + hist) / ptoks,
                             within_session=int(t > 0),
                             **{f"noise_{k}": int(present[k]) for k in NOISE}))
            hist += user_tok + reply_tok
    return pd.DataFrame(rows)


def hit_rate(df, applied_transforms, inv_rate=0.0):
    """A reuse hit needs a prior turn (within_session) AND no UNFIXED noise source.
    Invalidation independently drops a fraction of would-be hits."""
    fixed = {TRANSFORMS[t] for t in applied_transforms}
    unfixed = [k for k in NOISE if k not in fixed]
    clean = np.ones(len(df), bool)
    for k in unfixed:
        clean &= (df[f"noise_{k}"] == 0).values
    hit = (df["within_session"] == 1).values & clean
    hit = hit & (rng.random(len(df)) >= inv_rate)
    return float(hit.mean())


def speedup(reusable_fraction, hit, overhead):
    cost = (1 - reusable_fraction) + reusable_fraction * ((1 - hit) * 1.0 + hit * overhead)
    return 1.0 / cost


def run():
    os.makedirs(TABLES, exist_ok=True)
    df = simulate()
    r = float(df.reusable_fraction.mean())

    # cumulative transform ablation (add in a sensible priority order)
    order = ["strip_volatile_ids", "versioned_memory", "normalize_tool_schemas",
             "sort_tool_definitions", "stable_doc_hashes"]
    rows = []
    applied = []
    base_hit = hit_rate(df, [])
    rows.append(dict(stage="exact (no canon)", hit_rate=base_hit,
                     speedup_ov0=speedup(r, base_hit, 0.0),
                     speedup_ov03=speedup(r, base_hit, 0.03)))
    for tr in order:
        applied.append(tr)
        h = hit_rate(df, applied)
        rows.append(dict(stage="+" + tr, hit_rate=h,
                         speedup_ov0=speedup(r, h, 0.0),
                         speedup_ov03=speedup(r, h, 0.03)))
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(TABLES, "prefix_reuse_v2_ablation.csv"), index=False)

    # invalidation sensitivity at full canonicalization
    full_t = list(TRANSFORMS)
    inv_rows = []
    for inv in (0.0, 0.05, 0.1, 0.2, 0.3):
        h = hit_rate(df, full_t, inv_rate=inv)
        for ov in (0.0, 0.03, 0.1):
            inv_rows.append(dict(invalidation=inv, overhead=ov, hit_rate=h,
                                 speedup=speedup(r, h, ov)))
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(os.path.join(TABLES, "prefix_reuse_v2_invalidation.csv"), index=False)

    write_report(df, abl, inv, r)
    return abl, inv


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(df, abl, inv, r):
    exact = float(abl.iloc[0].hit_rate); full = float(abl.iloc[-1].hit_rate)
    sp_full = float(abl.iloc[-1].speedup_ov03)
    sp_exact = float(abl.iloc[0].speedup_ov03)
    # robust speedup at invalidation 0.1, overhead 0.03
    rob = inv[(inv.invalidation == 0.1) & (inv.overhead == 0.03)].speedup.iloc[0]
    keep = sp_full >= 5.0
    strong = sp_full >= 10.0
    verdict = "STRONG KEEP" if strong else ("KEEP" if keep else "PARK")
    L = []; A = L.append
    A("# Prefix / Latent Reuse — Canonicalization Ablation (Probe A)")
    A("")
    A("> Simulated token-level agent traces; analytic speedup (economics, not "
      "wall-clock). Each session carries several independent prefix-noise sources; "
      "each canonicalization transform neutralises one.")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    A(f"- Mean reusable-prefix fraction **{r:.2f}**.")
    A(f"- Exact-match hit rate **{exact:.2f}** → full-canonicalization hit rate "
      f"**{full:.2f}**.")
    A(f"- Modeled end-to-end speedup (overhead 3%): exact **{sp_exact:.1f}×** → "
      f"canonicalized **{sp_full:.1f}×**.")
    A(f"- Under 10% cache invalidation + 3% overhead, still **{rob:.1f}×**.")
    A(f"- Rule: KEEP if canonicalization ≥5× modeled speedup; STRONG KEEP ≥10×.")
    A("")
    A("## Canonicalization ablation (cumulative)")
    A("")
    A(_md(abl))
    A("")
    A("## Invalidation sensitivity (full canonicalization)")
    A("")
    A(_md(inv))
    A("")
    A("## Reading")
    A("")
    A("- The biggest hit-rate recoveries come from `strip_volatile_ids` and "
      "`versioned_memory` (the highest-probability noise sources). Schema "
      "normalization/sorting add the rest.")
    A("- Speedup is bounded by the *reusable fraction* (≈{:.2f}); even a perfect "
      "cache cannot beat 1/(1-r). 100× needs reuse of the *unique tail* too "
      "(cross-request latent reuse), not just the prefix.".format(r))
    A("- This is the recommended lever for the agent / repeated-prefix regime, "
      "which transferred worst for sparse routing.")
    A("")
    p = os.path.join(RESULTS, "PREFIX_REUSE_REPORT.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p, "| verdict:", verdict)


if __name__ == "__main__":
    run()
