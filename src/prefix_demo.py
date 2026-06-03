"""TRACK A — prefix/latent reuse PRODUCT demo.

Runs synthetic agent traces through canonicalization + a KV/latent object store
and measures exact vs canonical vs invalidation-adjusted hit rate, reusable
fraction, and modeled TTFT / cost savings.

Cost model: prefill cost ∝ prompt tokens. On a prefix hit, the reusable-prefix
tokens are served from the cached KV/latent object (cost = `overhead` × those
tokens for the lookup/transfer); only the changing user tail is recomputed.

    cost_per_request = unique_tail_tokens
                       + reusable_tokens * (overhead if hit else 1.0)
    baseline         = total_tokens
    savings (TTFT/compute) = baseline / cost

Writes results/PREFIX_PRODUCT_DEMO.md + results/tables/prefix_demo_*.csv.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from prefix_cache import (Context, canonical_prefix_bytes, exact_prefix_bytes,
                          prefix_hash, PrefixStore, trace_sim)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_ID, TOK_ID = "qwen2.5-1.5b", "qwen2-bpe"
DTYPE_BYTES, N_LAYERS, N_KV, HEAD_DIM = 2, 28, 2, 128
DECODE_TOKENS = 250          # generated tokens/request (decode is NOT reuse-saved)


def kv_bytes(tokens):
    return 2 * N_LAYERS * N_KV * HEAD_DIM * tokens * DTYPE_BYTES


def mem_version(ctx: Context) -> int:
    return max((int(b.get("version", 0)) for b in ctx.memory), default=0)


def run_policy(requests, mode, transforms=None, overhead=0.03):
    """mode in {exact, canonical}. Returns per-request records."""
    store = PrefixStore()
    rows = []
    for (s, t, ctx, mem_event) in requests:
        if mode == "exact":
            ph = prefix_hash(exact_prefix_bytes(ctx)); ver = 0
        else:
            ph = prefix_hash(canonical_prefix_bytes(ctx, transforms))
            ver = mem_version(ctx)            # memory edit -> new version -> miss
        key = store.make_key(MODEL_ID, TOK_ID, ph, ver)
        hit = store.get(key) is not None
        rtok = ctx.reusable_tokens(); ttok = ctx.total_tokens()
        if not hit:
            store.put(key, rtok, kv_bytes(rtok))
        prefill = (ttok - rtok) + rtok * (overhead if hit else 1.0)
        # TTFT = prefill only; total cost = prefill + decode
        rows.append(dict(session=s, turn=t, mode=mode, hit=int(hit),
                         mem_event=int(mem_event), reusable_tokens=rtok,
                         total_tokens=ttok, reusable_fraction=rtok / ttok,
                         ttft_cost=prefill, ttft_base=ttok,
                         total_cost=prefill + DECODE_TOKENS,
                         total_base=ttok + DECODE_TOKENS))
    return pd.DataFrame(rows)


def summarize(df):
    return dict(hit_rate=df.hit.mean(), reusable_fraction=df.reusable_fraction.mean(),
                ttft_savings=df.ttft_base.sum() / df.ttft_cost.sum(),
                total_cost_savings=df.total_base.sum() / df.total_cost.sum(),
                n=len(df))


def run():
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    reqs = trace_sim.generate(n_sessions=50, turns=8, seed=0)

    ex = run_policy(reqs, "exact")
    ca = run_policy(reqs, "canonical")
    # invalidation-free canonical (ignore memory version) to isolate invalidation cost
    ca_nv = run_policy([(s, t, Context(c.system, c.tools, c.docs,
                        [dict(b, version=1) for b in c.memory], c.user), m)
                       for (s, t, c, m) in reqs], "canonical")

    main = pd.DataFrame([{**summarize(ex), "policy": "exact"},
                         {**summarize(ca), "policy": "canonical(+versioned mem)"},
                         {**summarize(ca_nv), "policy": "canonical(no invalidation)"}])
    main.to_csv(os.path.join(RESULTS, "tables", "prefix_demo_main.csv"), index=False)

    # transform ablation (cumulative) on canonical hit rate + savings
    order = ["system", "memory", "tools", "docs"]
    abl = []
    applied = set()
    for label, tset in [("exact (none)", set())] + [
            ("+" + o, None) for o in order]:
        if tset is None:
            applied.add(order[len(applied)])
            tset = set(applied)
        d = run_policy(reqs, "canonical", transforms=tset)
        abl.append({"stage": label, **summarize(d)})
    abl = pd.DataFrame(abl)
    abl.to_csv(os.path.join(RESULTS, "tables", "prefix_demo_ablation.csv"), index=False)

    # overhead / TTL sensitivity at full canonicalization
    sens = []
    for ov in (0.0, 0.01, 0.03, 0.05, 0.1):
        d = run_policy(reqs, "canonical", overhead=ov)
        sens.append({"overhead": ov, **summarize(d)})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(RESULTS, "tables", "prefix_demo_overhead.csv"), index=False)

    write_report(main, abl, sens, ex, ca)
    return main


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(main, abl, sens, ex, ca):
    g = main.set_index("policy")
    sav_canon = float(g.loc["canonical(+versioned mem)", "ttft_savings"])
    sav_exact = float(g.loc["exact", "ttft_savings"])
    tot_canon = float(g.loc["canonical(+versioned mem)", "total_cost_savings"])
    hit_canon = float(g.loc["canonical(+versioned mem)", "hit_rate"])
    hit_exact = float(g.loc["exact", "hit_rate"])
    verdict = ("STRONG KEEP" if sav_canon >= 5.0 else
               ("KEEP" if sav_canon >= 3.0 else "PARK"))
    L = []; A = L.append
    A("# Prefix / Latent Reuse — Product Demo (Track A)")
    A("")
    A("A runnable prototype: agent traces → **canonicalization** "
      "(`prefix_cache/canonicalize.py`) → **KV/latent object store** "
      "(`prefix_cache/store.py`, key = model+tokenizer+canonical_hash+version). "
      "Token counts are char/4 estimates; the cache stores object *metadata* "
      "(KV bytes modelled), not real GPU tensors. Economics, not wall-clock.")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    A(f"- **Canonical (versioned-memory invalidation): {sav_canon:.1f}× modeled "
      f"TTFT (prefill) savings** at hit rate {hit_canon:.2f}.")
    A(f"- **Total-cost savings (prefill+decode, {DECODE_TOKENS}-tok generation): "
      f"{tot_canon:.1f}×** — decode is not reuse-saved, so $ savings are smaller "
      f"than TTFT; reuse is primarily a *latency* (TTFT) lever.")
    A(f"- Exact-match cache only: {sav_exact:.1f}× TTFT at hit rate {hit_exact:.2f} "
      f"— per-request volatile IDs / tool reordering / doc-path noise break it.")
    A(f"- Mean reusable-prefix fraction {float(g.loc['exact','reusable_fraction']):.2f}.")
    A(f"- Rule (TTFT): STRONG KEEP ≥5× after invalidation; KEEP ≥3×; PARK <3×.")
    A("")
    A("## Policies")
    A("")
    A(_md(main[["policy", "hit_rate", "reusable_fraction", "ttft_savings",
                "total_cost_savings", "n"]]))
    A("")
    A("## Canonicalization ablation (cumulative, TTFT savings)")
    A("")
    A(_md(abl[["stage", "hit_rate", "ttft_savings", "total_cost_savings"]]))
    A("")
    A("## Overhead sensitivity (full canonicalization, versioned memory)")
    A("")
    A(_md(sens[["overhead", "hit_rate", "ttft_savings", "total_cost_savings"]]))
    A("")
    A("## What the prototype shows")
    A("")
    A("- **Canonicalization is the product.** It converts a near-useless exact "
      f"cache ({sav_exact:.1f}× TTFT) into a {sav_canon:.1f}× TTFT engine by "
      "mapping meaning-equivalent prefixes to one key, while versioned memory "
      "blocks give correct invalidation on real edits.")
    A("- The store key (model+tokenizer+canonical_hash+version) is the addressing "
      "scheme a real KV/latent object store would use; swapping the metadata value "
      "for an actual paged-KV / latent handle is the only remaining engineering.")
    A("- Savings are bounded by the reusable fraction; the changing user tail is "
      "always recomputed. 100× needs cross-request *latent* reuse (shared tool "
      "results / sub-dialogues), the Track B/persistent-state direction.")
    A("")
    p = os.path.join(RESULTS, "PREFIX_PRODUCT_DEMO.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p, "| verdict:", verdict)


if __name__ == "__main__":
    run()
