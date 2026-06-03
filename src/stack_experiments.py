"""Capture pass feeding MAIN STACK Experiment 1 (adaptive budget) and
Experiment 2 (block locality / hardware realizability).

For each (model, context, prompt, layer, head, last-token query) it computes:
  * oracle sparse rel-L2 at the full budget ladder
    {0.25,0.5,1,2,5,10,15}% and full -> pass/fail at {0.05,0.10,0.20}
    (one argsort, prefixes reused -> cheap);
  * the cheap deployable feature vector (reused from build_dataset.case_row);
  * block-locality metrics of the oracle selection at a reference budget
    (contiguity, #blocks at sizes 16/32/64/128, block utilization, recent/sink
    fraction, softmax length S).

Per prompt it also records cross-head and cross-layer Jaccard overlap of the
selected token sets (hardware reuse signal).

Outputs (results/tables/stack/):
  budget_{tag}_{ctx}.csv     -- one row per case (features + rl2 ladder + locality)
  overlap_{tag}_{ctx}.csv    -- one row per (prompt) cross-head/layer overlap
Resumable: skips a (tag,ctx) whose budget CSV already exists.
"""
from __future__ import annotations
import os, sys, gc, json, time, argparse, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_dataset as BD
import attention_ops as AO
import sparse_routing as SR

STACKDIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "stack")
SEED = 0
BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]   # fractions; 'full' implicit (rl2=0)
THRESHS = [0.05, 0.10, 0.20]
REF_BUDGET = 0.02     # reference selection for locality/overlap metrics
BLOCK_SIZES = [16, 32, 64, 128]
RECENT, SINK = 64, 4
MEM_LIMIT_GB = 13.0

PLAN = {
    "0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", [4096, 8192, 16384, 32768]),
    "1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", [8192]),
}


def _ceilS(r, T):
    return max(1, min(T, int(np.ceil(r * T))))


def block_metrics(idx, T):
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    S = len(idx)
    runs = 1 + int(np.sum(np.diff(idx) > 1)) if S > 0 else 0
    m = dict(S=S, contig_runs=runs, run_frac=(runs / S) if S else 1.0,
             recent_frac=float(np.mean(idx >= T - RECENT)) if S else 0.0,
             sink_frac=float(np.mean(idx < SINK)) if S else 0.0)
    for b in BLOCK_SIZES:
        blocks = np.unique(idx // b)
        m[f"nblocks_{b}"] = int(len(blocks))
        m[f"blockutil_{b}"] = float(S / (len(blocks) * b)) if len(blocks) else 0.0
    return m


def case_plus(name, regime, model, ctx, T, li, h, q, K, V, scaling):
    """Feature row (reuses build_dataset.case_row) + full budget ladder + locality.
    Returns (row_dict, ref_selection_indices)."""
    row = BD.case_row(name, regime, T, li, h, q, K, V, scaling)
    row.update(dict(model=model, ctx=ctx))
    logits = AO.compute_logits(q, K, scaling)
    probs = AO.softmax(logits)
    O_full = probs @ V
    order = np.argsort(logits)[::-1]
    ref_idx = None
    for r in BUDGETS:
        S = _ceilS(r, T)
        idx = order[:S]
        Os, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
        rl2 = AO.np.linalg.norm(Os - O_full) / (np.linalg.norm(O_full) + 1e-12)
        row[f"orl2_{r}"] = float(rl2)
        if abs(r - REF_BUDGET) < 1e-9:
            ref_idx = idx
    row["orl2_full"] = 0.0
    if ref_idx is None:
        ref_idx = order[:_ceilS(REF_BUDGET, T)]
    row.update(block_metrics(ref_idx, T))
    return row, set(int(i) for i in ref_idx)


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def build_one(model, tok, q_heads, model_gb, tag, ctx):
    import capture_lasttok as C
    bpath = os.path.join(STACKDIR, f"budget_{tag}_{ctx}.csv")
    if os.path.exists(bpath):
        print(f"  RESUME {tag}@{ctx}")
        return
    chunk = min(ctx, 4096)
    est = q_heads * chunk * ctx * 2 / 1e9 + model_gb + 1.5
    if est > MEM_LIMIT_GB:
        print(f"  SKIP {tag}@{ctx}: est {est:.1f}GB"); return
    from capture_qkv import build_long_prompts
    prompts, seen = [], set()
    for p in build_long_prompts(ctx):
        if p["regime"] in seen:
            continue
        if p["regime"] == "needle" and "d50" not in p["name"]:
            continue
        prompts.append(p); seen.add(p["regime"])
    rows, overlap_rows = [], []
    for p in prompts:
        t0 = time.time()
        cap = C.capture_prompt(model, tok, p["text"], max_tokens=ctx)
        err, ok = C.verify_capture(cap)
        qi = len(cap["query_positions"]) - 1; tpos = cap["query_positions"][qi]
        groups = cap["groups"]; scaling = cap["scaling"]
        sel = {}  # (layer,head)->set for overlap
        for L in cap["layers"]:
            for h in range(cap["n_heads"]):
                kv = h // groups
                q = L["Q"][qi, h].astype(np.float64)
                K = L["K"][kv][: tpos + 1].astype(np.float64)
                V = L["V"][kv][: tpos + 1].astype(np.float64)
                row, s = case_plus(p["name"], p["regime"], tag, ctx, cap["T"],
                                   L["layer_idx"], h, q, K, V, scaling)
                rows.append(row); sel[(L["layer_idx"], h)] = s
        # cross-head overlap (within layer), cross-layer overlap (within head)
        layers = sorted({l for (l, _) in sel})
        heads = sorted({h for (_, h) in sel})
        ch = []
        for l in layers:
            hs = [sel[(l, h)] for h in heads if (l, h) in sel]
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    ch.append(_jaccard(hs[i], hs[j]))
        cl = []
        for h in heads:
            ls = [sel[(l, h)] for l in layers if (l, h) in sel]
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    cl.append(_jaccard(ls[i], ls[j]))
        overlap_rows.append(dict(model=tag, ctx=ctx, prompt=p["name"], regime=p["regime"],
            T=cap["T"], cross_head_jaccard=float(np.mean(ch)) if ch else np.nan,
            cross_layer_jaccard=float(np.mean(cl)) if cl else np.nan))
        print(f"  {tag}@{ctx} {p['regime']:8s} T={cap['T']:6d} verify={err:.1e} ({time.time()-t0:.0f}s)")
        del cap; gc.collect()
    os.makedirs(STACKDIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(bpath, index=False)
    pd.DataFrame(overlap_rows).to_csv(os.path.join(STACKDIR, f"overlap_{tag}_{ctx}.csv"), index=False)


def run(plan=PLAN):
    os.makedirs(STACKDIR, exist_ok=True)
    import capture_lasttok as C
    from transformers import AutoConfig
    for tag, (hf, ctxs) in plan.items():
        cfg = AutoConfig.from_pretrained(hf)
        q_heads = cfg.num_attention_heads
        model_gb = {"0.5B": 1.0, "1.5B": 3.0, "3B": 6.0}.get(tag, 3.0)
        print(f"=== {tag} ({hf}) ===")
        model, tok = C.load_model(hf)
        for ctx in ctxs:
            try:
                build_one(model, tok, q_heads, model_gb, tag, ctx)
            except Exception as e:
                print(f"  ERROR {tag}@{ctx}: {e}"); traceback.print_exc()
            gc.collect()
        del model; gc.collect()
    print("stack capture done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5B,1.5B")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    plan = {k: v for k, v in PLAN.items() if k in args.models.split(",")}
    run(plan)
