"""MAIN NEXT — Experiments 1 & 2 capture/analysis: block-oracle upper bound and
locality-regularized block routers.

Question: can sparse-safe routing be made hardware-realizable by selecting
contiguous BLOCKS instead of scattered tokens? We compute, per real case:

  * token-oracle sparse (reference upper bound)
  * block-oracle sparse at block sizes {16,32,64,128,256} (select best blocks by
    true attention mass) -- Experiment 1 headroom
  * cheap NON-oracle block routers -- Experiment 2:
      centroid, centroid+radius upper-bound, max-sketch, recent/sink+routed
  * interval-cover of the token-oracle selection (hardware over-read of trying
    to gather the scattered token choice as whole blocks)

Equal-read-budget framing: every policy reads ~r*T tokens (rounded to whole
blocks); the question is the resulting rel-L2. Plus a cover framing for over-read.

Outputs (results/tables/block/):
  agg_{tag}_{ctx}.csv        -- aggregated by (regime,policy,block_size,budget)
  percase_{tag}_{ctx}.csv    -- per-case rel-L2 for block sizes {16,32}, key
                                policies (for the selective classifier / Exp 3)
Resumable. Long-context focus.
"""
from __future__ import annotations
import os, sys, gc, time, argparse, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_dataset as BD
import attention_ops as AO

BLOCKDIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "block")
SEED = 0
BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
BLOCK_SIZES = [16, 32, 64, 128, 256]
PERCASE_SIZES = [16, 32]
PERCASE_POLICIES = ["block_oracle", "centroid_ub", "maxsketch"]
REF_BUDGET = 0.02
RECENT, SINK = 64, 4
SKETCH_R = 16
MEM_LIMIT_GB = 13.0

PLAN = {
    "0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", [8192, 16384, 32768]),
    "1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", [8192]),
}


def _ceilS(r, T):
    return max(1, min(T, int(np.ceil(r * T))))


def _starts(T, b):
    return np.arange(0, T, b, dtype=np.int64)


def _gather(starts, counts, sel_blocks, T):
    """Token indices for selected blocks."""
    out = []
    for bi in sel_blocks:
        a = starts[bi]; out.append(np.arange(a, min(a + counts[bi], T)))
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def case_analyze(q, K, V, scaling, logits, probs, O_full, sketch_P):
    T, d = K.shape
    order = np.argsort(logits)[::-1]
    qn = float(np.linalg.norm(q))
    Kp = K @ sketch_P                 # [T, r]
    qp = q @ sketch_P                 # [r]
    tok_sketch = Kp @ qp              # [T]  (cheap max-logit proxy)

    rows = []          # aggregated rows for this case
    percase = {}       # per-case rl2 values for selected configs

    # token oracle reference
    tok_rl2 = {}
    for r in BUDGETS:
        S = _ceilS(r, T); idx = order[:S]
        Os, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
        rl2 = float(np.linalg.norm(Os - O_full) / (np.linalg.norm(O_full) + 1e-12))
        tok_rl2[r] = rl2
        rows.append(dict(policy="token_oracle", block_size=0, budget=r,
                         rl2=rl2, read_ratio=S / T, mass=float(probs[idx].sum()),
                         support=1.0, n_blocks=np.nan, over_read=1.0))

    for b in BLOCK_SIZES:
        starts = _starts(T, b)
        counts = np.minimum(starts + b, T) - starts
        nb_total = len(starts)
        # block scores
        block_mass = np.add.reduceat(probs, starts)                  # oracle
        cen = np.add.reduceat(K, starts, axis=0) / counts[:, None]   # [nb,d]
        cen_score = cen @ q
        # radius: max ||k - c|| per block
        diff = K - np.repeat(cen, counts, axis=0)
        dist = np.linalg.norm(diff, axis=1)
        radius = np.maximum.reduceat(dist, starts)
        ub_score = cen_score + qn * radius
        sketch_score = np.maximum.reduceat(tok_sketch, starts)
        policies = dict(block_oracle=block_mass, centroid=cen_score,
                        centroid_ub=ub_score, maxsketch=sketch_score)

        for r in BUDGETS:
            S = _ceilS(r, T)
            nb = max(1, int(round(r * T / b)))
            nb = min(nb, nb_total)
            read = min(nb * b, T)
            for pname, score in policies.items():
                sel = np.argsort(score)[::-1][:nb]
                idx = _gather(starts, counts, sel, T)
                Os, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
                rl2 = float(np.linalg.norm(Os - O_full) / (np.linalg.norm(O_full) + 1e-12))
                supp = float(np.isin(order[:S], idx).mean())
                rows.append(dict(policy=pname, block_size=b, budget=r, rl2=rl2,
                    read_ratio=len(idx) / T, mass=float(probs[idx].sum()),
                    support=supp, n_blocks=nb, over_read=len(idx) / S))
                if b in PERCASE_SIZES and pname in PERCASE_POLICIES:
                    percase[f"brl2_{pname}_{b}_{r}"] = rl2

        # interval-cover: gather the token-oracle top-S as whole blocks (hardware over-read)
        for r in (0.02, 0.05):
            S = _ceilS(r, T); needed = order[:S]
            cov_blocks = np.unique(needed // b)
            idx = _gather(starts, counts, cov_blocks, T)
            Os, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
            rl2 = float(np.linalg.norm(Os - O_full) / (np.linalg.norm(O_full) + 1e-12))
            rows.append(dict(policy="interval_cover", block_size=b, budget=r, rl2=rl2,
                read_ratio=len(idx) / T, mass=float(probs[idx].sum()), support=1.0,
                n_blocks=len(cov_blocks), over_read=len(idx) / S))

        # recent/sink + routed blocks (block size only)
        for r in BUDGETS:
            nb = max(1, int(round(r * T / b))); nb = min(nb, nb_total)
            forced = set([0])                                  # sink block
            forced |= set(np.unique(np.arange(max(0, T - RECENT), T) // b).tolist())
            remaining = max(0, nb - len(forced))
            route_order = [bi for bi in np.argsort(cen_score)[::-1] if bi not in forced]
            sel = list(forced) + route_order[:remaining]
            idx = _gather(starts, counts, np.array(sel), T)
            Os, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
            rl2 = float(np.linalg.norm(Os - O_full) / (np.linalg.norm(O_full) + 1e-12))
            rows.append(dict(policy="recent_sink_routed", block_size=b, budget=r, rl2=rl2,
                read_ratio=len(idx) / T, mass=float(probs[idx].sum()),
                support=float(np.isin(order[:_ceilS(r,T)], idx).mean()),
                n_blocks=len(sel), over_read=len(idx) / _ceilS(r, T)))

    return rows, percase, cen_score_ref(K, V, q, probs, logits, O_full, scaling)


def cen_score_ref(K, V, q, probs, logits, O_full, scaling):
    """Reference block selection (size 32, centroid_ub, REF_BUDGET) for shared-block
    overlap -- returns the selected block-id set at size 32."""
    T = K.shape[0]; b = 32
    starts = _starts(T, b); counts = np.minimum(starts + b, T) - starts
    cen = np.add.reduceat(K, starts, axis=0) / counts[:, None]
    score = cen @ q + np.linalg.norm(q) * np.maximum.reduceat(
        np.linalg.norm(K - np.repeat(cen, counts, axis=0), axis=1), starts)
    nb = max(1, int(round(REF_BUDGET * T / b)))
    return set(np.argsort(score)[::-1][:nb].tolist())


def build_one(model, tok, q_heads, model_gb, tag, ctx):
    import capture_lasttok as C
    apath = os.path.join(BLOCKDIR, f"agg_{tag}_{ctx}.csv")
    if os.path.exists(apath):
        print(f"  RESUME {tag}@{ctx}"); return
    if q_heads * min(ctx, 4096) * ctx * 2 / 1e9 + model_gb + 1.5 > MEM_LIMIT_GB:
        print(f"  SKIP {tag}@{ctx}"); return
    from capture_qkv import build_long_prompts
    prompts, seen = [], set()
    for p in build_long_prompts(ctx):
        if p["regime"] in seen:
            continue
        if p["regime"] == "needle" and "d50" not in p["name"]:
            continue
        prompts.append(p); seen.add(p["regime"])
    rng = np.random.default_rng(SEED)
    agg_rows, pc_rows, shared_rows = [], [], []
    for p in prompts:
        t0 = time.time()
        cap = C.capture_prompt(model, tok, p["text"], max_tokens=ctx)
        err, _ = C.verify_capture(cap)
        qi = len(cap["query_positions"]) - 1; tpos = cap["query_positions"][qi]
        groups = cap["groups"]; scaling = cap["scaling"]; d = cap["d"]
        sketch_P = rng.standard_normal((d, SKETCH_R)) / np.sqrt(SKETCH_R)
        shared_sets = {}  # (layer,head)->block set @32 for shared-block overlap
        for L in cap["layers"]:
            for h in range(cap["n_heads"]):
                kv = h // groups
                q = L["Q"][qi, h].astype(np.float64)
                K = L["K"][kv][: tpos + 1].astype(np.float64)
                V = L["V"][kv][: tpos + 1].astype(np.float64)
                logits = AO.compute_logits(q, K, scaling)
                probs = AO.softmax(logits); O_full = probs @ V
                rows, percase, refset = case_analyze(q, K, V, scaling, logits, probs,
                                                     O_full, sketch_P)
                for rr in rows:
                    rr.update(dict(regime=p["regime"], layer=L["layer_idx"], head=h))
                    agg_rows.append(rr)
                feat = BD.case_row(p["name"], p["regime"], cap["T"], L["layer_idx"], h,
                                   q, K, V, scaling)
                feat.update(percase); feat.update(dict(model=tag, ctx=ctx))
                pc_rows.append(feat)
                shared_sets[(L["layer_idx"], h)] = refset
        # shared-block across heads (within layer): union of per-head @32 ref sets,
        # measure how much bigger the shared set is vs per-head (read amplification)
        layers = sorted({l for (l, _) in shared_sets})
        for l in layers:
            hs = [shared_sets[(l, hh)] for (ll, hh) in shared_sets if ll == l]
            if not hs:
                continue
            union = set().union(*hs)
            mean_indiv = np.mean([len(s) for s in hs])
            shared_rows.append(dict(model=tag, ctx=ctx, regime=p["regime"], layer=l,
                mean_blocks_per_head=mean_indiv, shared_union_blocks=len(union),
                read_amplification=len(union) / (mean_indiv + 1e-9)))
        print(f"  {tag}@{ctx} {p['regime']:8s} T={cap['T']:6d} verify={err:.1e} ({time.time()-t0:.0f}s)")
        del cap; gc.collect()
    os.makedirs(BLOCKDIR, exist_ok=True)
    # aggregate
    ad = pd.DataFrame(agg_rows)
    g = ad.groupby(["regime", "policy", "block_size", "budget"]).agg(
        rl2_med=("rl2", "median"), rl2_p90=("rl2", lambda x: x.quantile(0.9)),
        read_ratio=("read_ratio", "mean"), pass10=("rl2", lambda x: (x <= 0.10).mean()),
        mass=("mass", "mean"), support=("support", "mean"),
        over_read=("over_read", "mean"), n=("rl2", "size")).reset_index()
    g["model"] = tag; g["ctx"] = ctx
    g.to_csv(apath, index=False)
    pd.DataFrame(pc_rows).to_csv(os.path.join(BLOCKDIR, f"percase_{tag}_{ctx}.csv"), index=False)
    pd.DataFrame(shared_rows).to_csv(os.path.join(BLOCKDIR, f"shared_{tag}_{ctx}.csv"), index=False)


def run(plan=PLAN):
    os.makedirs(BLOCKDIR, exist_ok=True)
    import capture_lasttok as C
    from transformers import AutoConfig
    for tag, (hf, ctxs) in plan.items():
        cfg = AutoConfig.from_pretrained(hf)
        q_heads = cfg.num_attention_heads
        model_gb = {"0.5B": 1.0, "1.5B": 3.0}.get(tag, 3.0)
        print(f"=== {tag} ===")
        model, tok = C.load_model(hf)
        for ctx in ctxs:
            try:
                build_one(model, tok, q_heads, model_gb, tag, ctx)
            except Exception as e:
                print(f"  ERROR {tag}@{ctx}: {e}"); traceback.print_exc()
            gc.collect()
        del model; gc.collect()
    print("block capture done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5B,1.5B")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    run({k: v for k, v in PLAN.items() if k in args.models.split(",")})
