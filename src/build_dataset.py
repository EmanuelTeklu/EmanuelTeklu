"""Build the supervised sparse-safety dataset.

Each row = one analysis case (prompt, layer, query-head, last-token query).
We re-capture real Q/K/V from the model (the raw tensors were not all persisted)
and compute, per case:

  * features -- tagged CHEAP (computable without the full softmax over true
    logits; deployable at decode time) vs ORACLE (need the true attention
    distribution; an upper bound on predictability).
  * labels   -- oracle sparse and best-non-oracle-router (sink+recent+route)
    rel-L2 at read ratios {1,2,5,10,15,25}%, thresholded at {0.05,0.10,0.20}.

The deployable selective policy uses the ROUTER at a fixed read budget when the
classifier says "safe"; so the load-bearing label is router-based, with the
oracle label reported alongside as the ceiling.

Output: results/tables/classifier_dataset.csv
"""
from __future__ import annotations
import os, sys, math, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import metrics as MET
import attention_ops as AO
import sparse_routing as SR

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES = os.path.join(RESULTS, "tables")
SEED = 0
BLOCK = 32
READ_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.25]
THRESHOLDS = [0.05, 0.10, 0.20]
RECENT = 64
SINK = 4

# Which feature columns are cheap (no full true-logit softmax required).
CHEAP_FEATURES = [
    "log_T", "q_norm",
    "k_norm_mean", "k_norm_std", "k_norm_max",
    "v_norm_mean", "v_norm_std", "v_norm_max",
    "value_spectral",
    "cent_top1", "cent_margin12", "cent_entropy", "cent_top_block_mass",
    "cent_recent_frac", "qk_align_max",
]
ORACLE_FEATURES = [
    "entropy", "gini", "mass_top8", "mass_top16", "mass_top32",
    "margin_top8", "margin_top16", "sink_mass", "recent_mass",
]
ID_FEATURES = ["layer", "head"]  # used only in random-split variant


def _ceil_S(r, T):
    return max(1, min(T, math.ceil(r * T)))


def case_row(name, regime, T, li, h, q, K, V, scaling):
    logits = AO.compute_logits(q, K, scaling)
    probs = AO.softmax(logits)
    O_full = probs @ V
    d = K.shape[1]

    # --- cheap features (structure of q/K/V; no true softmax) ---
    qn = float(np.linalg.norm(q))
    kn = np.linalg.norm(K, axis=1)
    vn = np.linalg.norm(V, axis=1)
    cK, mV, rad, spans = SR.block_centroid_summaries(K, V, BLOCK)
    cent_scores = cK @ q                       # cheap per-block scores
    cs_sorted = np.sort(cent_scores)[::-1]
    cent_p = AO.softmax(cent_scores)
    nb = len(cent_scores)
    # fraction of top-ceil(10%) blocks that lie in the recent window
    top_blocks = np.argsort(cent_scores)[::-1][:max(1, nb // 10)]
    recent_lo = T - RECENT
    cent_recent_frac = float(np.mean([spans[b][0] >= recent_lo for b in top_blocks]))
    # cheap query-key alignment proxy: max cosine over a subsample of keys
    qk_align_max = float((K @ q / (kn * qn + 1e-9)).max())

    feats = dict(
        layer=li, head=h, prompt=name, regime=regime, T=T,
        log_T=math.log(T), q_norm=qn,
        k_norm_mean=float(kn.mean()), k_norm_std=float(kn.std()), k_norm_max=float(kn.max()),
        v_norm_mean=float(vn.mean()), v_norm_std=float(vn.std()), v_norm_max=float(vn.max()),
        value_spectral=MET.value_spectral_decay(V),
        cent_top1=float(cs_sorted[0]),
        cent_margin12=float(cs_sorted[0] - cs_sorted[1]) if nb > 1 else 0.0,
        cent_entropy=MET.attention_entropy(cent_p),
        cent_top_block_mass=float(cent_p.max()),
        cent_recent_frac=cent_recent_frac,
        qk_align_max=qk_align_max,
    )

    # --- oracle features (true attention distribution) ---
    feats.update(dict(
        entropy=MET.attention_entropy(probs),
        gini=MET.gini_concentration(probs),
        mass_top8=MET.mass_retained(probs, np.argsort(probs)[::-1][:min(8, T)]),
        mass_top16=MET.mass_retained(probs, np.argsort(probs)[::-1][:min(16, T)]),
        mass_top32=MET.mass_retained(probs, np.argsort(probs)[::-1][:min(32, T)]),
        margin_top8=MET.topk_margin(logits, min(8, T - 1)),
        margin_top16=MET.topk_margin(logits, min(16, T - 1)),
        sink_mass=float(probs[:SINK].sum()),
        recent_mass=float(probs[recent_lo:].sum()),
    ))

    # --- labels: oracle sparse at each read ratio ---
    for r in READ_RATIOS:
        S = _ceil_S(r, T)
        idx = SR.oracle_topk(q, K, scaling, S, logits=logits)
        Os, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
        feats[f"oracle_rl2_{int(r*100)}"] = MET.rel_l2(Os, O_full)

    # --- labels: best non-oracle router (sink+recent+route) at 10% & 15% ---
    for r in (0.10, 0.15):
        budget = _ceil_S(r, T)
        ridx = SR.sink_recent_router_select(q, K, V, budget, sink=SINK,
                                             recent=min(RECENT, budget), block=BLOCK)
        Or, _ = AO.sparse_attention(q, K, V, scaling, ridx, logits=logits)
        feats[f"router_rl2_{int(r*100)}"] = MET.rel_l2(Or, O_full)
        feats[f"router_read_{int(r*100)}"] = len(ridx) / T
    return feats


def build(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_tokens=4096):
    os.makedirs(TABLES, exist_ok=True)
    import torch
    torch.manual_seed(SEED)
    from capture_qkv import load_model, capture_prompt, build_prompts
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x

    model, tok = load_model(model_name)
    rows = []
    for p in build_prompts():
        cap = capture_prompt(model, tok, p["text"], query_positions=("last",), max_tokens=max_tokens)
        qi = len(cap["query_positions"]) - 1
        tpos = cap["query_positions"][qi]
        groups = cap["groups"]; scaling = cap["scaling"]
        print(f"  {p['name']:16s} T={cap['T']}")
        for L in tqdm(cap["layers"], desc=p["name"], leave=False):
            for h in range(cap["n_heads"]):
                kv = h // groups
                q = L["Q"][qi, h].astype(np.float64)
                K = L["K"][kv][: tpos + 1].astype(np.float64)
                V = L["V"][kv][: tpos + 1].astype(np.float64)
                rows.append(case_row(p["name"], p["regime"], cap["T"],
                                     L["layer_idx"], h, q, K, V, scaling))
    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(TABLES, "classifier_dataset.csv")
    df.to_csv(out, index=False)
    meta = dict(cheap_features=CHEAP_FEATURES, oracle_features=ORACLE_FEATURES,
                id_features=ID_FEATURES, read_ratios=READ_RATIOS,
                thresholds=THRESHOLDS, n_rows=len(df), block=BLOCK,
                recent=RECENT, sink=SINK, seed=SEED)
    with open(os.path.join(TABLES, "classifier_dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out}  rows={len(df)} cols={len(df.columns)}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    build(args.model, args.max_tokens)
