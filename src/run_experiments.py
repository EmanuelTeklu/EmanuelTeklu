"""Run the full inference-quotient benchmark on real Q/K/V tensors.

Experiments:
  C  oracle sparse upper bound
  D  non-oracle routing
  E  sparse-plus-residual correction
  F  margin-aware KV quantization
  G  per-head/layer covariates (for predictor correlations)
  H  prefix/latent reuse economics   (economics.py)
  I  Amdahl end-to-end                (economics.py)

Each analysis "case" = (prompt, layer, query-head) at the last token position
(decode-style). GQA: query head h reads KV head h // groups.

Outputs CSVs under results/tables/ and a copy of one raw capture under
results/raw/. Set INFERENCE_QUOTIENT_SYNTHETIC=1 to force the labeled synthetic
fallback (NOT decisive).
"""
from __future__ import annotations
import os, sys, json, time, platform, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import metrics as MET
import attention_ops as AO
import sparse_routing as SR
import residual_correction as RC
import quantization as QZ
import economics as ECON

SEED = 0
np.random.seed(SEED)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES = os.path.join(RESULTS, "tables")
RAW = os.path.join(RESULTS, "raw")

S_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ROUTER_BUDGETS = [8, 16, 32, 64, 128]
RESIDUAL_BUDGETS = [8, 16, 32, 64]
BLOCK = 32
JL_DIMS = [8, 16, 32, 64, 128]
QUANT_BITS = [8, 4, 3, 2]


# ---------------------------------------------------------------------------
def iter_cases(cap):
    """Yield (layer_idx, head, q, K, V, scaling, O_ref) for the last position."""
    qi = len(cap["query_positions"]) - 1
    tpos = cap["query_positions"][qi]
    groups = cap["groups"]
    scaling = cap["scaling"]
    for L in cap["layers"]:
        for h in range(cap["n_heads"]):
            kv = h // groups
            q = L["Q"][qi, h].astype(np.float64)
            K = L["K"][kv][: tpos + 1].astype(np.float64)
            V = L["V"][kv][: tpos + 1].astype(np.float64)
            O_ref = L["out"][qi, h].astype(np.float64)
            yield L["layer_idx"], h, q, K, V, scaling, O_ref


# ---------------------------------------------------------------------------
def run_all(model_name, max_tokens, synthetic=False, limit_layers=None):
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)

    rows_oracle, rows_route, rows_resid, rows_quant, rows_pred = [], [], [], [], []
    env = dict(model=model_name, synthetic=synthetic, max_tokens=max_tokens,
               seed=SEED, block=BLOCK,
               python=platform.python_version(), platform=platform.platform())

    # --- capture -----------------------------------------------------------
    if synthetic:
        from synthetic_fallback import build_synthetic_captures
        captures = build_synthetic_captures(max_tokens)
        env["note"] = "SYNTHETIC FALLBACK -- NOT DECISIVE"
    else:
        import torch
        torch.manual_seed(SEED)
        from capture_qkv import load_model, capture_prompt, build_prompts, verify_capture
        t0 = time.time()
        model, tok = load_model(model_name)
        env["torch"] = torch.__version__
        import transformers
        env["transformers"] = transformers.__version__
        prompts = build_prompts()
        captures = []
        verify_errs = []
        for p in prompts:
            cap = capture_prompt(model, tok, p["text"], query_positions=("last",),
                                 max_tokens=max_tokens)
            cap["name"] = p["name"]; cap["regime"] = p["regime"]
            err, ok = verify_capture(cap)
            verify_errs.append((p["name"], cap["T"], err, ok))
            captures.append(cap)
            print(f"  captured {p['name']:16s} T={cap['T']:5d} verify_err={err:.2e}")
        env["capture_seconds"] = round(time.time() - t0, 1)
        env["verify"] = [dict(name=n, T=t, rel_err=e, ok=bool(o)) for n, t, e, o in verify_errs]
        env["context_lengths"] = {c["name"]: c["T"] for c in captures}
        env["max_context_tested"] = max(c["T"] for c in captures)
        # persist one raw capture (needle) for reproducibility
        for c in captures:
            if c["regime"] == "needle":
                np.savez_compressed(os.path.join(RAW, "sample_capture_needle.npz"),
                                    **_flatten_capture(c))
                break

    # --- per-case experiments ---------------------------------------------
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x

    n_cases = 0
    for cap in captures:
        layers = cap["layers"]
        if limit_layers:
            layers = [L for L in layers if L["layer_idx"] in limit_layers]
            cap = {**cap, "layers": layers}
        name, regime, T = cap["name"], cap["regime"], cap["T"]
        for (li, h, q, K, V, scaling, O_ref) in tqdm(
                list(iter_cases(cap)), desc=f"{name}", leave=False):
            n_cases += 1
            _do_case(name, regime, T, li, h, q, K, V, scaling, O_ref,
                     rows_oracle, rows_route, rows_resid, rows_quant, rows_pred)

    env["n_cases"] = n_cases
    print(f"total cases: {n_cases}")

    # --- write per-experiment CSVs ----------------------------------------
    df_oracle = pd.DataFrame(rows_oracle); df_oracle.to_csv(os.path.join(TABLES, "oracle_sparse.csv"), index=False)
    df_route = pd.DataFrame(rows_route); df_route.to_csv(os.path.join(TABLES, "routing.csv"), index=False)
    df_resid = pd.DataFrame(rows_resid); df_resid.to_csv(os.path.join(TABLES, "residual.csv"), index=False)
    df_quant = pd.DataFrame(rows_quant); df_quant.to_csv(os.path.join(TABLES, "quantization.csv"), index=False)
    df_pred = pd.DataFrame(rows_pred); df_pred.to_csv(os.path.join(TABLES, "predictors.csv"), index=False)

    # --- economics + amdahl (analytic) ------------------------------------
    ECON.amdahl_table().to_csv(os.path.join(TABLES, "amdahl.csv"), index=False)
    ECON.prefix_cache_table().to_csv(os.path.join(TABLES, "prefix_cache.csv"), index=False)

    # --- config + aggregate ----------------------------------------------
    with open(os.path.join(RESULTS, "config.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)
    _aggregate(df_oracle, df_route, df_resid, df_quant)
    print("done.")
    return env


def _flatten_capture(c):
    out = {f"meta_{k}": np.array(c[k]) for k in ["T", "n_layers", "n_heads",
            "n_kv_heads", "groups", "d", "scaling"]}
    # store only first 4 layers to bound size
    for L in c["layers"][:4]:
        i = L["layer_idx"]
        out[f"L{i}_K"] = L["K"].astype(np.float16)
        out[f"L{i}_V"] = L["V"].astype(np.float16)
        out[f"L{i}_Q"] = L["Q"].astype(np.float16)
    return out


# ---------------------------------------------------------------------------
def _do_case(name, regime, T, li, h, q, K, V, scaling, O_ref,
             rows_oracle, rows_route, rows_resid, rows_quant, rows_pred):
    logits = AO.compute_logits(q, K, scaling)
    probs = AO.softmax(logits)
    O_full, _, _ = AO.full_attention(q, K, V, scaling)  # == O_ref up to fp
    entropy = MET.attention_entropy(probs)
    gini = MET.gini_concentration(probs)
    summaries = SR.block_centroid_summaries(K, V, BLOCK)
    vbar_global = V.mean(axis=0)
    basis = RC.head_value_basis(V, k=8)
    spec = MET.value_spectral_decay(V)
    oracle_idx_cache = {}

    # ---- C oracle sparse ----
    for S in S_SWEEP:
        if S > T:
            continue
        idx = SR.oracle_topk(q, K, scaling, S, logits=logits)
        oracle_idx_cache[S] = idx
        O_s, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
        rl2 = MET.rel_l2(O_s, O_full)
        full_probs_on_idx = np.zeros(T); full_probs_on_idx[idx] = probs[idx]
        rows_oracle.append(dict(
            prompt=name, regime=regime, layer=li, head=h, T=T, S=S,
            read_ratio=S / T, component_read_gain=T / S,
            rel_l2=rl2, cosine=MET.cosine(O_s, O_full),
            js=MET.js_div(probs, full_probs_on_idx / (full_probs_on_idx.sum() + 1e-12)),
            topk_overlap=1.0, mass_retained=MET.mass_retained(probs, idx),
            entropy=entropy, gini=gini,
            pass_strict=rl2 <= 0.05, pass_mod=rl2 <= 0.10, pass_weak=rl2 <= 0.20))

    # ---- D non-oracle routing ----
    for B in ROUTER_BUDGETS:
        if B > T:
            continue
        oracle_idx = SR.oracle_topk(q, K, scaling, B, logits=logits)
        # JL (sweep r) -- pick representative r=32 for main table, log all
        for r in JL_DIMS:
            idx = SR.jl_projection_select(q, K, B, r, seed=SEED)
            _route_row(rows_route, name, regime, li, h, T, "jl", r, B, idx,
                       q, K, V, scaling, logits, probs, O_full, oracle_idx)
        # block centroid
        idx, _ = SR.block_centroid_select(q, K, V, B, block=BLOCK, summaries=summaries)
        _route_row(rows_route, name, regime, li, h, T, "block_centroid", BLOCK, B, idx,
                   q, K, V, scaling, logits, probs, O_full, oracle_idx)
        # block upper-bound certificate
        idx, _ = SR.block_upperbound_select(q, K, V, B, block=BLOCK, summaries=summaries)
        _route_row(rows_route, name, regime, li, h, T, "block_upperbound", BLOCK, B, idx,
                   q, K, V, scaling, logits, probs, O_full, oracle_idx)
        # recency hybrid
        idx = SR.recency_router_select(q, K, V, B, recent=min(64, B), block=BLOCK)
        _route_row(rows_route, name, regime, li, h, T, "recency_router", 64, B, idx,
                   q, K, V, scaling, logits, probs, O_full, oracle_idx)
        # sink + recent + route
        idx = SR.sink_recent_router_select(q, K, V, B, sink=4, recent=min(64, B), block=BLOCK)
        _route_row(rows_route, name, regime, li, h, T, "sink_recent_router", 64, B, idx,
                   q, K, V, scaling, logits, probs, O_full, oracle_idx)

    # ---- E residual correction ----
    for B in RESIDUAL_BUDGETS:
        if B > T:
            continue
        idx = SR.oracle_topk(q, K, scaling, B, logits=logits)  # fix selection, vary correction
        rr = B / T
        base = RC.sparse_renorm(logits, V, idx)
        variants = {
            "sparse_renorm": (base, 0),
            "cheap_block_residual": (RC.correct_block_residual(q, K, V, scaling, logits, idx, BLOCK, summaries),
                                      (T // BLOCK) * (K.shape[1] * 2)),
            "oracle_mass_global": (RC.oracle_mass_global(logits, V, idx, vbar_global), K.shape[1]),
            "oracle_mass_block": (RC.oracle_mass_block(logits, V, idx, BLOCK, summaries),
                                   (T // BLOCK) * K.shape[1]),
            "lowrank_k8": (RC.correct_lowrank(logits, V, idx, basis), 8 * K.shape[1]),
        }
        for meth, (Ohat, extra) in variants.items():
            rl2 = MET.rel_l2(Ohat, O_full)
            rows_resid.append(dict(prompt=name, regime=regime, layer=li, head=h, T=T,
                budget=B, read_ratio=rr, method=meth, rel_l2=rl2,
                cosine=MET.cosine(Ohat, O_full), extra_floats=extra))

    # ---- F quantization (frontier sweep) ----
    routed_idx = oracle_idx_cache.get(32, SR.oracle_topk(q, K, scaling, min(32, T), logits=logits))
    schemes = []
    # uniform baselines (KIVI-style: K per-channel, V per-token)
    for b in QUANT_BITS:
        Khat, Vhat, eb = QZ.uniform_kv(K, V, b)
        schemes.append((f"uniform_{b}bit", eb, Khat, Vhat, eb))
    # adaptive token-importance frontier
    for hi, lo, frac in [(8, 4, 0.10), (8, 4, 0.25), (8, 2, 0.10), (8, 2, 0.25), (4, 2, 0.25)]:
        Khat, Vhat, eb, _ = QZ.adaptive_token_importance(K, V, probs, hi_bits=hi, lo_bits=lo, frac_hi=frac)
        schemes.append((f"adapt_imp_h{hi}l{lo}f{int(frac*100)}", eb, Khat, Vhat, eb))
    # adaptive margin-fragility frontier
    for hi, lo, frac in [(8, 4, 0.10), (8, 4, 0.25), (8, 2, 0.25)]:
        Khat, Vhat, eb, _ = QZ.adaptive_margin_fragility(K, V, logits, hi_bits=hi, lo_bits=lo, frac_hi=frac)
        schemes.append((f"adapt_margin_h{hi}l{lo}f{int(frac*100)}", eb, Khat, Vhat, eb))
    # recent-exact + old-quant frontier
    for rec, rb, ob in [(64, 8, 2), (128, 8, 2), (64, 8, 4), (128, 8, 4)]:
        Khat, Vhat, eb, _ = QZ.recent_exact_old_quant(K, V, recent=rec, recent_bits=rb, old_bits=ob)
        schemes.append((f"recent{rec}_r{rb}o{ob}", eb, Khat, Vhat, eb))
    # routed-exact + background-quant
    for rb, bg in [(8, 2), (8, 4), (16, 2)]:
        Khat, Vhat, eb, _ = QZ.routed_exact_background_quant(K, V, routed_idx, routed_bits=rb, bg_bits=bg)
        schemes.append((f"routed_r{rb}bg{bg}", eb, Khat, Vhat, eb))

    for nm, b, Khat, Vhat, eb in schemes:
        lg = AO.compute_logits(q, Khat, scaling)
        pr = AO.softmax(lg)
        Oq = pr @ Vhat
        rl2 = MET.rel_l2(Oq, O_full)
        rows_quant.append(dict(prompt=name, regime=regime, layer=li, head=h, T=T,
            method=nm, eff_bits=eb, storage_ratio=eb / 16.0,
            rel_l2=rl2, cosine=MET.cosine(Oq, O_full),
            js=MET.js_div(probs, pr), topk_overlap=MET.topk_overlap(
                np.argsort(lg)[::-1][:8], np.argsort(logits)[::-1][:8])))

    # ---- G predictors (one row per case) ----
    o16 = oracle_idx_cache.get(16)
    if o16 is None and T >= 16:
        o16 = SR.oracle_topk(q, K, scaling, 16, logits=logits)
    O_s16, _ = AO.sparse_attention(q, K, V, scaling, o16, logits=logits)
    jl32 = SR.jl_projection_select(q, K, min(16, T), 32, seed=SEED)
    O_jl, _ = AO.sparse_attention(q, K, V, scaling, jl32, logits=logits)
    Kq4, Vq4, _ = QZ.uniform_kv(K, V, 4)
    lg4 = AO.compute_logits(q, Kq4, scaling); O_q4 = AO.softmax(lg4) @ Vq4
    rows_pred.append(dict(prompt=name, regime=regime, layer=li, head=h, T=T,
        entropy=entropy, gini=gini, value_spectral=spec,
        margin_top16=MET.topk_margin(logits, min(16, T - 1)),
        mass_top16=MET.mass_retained(probs, o16),
        oracle_rl2_s16=MET.rel_l2(O_s16, O_full),
        jl_rl2_s16=MET.rel_l2(O_jl, O_full),
        quant4_rl2=MET.rel_l2(O_q4, O_full)))


def _route_row(rows, name, regime, li, h, T, method, param, B, idx,
               q, K, V, scaling, logits, probs, O_full, oracle_idx):
    O_r, _ = AO.sparse_attention(q, K, V, scaling, idx, logits=logits)
    rl2 = MET.rel_l2(O_r, O_full)
    recall = MET.topk_overlap(idx, oracle_idx)
    rows.append(dict(prompt=name, regime=regime, layer=li, head=h, T=T,
        method=method, param=param, budget=B, read=len(idx),
        read_ratio=len(idx) / T, rel_l2=rl2, cosine=MET.cosine(O_r, O_full),
        mass_retained=MET.mass_retained(probs, idx), oracle_recall=recall,
        pass_mod=rl2 <= 0.10))


# ---------------------------------------------------------------------------
def _aggregate(df_oracle, df_route, df_resid, df_quant):
    """Write a compact aggregate_results.csv summarizing headline numbers."""
    rows = []
    # oracle: median read gain to hit moderate pass per case
    for S in sorted(df_oracle["S"].unique()):
        sub = df_oracle[df_oracle["S"] == S]
        rows.append(dict(experiment="oracle_sparse", key=f"S={S}",
            mean_read_ratio=sub["read_ratio"].mean(),
            mean_rel_l2=sub["rel_l2"].mean(), median_rel_l2=sub["rel_l2"].median(),
            frac_pass_mod=sub["pass_mod"].mean(), mean_mass=sub["mass_retained"].mean()))
    for meth in df_route["method"].unique():
        sub = df_route[df_route["method"] == meth]
        rows.append(dict(experiment="routing", key=meth,
            mean_read_ratio=sub["read_ratio"].mean(),
            mean_rel_l2=sub["rel_l2"].mean(), median_rel_l2=sub["rel_l2"].median(),
            frac_pass_mod=sub["pass_mod"].mean(), mean_oracle_recall=sub["oracle_recall"].mean()))
    for meth in df_resid["method"].unique():
        sub = df_resid[df_resid["method"] == meth]
        rows.append(dict(experiment="residual", key=meth,
            mean_rel_l2=sub["rel_l2"].mean(), median_rel_l2=sub["rel_l2"].median()))
    for meth in df_quant["method"].unique():
        sub = df_quant[df_quant["method"] == meth]
        rows.append(dict(experiment="quant", key=meth,
            mean_eff_bits=sub["eff_bits"].mean(), mean_storage_ratio=sub["storage_ratio"].mean(),
            mean_rel_l2=sub["rel_l2"].mean(), median_rel_l2=sub["rel_l2"].median()))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "aggregate_results.csv"), index=False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--limit-layers", type=str, default=None,
                    help="comma list of layer idxs to restrict (speed)")
    args = ap.parse_args()
    ll = [int(x) for x in args.limit_layers.split(",")] if args.limit_layers else None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    run_all(args.model, args.max_tokens, synthetic=args.synthetic, limit_layers=ll)
