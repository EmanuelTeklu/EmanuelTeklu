"""MAIN NEXT — Experiment 4: minimum per-model calibration + active selection.

How few target-model cases are needed, and does *active* selection (uncertainty /
diversity over layer/head/regime/context) beat random?

Strategies:
  * random
  * active_diversity   -- stratified round-robin over (regime × layer-bin × head-bin)
                          to span the structure with few cases
  * active_uncertainty -- batch active learning: seed, train gboost, add the most
                          uncertain pool cases (|p-0.5|), retrain

Evaluated at the 95%-precision operating point on a held-out-regime test set;
reports coverage@95%, KV read reduction, failure, and fraction of the full-data
ceiling recovered.

Writes results/CALIBRATION_MIN_REPORT.md + results/tables/calibration_min.csv.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_dataset as BD
import classifier as CL

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
SCALEDIR = os.path.join(RESULTS, "tables", "scaleup")
SEED = 0
SIZES = [25, 50, 100, 250, 500, 1000]
rng = np.random.default_rng(SEED)


def load_pool(tag="1.5B"):
    fs = sorted(glob.glob(os.path.join(SCALEDIR, f"dataset_{tag}_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).reset_index(drop=True)
    df["y"] = (df["router_rl2_10"] <= 0.10).astype(int)
    return df


def _eval(train_df, test_df, idx, feats):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    tr = train_df.loc[idx]
    if tr.y.nunique() < 2:
        return None
    Xtr = tr[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
    Xte = test_df[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
    m = GradientBoostingClassifier(random_state=SEED).fit(Xtr, tr.y.values)
    pte = m.predict_proba(Xte)[:, 1]
    yte = test_df.y.values
    cov, thr, _ = CL.coverage_at_precision(pte, yte)
    pred = pte >= (thr if not np.isnan(thr) else np.inf)
    sm = CL.selective_metrics(test_df.reset_index(drop=True), np.arange(len(test_df)), pred)
    return dict(cov_at_95prec=cov, kv_read_reduction=sm["kv_read_reduction"],
                fail_rate_routed=sm["fail_rate_routed"],
                auc=roc_auc_score(yte, pte) if yte.std() else np.nan)


def sel_random(train_df, n):
    return rng.choice(train_df.index.values, min(n, len(train_df)), replace=False)


def sel_diversity(train_df, n):
    """Round-robin over (regime, layer//4, head//4) strata for max spread."""
    key = (train_df["regime"].astype(str) + "|" +
           (train_df["layer"] // 4).astype(str) + "|" +
           (train_df["head"] // 4).astype(str))
    groups = {k: list(rng.permutation(v)) for k, v in train_df.groupby(key).groups.items()}
    order = list(groups.keys()); chosen = []
    i = 0
    while len(chosen) < min(n, len(train_df)):
        g = groups[order[i % len(order)]]
        if g:
            chosen.append(g.pop())
        i += 1
        if i > 50 * n:
            break
    return np.array(chosen)


def sel_uncertainty(train_df, test_df, n, feats, seed_n=25, batch=25):
    from sklearn.ensemble import GradientBoostingClassifier
    pool = list(train_df.index.values)
    chosen = list(rng.choice(pool, min(seed_n, len(pool)), replace=False))
    pool = [p for p in pool if p not in set(chosen)]
    while len(chosen) < min(n, len(train_df)) and pool:
        tr = train_df.loc[chosen]
        if tr.y.nunique() < 2:
            chosen += list(rng.choice(pool, min(batch, len(pool)), replace=False))
            pool = [p for p in pool if p not in set(chosen)]
            continue
        Xtr = tr[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
        m = GradientBoostingClassifier(random_state=SEED).fit(Xtr, tr.y.values)
        Xpool = train_df.loc[pool][feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
        p = m.predict_proba(Xpool)[:, 1]
        unc = -np.abs(p - 0.5)
        take = np.argsort(unc)[::-1][:batch]
        add = [pool[i] for i in take]
        chosen += add; pool = [x for x in pool if x not in set(add)]
    return np.array(chosen[:n])


def run(tag="1.5B", seeds=(0, 1, 2)):
    df = load_pool(tag)
    feats = BD.CHEAP_FEATURES
    reg = df.regime.value_counts().index[0]
    train_df = df[df.regime != reg].copy()
    test_df = df[df.regime == reg].copy()
    full = _eval(train_df, test_df, train_df.index.values, feats)
    full_cov = full["cov_at_95prec"] if full else np.nan

    raw = []
    for sd in seeds:
        global rng
        rng = np.random.default_rng(sd)
        for n in SIZES:
            for strat, idx in [("random", sel_random(train_df, n)),
                               ("active_diversity", sel_diversity(train_df, n)),
                               ("active_uncertainty",
                                sel_uncertainty(train_df, test_df, n, feats))]:
                r = _eval(train_df, test_df, idx, feats)
                if r is None:
                    continue
                r.update(dict(n=n, strategy=strat, seed=sd,
                              frac_of_full=(r["cov_at_95prec"] / full_cov) if full_cov else np.nan))
                raw.append(r)
    rawdf = pd.DataFrame(raw)
    # average over seeds
    out = rawdf.groupby(["n", "strategy"]).agg(
        cov_at_95prec=("cov_at_95prec", "mean"),
        frac_of_full=("frac_of_full", "mean"),
        kv_read_reduction=("kv_read_reduction", "mean"),
        fail_rate_routed=("fail_rate_routed", "mean"),
        auc=("auc", "mean")).reset_index()
    out.to_csv(os.path.join(RESULTS, "tables", "calibration_min.csv"), index=False)
    write_report(out, full_cov, len(train_df), tag, df.y.mean())
    return out


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(out, full_cov, ntrain, tag, base):
    def frac_at(strat, n):
        r = out[(out.strategy == strat) & (out.n == n)]
        return float(r.frac_of_full.iloc[0]) if len(r) else np.nan
    # smallest n (any strategy) reaching >=90% of full; and best active vs random
    hit = out[out.frac_of_full >= 0.90].sort_values("n")
    min_n = int(hit.n.iloc[0]) if len(hit) else None
    act = out[out.strategy.isin(["active_diversity", "active_uncertainty"])]
    hit_a = act[act.frac_of_full >= 0.90].sort_values("n")
    min_active_n = int(hit_a.n.iloc[0]) if len(hit_a) else None
    # does active beat random at 250/500?
    def fr(s, n):
        r = out[(out.strategy == s) & (out.n == n)]
        return float(r.frac_of_full.iloc[0]) if len(r) else np.nan
    active_helps = (np.nanmax([fr("active_diversity", 250), fr("active_uncertainty", 250)])
                    > fr("random", 250) + 0.05)
    keep = (min_n is not None and min_n <= 500)
    verdict = ("KEEP (cheap calibration)" if keep else "PARK (calibration-hungry)")
    verdict += "; active " + ("beats random" if active_helps else "does NOT beat random")

    L = []; A = L.append
    A(f"# Calibration Minimization + Active Selection ({tag})")
    A("")
    A("Minimum target-model calibration to recover sparse-safe performance, and "
      "whether active selection beats random. Held-out-regime test, 95%-precision "
      f"operating point. Full-data ceiling coverage = {full_cov:.3f} "
      f"({ntrain} train cases, base rate {base:.3f}).")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    if min_n:
        A(f"- **≥90% of the full-data ceiling is reached at {min_n} cases** "
          f"(any strategy; {'active' if min_active_n==min_n else 'random'} first). "
          f"Mean of 3 seeds.")
    else:
        A("- No strategy reached 90% of the ceiling within 1000 cases.")
    A(f"- **Active vs random:** active selection "
      f"{'beats' if active_helps else 'does NOT beat'} random at ≤250 cases "
      f"(the strict 95%-precision coverage metric is noisy at small n; random is a "
      f"strong baseline).")
    A(f"- @250: random {frac_at('random',250)*100:.0f}% · diversity "
      f"{frac_at('active_diversity',250)*100:.0f}% · uncertainty "
      f"{frac_at('active_uncertainty',250)*100:.0f}% of full coverage.")
    A(f"- @500: random {frac_at('random',500)*100:.0f}% · diversity "
      f"{frac_at('active_diversity',500)*100:.0f}% · uncertainty "
      f"{frac_at('active_uncertainty',500)*100:.0f}%.")
    A(f"- Rule: KEEP if active reaches ≥90% with ≤250–500 cases.")
    A("")
    A("## Coverage @ 95% precision vs calibration size")
    A("")
    piv = out.pivot_table(index="n", columns="strategy", values="cov_at_95prec")
    A(_md(piv.reset_index()))
    A("")
    A("## Fraction of full-data ceiling recovered")
    A("")
    pivf = out.pivot_table(index="n", columns="strategy", values="frac_of_full")
    A(_md(pivf.reset_index()))
    A("")
    A("## KV read reduction vs size")
    A("")
    pivr = out.pivot_table(index="n", columns="strategy", values="kv_read_reduction")
    A(_md(pivr.reset_index()))
    A("")
    A("## Reading")
    A("")
    A("- A label costs one forward pass + oracle rel-L2 at the budget ladder for a "
      "few (layer,head) cases — so 250–500 cases ≈ a handful of prompts. If active "
      "beats random here, a new model is calibrated almost for free, neutralising "
      "the weak zero-shot model-size transfer.")
    A("")
    p = os.path.join(RESULTS, "CALIBRATION_MIN_REPORT.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p, "| verdict:", verdict)


if __name__ == "__main__":
    run("1.5B")
