"""MAIN STACK Experiment 3 — per-model calibration data efficiency.

Model-size transfer is weak (a 0.5B-trained threshold drops to ~0.68 precision
on 1.5B). Question: how much *target-model* calibration data is needed to
recover most of the achievable sparse-safe performance?

Method: pool the existing 1.5B scaleup datasets, hold out a fixed test set
(by regime, by layer, by head, and random), then train the cheap classifier
from scratch on increasing calibration sizes {50,100,250,500,1000,2000} and
measure coverage @ 95% precision and selective KV read reduction on the held-out
test set. Compare to the full-data ceiling.

Writes results/CALIBRATION_REPORT.md + results/tables/calibration.csv.
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
SIZES = [50, 100, 250, 500, 1000, 2000]
rng = np.random.default_rng(SEED)


def load_model_pool(tag):
    files = sorted(glob.glob(os.path.join(SCALEDIR, f"dataset_{tag}_*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["y"] = (df["router_rl2_10"] <= 0.10).astype(int)
    return df


def _fit_eval(train, test, feats, n):
    from sklearn.ensemble import GradientBoostingClassifier
    # stratified subsample of train to size n
    if n < len(train):
        idx0 = train.index[train.y == 0].to_numpy()
        idx1 = train.index[train.y == 1].to_numpy()
        n1 = int(round(n * train.y.mean())); n0 = n - n1
        n0 = min(n0, len(idx0)); n1 = min(n1, len(idx1))
        take = np.concatenate([rng.choice(idx0, n0, replace=False),
                               rng.choice(idx1, n1, replace=False)])
        tr = train.loc[take]
    else:
        tr = train
    if tr.y.nunique() < 2:
        return None
    Xtr = tr[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
    Xte = test[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
    m = GradientBoostingClassifier(random_state=SEED).fit(Xtr, tr.y.values)
    pte = m.predict_proba(Xte)[:, 1]
    yte = test.y.values
    cov95, thr, _ = CL.coverage_at_precision(pte, yte)
    pred = pte >= (thr if not np.isnan(thr) else np.inf)
    sm = CL.selective_metrics(test.reset_index(drop=True), np.arange(len(test)), pred)
    from sklearn.metrics import roc_auc_score
    return dict(n_train=len(tr), cov_at_95prec=cov95,
                auc=roc_auc_score(yte, pte) if yte.std() else np.nan,
                kv_read_reduction=sm["kv_read_reduction"],
                fail_rate_routed=sm["fail_rate_routed"])


def run(tag="1.5B"):
    df = load_model_pool(tag)
    feats = BD.CHEAP_FEATURES
    rows = []
    # held-out test definitions
    splits = {}
    # by regime: hold out the largest regime
    reg = df.regime.value_counts().index[0]
    splits["heldout_regime"] = (df.regime != reg, df.regime == reg)
    splits["heldout_layer"] = (df["layer"] % 4 != 0, df["layer"] % 4 == 0)
    splits["heldout_head"] = (df["head"] % 4 != 0, df["head"] % 4 == 0)
    # random 70/30
    mask = rng.random(len(df)) < 0.7
    splits["random"] = (mask, ~mask)

    for sname, (trm, tem) in splits.items():
        train = df[trm.values if hasattr(trm, "values") else trm].copy()
        test = df[tem.values if hasattr(tem, "values") else tem].copy()
        train.index = range(len(train))
        full = _fit_eval(train, test, feats, len(train))
        for n in SIZES + [len(train)]:
            r = _fit_eval(train, test, feats, n)
            if r is None:
                continue
            r.update(dict(split=sname, target_n=n,
                          frac_of_full_cov=(r["cov_at_95prec"] / full["cov_at_95prec"])
                          if full and full["cov_at_95prec"] > 0 else np.nan,
                          full_cov=full["cov_at_95prec"] if full else np.nan))
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "tables", "calibration.csv"), index=False)
    write_report(out, tag, df)
    return out


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(out, tag, df):
    # at n=1000, fraction of full coverage recovered (mean over splits)
    at1000 = out[out.target_n == 1000]
    frac1000 = float(at1000.frac_of_full_cov.mean()) if len(at1000) else np.nan
    at250 = out[out.target_n == 250]
    frac250 = float(at250.frac_of_full_cov.mean()) if len(at250) else np.nan
    keep = frac1000 >= 0.90
    verdict = "KEEP (cheap calibration)" if keep else "PARK (calibration-hungry)"

    L = []; A = L.append
    A(f"# Per-Model Calibration Report ({tag})")
    A("")
    A("**Question:** model-size transfer is weak — how many *target-model* "
      "calibration cases are needed to recover most sparse-safe performance? "
      "Cheap features, gradient-boosted classifier trained from scratch, evaluated "
      "at the 95%-precision operating point on held-out regime / layer / head / random.")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    A(f"- At **1000** calibration cases, the classifier recovers "
      f"**{frac1000*100:.0f}%** of the full-data coverage (mean over splits); at "
      f"**250** cases it already recovers **{frac250*100:.0f}%**.")
    A(f"- Pool: {len(df)} 1.5B cases across contexts; base rate {df.y.mean():.3f}.")
    A(f"- {'≤1000 cases recover most performance → cheap to calibrate a new model.' if keep else 'Recovery needs large calibration → expensive per model.'}")
    A("")
    A("## Coverage @ 95% precision vs calibration size")
    A("")
    piv = out.pivot_table(index="target_n", columns="split", values="cov_at_95prec")
    A(_md(piv.reset_index()))
    A("")
    A("## Fraction of full-data coverage recovered")
    A("")
    pivf = out.pivot_table(index="target_n", columns="split", values="frac_of_full_cov")
    A(_md(pivf.reset_index()))
    A("")
    A("## KV read reduction vs calibration size (held-out regime)")
    A("")
    hr = out[out.split == "heldout_regime"][["target_n", "cov_at_95prec",
              "kv_read_reduction", "fail_rate_routed", "auc"]]
    A(_md(hr))
    A("")
    A("## Reading")
    A("")
    A("- If coverage saturates by ~250–1000 cases, a new model can be calibrated "
      "with a tiny labelled set (each label = one forward + oracle rel-L2 at the "
      "budget ladder), making the weak zero-shot model-transfer a non-issue.")
    A("- Held-out-layer/head splits test whether a few layers/heads of calibration "
      "generalize to the rest of the same model.")
    A("")
    p = os.path.join(RESULTS, "CALIBRATION_REPORT.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    print("wrote", p, "| verdict:", verdict)


if __name__ == "__main__":
    run("1.5B")
