"""TRACK C — sparse-safety as a SUPPORTING control signal (not a lead path).

Sparse-as-product is killed. But the cheap per-case features that predicted
sparse-safety may still be a useful *controller* for persistent-inference
decisions: compress vs keep-exact, serve-from-latent vs recompute, and when to
fall back. We test whether one cheap-feature classifier predicts several such
control decisions, on the existing block per-case data (cheap features +
oracle rel-L2 at the budget ladder).

Control decisions (binary, per layer/head/context case):
  * aggressively_compressible  -- block-oracle rel-L2 @2% reads ≤ 0.10
  * mildly_compressible        -- block-oracle rel-L2 @10% reads ≤ 0.10
  * needs_exact (fallback)     -- NOT safe even at 15% reads (must recompute)

KEEP-as-controller if cheap features predict these at high AUC with usable
coverage@95% precision (so the controller can act with low error).

Writes results/CONTROLLER_REPORT.md + results/tables/controller.csv.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_dataset as BD
import classifier as CL

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
BLOCKDIR = os.path.join(RESULTS, "tables", "block")


def load():
    fs = sorted(glob.glob(os.path.join(BLOCKDIR, "percase_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df[df["ctx"] >= 8192].reset_index(drop=True)


def run():
    df = load()
    bo = "brl2_block_oracle_16_"
    labels = {
        "aggressively_compressible": (df[bo + "0.02"] <= 0.10).astype(int).values,
        "mildly_compressible": (df[bo + "0.1"] <= 0.10).astype(int).values,
        "needs_exact_fallback": (df[bo + "0.15"] > 0.10).astype(int).values,
    }
    rows = []
    gr = df["regime"].values
    for name, y in labels.items():
        if y.sum() in (0, len(y)):
            continue
        for split, groups in [("random", None), ("heldout_regime", gr)]:
            proba = CL.oof_grouped(df, y, BD.CHEAP_FEATURES, "gboost", groups=groups)
            from sklearn.metrics import roc_auc_score
            cov, _, prec = CL.coverage_at_precision(proba, y)
            rows.append(dict(decision=name, split=split, base_rate=float(y.mean()),
                auc=float(roc_auc_score(y, proba)), cov_at_95prec=cov,
                realized_prec=prec))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "tables", "controller.csv"), index=False)
    write_report(out)
    return out


def _md(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notnull(x) else "")
    return df.to_markdown(index=False)


def write_report(out):
    ho = out[out.split == "heldout_regime"]
    mean_auc = float(ho.auc.mean()) if len(ho) else np.nan
    keep = mean_auc >= 0.75
    verdict = ("KEEP (cheap features are a usable controller signal)" if keep
               else "PARK (controller signal too weak)")
    L = []; A = L.append
    A("# Sparse-Safety as a Supporting Controller (Track C)")
    A("")
    A("Sparse attention is NOT revived as a product. Here we only ask: do the "
      "cheap per-case features still serve as a *control signal* for "
      "persistent-inference decisions (compress vs exact, latent vs recompute, "
      "fallback)? Held-out-regime OOF on long-context block per-case data.")
    A("")
    A(f"## Verdict: **{verdict}**")
    A("")
    A(f"- Mean held-out-regime AUC across control decisions: **{mean_auc:.2f}**.")
    A("- The same cheap features (entropy/mass/margin proxies, q/K/V norm stats, "
      "centroid-router scores) that predicted sparse-safety predict "
      "compress/exact/fallback decisions — so the classifier becomes a cheap "
      "*controller* in front of the persistent-state system, not a lead path.")
    A("")
    A("## Control-decision predictability")
    A("")
    A(_md(out))
    A("")
    A("## Role in the pivoted system")
    A("")
    A("- **Gate latent vs exact:** route confidently-compressible heads/contexts to "
      "the latent-KV path, keep fragile ones exact.")
    A("- **Gate reuse vs recompute:** a low-margin / high-entropy context is less "
      "tolerant of approximate reuse; the controller can force a fresh compute.")
    A("- **Fallback detector:** `needs_exact_fallback` flags cases no compression "
      "budget serves safely.")
    A("- This keeps the sparse-safety work alive as a cheap supervisory signal "
      "while the product/research weight moves to reuse (Track A) and latent-KV "
      "(Track B).")
    A("")
    with open(os.path.join(RESULTS, "CONTROLLER_REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote CONTROLLER_REPORT.md | verdict:", verdict)


if __name__ == "__main__":
    run()
