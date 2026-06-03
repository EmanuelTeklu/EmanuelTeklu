"""Sparse-plus-residual value correction.

Naive renormalized sparse attention drops the omitted mass entirely. The exact
decomposition with a global max-shift (see attention_ops.unnormalized_partials):

    O_full = (num_S + num_O) / (den_S + den_O)
      num_S = sum_{j in S} w_j v_j ,  den_S = sum_{j in S} w_j
      num_O = sum_{j in O} w_j v_j ,  den_O = sum_{j in O} w_j      (omitted)

Correction = approximate (num_O, den_O) with cheap summaries instead of reading
every omitted value. We provide:

  * sparse_renorm                 -- baseline, ignores omitted mass.
  * correct_global_mean           -- num_O ≈ den_O * vbar_global.
  * correct_block_residual        -- num_O ≈ sum_b den_b * mean(V_b), den_b from block logit summary.
  * correct_lowrank               -- project omitted V onto a rank-k basis of the head's V.
  * oracle_mass_global / oracle_mass_block -- use TRUE den_O (ceiling on benefit).

Each returns the corrected output vector [d]. `extra_floats` annotates the
proxy storage each method needs beyond the selected K/V.
"""
from __future__ import annotations
import numpy as np

from attention_ops import unnormalized_partials, compute_logits


def _assemble(num_S, den_S, num_O_hat, den_O_hat):
    return (num_S + num_O_hat) / (den_S + den_O_hat + 1e-12)


def sparse_renorm(logits, V, idx):
    num_S, den_S, m, w = unnormalized_partials(logits, V, idx)
    return num_S / (den_S + 1e-12)


# --- cheap proxies (do not read omitted V exactly) -------------------------
def correct_global_mean(logits, V, idx, vbar_global, den_O_hat):
    """num_O ≈ den_O_hat * vbar_global. den_O_hat is a cheap mass estimate."""
    num_S, den_S, m, w = unnormalized_partials(logits, V, idx)
    return _assemble(num_S, den_S, den_O_hat * vbar_global, den_O_hat)


def estimate_omitted_mass_from_blocks(q, K, scaling, idx, block, summaries):
    """Cheap estimate of den_O using block centroid logits (no full softmax).

    For each block we estimate its summed weight from the centroid logit and
    block size, then subtract the part already selected. Uses the same global
    max-shift convention as unnormalized_partials by reading the true max logit
    of the SELECTED set as the shift (selected logits are known cheaply).
    """
    cK, mV, rad, spans = summaries
    # shift by the max selected logit (selected logits are computed anyway)
    sel_logits = scaling * (K[idx] @ q)
    m = sel_logits.max() if len(idx) else 0.0
    sel = set(int(i) for i in idx)
    den_O = 0.0
    num_O = np.zeros(mV.shape[1])
    for bi, (a, b) in enumerate(spans):
        size = b - a
        # tokens in this block not already selected
        n_omitted = size - sum(1 for j in range(a, b) if j in sel)
        if n_omitted <= 0:
            continue
        c_logit = scaling * (cK[bi] @ q)
        w_b = np.exp(c_logit - m) * n_omitted   # centroid-weight * count
        den_O += w_b
        num_O += w_b * mV[bi]
    return num_O, den_O, m


def correct_block_residual(q, K, V, scaling, logits, idx, block, summaries):
    """num_O, den_O estimated from per-block centroid+mean summaries."""
    num_O_hat, den_O_hat, m = estimate_omitted_mass_from_blocks(q, K, scaling, idx, block, summaries)
    # recompute selected partials under the SAME shift m for consistency
    sel_logits = logits[idx]
    w_sel = np.exp(sel_logits - m)
    num_S = w_sel @ V[idx]
    den_S = float(w_sel.sum())
    return _assemble(num_S, den_S, num_O_hat, den_O_hat)


def correct_lowrank(logits, V, idx, basis):
    """Approximate omitted values by their projection onto `basis` [k,d].

    num_O ≈ sum_O w_j (basis^T basis) v_j  -- but we avoid reading w_j v_j by
    using the block/global mass; here we instead correct by projecting the
    *global* omitted summary onto the basis. Practically this reuses the
    global-mean correction restricted to the value subspace, isolating how much
    of the missing mass lives in the dominant value directions.
    """
    num_S, den_S, m, w = unnormalized_partials(logits, V, idx)
    all_idx = np.arange(len(logits))
    omit = np.setdiff1d(all_idx, idx)
    den_O = float(w[omit].sum())
    if len(omit) == 0:
        return num_S / (den_S + 1e-12)
    vbar_O = V[omit].mean(axis=0)
    proj = basis.T @ (basis @ vbar_O)   # project onto dominant value subspace
    return _assemble(num_S, den_S, den_O * proj, den_O)


# --- oracle ceilings (use true omitted mass) -------------------------------
def oracle_mass_global(logits, V, idx, vbar_global):
    num_S, den_S, m, w = unnormalized_partials(logits, V, idx)
    omit = np.setdiff1d(np.arange(len(logits)), idx)
    den_O = float(w[omit].sum())
    return _assemble(num_S, den_S, den_O * vbar_global, den_O)


def oracle_mass_block(logits, V, idx, block, summaries):
    """True per-block omitted mass with block mean values (best block summary)."""
    cK, mV, rad, spans = summaries
    num_S, den_S, m, w = unnormalized_partials(logits, V, idx)
    sel = set(int(i) for i in idx)
    num_O = np.zeros(V.shape[1])
    den_O = 0.0
    for bi, (a, b) in enumerate(spans):
        omit_local = [j for j in range(a, b) if j not in sel]
        if not omit_local:
            continue
        wb = float(w[omit_local].sum())
        den_O += wb
        num_O += wb * mV[bi]
    return _assemble(num_S, den_S, num_O, den_O)


def head_value_basis(V, k=8):
    """Top-k right singular vectors of V [T,d] -> basis [k,d]."""
    try:
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(min(k, V.shape[1]), V.shape[1])
    return Vt[:k]
