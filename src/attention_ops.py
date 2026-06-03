"""Exact and sparse single-query attention over captured K/V tensors.

Conventions
-----------
A single analysis "case" is one (prompt, layer, query-head, query-position).
We work with a decode-style query: one query vector q (dim d) attending over
T context key/value vectors under a causal mask (q attends to tokens 0..t).

    logits_j = scaling * (q . k_j)        for j in 0..T-1   (already masked)
    probs    = softmax(logits)
    O_full   = sum_j probs_j * v_j

All arrays are numpy float32/float64.
"""
from __future__ import annotations
import numpy as np


def compute_logits(q: np.ndarray, K: np.ndarray, scaling: float) -> np.ndarray:
    """q [d], K [T,d] -> logits [T] (scaled dot products)."""
    return scaling * (K.astype(np.float64) @ q.astype(np.float64))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def full_attention(q, K, V, scaling):
    """Return (output [d], probs [T], logits [T])."""
    logits = compute_logits(q, K, scaling)
    probs = softmax(logits)
    out = probs @ V.astype(np.float64)
    return out, probs, logits


def sparse_attention(q, K, V, scaling, idx, logits=None):
    """Renormalized sparse attention restricted to indices `idx`.

    O_sparse = softmax(logits[idx]) @ V[idx]
    Returns (output [d], probs_local [len(idx)]).
    """
    if logits is None:
        logits = compute_logits(q, K, scaling)
    idx = np.asarray(idx, dtype=np.int64)
    sub = logits[idx]
    p = softmax(sub)
    out = p @ V[idx].astype(np.float64)
    return out, p


def unnormalized_partials(logits, V, idx):
    """Exact unnormalized numerator/denominator for a subset `idx`.

    Uses a global max-shift so selected and omitted partials are comparable.

        w_j   = exp(logits_j - max_logit)
        num_S = sum_{j in idx} w_j v_j     [d]
        den_S = sum_{j in idx} w_j         scalar

    Returns (num_S, den_S, max_logit, w_all) where w_all are the full shifted
    weights (handy for building omitted summaries).
    """
    logits = logits.astype(np.float64)
    m = logits.max()
    w = np.exp(logits - m)
    idx = np.asarray(idx, dtype=np.int64)
    num_S = w[idx] @ V[idx].astype(np.float64)
    den_S = float(w[idx].sum())
    return num_S, den_S, m, w


def causal_logits_for_position(q, K, scaling, t):
    """Logits for a query at position t attending to keys 0..t (inclusive)."""
    logits = compute_logits(q, K[: t + 1], scaling)
    return logits
