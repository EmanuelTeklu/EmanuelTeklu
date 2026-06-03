"""Token / block selectors for sparse attention.

Two families:
  * oracle_topk            -- uses true logits (upper bound).
  * cheap, non-oracle routers that DO NOT use the exact full logits:
      - jl_projection_select
      - block_centroid_select
      - block_upperbound_select   (certificate-like)
      - recency_router_select     (recent exact + route older)
      - sink_recent_router_select (sink + recent + route middle)

Selectors return integer arrays of selected token indices. Block routers also
expose the per-block summaries they used so residual_correction can reuse them.

`budget` is expressed as a number of tokens to read; block routers round to a
whole number of blocks.
"""
from __future__ import annotations
import numpy as np

from attention_ops import compute_logits


# ---------------------------------------------------------------------------
def oracle_topk(q, K, scaling, S, logits=None):
    if logits is None:
        logits = compute_logits(q, K, scaling)
    S = min(S, len(logits))
    return np.argsort(logits)[::-1][:S].astype(np.int64)


# ---------------------------------------------------------------------------
def jl_projection_select(q, K, S, r, seed=0):
    """Random-projection (Johnson-Lindenstrauss) selector.

    Project q,K to r dims with a fixed Gaussian matrix, rank tokens by the
    projected dot product. Cheap: cost ~ r*d per token instead of full d when
    r<d, and the projection of K can be precomputed/cached.
    """
    d = K.shape[1]
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((d, r)) / np.sqrt(r)
    qp = q @ P            # [r]
    Kp = K @ P            # [T,r]
    score = Kp @ qp       # [T]
    S = min(S, len(score))
    return np.argsort(score)[::-1][:S].astype(np.int64)


# ---------------------------------------------------------------------------
def _make_blocks(T, block):
    edges = list(range(0, T, block)) + [T]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def block_centroid_summaries(K, V, block):
    """Per-block centroid of K, mean of V, and K radius (max ||k-c||)."""
    T = K.shape[0]
    blocks = _make_blocks(T, block)
    cK, mV, rad, spans = [], [], [], []
    for (a, b) in blocks:
        Kb = K[a:b]
        c = Kb.mean(axis=0)
        cK.append(c)
        mV.append(V[a:b].mean(axis=0))
        rad.append(float(np.linalg.norm(Kb - c, axis=1).max()) if b - a > 0 else 0.0)
        spans.append((a, b))
    return np.array(cK), np.array(mV), np.array(rad), spans


def block_centroid_select(q, K, V, budget, block=32, summaries=None):
    """Select whole blocks by q·centroid until the token budget is met."""
    if summaries is None:
        summaries = block_centroid_summaries(K, V, block)
    cK, mV, rad, spans = summaries
    score = cK @ q
    order = np.argsort(score)[::-1]
    chosen, idx = [], []
    for bi in order:
        a, b = spans[bi]
        idx.extend(range(a, b))
        chosen.append(bi)
        if len(idx) >= budget:
            break
    return np.array(sorted(idx), dtype=np.int64), np.array(chosen, dtype=np.int64)


def block_upperbound_select(q, K, V, budget, block=32, summaries=None):
    """Certificate-like router: rank blocks by an UPPER BOUND on q·k.

        upper_b = q·centroid_b + ||q|| * radius_b   >=  max_{j in b} q·k_j
    Blocks with the highest possible contribution are read first; a block can be
    *safely rejected* when its upper bound is below the current top score.
    """
    if summaries is None:
        summaries = block_centroid_summaries(K, V, block)
    cK, mV, rad, spans = summaries
    qn = np.linalg.norm(q)
    upper = cK @ q + qn * rad
    order = np.argsort(upper)[::-1]
    chosen, idx = [], []
    for bi in order:
        a, b = spans[bi]
        idx.extend(range(a, b))
        chosen.append(bi)
        if len(idx) >= budget:
            break
    return np.array(sorted(idx), dtype=np.int64), np.array(chosen, dtype=np.int64)


# ---------------------------------------------------------------------------
def recency_router_select(q, K, V, budget, recent=64, block=32):
    """Always keep `recent` newest tokens; route older blocks for the rest."""
    T = K.shape[0]
    recent = min(recent, T)
    recent_idx = list(range(T - recent, T))
    remaining = max(0, budget - recent)
    older = K[: T - recent]
    if remaining > 0 and older.shape[0] > 0:
        idx_old, _ = block_centroid_select(q, older, V[: T - recent], remaining, block=block)
    else:
        idx_old = np.array([], dtype=np.int64)
    idx = np.array(sorted(set(recent_idx) | set(idx_old.tolist())), dtype=np.int64)
    return idx


def sink_recent_router_select(q, K, V, budget, sink=4, recent=64, block=32):
    """Keep first `sink` tokens + `recent` newest + route the middle."""
    T = K.shape[0]
    sink = min(sink, T)
    recent = min(recent, T - sink) if T > sink else 0
    sink_idx = list(range(sink))
    recent_idx = list(range(T - recent, T)) if recent > 0 else []
    fixed = set(sink_idx) | set(recent_idx)
    remaining = max(0, budget - len(fixed))
    mid_lo, mid_hi = sink, T - recent
    if remaining > 0 and mid_hi > mid_lo:
        Kmid, Vmid = K[mid_lo:mid_hi], V[mid_lo:mid_hi]
        idx_mid, _ = block_centroid_select(q, Kmid, Vmid, remaining, block=block)
        idx_mid = idx_mid + mid_lo
    else:
        idx_mid = np.array([], dtype=np.int64)
    idx = np.array(sorted(fixed | set(idx_mid.tolist())), dtype=np.int64)
    return idx
