"""Output- and distribution-level metrics for attention approximations.

All functions operate on numpy float64 internally for stability. Inputs are
typically single-query attention outputs (vectors of dim d) or probability
vectors over T context tokens.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def rel_l2(approx: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error ||approx - ref|| / ||ref||."""
    a = approx.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    denom = np.linalg.norm(r) + EPS
    return float(np.linalg.norm(a - r) / denom)


def cosine(approx: np.ndarray, ref: np.ndarray) -> float:
    a = approx.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    na = np.linalg.norm(a) + EPS
    nr = np.linalg.norm(r) + EPS
    return float(np.dot(a, r) / (na * nr))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over probability vectors (nats)."""
    p = p.astype(np.float64) + EPS
    q = q.astype(np.float64) + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats), symmetric, bounded by ln2."""
    p = p.astype(np.float64) + EPS
    q = q.astype(np.float64) + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def topk_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Fraction of idx_b recovered by idx_a (|A ∩ B| / |B|)."""
    if len(idx_b) == 0:
        return 1.0
    sa = set(int(i) for i in idx_a)
    sb = set(int(i) for i in idx_b)
    return len(sa & sb) / len(sb)


def attention_entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of an attention probability vector."""
    p = p.astype(np.float64) + EPS
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def mass_retained(probs: np.ndarray, idx: np.ndarray) -> float:
    """Fraction of total attention probability mass on the selected indices."""
    if len(idx) == 0:
        return 0.0
    return float(probs[idx].sum() / (probs.sum() + EPS))


def topk_margin(logits: np.ndarray, k: int) -> float:
    """Logit gap between the k-th and (k+1)-th largest logits.

    A large margin means the top-k support is well separated (robust to
    perturbation / quantization). k is clipped to len-1.
    """
    s = np.sort(logits.astype(np.float64))[::-1]
    if len(s) <= k:
        return float(s[0] - s[-1]) if len(s) > 1 else 0.0
    return float(s[k - 1] - s[k])


def value_spectral_decay(V: np.ndarray) -> float:
    """Participation-ratio-style effective rank fraction of value matrix V [T,d].

    Returns (effective_rank / d) in (0,1]; small => values are low-rank /
    redundant => more amenable to residual/low-rank correction.
    """
    V = V.astype(np.float64)
    if V.shape[0] < 2:
        return 1.0
    # center is not applied: we care about raw value subspace energy
    try:
        s = np.linalg.svd(V, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1.0
    s2 = s ** 2
    if s2.sum() <= EPS:
        return 1.0
    eff_rank = (s2.sum() ** 2) / (np.sum(s2 ** 2) + EPS)  # participation ratio
    return float(eff_rank / V.shape[1])


def gini_concentration(probs: np.ndarray) -> float:
    """Gini coefficient of attention mass (0=uniform, ->1 concentrated)."""
    p = np.sort(probs.astype(np.float64))
    n = len(p)
    if n == 0 or p.sum() <= EPS:
        return 0.0
    cum = np.cumsum(p)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
