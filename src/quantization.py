"""KV quantization: uniform baselines and adaptive / margin-aware schemes.

We model storage as *effective bits per stored scalar* and compute the
attention output under dequantized K/V to measure quality. Quantization is
per-token symmetric absmax (the standard practical KV-cache scheme): each
token's k (and v) vector is quantized with its own scale.

Returned dicts include 'eff_bits' = average bits/scalar across the K and V
tensors so adaptive schemes can be compared on a common bits-per-value axis.
"""
from __future__ import annotations
import numpy as np


def quantize_symmetric(X: np.ndarray, bits: int, axis: int = 1) -> np.ndarray:
    """Symmetric absmax quantize+dequantize. X [T,d] -> Xhat [T,d].

    axis=1 -> per-token (each row shares a scale); axis=0 -> per-channel (each
    column shares a scale). bits>=16 is treated as exact.
    """
    if bits >= 16:
        return X.astype(np.float64).copy()
    X = X.astype(np.float64)
    qmax = (1 << (bits - 1)) - 1  # bits=4 -> 7
    if qmax < 1:
        qmax = 1
    absmax = np.abs(X).max(axis=axis, keepdims=True)
    absmax = np.where(absmax < 1e-12, 1.0, absmax)
    scale = absmax / qmax
    q = np.clip(np.round(X / scale), -qmax, qmax)
    return q * scale


def quantize_per_token(X: np.ndarray, bits: int) -> np.ndarray:
    return quantize_symmetric(X, bits, axis=1)


def quantize_kv(K, V, k_bits, v_bits, k_axis=0, v_axis=1):
    """KIVI-style default: K per-channel (axis=0), V per-token (axis=1)."""
    return (quantize_symmetric(K, k_bits, axis=k_axis),
            quantize_symmetric(V, v_bits, axis=v_axis))


def quantize_rows_variable_bits(X: np.ndarray, bits_per_token: np.ndarray) -> np.ndarray:
    """Quantize each token row with its own bit budget (adaptive schemes)."""
    X = X.astype(np.float64)
    out = np.empty_like(X)
    for b in np.unique(bits_per_token):
        mask = bits_per_token == b
        out[mask] = quantize_per_token(X[mask], int(b))
    return out


def eff_bits(bits_per_token_K, bits_per_token_V, d):
    """Average bits/scalar over all K and V scalars given per-token bit budgets."""
    total = (bits_per_token_K.sum() + bits_per_token_V.sum()) * d
    n = (len(bits_per_token_K) + len(bits_per_token_V)) * d
    return float(total / n)


# ---------------------------------------------------------------------------
# Uniform baselines
# ---------------------------------------------------------------------------
def uniform_kv(K, V, bits, k_axis=0, v_axis=1):
    """Uniform K/V quant. Default K per-channel + V per-token (KIVI-style),
    the strong standard baseline. eff_bits is the nominal bit width."""
    Khat = quantize_symmetric(K, bits, axis=k_axis)
    Vhat = quantize_symmetric(V, bits, axis=v_axis)
    T = K.shape[0]
    bpt = np.full(T, min(bits, 16))
    return Khat, Vhat, eff_bits(bpt, bpt, K.shape[1])


# ---------------------------------------------------------------------------
# Adaptive schemes. Each returns (Khat, Vhat, eff_bits, name).
# `probs` is the true full-attention probability vector (importance signal);
# `logits` are the true logits (margin signal). These are oracle signals used
# to UPPER-BOUND adaptive benefit; cheap proxies are discussed in the report.
# ---------------------------------------------------------------------------
def adaptive_token_importance(K, V, probs, hi_bits=8, lo_bits=2, frac_hi=0.10):
    """High-mass tokens get hi_bits, the rest lo_bits."""
    T = K.shape[0]
    n_hi = max(1, int(round(frac_hi * T)))
    hi_idx = np.argsort(probs)[::-1][:n_hi]
    bpt = np.full(T, lo_bits)
    bpt[hi_idx] = hi_bits
    Khat = quantize_rows_variable_bits(K, bpt)
    Vhat = quantize_rows_variable_bits(V, bpt)
    return Khat, Vhat, eff_bits(bpt, bpt, K.shape[1]), "adaptive_token_importance"


def adaptive_margin_fragility(K, V, logits, k=8, hi_bits=8, lo_bits=2, frac_hi=0.10):
    """Tokens whose logit is near the top-k decision boundary get hi_bits.

    Fragility = -|logit_j - boundary|; tokens closest to the boundary (most
    likely to flip in/out of the top-k support under perturbation) are
    protected with more bits.
    """
    T = K.shape[0]
    s = np.sort(logits)[::-1]
    boundary = s[k] if T > k else s[-1]
    fragility = -np.abs(logits - boundary)
    n_hi = max(1, int(round(frac_hi * T)))
    hi_idx = np.argsort(fragility)[::-1][:n_hi]
    bpt = np.full(T, lo_bits)
    bpt[hi_idx] = hi_bits
    Khat = quantize_rows_variable_bits(K, bpt)
    Vhat = quantize_rows_variable_bits(V, bpt)
    return Khat, Vhat, eff_bits(bpt, bpt, K.shape[1]), "adaptive_margin_fragility"


def recent_exact_old_quant(K, V, recent=128, recent_bits=8, old_bits=2):
    """Recent window quantized at recent_bits, older context at old_bits."""
    T = K.shape[0]
    bpt = np.full(T, old_bits)
    bpt[max(0, T - recent):] = recent_bits
    Khat = quantize_rows_variable_bits(K, bpt)
    Vhat = quantize_rows_variable_bits(V, bpt)
    return Khat, Vhat, eff_bits(bpt, bpt, K.shape[1]), "recent_exact_old_quant"


def routed_exact_background_quant(K, V, routed_idx, routed_bits=8, bg_bits=2):
    """Router-selected tokens at routed_bits, omitted background at bg_bits."""
    T = K.shape[0]
    bpt = np.full(T, bg_bits)
    bpt[np.asarray(routed_idx, dtype=np.int64)] = routed_bits
    Khat = quantize_rows_variable_bits(K, bpt)
    Vhat = quantize_rows_variable_bits(V, bpt)
    return Khat, Vhat, eff_bits(bpt, bpt, K.shape[1]), "routed_exact_background_quant"
