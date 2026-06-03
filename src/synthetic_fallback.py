"""LABELED synthetic Q/K/V fallback. NOT DECISIVE.

Only used when real-model download/inference is impossible. It plants the same
regimes the real suite targets (needle, diffuse, fragile-margin) so the harness
runs end-to-end, but results from this mode must never be reported as evidence
about real transformers.
"""
from __future__ import annotations
import numpy as np

SEED = 0


def _make_capture(name, regime, T, n_layers=6, n_heads=8, n_kv=2, d=64, mode="needle"):
    rng = np.random.default_rng(abs(hash((name, regime))) % (2**32))
    groups = n_heads // n_kv
    scaling = 1.0 / np.sqrt(d)
    layers = []
    for li in range(n_layers):
        K = rng.standard_normal((n_kv, T, d)).astype(np.float32)
        V = rng.standard_normal((n_kv, T, d)).astype(np.float32)
        Q = rng.standard_normal((1, n_heads, d)).astype(np.float32)
        if mode == "needle":
            for hkv in range(n_kv):
                j = rng.integers(0, T)
                K[hkv, j] *= 6.0
        elif mode == "diffuse":
            K *= 0.3
        # last-position query already [1, H, d]; compute baseline out
        out = np.zeros((1, n_heads, d), np.float32)
        for h in range(n_heads):
            kv = h // groups
            lg = scaling * (K[kv] @ Q[0, h])
            p = np.exp(lg - lg.max()); p /= p.sum()
            out[0, h] = p @ V[kv]
        layers.append(dict(layer_idx=li, Q=Q, K=K, V=V, out=out))
    return dict(T=T, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv,
                groups=groups, d=d, scaling=scaling, query_positions=[T - 1],
                layers=layers, name=name, regime=regime)


def build_synthetic_captures(max_tokens=512):
    T = min(512, max_tokens)
    return [
        _make_capture("syn_needle", "needle", T, mode="needle"),
        _make_capture("syn_diffuse", "diffuse", T, mode="diffuse"),
        _make_capture("syn_normal", "short", T, mode="normal"),
    ]
