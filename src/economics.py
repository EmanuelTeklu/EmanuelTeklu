"""Analytic cost models: prefix/latent reuse (H) and Amdahl end-to-end (I).

These are economics, NOT measured wall-clock. They translate a claimed
*component* reduction into an *end-to-end* speedup envelope.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


def amdahl_table():
    """End-to-end speedup when a fraction f of runtime is sped up by factor k.

        speedup = 1 / ((1 - f) + f / k)
    """
    fracs = [0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99]
    reductions = [2, 4, 8, 16, 32, 64, 128]
    rows = []
    for f, k in itertools.product(fracs, reductions):
        sp = 1.0 / ((1 - f) + f / k)
        rows.append(dict(component_fraction=f, component_reduction=k,
                         end_to_end_speedup=round(sp, 3),
                         reaches_10x=sp >= 10.0, reaches_100x=sp >= 100.0))
    return pd.DataFrame(rows)


def prefix_cache_table():
    """Speedup for repeated-prefix workloads.

    A request costs: reusable_fraction (prefix) + (1-reusable_fraction) (unique).
    With cache hit rate h, a fraction h of the prefix cost is replaced by a
    lookup of relative cost `overhead`. Expected per-request cost:

        cost = (1-r) + r*[(1-h)*1 + h*overhead]
        speedup = 1 / cost
    """
    rs = [0.5, 0.7, 0.9, 0.95, 0.99]
    hs = [0.5, 0.7, 0.9, 0.95, 0.99]
    ovs = [0.0, 0.01, 0.03, 0.05, 0.1]
    rows = []
    for r, h, ov in itertools.product(rs, hs, ovs):
        cost = (1 - r) + r * ((1 - h) * 1.0 + h * ov)
        sp = 1.0 / cost
        rows.append(dict(reusable_fraction=r, hit_rate=h, overhead=ov,
                         end_to_end_speedup=round(sp, 3),
                         reaches_10x=sp >= 10.0, reaches_100x=sp >= 100.0))
    return pd.DataFrame(rows)
