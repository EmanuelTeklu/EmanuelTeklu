# Sparse-Safety as a Supporting Controller (Track C)

Sparse attention is NOT revived as a product. Here we only ask: do the cheap per-case features still serve as a *control signal* for persistent-inference decisions (compress vs exact, latent vs recompute, fallback)? Held-out-regime OOF on long-context block per-case data.

## Verdict: **KEEP (cheap features are a usable controller signal)**

- Mean held-out-regime AUC across control decisions: **0.89**.
- The same cheap features (entropy/mass/margin proxies, q/K/V norm stats, centroid-router scores) that predicted sparse-safety predict compress/exact/fallback decisions — so the classifier becomes a cheap *controller* in front of the persistent-state system, not a lead path.

## Control-decision predictability

| decision                  | split          |   base_rate |   auc |   cov_at_95prec |   realized_prec |
|:--------------------------|:---------------|------------:|------:|----------------:|----------------:|
| aggressively_compressible | random         |       0.569 | 0.908 |           0.348 |           0.95  |
| aggressively_compressible | heldout_regime |       0.569 | 0.881 |           0.29  |           0.95  |
| mildly_compressible       | random         |       0.776 | 0.916 |           0.67  |           0.95  |
| mildly_compressible       | heldout_regime |       0.776 | 0.89  |           0.608 |           0.95  |
| needs_exact_fallback      | random         |       0.168 | 0.919 |           0.031 |           0.953 |
| needs_exact_fallback      | heldout_regime |       0.168 | 0.893 |           0.001 |           1     |

## Role in the pivoted system

- **Gate latent vs exact:** route confidently-compressible heads/contexts to the latent-KV path, keep fragile ones exact.
- **Gate reuse vs recompute:** a low-margin / high-entropy context is less tolerant of approximate reuse; the controller can force a fresh compute.
- **Fallback detector:** `needs_exact_fallback` flags cases no compression budget serves safely.
- This keeps the sparse-safety work alive as a cheap supervisory signal while the product/research weight moves to reuse (Track A) and latent-KV (Track B).
