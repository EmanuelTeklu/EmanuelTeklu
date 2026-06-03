# Prefix / Latent Reuse — Product Demo (Track A)

A runnable prototype: agent traces → **canonicalization** (`prefix_cache/canonicalize.py`) → **KV/latent object store** (`prefix_cache/store.py`, key = model+tokenizer+canonical_hash+version). Token counts are char/4 estimates; the cache stores object *metadata* (KV bytes modelled), not real GPU tensors. Economics, not wall-clock.

## Verdict: **STRONG KEEP**

- **Canonical (versioned-memory invalidation): 13.2× modeled TTFT (prefill) savings** at hit rate 0.96.
- **Total-cost savings (prefill+decode, 250-tok generation): 3.2×** — decode is not reuse-saved, so $ savings are smaller than TTFT; reuse is primarily a *latency* (TTFT) lever.
- Exact-match cache only: 1.0× TTFT at hit rate 0.00 — per-request volatile IDs / tool reordering / doc-path noise break it.
- Mean reusable-prefix fraction 0.99.
- Rule (TTFT): STRONG KEEP ≥5× after invalidation; KEEP ≥3×; PARK <3×.

## Policies

| policy                     |   hit_rate |   reusable_fraction |   ttft_savings |   total_cost_savings |   n |
|:---------------------------|-----------:|--------------------:|---------------:|---------------------:|----:|
| exact                      |      0     |               0.989 |          1     |                1     | 400 |
| canonical(+versioned mem)  |      0.963 |               0.989 |         13.182 |                3.228 | 400 |
| canonical(no invalidation) |      0.98  |               0.989 |         16.756 |                3.359 | 400 |

## Canonicalization ablation (cumulative, TTFT savings)

| stage        |   hit_rate |   ttft_savings |   total_cost_savings |
|:-------------|-----------:|---------------:|---------------------:|
| exact (none) |      0     |          1     |                1     |
| +system      |      0.772 |          3.846 |                2.236 |
| +memory      |      0.772 |          3.846 |                2.236 |
| +tools       |      0.782 |          3.981 |                2.269 |
| +docs        |      0.963 |         13.182 |                3.228 |

## Overhead sensitivity (full canonicalization, versioned memory)

|   overhead |   hit_rate |   ttft_savings |   total_cost_savings |
|-----------:|-----------:|---------------:|---------------------:|
|       0    |      0.963 |         21.15  |                3.467 |
|       0.01 |      0.963 |         17.603 |                3.383 |
|       0.03 |      0.963 |         13.182 |                3.228 |
|       0.05 |      0.963 |         10.535 |                3.086 |
|       0.1  |      0.963 |          7.015 |                2.781 |

## What the prototype shows

- **Canonicalization is the product.** It converts a near-useless exact cache (1.0× TTFT) into a 13.2× TTFT engine by mapping meaning-equivalent prefixes to one key, while versioned memory blocks give correct invalidation on real edits.
- The store key (model+tokenizer+canonical_hash+version) is the addressing scheme a real KV/latent object store would use; swapping the metadata value for an actual paged-KV / latent handle is the only remaining engineering.
- Savings are bounded by the reusable fraction; the changing user tail is always recomputed. 100× needs cross-request *latent* reuse (shared tool results / sub-dialogues), the Track B/persistent-state direction.
