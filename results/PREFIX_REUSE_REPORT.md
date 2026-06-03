# Prefix / Latent Reuse Economics — Agent Workloads (Probe)

> Economics, not measured wall-clock. Token-level **simulated** agent traces; the speedup model is analytic. Motivation: the agent / repeated-prefix regime is where sparse-safe routing transferred *worst*, so reuse is the natural alternative lever there.

## Headline

- Simulated traces: 40 sessions × 8 turns. Mean **reusable-prefix fraction = 0.98**.
- **Exact-match** prefix cache hit rate: **0.48**; **canonicalized** prefix hit rate: **0.88** (canonicalization recovers the sessions whose headers carry formatting noise — a cheap, high-leverage fix).
- At the measured operating point, end-to-end speedup is **6.5×** (canonical, overhead=0.01) vs **1.9×** (exact).
- Across the full economics grid, **11/125** points reach 10× and **0/125** reach 100×; the best is 50.2× (reusable_fraction=0.99, hit_rate=0.99, overhead=0.0).

## Speedup at the measured trace operating point

reusable_fraction=0.98

**Exact-match cache** (hit=0.48):

|   overhead |   end_to_end_speedup |
|-----------:|---------------------:|
|       0    |                1.887 |
|       0.01 |                1.87  |
|       0.03 |                1.838 |
|       0.05 |                1.806 |
|       0.1  |                1.733 |

**Canonical-prefix cache** (hit=0.88):

|   overhead |   end_to_end_speedup |
|-----------:|---------------------:|
|       0    |                6.869 |
|       0.01 |                6.488 |
|       0.03 |                5.84  |
|       0.05 |                5.31  |
|       0.1  |                4.328 |

## When are 10× / 100× possible?

- **10×** requires reusable_fraction ≥ ~0.9 AND hit_rate ≥ ~0.9 with low overhead — realistic for tight agent loops with a big pinned prefix and a warm cache, but not for one-shot/cold requests.
- **100×** is essentially unreachable by prefix caching alone (saturates around ~50× even at r=h=0.99): the non-reusable unique tail bounds it. Reaching 100× needs *latent* reuse — reusing computed state across *different* requests (shared tool results, canonical sub-dialogues), not just the literal prefix.

## Trace conditions required

| Target | reusable_fraction | hit_rate | extra |
|---|---|---|---|
| 10× | ≥0.90 | ≥0.90 | overhead ≤0.03, warm cache |
| 30× | ≥0.97 | ≥0.95 | canonicalization on, session pinning |
| 100× | ≥0.99 | ≥0.99 | + cross-request latent reuse (beyond prefix) |

## Verdict

**Serious parallel branch for agent workloads.** For repeated-prefix agents, prefix-cache reuse delivers a cleaner 5–20× than sparse routing does there, and canonicalization is a cheap multiplier. It is orthogonal to and stacks with sparse-safe routing on the *unique* tail. 100× remains an economics claim contingent on cross-request latent reuse and a real trace.
