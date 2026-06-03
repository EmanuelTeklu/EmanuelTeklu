# Next Stack — Integrated Summary: Can Sparse-Safe Routing Be Made Hardware-Realizable?

Four main experiments + two parallel probes. Real Qwen2.5 Q/K/V (0.5B at 8k–32k,
1.5B at 8k), memory-safe last-token capture, CPU, component-level (not wall-clock).
**Goal:** turn the earlier GPU-hostile finding into a realizable win via
block-level routing + smarter risk allocation.

## Verdict table

| # | Experiment | Verdict | Headline number |
|---|---|---|---|
| 1 | Block-oracle upper bound | **PARK** | block-16 retains **26%** of token-oracle gain; 50× at *median* but only **2.3× under safe selective gating** |
| 2 | Locality-regularized routing | **PARK** | cheap block routers reach 10–51× at *median* (rl2≤0.10) but **collapse to ~1× at ≤5% routed failure** |
| 3 | Smarter adaptive risk | **KEEP (token) / KILL (block)** | token: expected-cost **6.4× @4.3% fail** (1.42× over fixed-10%); block-oracle ceiling **2.3×**; cheap-block **1.0×** |
| 4 | Per-model calibration min | **KEEP; active=no** | **~500 random cases → ~95%** of ceiling; active selection does NOT beat random |
| A | Prefix/latent reuse (agents) | **KEEP** | canonicalization lifts hit rate **0.015→0.875**, **5.8× modeled speedup** (10% invalidation still ~5×) |
| B | MLA / latent-KV stack | **KEEP** | latent mixing **preserves** selection structure (rank-corr **0.98 @4×**, support 0.83); byte win modest on GQA-2, large on MHA |

## The decisive finding

**The hardware-realizability path hits a structural wall on this model class.**
The three properties we need — (i) GPU-friendly *block* selection, (ii) a *cheap*
non-oracle router, (iii) *safe* operation at ≤5% routed failure — do not co-exist:

```
                              safe read reduction (≤5% routed failure, long-ctx)
  token-oracle  selective ............ 6.4×   (but selections are GPU-HOSTILE/scattered)
  block-oracle  selective ............ 2.3×   (GPU-friendly, but block granularity costs ~3×)
  cheap block router selective ....... 1.0×   (router too inaccurate to route safely)
```

Two compounding taxes:
1. **Block granularity tax (~3×):** even *oracle* block selection, gated by a
   cheap-feature classifier at 95% precision, drops from the token path's 6.4× to
   **2.3×**. Block-oracle passes (median) at 2% read vs token-oracle's 0.5%.
2. **Cheap-router tax (~2.3×):** the cheap block routers (centroid, centroid+radius,
   max-sketch) are far less accurate than the block-oracle, so under safe selective
   gating they route almost nothing → **~1.0×**.

Critically, the **block-oracle selective ceiling (2.3×) is already below the 4×
KEEP bar.** So *no* cheap block router can clear 4× at ≤5% failure on this model
class — the wall is the block-granularity + cheap-feature-gating combination, not
just router quality. Better block *selection* cannot fix it; only better cheap
*predictive features* (the gate) or a different model regime could.

What we *did* fix: Experiment 3 shows the naive cascade's compounding failure is
solvable — **expected-cost risk allocation** holds ≤5% failure and recovers the
token path to **6.4×** (1.42× over fixed-10%). But that win lives on the
GPU-hostile token path, so it does not translate to wall-clock without a kernel
we've shown we shouldn't build.

## Updated keep / kill / park

| Direction | Status | Why |
|---|---|---|
| Token sparse + smart risk allocation | KEEP (research) | 6.4× safe component reduction — but GPU-hostile, no kernel |
| Block sparse (locality routing) | **PARK / near-KILL** | safe ceiling 2.3× < 4× even at oracle; cheap router ~1× |
| Per-model calibration | KEEP | ~500 random cases recover 95%; transfer non-issue |
| Triton block-sparse kernel | **DO NOT BUILD** | no safe ≥4× realizable operating point exists |
| Prefix/latent reuse (agents) | **KEEP — promote** | 5.8× modeled, no kernel, immediately shippable |
| MLA / latent-KV (MHA/large) | **KEEP — parallel branch** | preserves sparse structure; stacks; big byte win off GQA-2 |

## Next step: not Triton, not more block routing — prefix product + MLA

1. **Do NOT build a GPU kernel.** The block-oracle safe ceiling (2.3×) is below
   the threshold; there is no realizable ≥4× to accelerate.
2. **Stop pushing within-attention block routing on small GQA models** — it is
   capped below the bar even at the oracle. Revisit only (a) with better *cheap
   predictive features* for block-safety (the gate, not the selector), or (b) on
   MHA/large models where the structure may differ.
3. **Promote prefix/latent reuse to the product track** for agent workloads:
   5.8× modeled at realistic hit rates, canonicalization is the cheap multiplier,
   no kernel required.
4. **Open the MLA/latent-KV branch** for MHA/large models: latent mixing preserves
   the sparse-safe selection structure (so it stacks), and its byte win is large
   exactly where GQA-small gets none.
5. **Cheap research bet:** better cheap features for sparse-safety prediction
   (lifts the 95%-precision gate, the real ceiling on the selective path) and
   selection stability across decode steps.

Reports: `BLOCK_ORACLE_REPORT.md`, `LOCALITY_ROUTER_REPORT.md`,
`ADAPTIVE_RISK_REPORT.md` (+ `_BLOCK_*`), `CALIBRATION_MIN_REPORT.md`,
`PREFIX_REUSE_REPORT.md`, `MLA_STACK_REPORT.md`. No GPU kernel, no chip design,
JaneHive untouched (per brief).
