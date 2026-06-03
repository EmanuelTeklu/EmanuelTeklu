# Stacked Experiments — Integrated Summary

Three main-stack experiments + two parallel probes, all on the inference-quotient
/ sparse-safe-attention thesis. Real Qwen2.5 Q/K/V (0.5B at 2k–32k, 1.5B at
4k–16k), CPU, memory-safe last-token capture. Component-level (not wall-clock).

| # | Experiment | Verdict | Headline |
|---|---|---|---|
| 1 | Adaptive budget | **KEEP** | 16k: **7.8× KV read reduction @4.7% fail** (fixed-10% = 5.6×); but naive cascade pooled (≥8k) = 3.0×, *below* fixed-10%'s 4.5× — compounding eats the failure budget |
| 2 | Block locality / hardware | **GPU-HOSTILE / gather-bound** | block-util(32)=0.15, **~9× over-read**, cross-head overlap 0.22 → selections scattered, **kernel gate NOT cleared** |
| 3 | Per-model calibration | **KEEP** | **~1000 target cases recover ~95%** of full coverage (250 → ~50%) → weak model transfer is a non-issue |
| P1 | Prefix/latent reuse | **Serious parallel branch** (agents) | reusable frac 0.98; canonical hit **0.88** vs exact 0.48; 10× needs r≥0.9 & h≥0.9; 100× needs cross-request latent reuse |
| P2 | MLA / latent-KV | **Parallel branch for MHA/large** | MLA neutral-to-negative for GQA-2 small (0.5B: 2.3× *larger*), **28×** smaller for MHA-70B-like; orthogonal to sparse, **stacks to ~62×** |

## The decisive synthesis

**The bottleneck has moved from "is there sparse headroom?" to "is it hardware-realizable?"**

1. **Headroom is not budget-capped (Exp 1).** Replacing the fixed 10% budget with
   an adaptive cascade lifts 0.5B@16k from 5.6× to **7.8×** at matched ≤5% failure,
   and the *uncapped* cascade shows **10–24×** is physically present. So the read
   reduction ceiling reported earlier was an artifact of the fixed budget, not of
   the attention structure.

2. **…but capturing it safely is unsolved.** Holding overall routed failure ≤5%
   across the 7-stage cascade forces each stage to ~99% precision, which throttles
   the aggressive small budgets — so the *naive* cascade pooled across contexts
   (3.0×) actually loses to plain fixed-10% (4.5×). Smarter per-stage
   failure-budget allocation is the missing piece.

3. **The real blocker is block locality (Exp 2).** Even where coverage and budget
   are favorable, the selected tokens are **scattered**: ~15% sit in the recent
   window, block utilization is 0.08–0.20, whole-block gathers over-read 6–19×,
   and heads share only ~22% of their selections. A naive Triton block-sparse
   kernel would spend most of the sparse saving on padding. **The kernel gate is
   not cleared** — do not build a GPU kernel yet.

4. **Per-model cost is cheap (Exp 3).** The weak 0.5B→1.5B transfer found earlier
   is a non-issue: ~1000 labelled cases (a handful of forward passes) recover ~95%
   of a model's achievable coverage. Calibrate per model, don't transfer.

5. **The two regimes where sparse is weakest have better levers (P1, P2).**
   - *Agent / repeated-prefix* (worst sparse transfer): prefix-cache **reuse**
     gives a cleaner 5–20×, and canonicalizing the prefix lifts the hit rate from
     0.48 to 0.88. Orthogonal to sparse; stacks on the unique tail.
   - *Large / MHA models*: **latent-KV (MLA / MHA→MLA)** shrinks bytes-per-token
     ~28× where GQA-small gets nothing, and multiplies with sparse routing
     (~62× combined). Neutral for GQA-2 small models (where sparse is the only KV
     lever).

## What to do next (no kernel yet)

1. **Locality-regularized routing** — bias the router toward contiguous blocks /
   shared head selections, then re-measure block utilization vs the rel-L2 cost.
   This is the make-or-break for any block-sparse kernel.
2. **Smarter cascade** — allocate the 5% failure budget across stages (tight on
   tiny budgets, loose on large) so adaptive dominates fixed-10% everywhere, not
   just at 16k.
3. **Selection stability** — measure whether a head's scattered selection is
   *stable across decode steps*; if so, gather sets can be precomputed/amortized
   (a different hardware bet than block-sparse).
4. **Parallel:** prototype canonical prefix-cache reuse for an agent trace; and a
   one-checkpoint MHA→MLA conversion to test whether sparse structure survives the
   latent mixing.

Per the brief: **no GPU kernel** (block locality failed the gate), **no chip
design** (hardware report only), JaneHive untouched.

Individual reports: `ADAPTIVE_BUDGET_REPORT.md`, `HARDWARE_REALIZABILITY_REPORT.md`,
`CALIBRATION_REPORT.md`, `PREFIX_REUSE_REPORT.md`, `MLA_LATENT_KV_REPORT.md`.
