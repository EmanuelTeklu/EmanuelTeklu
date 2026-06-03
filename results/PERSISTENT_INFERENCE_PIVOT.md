# Persistent Inference State — The Pivot

The lead direction moves from *sparse attention* to **persistent inference
state**: reuse and compress KV/latent state across requests, with the
sparse-safety classifier demoted to a supporting controller. This document
records why, with the evidence from every prior stage.

## 1. Why the sparse block-GPU path is killed (lead path)

The realizability stack (`NEXT_STACK_SUMMARY.md`) showed the three things a
shippable sparse kernel needs cannot co-exist on this model class:

| Selection | safe read reduction (≤5% routed failure, long-ctx) | hardware |
|---|---|---|
| token-oracle, selective | 6.4× | **GPU-HOSTILE** (scattered: block-util 0.08–0.20, 6–19× over-read) |
| block-oracle, selective | **2.3×** | GPU-friendly, but **below the 4× bar even at the oracle** |
| cheap block router, selective | **~1.0×** | GPU-friendly but router too inaccurate to route safely |

Two compounding taxes — block granularity (~3×) and cheap-router inaccuracy
(~2.3×) — leave no realizable ≥4× operating point. Because the *block-oracle*
ceiling (2.3×) already fails the bar, no amount of router engineering fixes it on
small GQA models. **Decision: no Triton kernel, sparse not the lead product.**

## 2. Why prefix / latent reuse becomes the main product (Track A)

A runnable prototype (`src/prefix_cache/`, `PREFIX_PRODUCT_DEMO.md`):
canonicalization → KV/latent object store (key = model+tokenizer+canonical_hash
+version) → measured on synthetic agent traces.

- Exact-match prefix cache: hit rate **0.00**, **1.0×** — per-request volatile
  IDs / tool reordering / doc-path noise destroy it.
- **Canonical reuse (versioned-memory invalidation): hit 0.96, 13.2× TTFT
  savings, 3.2× total-cost savings.** → **STRONG KEEP.**
- Canonicalization *is* the moat: stripping volatile IDs lifts hit 0.00→0.77;
  stable doc-hashes 0.78→0.96. Versioned memory gives correct invalidation.
- Honest bound: reuse is primarily a **TTFT/latency** lever (13×); decode isn't
  reuse-saved so **$-cost** savings are ~3×. 100× needs cross-request *latent*
  reuse (shared tool results / sub-dialogues), not just the literal prefix.

This is shippable now: no kernel, the only remaining engineering is swapping the
store's metadata value for a real paged-KV / latent handle.

## 3. Why MLA / latent-KV becomes the main research (Track B)

Tested on a real **MHA** model (GPT-2, 12 heads, no GQA — native KV 1536
elems/token/layer), `MLA_LARGE_MODEL_REPORT.md`:

- Joint low-rank latent compresses GPT-2 K|V to **dc=64 → 24× smaller** at
  full-attention rel-L2 **0.036**, with **logit rank-corr 1.0 and oracle-support
  1.0** — the selection/reuse structure is perfectly preserved.
- Contrast: on GQA-2 (Qwen-0.5B, native KV 256) the MLA latent (576) is *larger*
  than native — latent-KV is **neutral-to-negative on small GQA, a major win on
  MHA/high-KV**. → **KEEP, scoped to MHA/large models** (and via MHA→MLA
  conversion to avoid pretraining).
- Critically, latent mixing does **not** destroy the structure that reuse and the
  controller depend on — so **latent-KV stacks with persistent-state reuse**:
  latent shrinks bytes/token, reuse shrinks recomputed tokens.

## 4. How sparse-safety remains a controller (Track C)

`CONTROLLER_REPORT.md`: the cheap per-case features that predicted sparse-safety
predict persistent-inference *control* decisions at **held-out-regime AUC ≈
0.89**:

- gate **latent vs exact** (compress confidently-tolerant heads, keep fragile
  ones exact);
- gate **reuse vs recompute** (force fresh compute on low-margin / high-entropy
  contexts);
- **fallback detector** for cases no compression budget serves safely.

So the sparse work is preserved as a cheap supervisory signal in front of the
persistent-state system — not a lead path.

## 5. Implications for future KV / latent-state hardware

The bottleneck has moved from *computing attention sparsely* to **storing,
addressing, and transferring persistent state**:

- **Content-addressed KV/latent object store** is the core primitive — a
  canonical-hash-keyed, versioned cache of KV/latent blobs, shared across
  requests and sessions. The win is bandwidth/latency (skip prefill), not FLOPs.
- **Cheap latent up-projection** (MLA-style) is the compression primitive that
  makes those blobs small enough to store/transfer at scale — most valuable on
  MHA/high-KV models, and it preserves selection structure.
- **Not** a sparse-attention gather chip: the falsified block path showed
  scattered-gather hardware would chase a ≤2.3× ceiling. The realizable hardware
  bet is a **latent KV cache tier** (store + version + up-project), with a cheap
  controller deciding latent-vs-exact / reuse-vs-recompute per head/context.

## 6. Keep / Kill / Park — pivoted

| Direction | Status |
|---|---|
| Sparse block-GPU kernel | **KILL** (no realizable ≥4×) |
| Prefix/latent reuse (canonicalization + KV object store) | **STRONG KEEP — lead product** |
| MLA / latent-KV on MHA/large (+ MHA→MLA conversion) | **KEEP — lead research** |
| Sparse-safety classifier as controller | **KEEP — supporting signal** |
| Token sparse + smart risk allocation (6.4×) | PARK (GPU-hostile; only if a non-gather realization appears) |

Reports: `PREFIX_PRODUCT_DEMO.md`, `MLA_LARGE_MODEL_REPORT.md`,
`CONTROLLER_REPORT.md`, and the upstream `NEXT_STACK_SUMMARY.md`. No GPU kernel,
no chip design (hardware implications only), JaneHive untouched.
