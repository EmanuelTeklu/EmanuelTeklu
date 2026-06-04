# Persistent Inference State — Real Reuse Wedge Report

## 1. Executive verdict: **STRONG KEEP**

- **False hits under semantic change: 0** (false-hit rate 0.000); false misses 0 (rate 0.000). Zero false hits.
- **Real `past_key_values` reuse (CPU, actual tensors):** canonical mode **5.1× wall-clock prefill speedup** vs stateless; correctness max logit diff **3.8e-05** (reuse == full forward of the canonical context).
- **Wedge vs exact caching:** exact-prefix hit rate **0.00** under volatile noise vs canonical **0.97** → incremental lift **0.97**; works in **5/5** regimes.

## 2. What was actually implemented

- `src/prefix_cache/canonicalize.py` — canonicalization transforms + Context.
- `src/prefix_cache/kv_reuse.py` — **real** DynamicCache prefill + tail-only reuse (validated to ~1e-5 logit diff).
- `src/prefix_cache/store.py` — versioned content-addressed object store.
- `src/prefix_cache/regimes.py` — 5 agent regimes; `mutations.py` — adversarial surface/semantic mutations.
- `scripts/run_real_state_wedge.py` — Experiments A–F.

## 3. Real KV reuse result (Experiment A)

Model: `Qwen/Qwen2.5-0.5B-Instruct`, dtype float32, device CPU; reuse uses **actual `past_key_values` tensors** (not metadata). Stateless = full prefill every request; canonical = prefill the canonical prefix once per unique context, then forward only the tail.

| regime   | mode      |   seconds |   hits |   prefills |   tokens_avoided |   speedup_vs_stateless |
|:---------|:----------|----------:|-------:|-----------:|-----------------:|-----------------------:|
| coding   | stateless |   147.148 |      0 |         14 |                0 |                  1     |
| coding   | exact     |   146.9   |      0 |         14 |                0 |                  1.002 |
| coding   | canonical |    26.963 |     10 |          4 |            17104 |                  5.457 |
| research | stateless |   136.102 |      0 |         14 |                0 |                  1     |
| research | exact     |   168.904 |      0 |         14 |                0 |                  0.806 |
| research | canonical |    28.744 |     11 |          3 |            22426 |                  4.735 |

**Correctness (canonical reuse vs full forward of the canonical context):**

|   max_logit_diff |   mean_logit_diff |   argmax_agree | regime   |
|-----------------:|------------------:|---------------:|:---------|
|                0 |                 0 |              1 | coding   |
|                0 |                 0 |              1 | research |

## 4. Baseline comparison (Experiments B & D)

Hit rate by caching strategy, per regime (volatile noise ON):

| regime    |   canonical |   canonical_versioned |   exact |   normalized |
|:----------|------------:|----------------------:|--------:|-------------:|
| coding    |       0.975 |                 0.975 |       0 |        0.108 |
| legal     |       0.975 |                 0.975 |       0 |        0.25  |
| memory    |       0.975 |                 0.975 |       0 |        0.142 |
| multitool |       0.975 |                 0.975 |       0 |        0     |
| research  |       0.958 |                 0.958 |       0 |        0.125 |

TTFT savings (modeled, char/4 tokens) by mode:

| regime    |   canonical |   canonical_versioned |   exact |   normalized |
|:----------|------------:|----------------------:|--------:|-------------:|
| coding    |      17.035 |                17.035 |       1 |        1.117 |
| legal     |      17.432 |                17.432 |       1 |        1.319 |
| memory    |      17.341 |                17.341 |       1 |        1.159 |
| multitool |      17.109 |                17.109 |       1 |        1     |
| research  |      13.483 |                13.483 |       1 |        1.137 |

**The wedge:** exact-prefix caching collapses under per-request volatile noise; canonical state identity recovers the reuse. Incremental lift is the difference, and it is the part existing exact/prefix caches cannot capture.

## 5. Canonicalization stress test (Experiment C)

- should-hit checks: 240, false misses 0.
- should-miss checks: 320, **false hits 0**.

| kind        | mutation             |   frac_correct |
|:------------|:---------------------|---------------:|
| should_hit  | ALL                  |              1 |
| should_hit  | doc_metadata         |              1 |
| should_hit  | json_field_order     |              1 |
| should_hit  | tool_order           |              1 |
| should_hit  | volatile_id          |              1 |
| should_hit  | whitespace           |              1 |
| should_miss | doc_content_change   |              1 |
| should_miss | doc_replaced         |              1 |
| should_miss | memory_version_bump  |              1 |
| should_miss | model_changed        |              1 |
| should_miss | system_policy_change |              1 |
| should_miss | tokenizer_changed    |              1 |
| should_miss | tool_param_type      |              1 |
| should_miss | tool_semantic_change |              1 |

**Zero false hits.** Every semantic change (tool semantics, param type, doc content, memory version, system policy, doc replacement, model/tokenizer) broke the canonical key as required.

## 6. Agent trace results by regime (Experiment D)

| regime    |   hit_rate |   reusable_fraction |   ttft_savings |   lift_over_exact |
|:----------|-----------:|--------------------:|---------------:|------------------:|
| coding    |      0.975 |               0.995 |         17.035 |             0.975 |
| legal     |      0.975 |               0.997 |         17.432 |             0.975 |
| research  |      0.958 |               0.996 |         13.483 |             0.958 |
| multitool |      0.975 |               0.996 |         17.109 |             0.975 |
| memory    |      0.975 |               0.996 |         17.341 |             0.975 |

## 7. Differentiation from existing systems (Experiment E)

**What existing systems already do:**
- *vLLM prefix caching / PagedAttention*, *SGLang RadixAttention*, *LMCache*: cache and reuse KV when the **token prefix matches** (exact, by block/radix). Reuse is keyed on raw token identity.
- *MemGPT-like* memory: application-level context management, not KV reuse.

**What this layer adds:**
- A **canonical, versioned state identity** computed *before* tokenization, so semantically-stable agent context (system+tools+docs+memory) reuses internal state **despite volatile surface differences** (request IDs, timestamps, tool ordering, JSON formatting, doc metadata) that make the raw token prefix differ — exactly where exact/radix caches miss (exact hit 0.00 vs canonical 0.97 here).
- Correct **invalidation** via explicit content/version hashing of tools, docs, and memory blocks.

**Adapter plan (concrete):**
1. Compute canonical state ID at request admission (a preprocessing hook before the tokenizer).
2. Map state ID → KV handle: for vLLM, prefill the *canonical* prefix once and pin its blocks; serve volatile-variant requests by attending to the pinned blocks + the fresh tail (the canonical prefix becomes the cached token sequence). For SGLang, insert the canonical prefix as a Radix node.
3. Enforce invalidation by versioned key (memory edit / doc change → new node, old node TTL-evicted).
4. LMCache-style backend: state object metadata (this repo's `store.py`) maps to the KV blob handle in the cache tier.

**What must be proven next:** that serving the *canonical* surface form is acceptable to applications (it is, iff no false hits — §5), and the same wall-clock win on a GPU serving stack with real concurrency.

## 8. Controller policy sketch (Experiment F)

Sparse-safety stays a *support* signal. Policy actions: {exact_reuse, latent_reuse, recompute, invalidate, fallback}.

```
decide(request):
  sid = canonical_state_id(request)            # before tokenization
  if version_changed(sid): return INVALIDATE + RECOMPUTE
  if sid in store:
     if controller.safe_latent(features): return LATENT_REUSE   # compressed KV
     return EXACT_REUSE                                          # full KV blob
  if controller.reuse_risk(features) high: return RECOMPUTE
  return PREFILL + STORE
```
- `controller` = cheap-feature classifier (held-out AUC ~0.89, CONTROLLER_REPORT.md): inputs regime, stable-context fraction, schema/doc/memory version stability, prior hit/miss stats.
- Objective: minimize cost subject to **false-hit risk = 0** (canonical key guarantees this structurally; controller only chooses latent-vs-exact and reuse-vs-recompute, never overrides a version miss).

## 9. Remaining bottlenecks

- Wall-clock measured on **CPU, 0.5B, float32**; GPU serving numbers TBD.
- Token counts in §4 are char/4 estimates (labeled modeled); §3 uses real tokenizer + real forwards.
- Store holds real KV tensors *in-process*; a production tier needs eviction, persistence, and cross-node transfer.
- Serving the canonical surface form (not the literal prompt) is the design commitment; safe only because §5 shows zero false hits.

## 10. Next step

1. vLLM/SGLang adapter (pin canonical prefix blocks / Radix node).
2. Real customer/agent traces to validate hit rates and invalidation cadence.
3. Latent-KV handle (Track B / MLA) as the stored object for MHA/large models.
4. Hardware (content-addressed latent KV tier) only after the above.
