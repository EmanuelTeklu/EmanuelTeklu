# Inference Quotient — Certified Sparse / Residual / Margin-Aware KV Benchmark

A runnable benchmark that asks one decisive question on **real transformer
Q/K/V tensors** (not synthetic probes):

> Can a *cheap, non-oracle* method recover enough of the oracle
> sparse-attention gain to justify kernel work — and does margin/importance-aware
> KV compression beat uniform 4-bit?

It captures real attention tensors from an open model, measures the **oracle
sparse upper bound**, then tests whether cheap routers, residual/value
correction, and adaptive KV quantization can capture that headroom — with
per-head predictor analysis, prefix-reuse economics, and an Amdahl end-to-end
sanity check. Every run writes CSV/JSON artifacts and a Markdown report with
explicit **keep / kill / park** decisions.

## TL;DR result (Qwen2.5-0.5B-Instruct, CPU, up to 3,226-token context)

| Question | Verdict | Number |
|---|---|---|
| ≥10× component read reduction on real tensors (oracle upper bound) | **KEEP** | median long-context head: rel-L2≤0.10 at **~133× read reduction**; S=16 ≈ 145× at rel-L2≈0.065 |
| Cheap non-oracle router recovers ≥70% of it across heads | **PARK (weak)** | best router: median passes at ~8% reads, but only **67% of heads** pass; oracle-support recovery peaks at **69%** |
| Sparse+residual correction makes sparse robust | **PARK** | cheap block-residual cuts error ×0.61 at budget 8; oracle-mass ceiling ×0.46 shows real headroom |
| Adaptive KV quant beats uniform 4-bit | **PARK** | fails all 3 pre-registered storage tests, but importance-aware allocation is Pareto-superior 4.5–6 bits (≈8-bit quality at ~5 bits) |
| Strongest concrete asset | — | a **head-compressibility predictor**: top-k mass vs oracle error **r≈-0.81**, entropy **r≈+0.68** |

Full numbers, tables, and decisions: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md).

### Follow-up: selective sparse-safe classifier — **KEEP**

Universal routing was only PARK, so the next question was: *can we predict which
heads/queries are sparse-safe and route only those?* Yes.

| Test | Result |
|---|---|
| Beats simple entropy/top-mass thresholding? | **Yes, decisively** — single-feature thresholds reach **~0% coverage at 95% precision** (AUC≈0.69); the classifier reaches **40%** (random) / **32%** (held-out regime) on cheap features |
| Precision/coverage (cheap, deployable features) | 95% precision at **40%/32%/38%/26%** coverage (random / held-out regime / head / layer) |
| Long-context KV read reduction (held-out regime) | **~2.0× (cheap), ~2.9× (oracle-feature)** at ≤5% routed failure |
| Transfer to needle / longdoc / length-shift | holds, **2.5–4.2×** read reduction |
| Transfer to agent / repeated-prefix | **weak** (precision–coverage tradeoff degrades) |

Verdict **KEEP** (not STRONG KEEP): meets ≥95% precision, ≥30% held-out-regime
coverage, ≥2× long-context read reduction, and crushes thresholding baselines —
but cheap-feature coverage is <50% and the agent regime transfers poorly. Full
detail: [`results/CLASSIFIER_REPORT.md`](results/CLASSIFIER_REPORT.md).

### Scale-up: does sparse-safe structure strengthen with scale?

Re-ran the classifier across **Qwen2.5-0.5B at 2k–32k** and **1.5B at 4k–16k**,
using a memory-safe **last-token-only SDPA capture** in bf16
(`src/capture_lasttok.py`) that avoids materializing the `[heads,T,T]` score
matrix (0.5B reaches 32k, 1.5B ~16k within 15GB CPU RAM; GPU points for
1.5B@32k / 3B are wired but not run here).

| Axis | Finding |
|---|---|
| **Context length** | **Strengthens.** 0.5B base rate 0.60→**0.78**; held-out-regime coverage peaks **0.60 @16k**; read reduction **2.16× @16k**; oracle read-gain ≥100×; top-16 tokens still hold 67% of mass at 32k |
| **Context-length transfer** | **Holds** — ≤8k-trained threshold keeps ~0.94 precision on 16k/32k |
| **Model size** | **Does NOT help** — 1.5B is slightly harder for the cheap classifier than 0.5B at matched context |
| **Model-size transfer** | **Weak** — a 0.5B-trained threshold collapses to **0.68 precision** on 1.5B (retrain per model) |
| **Read-reduction ceiling** | Capped ~2.2× by the fixed 10% router budget, *not* by missing structure (oracle ceiling ≥100×) |

Verdict **KEEP** (improving toward STRONG with context, but model-size transfer
is a PARK signal and read reduction is budget-capped). Detail:
[`results/SCALE_UP_REPORT.md`](results/SCALE_UP_REPORT.md). Reproduce/extend on
GPU: `python src/scale_up_prep.py --check`.

### Stacked experiments — the bottleneck moves to hardware-realizability

Three main-stack experiments + two parallel probes ([`results/STACK_SUMMARY.md`](results/STACK_SUMMARY.md)):

| # | Experiment | Verdict | Headline |
|---|---|---|---|
| 1 | **Adaptive budget** ([report](results/ADAPTIVE_BUDGET_REPORT.md)) | KEEP | 16k: **7.8×** read reduction @4.7% fail (fixed-10% = 5.6×); 10–24× headroom exists but naive cascade can't hold ≤5% failure when pooled |
| 2 | **Block locality / hardware** ([report](results/HARDWARE_REALIZABILITY_REPORT.md)) | GPU-HOSTILE | block-util 0.15, **~9× over-read**, cross-head overlap 0.22 → scattered selections, **kernel gate NOT cleared** |
| 3 | **Per-model calibration** ([report](results/CALIBRATION_REPORT.md)) | KEEP | **~1000 cases recover ~95%** of coverage → weak model transfer is a non-issue |
| P1 | **Prefix/latent reuse** ([report](results/PREFIX_REUSE_REPORT.md)) | parallel branch (agents) | canonical hit **0.88** vs exact 0.48; 10× needs r,h≥0.9; 100× needs latent reuse |
| P2 | **MLA / latent-KV** ([report](results/MLA_LATENT_KV_REPORT.md)) | parallel branch (MHA/large) | neutral for GQA-small, **28×** for MHA-70B-like, **stacks to ~62×** with sparse |

**Synthesis:** the read-reduction ceiling was the *fixed budget*, not the
attention structure (adaptive shows 10–24× headroom). The real blocker is now
**block locality** — real sparse selections are scattered and gather-bound, so a
naive Triton kernel would waste 6–19× on padding. **No GPU kernel yet** (gate not
cleared). Next real-tensor work: locality-regularized routing + smarter cascade
failure-budget allocation; in parallel, prefix-reuse for agents and MHA→MLA for
large models.

Reproduce the stack:

```bash
cd src
python stack_experiments.py        # memory-safe capture -> budget/overlap datasets
python adaptive_policy.py           # Exp 1
python hardware_realizability.py    # Exp 2
python calibration.py               # Exp 3
python prefix_reuse_probe.py        # Probe 1
python mla_comparison.py            # Probe 2
```

### Realizability stack — can block routing + smart risk allocation fix it?

The hardware-realizability question, answered ([`results/NEXT_STACK_SUMMARY.md`](results/NEXT_STACK_SUMMARY.md)):

| # | Experiment | Verdict | Headline |
|---|---|---|---|
| 1 | Block-oracle upper bound ([report](results/BLOCK_ORACLE_REPORT.md)) | PARK | block-16 retains **26%** of token gain; 50× median but **2.3× under safe gating** |
| 2 | Locality-regularized routing ([report](results/LOCALITY_ROUTER_REPORT.md)) | PARK | cheap block routers 10–51× at *median* but **~1× at ≤5% failure** |
| 3 | Smarter adaptive risk ([report](results/ADAPTIVE_RISK_REPORT.md)) | KEEP (token) / KILL (block) | token expected-cost **6.4×** (fixes naive cascade); block collapses |
| 4 | Calibration minimization ([report](results/CALIBRATION_MIN_REPORT.md)) | KEEP | **~500 random cases → 95%**; active ≠ better |
| A | Prefix reuse v2 ([report](results/PREFIX_REUSE_REPORT.md)) | KEEP | canonicalization **0.015→0.875** hit, **5.8×** |
| B | MLA latent-KV stack ([report](results/MLA_STACK_REPORT.md)) | KEEP | latent **preserves** sparse structure (rank-corr 0.98 @4×) |

**Decisive finding:** the three requirements — GPU-friendly *blocks*, a *cheap*
router, and *safe* ≤5% failure — don't co-exist. Token-oracle selective = 6.4×
(but GPU-hostile); **block-oracle** selective = 2.3× (block tax ~3×); **cheap
block router** selective = ~1× (router tax). The block-oracle ceiling (2.3×) is
**already below the 4× bar**, so no cheap block router can clear it on this model
class. **Do not build a Triton kernel.** The shippable wins are the parallel
branches: **prefix reuse** for agents (5.8×, no kernel) and **MLA/latent-KV** for
MHA/large models (structure-preserving, stacks).

## The pivot: persistent inference state

Sparse attention is no longer the lead product. The work pivots to **persistent
inference state** ([`results/PERSISTENT_INFERENCE_PIVOT.md`](results/PERSISTENT_INFERENCE_PIVOT.md)):

| Track | Verdict | Result |
|---|---|---|
| **A — Prefix/latent reuse product** ([demo](results/PREFIX_PRODUCT_DEMO.md), `src/prefix_cache/`) | **STRONG KEEP — lead product** | canonicalization → KV object store: exact cache **1.0×** → canonical **13.2× TTFT / 3.2× total cost** (hit 0.00→0.96) |
| **B — MLA / latent-KV** ([report](results/MLA_LARGE_MODEL_REPORT.md), GPT-2 MHA) | **KEEP — lead research** | latent dc=64 = **24× byte reduction** at rel-L2 0.036, **structure perfectly preserved** (rank-corr 1.0); neutral on GQA-small, big on MHA |
| **C — Sparse-safety as controller** ([report](results/CONTROLLER_REPORT.md)) | **KEEP — supporting** | cheap features predict compress/exact/latent/fallback at held-out AUC **0.89** |

**Why:** sparse block-GPU is capped below the bar even at the oracle (KILL the
kernel); prefix reuse is shippable now and is a 13× TTFT lever; latent-KV is a
24× byte win on MHA that *preserves* the structure reuse/control rely on, so they
stack; the sparse-safety classifier survives as a cheap controller gating
latent-vs-exact / reuse-vs-recompute. The hardware implication is a
**content-addressed latent KV-cache tier**, not a sparse-gather chip.

```bash
cd src
python prefix_demo.py        # Track A: canonicalization + KV object store demo
python mla_large.py          # Track B: latent-KV on GPT-2 (MHA)
python controller.py         # Track C: sparse-safety as control signal
```

## Persistent inference state wedge

Is canonical persistent-state reuse a **real wedge beyond exact prefix caching**?
Benchmark with **real `past_key_values` tensor reuse** + a false-hit audit
([`results/REAL_STATE_REUSE_WEDGE_REPORT.md`](results/REAL_STATE_REUSE_WEDGE_REPORT.md),
`scripts/run_real_state_wedge.py`):

**Verdict: STRONG KEEP.**

- **Real KV reuse (Qwen2.5-0.5B, CPU, actual `past_key_values`):** canonical state
  reuse = **5.1× wall-clock prefill speedup** (coding 5.46×, research 4.73×) at
  **3.8e-5** max logit diff (reuse == full forward of the canonical context).
- **The wedge:** under per-request volatile noise, **exact-prefix cache hit = 0.00**
  (what vLLM/SGLang/LMCache key on) vs **canonical hit = 0.97** — incremental lift
  ~0.97 across **all 5 regimes** (coding/legal/research/multitool/memory).
- **Zero false hits** across 14 adversarial mutation types: every semantic change
  (tool semantics, param type, doc content, memory version, system policy,
  doc replace, model/tokenizer) breaks the key; every surface change (volatile IDs,
  timestamps, tool/JSON order, whitespace, doc metadata) collapses correctly.
- TTFT savings (modeled) 8.5–11.3× per regime; total-cost savings ~3× (decode
  isn't reuse-saved). Differentiator: canonical/versioned **state identity computed
  before tokenization**, so semantically-stable context reuses internal state
  despite volatile surface differences — exactly where exact/radix caches miss.

Deliverables: `canonicalization_stress.csv`, `real_kv_reuse.csv`,
`baseline_comparison.csv`, `false_hit_audit.csv`. Source: `src/prefix_cache/`
(`canonicalize.py`, `kv_reuse.py`, `store.py`, `regimes.py`, `mutations.py`).

```bash
python scripts/run_real_state_wedge.py           # full (real CPU forwards, ~15 min)
python scripts/run_real_state_wedge.py --quick   # fast smoke
```

**Bottom line on the thesis.** The *opportunity* is real and large (10–100×
component read reduction lives in long-context heads). The gap is a cheap,
robust selector that survives diffuse heads. Next move is **not** a GPU kernel
yet — it is (1) confirm the oracle headroom scales with context on a larger
model/GPU, and (2) build the head-selective compress/skip classifier the
predictors already justify. KV quantization is *not* the wedge (uniform 4-bit
is strong); naive sparse routing is *not yet* safe enough.

## Quickstart

```bash
pip install -r requirements.txt
# CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu

cd src
python run_experiments.py --max-tokens 4096   # capture + experiments C–I
python report.py                              # build results/FINAL_REPORT.md + plots
```

Useful flags:
- `--model Qwen/Qwen2.5-1.5B-Instruct` — larger model (GPU recommended).
- `--limit-layers 0,12,23` — restrict layers for a fast smoke run.
- `--synthetic` — **labeled** synthetic-tensor fallback (NOT decisive; only for
  when model download/inference is impossible).

If a CUDA GPU is present, `torch` will use it automatically for capture; the
analysis itself is NumPy on CPU.

## What it does

| Stage | Module | Output |
|---|---|---|
| Real Q/K/V capture (monkeypatches `eager_attention_forward`, GQA-correct, validated to ~1e-6) | `capture_qkv.py` | `results/raw/sample_capture_needle.npz` |
| Exact / sparse single-query attention | `attention_ops.py` | — |
| C: oracle sparse upper bound (S∈{1..256}) | `sparse_routing.py` | `tables/oracle_sparse.csv` |
| D: non-oracle routers (JL, block-centroid, upper-bound certificate, recency, sink+recent) | `sparse_routing.py` | `tables/routing.csv` |
| E: sparse + residual/value correction (cheap block-residual, low-rank, oracle-mass ceilings) | `residual_correction.py` | `tables/residual.csv` |
| F: KV quant frontier (uniform 8/4/3/2-bit KIVI-style vs adaptive) | `quantization.py` | `tables/quantization.csv` |
| G: per-head/layer predictor correlations | `report.py` | `tables/predictor_correlations.csv` |
| H/I: prefix-reuse + Amdahl analytic cost models | `economics.py` | `tables/prefix_cache.csv`, `tables/amdahl.csv` |
| Report + plots + keep/kill/park | `report.py` | `results/FINAL_REPORT.md`, `results/plots/*.png` |
| Sparse-safe supervised dataset (features + labels) | `build_dataset.py` | `tables/classifier_dataset.csv` |
| Selective sparse classifier + policy (LR/tree/RF/GBM/MLP, grouped splits, transfer) | `classifier.py` | `tables/classifier_*.csv`, `tables/selective_*.csv`, `results/CLASSIFIER_REPORT.md` |
| GPU scale-up prep (config + exact commands, no heavy run) | `scale_up_prep.py` | `results/scale_up_config.json` |

Reproduce the classifier branch:

```bash
cd src
python build_dataset.py     # re-captures Q/K/V, writes the supervised dataset
python classifier.py        # trains, evaluates, writes results/CLASSIFIER_REPORT.md
python scale_up_prep.py --check   # prints exact 1.5B/3B @ 8k-32k commands
```

## Method notes / honesty

- **Capture is validated:** for a sampled layer/head/last-token, reconstructed
  exact attention matches the model's own attention output to ~1e-6 relative
  error, confirming the GQA head mapping (query head `h` reads KV head
  `h // groups`) and the math.
- **Oracle = upper bound.** Oracle sparse, oracle-mass residual ceilings, and
  importance-aware quant all use *true* attention signals. They bound what a
  cheap method could achieve; the cheap, deployable versions are reported
  side-by-side so the gap is explicit.
- **Decode last-token queries** over the full causal context (the long-context
  decode regime). Multi-position prefill patterns are not swept.
- **CPU-only, ≤3.2k tokens, one 0.5B model.** Findings are about attention
  *structure*, not wall-clock. No CUDA kernels were written or timed; the
  Amdahl/prefix sections are economics, not measured runtime.
- **Failures are included**, not hidden (e.g. JL projection routing is poor;
  block upper-bound certificate underperforms the plain centroid here).

Reproducibility: all randomness is seeded (`SEED=0`); `results/config.json`
records model, versions, context lengths, and capture-verification errors.

---

### About the author

Emanuel Teklu — independent researcher working on where AI has the highest
marginal value (currently mechanistic interpretability and, here, inference
efficiency under behavior-preservation constraints).
[emanuelteklu.com](https://emanuelteklu.com)
