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
