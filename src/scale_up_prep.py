"""Scale-up preparation for GPU runs on larger models / longer contexts.

This DOES NOT RUN heavy work by default. It writes a config (results/scale_up_config.json)
and prints the exact commands to reproduce the benchmark + classifier on
Qwen2.5-1.5B / 3B at 8k/16k/32k context. Run with `--check` to estimate
feasibility on the current machine; pass `--run MODEL CTX` only if you have
confirmed the hardware can handle it.

Rationale for the next experiment (from CLASSIFIER_REPORT.md):
  * oracle sparse headroom is large and should GROW ~linearly with context;
  * cheap selective classifier is KEEP on 0.5B long-context (≥2× read reduction
    at ≥95% precision, held-out regime). The open question is whether coverage
    and read-reduction improve at 8k-32k on a 1.5B/3B model, where the safe
    base-rate should rise (more tokens => more skippable mass).
"""
from __future__ import annotations
import os, sys, json, argparse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}
CONTEXTS = [8192, 16384, 32768]

# Rough per-forward memory/compute notes (eager attention materializes the
# full [n_heads, T, T] scores for capture). T=32k * n_heads is the cost driver.
NOTES = {
    "1.5b": dict(layers=28, q_heads=12, kv_heads=2, hidden=1536, head_dim=128),
    "3b":   dict(layers=36, q_heads=16, kv_heads=2, hidden=2048, head_dim=128),
}


def build_config():
    cfg = {
        "models": MODELS,
        "contexts": CONTEXTS,
        "capture_note": (
            "Uses eager_attention_forward monkeypatch; eager attention "
            "materializes [n_heads, T, T] scores. At T=32768 that is ~"
            f"{32768**2/1e9:.1f}G entries per head per layer in fp32 for the score "
            "matrix during capture — capture ONE layer at a time or use bf16 + "
            "chunked last-token-only logits to stay in memory."),
        "recommended_hardware": {
            "1.5b@8k": "1x A10/24GB or better; CPU possible but slow",
            "1.5b@32k": "1x A100/40GB (chunk capture)",
            "3b@32k": "1x A100/80GB (chunk capture)",
        },
        "decode_query_only": True,
        "commands": {
            "capture+experiments": (
                "python run_experiments.py --model {model} --max-tokens {ctx}"),
            "build_dataset": (
                "python build_dataset.py   # edit MODEL_DEFAULT/max_tokens or "
                "parameterize; uses capture_qkv.load_model"),
            "classifier": "python classifier.py",
            "report": "python report.py",
        },
        "code_changes_needed_for_long_ctx": [
            "capture_qkv.capture_prompt: add last-token-only logit path (avoid full "
            "TxT scores) — compute q_last @ K^T directly instead of relying on the "
            "captured attn matrix; the monkeypatch already keeps q,k,v so this is a "
            "post-hoc slice.",
            "build longer prompts: extend _filler() / add real long documents to "
            "reach 8k-32k tokens.",
            "run on CUDA: load_model already honors GPU via torch; set dtype=bf16.",
        ],
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "scale_up_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def feasibility_check():
    import torch
    have_cuda = torch.cuda.is_available()
    msg = []
    if have_cuda:
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 1e9
        msg.append(f"CUDA: {p.name}, {gb:.0f}GB")
        msg.append("OK to try 1.5B@8k" + (", 32k with chunked capture" if gb >= 40 else ""))
    else:
        msg.append("No CUDA. CPU-only.")
        msg.append("Feasible: 0.5B/1.5B at <=4k tokens (slow). 8k+ not recommended on CPU.")
    return have_cuda, msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="estimate feasibility")
    ap.add_argument("--run", nargs=2, metavar=("MODEL_KEY", "CTX"),
                    help="actually run capture+experiments (only if hardware allows)")
    args = ap.parse_args()
    cfg = build_config()
    print(f"wrote {os.path.join(RESULTS, 'scale_up_config.json')}")
    print("\nExact reproduction commands (per model/context):")
    for mk, mn in MODELS.items():
        for ctx in CONTEXTS:
            print(f"  # {mk} @ {ctx}: "
                  f"python run_experiments.py --model {mn} --max-tokens {ctx} && "
                  f"python build_dataset.py --model {mn} --max-tokens {ctx} && "
                  f"python classifier.py && python report.py")
    if args.check:
        ok, msg = feasibility_check()
        print("\nFeasibility:")
        for m in msg:
            print("  -", m)
    if args.run:
        ok, _ = feasibility_check()
        mk, ctx = args.run
        if not ok and int(ctx) > 4096:
            print(f"\nREFUSING to run {mk}@{ctx} on CPU (too expensive). "
                  f"Re-run on GPU or choose ctx<=4096.")
            sys.exit(2)
        model = MODELS[mk]
        print(f"\nRunning {model} @ {ctx} ...")
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        os.system(f"python {os.path.join(os.path.dirname(__file__),'run_experiments.py')} "
                  f"--model {model} --max-tokens {ctx}")
