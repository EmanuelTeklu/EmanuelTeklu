"""Real-model Q/K/V capture for Qwen2-style decoders (transformers >= 5).

Capture strategy
----------------
We monkeypatch `transformers.models.qwen2.modeling_qwen2.eager_attention_forward`,
which is the exact internal function that receives, for every layer:

    eager_attention_forward(module, query, key, value, attention_mask, scaling)

  * query : [B, n_heads, T, d]      -- POST rotary, already at full head count
  * key   : [B, n_kv_heads, T, d]   -- POST rotary, native GQA head count
  * value : [B, n_kv_heads, T, d]   -- native GQA head count
  * module.num_key_value_groups     -- GQA repeat factor (n_heads / n_kv_heads)
  * scaling                          -- 1/sqrt(d)

GQA handling: we store K/V at their native `n_kv_heads` (that is the real
KV-cache footprint). In analysis, query head h reads KV head `h // groups`,
exactly mirroring `repeat_kv`. This is documented and asserted against the
model's own attention output.

The model is loaded in float32 on CPU with attn_implementation="eager".
"""
from __future__ import annotations
import os
import numpy as np
import torch

MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

_CAPTURE = []  # list of dicts per layer-call, cleared each forward


def _patch():
    from transformers.models.qwen2 import modeling_qwen2 as M
    orig = M.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, attn_w = orig(module, query, key, value, attention_mask, scaling,
                           dropout=dropout, **kw)
        _CAPTURE.append({
            "layer_idx": int(getattr(module, "layer_idx", len(_CAPTURE))),
            "groups": int(getattr(module, "num_key_value_groups", 1)),
            "scaling": float(scaling),
            "q": query.detach().to(torch.float32).cpu().numpy()[0],   # [H,T,d]
            "k": key.detach().to(torch.float32).cpu().numpy()[0],     # [Hkv,T,d]
            "v": value.detach().to(torch.float32).cpu().numpy()[0],   # [Hkv,T,d]
            "out": out.detach().to(torch.float32).cpu().numpy()[0],   # [T,H,d]
        })
        return out, attn_w

    M.eager_attention_forward = patched
    return orig


def load_model(model_name=MODEL_DEFAULT):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    _patch()
    return model, tok


def _chat(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_text


@torch.no_grad()
def capture_prompt(model, tok, text, query_positions=("last",), max_tokens=2048):
    """Run one forward pass and return captured per-layer tensors.

    query_positions: iterable of either "last" or float fractions in (0,1).
    Returns dict with layer arrays. Q is sliced to the requested positions to
    bound memory; K/V are kept full.
    """
    _CAPTURE.clear()
    prompt = _chat(tok, text)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    T = enc["input_ids"].shape[1]
    model(**enc)
    layers = sorted(_CAPTURE, key=lambda r: r["layer_idx"])
    # de-dup (one call per layer expected)
    seen, uniq = set(), []
    for r in layers:
        if r["layer_idx"] in seen:
            continue
        seen.add(r["layer_idx"])
        uniq.append(r)
    layers = uniq

    # resolve query positions to integer indices
    qpos = []
    for p in query_positions:
        if p == "last":
            qpos.append(T - 1)
        else:
            qpos.append(min(T - 1, max(0, int(round(float(p) * (T - 1))))))
    qpos = sorted(set(qpos))

    H = layers[0]["q"].shape[0]
    Hkv = layers[0]["k"].shape[0]
    groups = layers[0]["groups"]
    d = layers[0]["q"].shape[2]
    out = {
        "T": T, "n_layers": len(layers), "n_heads": H, "n_kv_heads": Hkv,
        "groups": groups, "d": d, "scaling": layers[0]["scaling"],
        "query_positions": qpos, "layers": [],
    }
    for r in layers:
        out["layers"].append({
            "layer_idx": r["layer_idx"],
            # Q only at query positions: [n_qp, H, d]
            "Q": r["q"][:, qpos, :].transpose(1, 0, 2).copy(),
            "K": r["k"].copy(),   # [Hkv, T, d]
            "V": r["v"].copy(),   # [Hkv, T, d]
            "out": r["out"][qpos].copy(),  # [n_qp, H, d] baseline attn output
        })
    return out


def verify_capture(cap, tol=1e-3):
    """Reconstruct full attention for one (layer,head,last-pos) and compare to
    the model's own captured output. Validates GQA mapping + math."""
    from attention_ops import full_attention
    L = cap["layers"][len(cap["layers"]) // 2]
    qi = len(cap["query_positions"]) - 1  # last position
    h = cap["n_heads"] // 2
    kv = h // cap["groups"]
    q = L["Q"][qi, h]
    K = L["K"][kv][: cap["query_positions"][qi] + 1]
    V = L["V"][kv][: cap["query_positions"][qi] + 1]
    o, _, _ = full_attention(q, K, V, cap["scaling"])
    ref = L["out"][qi, h]
    err = np.linalg.norm(o - ref) / (np.linalg.norm(ref) + 1e-9)
    return float(err), err < tol


# ---------------------------------------------------------------------------
# Prompt suite across regimes
# ---------------------------------------------------------------------------
def _filler(n_sentences=40):
    base = ("The quarterly logistics review noted that warehouse throughput "
            "in the northern region improved after the new scheduling policy. ")
    topics = [
        "Supply chain latency was attributed to customs delays at the eastern port. ",
        "Engineering reported that the caching layer reduced average query time. ",
        "The finance team flagged an anomaly in the reconciliation ledger. ",
        "Marketing observed seasonal demand shifts in the coastal markets. ",
        "Operations recommended consolidating two underused distribution centers. ",
    ]
    out = []
    for i in range(n_sentences):
        out.append(base if i % 3 == 0 else topics[i % len(topics)])
    return "".join(out)


def build_prompts():
    """Return list of {name, regime, text}. Kept CPU-feasible (~hundreds-2k tok)."""
    prompts = []

    # 1. short normal
    prompts.append(dict(name="short_qa", regime="short", text=(
        "Explain in two sentences why caching improves the latency of a web service.")))
    prompts.append(dict(name="short_chat", regime="short", text=(
        "What is the capital of France, and what river runs through it?")))

    # 2. long document QA (~3k tokens)
    doc = _filler(220)
    prompts.append(dict(name="longdoc_qa", regime="longdoc", text=(
        "Read the following report and answer the question at the end.\n\n" + doc +
        "\n\nQuestion: According to the report, what did the operations team recommend?")))

    # 3. synthetic needle-in-context, needle at mid depth (~2k tokens)
    needle_key = "The secret authorization code for project Halcyon is 7Q-ZX-4419."
    filler = _filler(150)
    half = len(filler) // 2
    needle_text = filler[:half] + " " + needle_key + " " + filler[half:]
    prompts.append(dict(name="needle", regime="needle", text=(
        "You are given a long context. Find and report the secret code.\n\n" +
        needle_text + "\n\nQuestion: What is the secret authorization code for project Halcyon?")))

    # 3b. needle placed late (~90% depth), longer context (~3k tokens)
    filler2 = _filler(230)
    cut = int(len(filler2) * 0.9)
    needle_late = filler2[:cut] + " " + needle_key + " " + filler2[cut:]
    prompts.append(dict(name="needle_late", regime="needle", text=(
        "You are given a long context. Find and report the secret code.\n\n" +
        needle_late + "\n\nQuestion: What is the secret authorization code for project Halcyon?")))

    # 4. repeated-prefix / agent tool-schema-like
    tool_schema = (
        "SYSTEM TOOLS AVAILABLE:\n"
        "1. search(query: str) -> list[str]: search the knowledge base.\n"
        "2. fetch(url: str) -> str: fetch a document.\n"
        "3. calculate(expr: str) -> float: evaluate an arithmetic expression.\n"
        "4. write_file(path: str, content: str) -> bool: persist a file.\n"
        "Always think step by step. Use tools only when necessary.\n"
    )
    prompts.append(dict(name="agent_prefix", regime="agent", text=(
        tool_schema * 10 +
        "\nUser: Search for the latency report and then calculate 3*47.\nAssistant:")))

    # 5. code / retrieval
    prompts.append(dict(name="code_retrieval", regime="code", text=(
        "Here is a Python module:\n\n"
        "def load_config(path):\n    with open(path) as f:\n        return json.load(f)\n\n"
        "def merge(a, b):\n    out = dict(a)\n    out.update(b)\n    return out\n\n"
        "def normalize(x, lo, hi):\n    return (x - lo) / (hi - lo + 1e-9)\n\n"
        "class Pipeline:\n    def __init__(self, cfg):\n        self.cfg = cfg\n"
        "    def run(self, data):\n        return [normalize(d, 0, 100) for d in data]\n\n"
        "Question: Which function would raise an error if the file does not exist?")))

    return prompts


def build_long_prompts(target_tokens=8192, approx_chars_per_tok=4):
    """Long-context prompts (~target_tokens) for the scale-up study.

    Regimes kept separable (longdoc / needle / length-shift / agent) so the
    classifier can be trained/tested per-regime. `target_tokens` controls the
    filler size; actual T is reported by the capture.
    """
    # empirically ~14 tokens/sentence for this filler; overshoot then truncate
    n_sent = max(20, int(target_tokens / 12) + 40)
    prompts = []

    doc = _filler(n_sent)
    prompts.append(dict(name=f"longdoc_{target_tokens}", regime="longdoc", text=(
        "Read the following report and answer the question at the end.\n\n" + doc +
        "\n\nQuestion: According to the report, what did the operations team recommend?")))

    needle_key = "The secret authorization code for project Halcyon is 7Q-ZX-4419."
    # needle at three depths to avoid position bias
    for depth in (0.15, 0.5, 0.85):
        f = _filler(n_sent)
        cut = int(len(f) * depth)
        nt = f[:cut] + " " + needle_key + " " + f[cut:]
        prompts.append(dict(name=f"needle_{target_tokens}_d{int(depth*100)}", regime="needle",
            text=("Find and report the secret code.\n\n" + nt +
                  "\n\nQuestion: What is the secret authorization code for project Halcyon?")))

    # length-shift: same task, ~half length (tests within-regime length transfer)
    docs = _filler(max(10, n_sent // 2))
    prompts.append(dict(name=f"lenshift_{target_tokens}", regime="lenshift", text=(
        "Summarize the key recommendation in the following report.\n\n" + docs +
        "\n\nQuestion: What single action does the report most strongly recommend?")))

    tool_schema = (
        "SYSTEM TOOLS AVAILABLE:\n"
        "1. search(query: str) -> list[str]: search the knowledge base.\n"
        "2. fetch(url: str) -> str: fetch a document.\n"
        "3. calculate(expr: str) -> float: evaluate an arithmetic expression.\n"
        "4. write_file(path: str, content: str) -> bool: persist a file.\n"
        "Always think step by step. Use tools only when necessary.\n")
    reps = max(3, n_sent // 6)
    prompts.append(dict(name=f"agent_{target_tokens}", regime="agent", text=(
        tool_schema * reps +
        "\nUser: Search for the latency report and then calculate 3*47.\nAssistant:")))
    return prompts


if __name__ == "__main__":
    m, t = load_model()
    cap = capture_prompt(m, t, build_prompts()[3]["text"])
    err, ok = verify_capture(cap)
    print(f"T={cap['T']} layers={cap['n_layers']} H={cap['n_heads']} "
          f"Hkv={cap['n_kv_heads']} groups={cap['groups']} d={cap['d']}")
    print(f"verify rel_err={err:.2e} ok={ok}")
