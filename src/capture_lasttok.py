"""Memory-safe, last-token-only Q/K/V capture for long contexts.

Why a separate path from capture_qkv.py:
  * The eager capture records the full attention-weight matrix [H,T,T] per
    layer. At T=16k-32k that is tens of GB and OOMs.
  * For the inference-quotient analysis we only need, per layer/head:
        - q at the LAST token (decode-style query)          [H, d]
        - K, V for all tokens at native GQA head count       [Hkv, T, d]
    and we recompute logits = q_last @ K^T ourselves.

How: switch the model to SDPA (PyTorch scaled_dot_product_attention, which
tiles and does NOT retain an [H,T,T] score matrix) and register a custom
attention interface that calls through to SDPA for the real (correct) hidden
states while recording only q_last + native K/V. GQA mapping is preserved
(query head h reads KV head h // groups), identical to capture_qkv.py, and is
re-verified against SDPA's own last-token output.

Returns the SAME cap dict schema as capture_qkv.capture_prompt so downstream
code (build_dataset, run_experiments) is unchanged.
"""
from __future__ import annotations
import os, numpy as np, torch

_CAP = []
_REGISTERED = False


def _register():
    global _REGISTERED
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as orig_sdpa

    def patched(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, is_causal=None, **kw):
        out, _ = orig_sdpa(module, query, key, value, attention_mask,
                           dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)
        # out: [B, T, H, d] (orig transposes before returning)
        _CAP.append(dict(
            layer_idx=int(getattr(module, "layer_idx", len(_CAP))),
            groups=int(getattr(module, "num_key_value_groups", 1)),
            scaling=float(scaling),
            q_last=query[:, :, -1, :].detach().to(torch.float32).cpu().numpy()[0],   # [H,d]
            k=key.detach().to(torch.float32).cpu().numpy()[0],                        # [Hkv,T,d]
            v=value.detach().to(torch.float32).cpu().numpy()[0],                      # [Hkv,T,d]
            out_last=out[:, -1, :, :].detach().to(torch.float32).cpu().numpy()[0],    # [H,d]
        ))
        return out, None

    ALL_ATTENTION_FUNCTIONS["sdpa_capture"] = patched
    _REGISTERED = True


def load_model(model_name, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not _REGISTERED:
        _register()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa_capture")
    model.eval()
    return model, tok


def _chat(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_text


@torch.no_grad()
def capture_prompt(model, tok, text, max_tokens=8192):
    """One forward pass; return cap dict with last-token Q and full K/V per layer."""
    _CAP.clear()
    prompt = _chat(tok, text)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    T = enc["input_ids"].shape[1]
    # prefer non-materializing CPU/GPU backends; MATH is the fallback that builds scores
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                           SDPBackend.MATH])
    except Exception:
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:
        model(**enc, use_cache=False)
    layers = sorted(_CAP, key=lambda r: r["layer_idx"])
    seen, uniq = set(), []
    for r in layers:
        if r["layer_idx"] in seen:
            continue
        seen.add(r["layer_idx"]); uniq.append(r)
    layers = uniq
    H = layers[0]["q_last"].shape[0]; Hkv = layers[0]["k"].shape[0]
    groups = layers[0]["groups"]; d = layers[0]["q_last"].shape[1]
    out = dict(T=T, n_layers=len(layers), n_heads=H, n_kv_heads=Hkv, groups=groups,
               d=d, scaling=layers[0]["scaling"], query_positions=[T - 1], layers=[])
    for r in layers:
        out["layers"].append(dict(
            layer_idx=r["layer_idx"],
            Q=r["q_last"][None, :, :].copy(),   # [1,H,d]
            K=r["k"].copy(), V=r["v"].copy(),   # [Hkv,T,d]
            out=r["out_last"][None, :, :].copy()))  # [1,H,d]
    return out


def verify_capture(cap, tol=6e-3):  # bf16 forward => ~1-3e-3 rounding is expected
    """Recompute last-token attention for a sampled head; compare to SDPA output."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from attention_ops import full_attention
    L = cap["layers"][len(cap["layers"]) // 2]
    h = cap["n_heads"] // 2
    kv = h // cap["groups"]
    q = L["Q"][0, h]
    K = L["K"][kv]; V = L["V"][kv]
    o, _, _ = full_attention(q, K, V, cap["scaling"])
    ref = L["out"][0, h]
    err = float(np.linalg.norm(o - ref) / (np.linalg.norm(ref) + 1e-9))
    return err, err < tol
