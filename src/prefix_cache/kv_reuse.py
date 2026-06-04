"""Real `past_key_values` reuse engine (CPU, actual tensors — not metadata).

Validated: reusing a prefill's KV cache and forwarding only the tail reproduces
the full forward's tail logits to float precision (max diff ~1e-5, argmax 1.0).

Canonical reuse semantics: we cache the KV of the *canonical* prefix. A request
whose stable context differs only by volatile surface noise canonicalizes to the
same key, so it reuses the canonical KV and is served the canonical-prefix
answer. That is EXACT w.r.t. the canonical context (verified here); whether
collapsing those surface forms is *semantically* safe is the job of the
false-hit audit (mutations stress test), not this engine.
"""
from __future__ import annotations
import copy, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class KVReuseEngine:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="eager").eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def full_forward(self, text):
        x = self.ids(text)
        t0 = time.time()
        out = self.model(x)
        return out.logits, x.shape[1], time.time() - t0

    @torch.no_grad()
    def prefill(self, prefix_text):
        """Compute and return (cache, prefix_len, seconds) for a prefix."""
        x = self.ids(prefix_text)
        cache = DynamicCache()
        t0 = time.time()
        self.model(x, use_cache=True, past_key_values=cache)
        return cache, x.shape[1], time.time() - t0

    @staticmethod
    def clone(cache):
        return copy.deepcopy(cache)

    @torch.no_grad()
    def reuse_forward(self, cache, prefix_len, tail_text):
        """Forward only the tail on top of a cloned prefix cache. Returns
        (tail_logits, tail_len, seconds)."""
        t = self.ids(tail_text)
        Lt = t.shape[1]
        c = self.clone(cache)
        cpos = torch.arange(prefix_len, prefix_len + Lt, device=self.device)
        am = torch.ones(1, prefix_len + Lt, dtype=torch.long, device=self.device)
        t0 = time.time()
        out = self.model(t, past_key_values=c, use_cache=True,
                         attention_mask=am, cache_position=cpos)
        return out.logits, Lt, time.time() - t0

    @torch.no_grad()
    def forward_store(self, prefix_text, tail_text):
        """A cache MISS in a real server: prefill the whole prompt once (the KV is
        a byproduct), and keep the prefix-cropped cache for future reuse. Returns
        (logits, prefix_len, prefix_cache, seconds). This is the fair miss cost —
        one full forward, equal to stateless."""
        pl = self.ids(prefix_text).shape[1]
        x = self.ids(prefix_text + tail_text)
        cache = DynamicCache()
        t0 = time.time()
        out = self.model(x, use_cache=True, past_key_values=cache)
        dt = time.time() - t0
        pref = copy.deepcopy(cache)
        pref.crop(pl)                      # keep only the reusable prefix KV
        return out.logits, pl, pref, dt

    @torch.no_grad()
    def correctness(self, prefix_text, tail_text):
        """Max/mean logit diff between full(prefix+tail) and prefix-KV-reuse+tail."""
        full, _, _ = self.full_forward(prefix_text + tail_text)
        pl = self.ids(prefix_text).shape[1]
        cache, pl2, _ = self.prefill(prefix_text)
        re, Lt, _ = self.reuse_forward(cache, pl2, tail_text)
        a = full[0, pl:pl + Lt, :]
        b = re[0, :, :]
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        return dict(max_logit_diff=float((a - b).abs().max()),
                    mean_logit_diff=float((a - b).abs().mean()),
                    argmax_agree=float((a.argmax(-1) == b.argmax(-1)).float().mean()))
