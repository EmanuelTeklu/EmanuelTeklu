"""Adversarial mutations for the canonicalization correctness stress test.

Model of reality: every request carries volatile surface noise (a fresh request
id/timestamp, possibly reordered tools, reformatted JSON, jittered doc refs).
So the meaningful tests are:

  SHOULD-HIT: two different *surface renderings* of the SAME semantic context must
    canonicalize EQUAL (else a false miss -> lost savings, tolerable).
  SHOULD-MISS: a semantic change must canonicalize DIFFERENT (else a **false hit**
    -> wrong state served, FATAL).

`surface(ctx, seed)` applies all surface-noise dimensions with a given seed.
Per-dimension noisers are provided for diagnostics. Semantic mutations change
meaning and must break the key.
"""
from __future__ import annotations
import copy, random


def _c(ctx):
    return copy.deepcopy(ctx)


def _vol_header(rng):
    rid = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    ts = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z"
    return f"req_id={rid} ts={ts} "


# ---- per-dimension surface noisers (should-hit; parameterized by seed) ----
def n_volatile(ctx, rng):
    c = _c(ctx); c.system = _vol_header(rng) + c.system; return c

def n_tool_order(ctx, rng):
    c = _c(ctx); rng.shuffle(c.tools); return c

def n_json_order(ctx, rng):
    c = _c(ctx)
    c.tools = [{k: t[k] for k in rng.sample(list(t), len(t))} for t in c.tools]
    return c

def n_whitespace(ctx, rng):
    c = _c(ctx)
    c.system = (" " * rng.randint(1, 4)).join(c.system.split()) + "\n" * rng.randint(0, 2)
    c.tools = [dict(t, description=" " * rng.randint(0, 3) + str(t.get("description", "")))
               for t in c.tools]
    return c

def n_doc_meta(ctx, rng):
    c = _c(ctx)
    c.docs = [(ref.split("?")[0] + f"?v={rng.randint(1,999)}&etag={rng.randint(0,9999)}", content)
              for ref, content in c.docs]
    return c

DIMENSIONS = {"volatile_id": n_volatile, "tool_order": n_tool_order,
              "json_field_order": n_json_order, "whitespace": n_whitespace,
              "doc_metadata": n_doc_meta}


def surface(ctx, seed):
    """Apply all surface-noise dimensions with a given seed (a full rendering)."""
    rng = random.Random(seed)
    c = ctx
    for fn in (n_doc_meta, n_json_order, n_tool_order, n_whitespace, n_volatile):
        c = fn(c, rng)
    return c


# ---- semantic mutations (should-miss) ----
def s_tool_semantic(ctx):
    c = _c(ctx)
    if c.tools:
        c.tools[0] = dict(c.tools[0], description="DELETES all files permanently")
    return c

def s_tool_param_type(ctx):
    c = _c(ctx)
    if c.tools:
        a = dict(c.tools[0].get("args", {}))
        if a:
            k = list(a)[0]; a[k] = "int" if a[k] != "int" else "float"
        else:
            a = {"x": "int"}
        c.tools[0] = dict(c.tools[0], args=a)
    return c

def s_doc_content(ctx):
    c = _c(ctx)
    if c.docs:
        ref, content = c.docs[0]
        c.docs[0] = (ref, content + " NEW CLAUSE: liability is unlimited and perpetual.")
    return c

def s_memory_version(ctx):
    c = _c(ctx)
    if c.memory:
        c.memory[0] = dict(c.memory[0], version=int(c.memory[0].get("version", 1)) + 1,
                           content=str(c.memory[0].get("content", "")) + " UPDATED CONSTRAINT")
    return c

def s_system_policy(ctx):
    c = _c(ctx); c.system = c.system + " NEW POLICY: never use external tools."; return c

def s_doc_replace(ctx):
    c = _c(ctx)
    if c.docs:
        ref, _ = c.docs[0]
        c.docs[0] = (ref, "Entirely different document body about unrelated topics. " * 30)
    return c

SEMANTIC = {"tool_semantic_change": s_tool_semantic, "tool_param_type": s_tool_param_type,
            "doc_content_change": s_doc_content, "memory_version_bump": s_memory_version,
            "system_policy_change": s_system_policy, "doc_replaced": s_doc_replace}
