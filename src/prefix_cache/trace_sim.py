"""Synthetic agent-trace generator producing real Context objects.

Models the conditions that matter for prefix reuse:
  * a small library of tool sets and document sets shared ACROSS sessions
    (cross-session canonical reuse);
  * within a session the prefix grows / repeats while the user tail changes
    (within-session reuse);
  * per-session VOLATILE noise (request ids, timestamps), tool REORDERING, and
    DOC-PATH changes that break EXACT matching but NOT canonical;
  * MEMORY INVALIDATION events that legitimately bump a memory block version
    (canonical key must change -> a real miss).
"""
from __future__ import annotations
import random
from .canonicalize import Context

TOOLSETS = {
    "search_calc": [
        {"name": "search", "description": "search the knowledge base", "args": {"query": "str"}},
        {"name": "calculate", "description": "evaluate an expression", "args": {"expr": "str"}},
    ],
    "files_fetch": [
        {"name": "fetch", "description": "fetch a url", "args": {"url": "str"}},
        {"name": "write_file", "description": "persist a file", "args": {"path": "str", "content": "str"}},
        {"name": "read_file", "description": "read a file", "args": {"path": "str"}},
    ],
}
DOCSETS = {
    "policy": [("doc/policy_v1.md", "The logistics policy mandates dual-sourcing for critical parts. " * 30)],
    "api": [("ref/api.md", "GET /v1/items returns a paged list of items with cursors. " * 30),
            ("ref/auth.md", "Auth uses bearer tokens scoped per tenant. " * 25)],
}
SYS_BASE = ("You are a careful agent. Think step by step and use tools only when "
            "necessary. Always cite sources.")


def _volatile_header(rng):
    rid = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    ts = f"2026-0{rng.randint(1,9)}-{rng.randint(10,28)}T{rng.randint(10,23)}:00:00Z"
    return f"req_id={rid} ts={ts} "


def generate(n_sessions=50, turns=8, seed=0, mem_invalidation_p=0.12,
             exact_noise=True):
    rng = random.Random(seed)
    toolset_keys = list(TOOLSETS); docset_keys = list(DOCSETS)
    requests = []  # list of (session, turn, Context, mem_event)
    for s in range(n_sessions):
        tk = rng.choice(toolset_keys); dk = rng.choice(docset_keys)
        tools = [dict(t) for t in TOOLSETS[tk]]
        docs = list(DOCSETS[dk])
        # exact-breaking noise that canonicalization fixes:
        if exact_noise:
            rng.shuffle(tools)                                  # tool reordering
            docs = [(f"{ref}?v={rng.randint(1,9)}", c) for ref, c in docs]  # path noise
        memory = [{"block_id": "profile", "version": 1,
                   "content": "user prefers concise answers"}]
        for t in range(turns):
            # per-REQUEST volatile header (fresh request id/timestamp each turn) ->
            # exact matching breaks every turn; canonicalization strips it.
            sys_noise = _volatile_header(rng) if exact_noise else ""
            mem_event = rng.random() < mem_invalidation_p and t > 0
            if mem_event:
                memory = [dict(memory[0], version=memory[0]["version"] + 1,
                               content="user prefers concise answers; new constraint added")]
            ctx = Context(system=sys_noise + SYS_BASE, tools=[dict(x) for x in tools],
                          docs=list(docs), memory=[dict(memory[0])],
                          user=f"question {s}-{t}: " + rng.choice(
                              ["summarize", "compute 3*47", "find the policy", "fetch the ref"]))
            requests.append((s, t, ctx, mem_event))
    return requests
