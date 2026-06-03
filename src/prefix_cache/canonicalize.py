"""Canonicalization of an agent context's reusable prefix.

The reusable prefix of an agent request = system prompt + tool schemas + pinned
documents + memory blocks (everything BEFORE the changing user tail). Two
identical-in-meaning prefixes often differ byte-for-byte due to:
  * volatile IDs / timestamps embedded in the system header or memory
  * tool definitions emitted in a different order or with different whitespace
  * documents referenced by an unstable path/id rather than by content
  * unversioned memory drift

Canonicalization maps meaning-equivalent prefixes to the SAME bytes (hence the
same cache key), while keeping genuinely different prefixes distinct and letting
real updates (memory edits) invalidate via an explicit version.

A `Context` is a structured request; `canonical_prefix_bytes()` is what we hash.
"""
from __future__ import annotations
import re, json, hashlib
from dataclasses import dataclass, field
from typing import Any

# volatile patterns: UUIDs, ISO-ish timestamps, long hex, request/trace ids, epochs
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_TS = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b")
_HEX = re.compile(r"\b[0-9a-f]{16,}\b")
_REQID = re.compile(r"\b(?:req|trace|span|session)[-_]?id[=:]\s*\S+", re.I)
_EPOCH = re.compile(r"\b1[0-9]{9}\b")
_WS = re.compile(r"\s+")


def strip_volatile(text: str) -> str:
    """Replace volatile IDs/timestamps with stable placeholders."""
    text = _UUID.sub("<UUID>", text)
    text = _TS.sub("<TS>", text)
    text = _REQID.sub("<REQID>", text)
    text = _EPOCH.sub("<EPOCH>", text)
    text = _HEX.sub("<HEX>", text)
    return text


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def canonical_system(system: str) -> str:
    return normalize_ws(strip_volatile(system))


def normalize_tool(tool: dict) -> dict:
    """Recursively sort keys; normalize string whitespace; strip volatile."""
    def norm(v):
        if isinstance(v, dict):
            return {k: norm(v[k]) for k in sorted(v)}
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, str):
            return normalize_ws(strip_volatile(v))
        return v
    return norm(tool)


def canonical_tools(tools: list[dict]) -> list[dict]:
    """Sort tool definitions by name, normalize each (order-invariant)."""
    normed = [normalize_tool(t) for t in tools]
    return sorted(normed, key=lambda t: t.get("name", json.dumps(t, sort_keys=True)))


def doc_hash(content: str) -> str:
    """Stable content hash for a pinned document (path/id-independent)."""
    return "doc:" + hashlib.sha256(normalize_ws(content).encode()).hexdigest()[:16]


def canonical_docs(docs: list[tuple[str, str]]) -> list[str]:
    """docs = list of (ref, content). Represent by content hash, sorted -> a doc
    set referenced by different paths but same content canonicalizes identically."""
    return sorted(doc_hash(content) for _ref, content in docs)


def canonical_memory(memory: list[dict]) -> list[tuple]:
    """memory blocks = list of {block_id, version, content}. Key by
    (block_id, version, content_hash); a real edit bumps version -> new key
    (legitimate invalidation), but re-serialization noise does not."""
    out = []
    for b in memory:
        ch = hashlib.sha256(normalize_ws(strip_volatile(str(b.get("content", "")))).encode()).hexdigest()[:12]
        out.append((str(b.get("block_id", "")), int(b.get("version", 0)), ch))
    return sorted(out)


@dataclass
class Context:
    system: str = ""
    tools: list[dict] = field(default_factory=list)
    docs: list[tuple[str, str]] = field(default_factory=list)   # (ref, content)
    memory: list[dict] = field(default_factory=list)            # {block_id,version,content}
    user: str = ""                                              # changing tail (NOT reusable)

    def reusable_text(self) -> str:
        """Raw (non-canonical) serialization of the reusable prefix."""
        parts = [self.system]
        parts += [json.dumps(t) for t in self.tools]
        parts += [f"{ref}:{content}" for ref, content in self.docs]
        parts += [json.dumps(b) for b in self.memory]
        return "\n".join(parts)

    def full_text(self) -> str:
        return self.reusable_text() + "\nUSER:" + self.user

    # --- token accounting (char/4 approximation, documented) ---
    def reusable_tokens(self) -> int:
        return max(1, len(self.reusable_text()) // 4)

    def total_tokens(self) -> int:
        return max(1, len(self.full_text()) // 4)


def exact_prefix_bytes(ctx: Context) -> bytes:
    return ctx.reusable_text().encode()


def canonical_prefix_bytes(ctx: Context, transforms: set[str] | None = None) -> bytes:
    """Canonical serialization of the reusable prefix. `transforms` selects which
    canonicalizations are applied (for ablation); None = all."""
    t = transforms if transforms is not None else {
        "system", "tools", "docs", "memory"}
    sys_ = canonical_system(ctx.system) if "system" in t else ctx.system
    tools_ = canonical_tools(ctx.tools) if "tools" in t else ctx.tools
    docs_ = canonical_docs(ctx.docs) if "docs" in t else [f"{r}:{c}" for r, c in ctx.docs]
    mem_ = canonical_memory(ctx.memory) if "memory" in t else ctx.memory
    obj = {"system": sys_, "tools": tools_, "docs": docs_, "memory": mem_}
    return json.dumps(obj, sort_keys=True, default=str).encode()


def prefix_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:24]
