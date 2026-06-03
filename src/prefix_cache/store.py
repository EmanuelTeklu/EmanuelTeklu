"""KV / latent object-store abstraction for persistent inference state.

Stores *metadata* about cached prefix KV/latent objects (no real GPU KV yet —
the point of this track is the addressing + economics, not the tensor blob).

Key = (model_id, tokenizer_id, canonical_context_hash, version).
Value = PrefixObject metadata (token_count, bytes estimate, hit count, ...).

Versioning: a memory edit (or explicit bump) changes the canonical hash via the
memory block version, so stale objects are naturally not found. We also support
explicit invalidation by (model, tokenizer, prefix_hash) to model TTL / eviction.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class PrefixObject:
    key: tuple
    token_count: int
    kv_bytes_est: int           # placeholder: 2 * n_layers * n_kv_heads * d * tok * dtype
    created_at: float
    hits: int = 0


@dataclass
class StoreStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    tokens_served_from_cache: int = 0
    tokens_recomputed: int = 0


class PrefixStore:
    def __init__(self, capacity: int | None = None):
        self._d: dict[tuple, PrefixObject] = {}
        self.capacity = capacity
        self.stats = StoreStats()

    @staticmethod
    def make_key(model_id: str, tokenizer_id: str, prefix_hash: str, version: int = 0):
        return (model_id, tokenizer_id, prefix_hash, version)

    def get(self, key) -> PrefixObject | None:
        self.stats.lookups += 1
        obj = self._d.get(key)
        if obj is None:
            self.stats.misses += 1
            return None
        obj.hits += 1
        self.stats.hits += 1
        self.stats.tokens_served_from_cache += obj.token_count
        return obj

    def put(self, key, token_count: int, kv_bytes_est: int = 0) -> PrefixObject:
        obj = PrefixObject(key=key, token_count=token_count, kv_bytes_est=kv_bytes_est,
                           created_at=time.time())
        self._d[key] = obj
        self.stats.tokens_recomputed += token_count
        if self.capacity and len(self._d) > self.capacity:    # naive LRU-ish evict
            oldest = min(self._d.values(), key=lambda o: o.created_at)
            self._d.pop(oldest.key, None)
        return obj

    def invalidate_prefix(self, model_id, tokenizer_id, prefix_hash):
        """Drop all versions of a prefix (TTL / external edit)."""
        drop = [k for k in self._d if k[0] == model_id and k[1] == tokenizer_id
                and k[2] == prefix_hash]
        for k in drop:
            self._d.pop(k, None)
            self.stats.invalidations += 1

    def hit_rate(self) -> float:
        return self.stats.hits / self.stats.lookups if self.stats.lookups else 0.0
