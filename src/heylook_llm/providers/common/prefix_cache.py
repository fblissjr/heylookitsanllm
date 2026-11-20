"""Heylook-owned Copy-on-Write single-row (B=1) MLX prefix cache.

Stores immutable MLX array slice views across standard KV, sliding-window
rotating KV, buffered rotating KV, and hybrid recurrent ``ArraysCache``
layers without synchronous ``mx.contiguous()`` copies or per-layer
``mx.eval()`` barriers on lookup or prefill store. Oversized rotating views
pinned by large prefill chunks are detached lazily after generation ends via
:meth:`PrefixCacheStore.compact_pending`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx


def _cache_classes():
    from mlx_vlm.models.cache import (
        ArraysCache,
        BufferedRotatingKVCache,
        CacheList,
        KVCache,
        RotatingKVCache,
    )

    return KVCache, RotatingKVCache, BufferedRotatingKVCache, ArraysCache, CacheList


def cache_layout_supported(caches: Sequence[Any]) -> bool:
    """True when every entry in ``caches`` uses supported B=1 CoW cache types."""
    if not caches:
        return False
    KVCache, RotatingKVCache, BufferedRotatingKVCache, ArraysCache, CacheList = (
        _cache_classes()
    )

    def _ok(c: Any) -> bool:
        t = type(c)
        if t is KVCache:
            return True
        if t is BufferedRotatingKVCache or t is RotatingKVCache:
            return int(getattr(c, "keep", 0) or 0) == 0 and int(
                getattr(c, "max_size", 0) or 0
            ) > 0
        if t is ArraysCache:
            return (
                not getattr(c, "is_speculating", False)
                and getattr(c, "_left_padding", None) is None
                and getattr(c, "_lengths", None) is None
                and int(getattr(c, "batch_size", 1)) == 1
            )
        if t is CacheList:
            children = getattr(c, "caches", ())
            return bool(children) and all(_ok(ch) for ch in children)
        return False

    return all(_ok(c) for c in caches)


def cache_has_recurrent(caches: Sequence[Any]) -> bool:
    """True when any leaf in ``caches`` is a non-trimmable ``ArraysCache``."""
    _, _, _, ArraysCache, CacheList = _cache_classes()

    def _has(c: Any) -> bool:
        t = type(c)
        if t is ArraysCache:
            return True
        if t is CacheList:
            return any(_has(ch) for ch in getattr(c, "caches", ()))
        return False

    return any(_has(c) for c in caches)


def _array_view(x: Optional[mx.array]) -> Optional[mx.array]:
    if x is None:
        return None
    return x[...]


def _nbytes(arr: Optional[mx.array]) -> int:
    if arr is None:
        return 0
    return int(arr.nbytes)


_ROTATING_SLACK_TOKENS = 16


@dataclass
class _LayerSnapshot:
    kind: str  # "kv", "rotating", "arrays", "cache_list"
    keys: Optional[mx.array] = None
    values: Optional[mx.array] = None
    offset: int = 0
    idx: int = 0
    max_size: int = 0
    keep: int = 0
    min_offset: int = 0
    needs_compact: bool = False
    arrays: Optional[list[Optional[mx.array]]] = None
    children: Optional[tuple["_LayerSnapshot", ...]] = None

    @property
    def trimmable(self) -> bool:
        return self.min_offset == 0

    @property
    def logical_bytes(self) -> int:
        if self.kind in ("kv", "rotating"):
            return _nbytes(self.keys) + _nbytes(self.values)
        if self.kind == "arrays" and self.arrays is not None:
            return sum(_nbytes(a) for a in self.arrays)
        if self.kind == "cache_list" and self.children is not None:
            return sum(ch.logical_bytes for ch in self.children)
        return 0


def _snapshot_rotating_slices(
    ok: Optional[mx.array],
    ov: Optional[mx.array],
    *,
    backing_len: int,
    valid: int,
    offset: int,
    max_size: int,
    keep: int,
    trim_tail: int,
    slack_cap: int = _ROTATING_SLACK_TOKENS,
) -> Optional[_LayerSnapshot]:
    snap_off = offset - trim_tail
    if keep != 0 or max_size <= 0 or trim_tail < 0 or snap_off < 0:
        return None
    if ok is None or ov is None or valid == 0 or snap_off == 0:
        if snap_off != 0:
            return None
        return _LayerSnapshot(
            kind="rotating",
            keys=None,
            values=None,
            offset=0,
            idx=0,
            max_size=max_size,
            keep=0,
            min_offset=0,
            needs_compact=False,
        )

    end = valid - trim_tail
    if end <= 0:
        return None

    if snap_off <= max_size:
        if end < snap_off:
            return None
        start = end - snap_off
        k = ok[..., start:end, :]
        v = ov[..., start:end, :]
        snap_idx = snap_off
        min_offset = 0
    else:
        if end < max_size:
            return None
        take = min(end, max_size + max(0, int(slack_cap)))
        start = end - take
        k = ok[..., start:end, :]
        v = ov[..., start:end, :]
        snap_idx = take
        min_offset = snap_off - (take - max_size)

    return _LayerSnapshot(
        kind="rotating",
        keys=k,
        values=v,
        offset=snap_off,
        idx=snap_idx,
        max_size=max_size,
        keep=0,
        min_offset=min_offset,
        needs_compact=(k is not None and backing_len > snap_idx + 256),
    )


def _snapshot_layer(
    c: Any,
    *,
    trim_tail: int = 0,
    target_len: int = 0,
    slack_cap: int = _ROTATING_SLACK_TOKENS,
) -> Optional[_LayerSnapshot]:
    KVCache, RotatingKVCache, BufferedRotatingKVCache, ArraysCache, CacheList = (
        _cache_classes()
    )
    if trim_tail < 0:
        return None

    t = type(c)
    if t is KVCache:
        off = int(c.offset)
        end = off - trim_tail
        if end < 0:
            return None
        if c.keys is None or c.values is None or end == 0:
            if end != 0:
                return None
            return _LayerSnapshot(
                kind="kv",
                keys=None,
                values=None,
                offset=0,
                idx=0,
                min_offset=0,
                needs_compact=False,
            )
        if int(c.keys.shape[2]) < end:
            return None
        k = c.keys[..., :end, :]
        v = c.values[..., :end, :]
        return _LayerSnapshot(
            kind="kv",
            keys=k,
            values=v,
            offset=end,
            idx=end,
            min_offset=0,
            needs_compact=False,
        )

    if t is BufferedRotatingKVCache:
        if c.keys is None or c.values is None:
            return _snapshot_rotating_slices(
                None,
                None,
                backing_len=0,
                valid=0,
                offset=int(c.offset),
                max_size=int(c.max_size),
                keep=int(getattr(c, "keep", 0) or 0),
                trim_tail=trim_tail,
                slack_cap=slack_cap,
            )
        return _snapshot_rotating_slices(
            c.keys,
            c.values,
            backing_len=int(c.keys.shape[2]),
            valid=int(c._idx),
            offset=int(c.offset),
            max_size=int(c.max_size),
            keep=int(getattr(c, "keep", 0) or 0),
            trim_tail=trim_tail,
            slack_cap=slack_cap,
        )

    if t is RotatingKVCache:
        if c.keys is None or c.values is None:
            return _snapshot_rotating_slices(
                None,
                None,
                backing_len=0,
                valid=0,
                offset=int(c.offset),
                max_size=int(c.max_size),
                keep=int(getattr(c, "keep", 0) or 0),
                trim_tail=trim_tail,
                slack_cap=slack_cap,
            )
        ok = c._temporal_order(c.keys)
        ov = c._temporal_order(c.values)
        backing_len = max(int(c.keys.shape[2]), int(ok.shape[2]))
        return _snapshot_rotating_slices(
            ok,
            ov,
            backing_len=backing_len,
            valid=int(ok.shape[2]),
            offset=int(c.offset),
            max_size=int(c.max_size),
            keep=int(getattr(c, "keep", 0) or 0),
            trim_tail=trim_tail,
            slack_cap=slack_cap,
        )

    if t is ArraysCache:
        if trim_tail != 0:
            return None
        if (
            getattr(c, "is_speculating", False)
            or getattr(c, "_left_padding", None) is not None
            or getattr(c, "_lengths", None) is not None
            or int(getattr(c, "batch_size", 1)) != 1
        ):
            return None
        return _LayerSnapshot(
            kind="arrays",
            min_offset=int(target_len),
            needs_compact=False,
            arrays=[_array_view(a) for a in c.cache],
        )

    if t is CacheList:
        children = []
        for ch in c.caches:
            snap = _snapshot_layer(
                ch,
                trim_tail=trim_tail,
                target_len=target_len,
                slack_cap=slack_cap,
            )
            if snap is None:
                return None
            children.append(snap)
        return _LayerSnapshot(
            kind="cache_list",
            min_offset=max((ch.min_offset for ch in children), default=0),
            needs_compact=any(ch.needs_compact for ch in children),
            children=tuple(children),
        )

    return None


def snapshot_rotating_fallbacks(
    prompt_cache: Sequence[Any],
    *,
    target_len: int,
    max_tokens: int,
    slack_cap: int = _ROTATING_SLACK_TOKENS,
) -> dict[int, _LayerSnapshot]:
    """Capture pre-generation snapshots only for rotating layers likely to evict prompt tail."""
    _, RotatingKVCache, BufferedRotatingKVCache, _, CacheList = _cache_classes()

    def _needs_fallback(entry: Any) -> bool:
        t = type(entry)
        if t is BufferedRotatingKVCache:
            return (
                int(entry.offset) + max_tokens > int(entry.max_size)
                and int(entry._idx) + max_tokens + 4
                > int(entry.max_size) + int(entry.buffer_size)
            )
        if t is RotatingKVCache:
            return int(entry.offset) + max_tokens > int(entry.max_size)
        if t is CacheList:
            return any(_needs_fallback(ch) for ch in entry.caches)
        return False

    fallbacks: dict[int, _LayerSnapshot] = {}
    for idx, c in enumerate(prompt_cache):
        if _needs_fallback(c):
            snap = _snapshot_layer(
                c, trim_tail=0, target_len=target_len, slack_cap=slack_cap
            )
            if snap is not None:
                fallbacks[idx] = snap
    return fallbacks


def capture_rotating_fallbacks_before_evict(
    prompt_cache: Sequence[Any],
    *,
    target_len: int,
    incoming: int,
    captured: dict[int, _LayerSnapshot],
    slack_cap: int = _ROTATING_SLACK_TOKENS,
) -> bool:
    """Capture pre-compaction rotating snapshots right before decode evicts prompt tail.

    Returns ``True`` while at least one rotating layer in ``prompt_cache`` still
    needs monitoring, or ``False`` once every rotating layer has been captured.
    """
    _, RotatingKVCache, BufferedRotatingKVCache, _, CacheList = _cache_classes()

    def _has_rotating(entry: Any) -> bool:
        t = type(entry)
        if t is BufferedRotatingKVCache or t is RotatingKVCache:
            return True
        if t is CacheList:
            return any(_has_rotating(ch) for ch in entry.caches)
        return False

    def _needs_capture(entry: Any) -> tuple[bool, int]:
        t = type(entry)
        if t is BufferedRotatingKVCache:
            if entry.keys is None or int(entry.offset) < target_len:
                return False, 0
            if int(entry._planned_drop(incoming)) > 0:
                return True, int(entry.offset) - target_len
            return False, 0
        if t is RotatingKVCache:
            if entry.keys is None or int(entry.offset) < target_len:
                return False, 0
            if int(entry.offset) + incoming > int(entry.max_size):
                return True, int(entry.offset) - target_len
            return False, 0
        if t is CacheList:
            for ch in entry.caches:
                should, tail = _needs_capture(ch)
                if should:
                    return True, tail
            return False, 0
        return False, 0

    to_capture: list[tuple[int, Any, int]] = []
    any_uncaptured = False
    for idx, c in enumerate(prompt_cache):
        if idx in captured or not _has_rotating(c):
            continue
        any_uncaptured = True
        should, trim_tail = _needs_capture(c)
        if should:
            to_capture.append((idx, c, trim_tail))

    if to_capture:
        mx.eval([c.state for _, c, _ in to_capture])
        for idx, c, trim_tail in to_capture:
            snap = _snapshot_layer(
                c, trim_tail=trim_tail, target_len=target_len, slack_cap=slack_cap
            )
            if snap is not None:
                captured[idx] = snap
        any_uncaptured = any(
            idx not in captured and _has_rotating(c)
            for idx, c in enumerate(prompt_cache)
        )

    return any_uncaptured


def _restore_layer(
    snap: _LayerSnapshot,
    *,
    match_len: int,
    use_buffered_rotating: bool,
    buffer_size: int = 64,
) -> Any:
    KVCache, RotatingKVCache, BufferedRotatingKVCache, ArraysCache, CacheList = (
        _cache_classes()
    )

    if snap.kind == "kv":
        c = KVCache()
        if snap.keys is not None and snap.values is not None and match_len > 0:
            if match_len < snap.offset:
                c.keys = snap.keys[..., :match_len, :]
                c.values = snap.values[..., :match_len, :]
            else:
                c.keys = snap.keys[...]
                c.values = snap.values[...]
            c.offset = match_len
        return c

    if snap.kind == "rotating":
        drop_tail = snap.offset - match_len
        if drop_tail > 0:
            end = snap.idx - drop_tail
            take = min(end, snap.max_size)
            start = end - take
            k = (
                snap.keys[..., start:end, :]
                if snap.keys is not None and take > 0
                else None
            )
            v = (
                snap.values[..., start:end, :]
                if snap.values is not None and take > 0
                else None
            )
            off = match_len
            idx = take
        else:
            take = min(snap.idx, snap.max_size)
            start = snap.idx - take
            k = (
                snap.keys[..., start : snap.idx, :]
                if snap.keys is not None and take > 0
                else None
            )
            v = (
                snap.values[..., start : snap.idx, :]
                if snap.values is not None and take > 0
                else None
            )
            off = snap.offset
            idx = take
        if use_buffered_rotating:
            c = BufferedRotatingKVCache(
                snap.max_size, keep=snap.keep, buffer_size=buffer_size
            )
            c.keys = k
            c.values = v
            c.offset = off
            c._idx = idx
            c.start_position = max(0, off - idx)
            return c
        c = RotatingKVCache(max_size=snap.max_size, keep=snap.keep)
        c.keys = k
        c.values = v
        c.offset = off
        c._idx = idx
        return c

    if snap.kind == "arrays":
        assert snap.arrays is not None
        c = ArraysCache(len(snap.arrays))
        c.cache = [_array_view(a) for a in snap.arrays]
        return c

    if snap.kind == "cache_list":
        assert snap.children is not None
        return CacheList(
            *[
                _restore_layer(
                    ch,
                    match_len=match_len,
                    use_buffered_rotating=use_buffered_rotating,
                    buffer_size=buffer_size,
                )
                for ch in snap.children
            ]
        )

    raise ValueError(f"Unsupported snapshot kind: {snap.kind!r}")


def _collect_compactions(
    snap: _LayerSnapshot,
    targets: list[mx.array],
    pending: list[tuple[_LayerSnapshot, mx.array, mx.array]],
) -> None:
    if snap.kind == "rotating" and snap.needs_compact:
        if snap.keys is not None and snap.values is not None:
            kc = mx.contiguous(snap.keys)
            vc = mx.contiguous(snap.values)
            targets.extend((kc, vc))
            pending.append((snap, kc, vc))
        else:
            snap.needs_compact = False
    elif snap.kind == "cache_list" and snap.children is not None:
        for ch in snap.children:
            _collect_compactions(ch, targets, pending)


def _refresh_compact_flag(snap: _LayerSnapshot) -> bool:
    if snap.kind == "cache_list" and snap.children is not None:
        snap.needs_compact = any(_refresh_compact_flag(ch) for ch in snap.children)
    return snap.needs_compact


@dataclass
class _CacheEntry:
    extra_hash: int
    token_ids: tuple[int, ...]
    layers: list[_LayerSnapshot]
    min_match_len: int
    nbytes: int
    rope_deltas: Optional[mx.array] = None
    needs_compact: bool = field(default=False)

    @property
    def trimmable_all(self) -> bool:
        return self.min_match_len == 0


def _longest_common_prefix(a: tuple[int, ...], b: tuple[int, ...], limit: int) -> int:
    end = min(len(a), len(b), limit)
    n = 0
    while n < end and a[n] == b[n]:
        n += 1
    return n


class PrefixCacheStore:
    """In-memory CoW prefix cache for single-row MLX inference."""

    def __init__(
        self,
        *,
        memory_max_bytes: int = 0,
        max_entries: int = 32,
        memory_reserve_bytes: int = 0,
    ) -> None:
        self.memory_max_bytes = max(0, int(memory_max_bytes))
        self.max_entries = max(1, int(max_entries))
        self.memory_reserve_bytes = max(0, int(memory_reserve_bytes))
        self._lock = threading.RLock()
        self._entries: OrderedDict[tuple[int, tuple[int, ...]], _CacheEntry] = (
            OrderedDict()
        )
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.stored = 0
        self.memory_evictions = 0

    def is_empty(self) -> bool:
        with self._lock:
            return not self._entries

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    def stats_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_bytes": self._total_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "stored": self.stored,
                "memory_evictions": self.memory_evictions,
            }

    def touch(
        self,
        token_ids: Sequence[int],
        boundaries: Sequence[int],
        *,
        extra_hash: int = 0,
    ) -> None:
        if not boundaries:
            return
        token_tuple = tuple(int(t) for t in token_ids)
        with self._lock:
            for n in boundaries:
                key = (extra_hash, token_tuple[: int(n)])
                if key in self._entries:
                    self._entries.move_to_end(key)

    def _evict_key_locked(self, key: tuple[int, tuple[int, ...]]) -> None:
        removed = self._entries.pop(key, None)
        if removed is not None:
            self._total_bytes = max(0, self._total_bytes - removed.nbytes)

    def _make_room_locked(self, needed_bytes: int) -> None:
        while len(self._entries) > self.max_entries:
            oldest_key, _ = next(iter(self._entries.items()))
            self._evict_key_locked(oldest_key)
        if self.memory_max_bytes > 0:
            while (
                self._entries
                and self._total_bytes + needed_bytes > self.memory_max_bytes
            ):
                oldest_key, _ = next(iter(self._entries.items()))
                self._evict_key_locked(oldest_key)
                self.memory_evictions += 1

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
        suffix_is_text_only: Optional[Callable[[int], bool]] = None,
        use_buffered_rotating: bool = False,
        buffer_size: int = 64,
    ) -> tuple[Optional[list[Any]], int, Optional[mx.array]]:
        token_tuple = tuple(int(t) for t in token_ids)
        n = len(token_tuple)
        max_len = n - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= max(0, int(min_prefix_tokens)):
            return None, 0, None

        with self._lock:
            best_key: Optional[tuple[int, tuple[int, ...]]] = None
            best_entry: Optional[_CacheEntry] = None
            best_len = max(0, int(min_prefix_tokens))

            for key, entry in self._entries.items():
                if entry.extra_hash != extra_hash:
                    continue
                stored = entry.token_ids
                if entry.min_match_len < len(stored):
                    cand = _longest_common_prefix(token_tuple, stored, max_len)
                    if cand < entry.min_match_len or cand <= best_len:
                        continue
                    if suffix_is_text_only is not None:
                        while (
                            cand >= entry.min_match_len
                            and cand > best_len
                            and not suffix_is_text_only(cand)
                        ):
                            cand -= 1
                    if cand < entry.min_match_len or cand <= best_len:
                        continue
                else:
                    cand = len(stored)
                    if cand <= best_len or cand > max_len:
                        continue
                    if token_tuple[:cand] != stored:
                        continue
                    if suffix_is_text_only is not None and not suffix_is_text_only(cand):
                        continue
                best_key = key
                best_entry = entry
                best_len = cand

            if best_entry is None or best_key is None or best_len <= 0:
                self.misses += 1
                return None, 0, None

            self._entries.move_to_end(best_key)
            self.hits += 1
            restored = [
                _restore_layer(
                    layer_snap,
                    match_len=best_len,
                    use_buffered_rotating=use_buffered_rotating,
                    buffer_size=buffer_size,
                )
                for layer_snap in best_entry.layers
            ]
            rd = _array_view(best_entry.rope_deltas)
            return restored, best_len, rd

    def store(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        trim_tail: int = 0,
        cache_trim_tail: Optional[int] = None,
        rope_deltas: Optional[mx.array] = None,
        layer_fallbacks: Optional[dict[int, _LayerSnapshot]] = None,
    ) -> bool:
        target_len = len(token_ids) - int(trim_tail)
        kv_trim = int(trim_tail) if cache_trim_tail is None else int(cache_trim_tail)
        if target_len <= 0 or kv_trim < 0 or not prompt_cache:
            return False

        layers: list[_LayerSnapshot] = []
        for idx, c in enumerate(prompt_cache):
            snap = _snapshot_layer(
                c, trim_tail=kv_trim, target_len=target_len
            )
            if snap is None and layer_fallbacks is not None:
                fb = layer_fallbacks.get(idx)
                if fb is not None and (fb.offset == target_len or fb.kind == "arrays"):
                    snap = fb
            if snap is None:
                return False
            layers.append(snap)

        rd_view = _array_view(rope_deltas)
        entry_bytes = sum(s.logical_bytes for s in layers) + _nbytes(rd_view)
        if self.memory_max_bytes > 0 and entry_bytes > self.memory_max_bytes:
            return False

        key = (extra_hash, tuple(int(t) for t in token_ids[:target_len]))
        entry = _CacheEntry(
            extra_hash=extra_hash,
            token_ids=key[1],
            layers=layers,
            min_match_len=max((s.min_offset for s in layers), default=0),
            nbytes=entry_bytes,
            rope_deltas=rd_view,
            needs_compact=any(s.needs_compact for s in layers),
        )

        with self._lock:
            self._evict_key_locked(key)
            self._make_room_locked(entry_bytes)
            if len(self._entries) >= self.max_entries:
                oldest_key, _ = next(iter(self._entries.items()))
                self._evict_key_locked(oldest_key)
            self._entries[key] = entry
            self._total_bytes += entry_bytes
            self.stored += 1
            return True

    def compact_pending(self) -> int:
        """Detach oversized chunk-prefill rotating slice views post-generation."""
        targets: list[mx.array] = []
        pending: list[tuple[_LayerSnapshot, mx.array, mx.array]] = []
        compacted_entries: list[_CacheEntry] = []

        with self._lock:
            for entry in self._entries.values():
                if not entry.needs_compact:
                    continue
                compacted_entries.append(entry)
                for layer_snap in entry.layers:
                    _collect_compactions(layer_snap, targets, pending)

            if not targets:
                for entry in compacted_entries:
                    entry.needs_compact = False
                return 0

            mx.eval(targets)
            for snap, kc, vc in pending:
                snap.keys = kc
                snap.values = vc
                snap.needs_compact = False
            for entry in compacted_entries:
                old_bytes = entry.nbytes
                entry.needs_compact = any(_refresh_compact_flag(s) for s in entry.layers)
                new_bytes = sum(s.logical_bytes for s in entry.layers) + _nbytes(
                    entry.rope_deltas
                )
                entry.nbytes = new_bytes
                self._total_bytes = max(0, self._total_bytes - old_bytes + new_bytes)
            return len(pending)
