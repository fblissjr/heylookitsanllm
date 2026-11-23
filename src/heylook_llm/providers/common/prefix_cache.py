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
import weakref

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


def _collect_detach_arrays(item: Any, out: list[mx.array]) -> None:
    if item is None:
        return
    if isinstance(mx.array, type) and isinstance(item, mx.array):
        out.append(item)
        return
    if isinstance(item, _LayerSnapshot):
        if item.keys is not None:
            out.append(item.keys)
        if item.values is not None:
            out.append(item.values)
        if item.arrays is not None:
            for arr in item.arrays:
                if arr is not None:
                    out.append(arr)
        if item.children is not None:
            for ch in item.children:
                _collect_detach_arrays(ch, out)
        return
    if isinstance(item, (list, tuple)):
        for sub in item:
            _collect_detach_arrays(sub, out)
        return
    if isinstance(item, dict):
        for sub in item.values():
            _collect_detach_arrays(sub, out)


def detach_for_cache(*items: Any) -> None:
    """Materialize every lazy MLX view in ``items`` on the producer stream.

    Evaluates slice views before publishing them to any cross-request store or
    module attribute so lazy ``Slice``/weight graphs never remain bound to a
    worker's thread-local stream (zero extra GPU copy for slices).
    """
    to_eval: list[mx.array] = []
    for item in items:
        _collect_detach_arrays(item, to_eval)
    if to_eval:
        mx.eval(to_eval)


class _BackingTracker:
    """Track CoW backing-buffer identity across snapshots of the same live cache."""

    def __init__(self) -> None:
        self._by_cache: weakref.WeakKeyDictionary[Any, tuple[Any, Any, tuple, tuple, int, object]] = (
            weakref.WeakKeyDictionary()
        )

    def token_for(
        self,
        cache_obj: Any,
        keys: Optional[mx.array],
        values: Optional[mx.array],
        *,
        offset: int,
    ) -> Optional[object]:
        if keys is None or values is None:
            return None
        k_shape = tuple(getattr(keys, "shape", ()))
        v_shape = tuple(getattr(values, "shape", ()))
        try:
            prev = self._by_cache.get(cache_obj)
        except TypeError:
            return object()
        if (
            prev is not None
            and prev[0] is keys
            and prev[1] is values
            and prev[2] == k_shape
            and prev[3] == v_shape
            and prev[4] == offset
        ):
            return prev[5]
        tok = object()
        try:
            self._by_cache[cache_obj] = (keys, values, k_shape, v_shape, offset, tok)
        except TypeError:
            pass
        return tok


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
    backing_token: Optional[object] = None
    backing_len: int = 0
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
    backing_token: Optional[object] = None,
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
            backing_token=None,
            backing_len=0,
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
        backing_token=backing_token,
        backing_len=backing_len,
    )


def _snapshot_kv(
    c: Any,
    *,
    trim_tail: int,
    target_len: int,
    slack_cap: int,
    allow_kv_compact: bool,
    backing_tracker: Optional[_BackingTracker],
) -> Optional[_LayerSnapshot]:
    del target_len, slack_cap
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
            backing_token=None,
            backing_len=0,
        )
    backing_len = int(c.keys.shape[2])
    if backing_len < end:
        return None
    token = (
        backing_tracker.token_for(c, c.keys, c.values, offset=off)
        if backing_tracker is not None
        else None
    )
    k = c.keys[..., :end, :]
    v = c.values[..., :end, :]
    return _LayerSnapshot(
        kind="kv",
        keys=k,
        values=v,
        offset=end,
        idx=end,
        min_offset=0,
        needs_compact=(allow_kv_compact and backing_len > end + 256),
        backing_token=token,
        backing_len=backing_len,
    )


def _snapshot_buffered_rotating(
    c: Any,
    *,
    trim_tail: int,
    target_len: int,
    slack_cap: int,
    allow_kv_compact: bool,
    backing_tracker: Optional[_BackingTracker],
) -> Optional[_LayerSnapshot]:
    del target_len, allow_kv_compact
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
    off = int(c.offset)
    token = (
        backing_tracker.token_for(c, c.keys, c.values, offset=off)
        if backing_tracker is not None
        else None
    )
    return _snapshot_rotating_slices(
        c.keys,
        c.values,
        backing_len=int(c.keys.shape[2]),
        valid=int(c._idx),
        offset=off,
        max_size=int(c.max_size),
        keep=int(getattr(c, "keep", 0) or 0),
        trim_tail=trim_tail,
        slack_cap=slack_cap,
        backing_token=token,
    )


def _snapshot_rotating(
    c: Any,
    *,
    trim_tail: int,
    target_len: int,
    slack_cap: int,
    allow_kv_compact: bool,
    backing_tracker: Optional[_BackingTracker],
) -> Optional[_LayerSnapshot]:
    del target_len, allow_kv_compact
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
    off = int(c.offset)
    backing_len = max(int(c.keys.shape[2]), int(ok.shape[2]))
    token = (
        backing_tracker.token_for(c, ok, ov, offset=off)
        if backing_tracker is not None and ok is c.keys and ov is c.values
        else None
    )
    return _snapshot_rotating_slices(
        ok,
        ov,
        backing_len=backing_len,
        valid=int(ok.shape[2]),
        offset=off,
        max_size=int(c.max_size),
        keep=int(getattr(c, "keep", 0) or 0),
        trim_tail=trim_tail,
        slack_cap=slack_cap,
        backing_token=token,
    )


def _snapshot_arrays(
    c: Any,
    *,
    trim_tail: int,
    target_len: int,
    slack_cap: int,
    allow_kv_compact: bool,
    backing_tracker: Optional[_BackingTracker],
) -> Optional[_LayerSnapshot]:
    del slack_cap, allow_kv_compact, backing_tracker
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


def _snapshot_cache_list(
    c: Any,
    *,
    trim_tail: int,
    target_len: int,
    slack_cap: int,
    allow_kv_compact: bool,
    backing_tracker: Optional[_BackingTracker],
) -> Optional[_LayerSnapshot]:
    children = []
    for ch in c.caches:
        snap = _snapshot_layer(
            ch,
            trim_tail=trim_tail,
            target_len=target_len,
            slack_cap=slack_cap,
            allow_kv_compact=allow_kv_compact,
            backing_tracker=backing_tracker,
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


def _snapshot_handlers() -> dict[type, Callable[..., Optional[_LayerSnapshot]]]:
    KVCache, RotatingKVCache, BufferedRotatingKVCache, ArraysCache, CacheList = (
        _cache_classes()
    )
    return {
        KVCache: _snapshot_kv,
        BufferedRotatingKVCache: _snapshot_buffered_rotating,
        RotatingKVCache: _snapshot_rotating,
        ArraysCache: _snapshot_arrays,
        CacheList: _snapshot_cache_list,
    }


def _snapshot_layer(
    c: Any,
    *,
    trim_tail: int = 0,
    target_len: int = 0,
    slack_cap: int = _ROTATING_SLACK_TOKENS,
    allow_kv_compact: bool = True,
    backing_tracker: Optional[_BackingTracker] = None,
) -> Optional[_LayerSnapshot]:
    if trim_tail < 0:
        return None
    handler = _snapshot_handlers().get(type(c))
    if handler is None:
        return None
    return handler(
        c,
        trim_tail=trim_tail,
        target_len=target_len,
        slack_cap=slack_cap,
        allow_kv_compact=allow_kv_compact,
        backing_tracker=backing_tracker,
    )


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
    if fallbacks:
        detach_for_cache(fallbacks)
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
        newly_captured: list[_LayerSnapshot] = []
        for idx, c, trim_tail in to_capture:
            snap = _snapshot_layer(
                c, trim_tail=trim_tail, target_len=target_len, slack_cap=slack_cap
            )
            if snap is not None:
                captured[idx] = snap
                newly_captured.append(snap)
        if newly_captured:
            detach_for_cache(newly_captured)
        any_uncaptured = any(
            idx not in captured and _has_rotating(c)
            for idx, c in enumerate(prompt_cache)
        )

    return any_uncaptured


def _restore_kv(
    snap: _LayerSnapshot,
    *,
    match_len: int,
    use_buffered_rotating: bool,
    buffer_size: int,
) -> Any:
    del use_buffered_rotating, buffer_size
    KVCache, _, _, _, _ = _cache_classes()
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


def _restore_rotating(
    snap: _LayerSnapshot,
    *,
    match_len: int,
    use_buffered_rotating: bool,
    buffer_size: int,
) -> Any:
    _, RotatingKVCache, BufferedRotatingKVCache, _, _ = _cache_classes()
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


def _restore_arrays(
    snap: _LayerSnapshot,
    *,
    match_len: int,
    use_buffered_rotating: bool,
    buffer_size: int,
) -> Any:
    del match_len, use_buffered_rotating, buffer_size
    _, _, _, ArraysCache, _ = _cache_classes()
    assert snap.arrays is not None
    c = ArraysCache(len(snap.arrays))
    c.cache = [_array_view(a) for a in snap.arrays]
    return c


def _restore_cache_list(
    snap: _LayerSnapshot,
    *,
    match_len: int,
    use_buffered_rotating: bool,
    buffer_size: int,
) -> Any:
    _, _, _, _, CacheList = _cache_classes()
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


_RESTORE_HANDLERS: dict[str, Callable[..., Any]] = {
    "kv": _restore_kv,
    "rotating": _restore_rotating,
    "arrays": _restore_arrays,
    "cache_list": _restore_cache_list,
}


def _restore_layer(
    snap: _LayerSnapshot,
    *,
    match_len: int,
    use_buffered_rotating: bool,
    buffer_size: int = 64,
) -> Any:
    handler = _RESTORE_HANDLERS.get(snap.kind)
    if handler is None:
        raise ValueError(f"Unsupported snapshot kind: {snap.kind!r}")
    return handler(
        snap,
        match_len=match_len,
        use_buffered_rotating=use_buffered_rotating,
        buffer_size=buffer_size,
    )


def _iter_leaf_snapshots(snaps: Sequence[_LayerSnapshot]) -> list[_LayerSnapshot]:
    out: list[_LayerSnapshot] = []

    def _walk(snap: _LayerSnapshot) -> None:
        if snap.kind == "cache_list" and snap.children is not None:
            for ch in snap.children:
                _walk(ch)
        else:
            out.append(snap)

    for s in snaps:
        _walk(s)
    return out


def _refresh_compact_flag(snap: _LayerSnapshot) -> bool:
    if snap.kind == "cache_list" and snap.children is not None:
        snap.needs_compact = any(_refresh_compact_flag(ch) for ch in snap.children)
    return snap.needs_compact


@dataclass
class CacheLookupResult:
    """Rich result returned by :meth:`PrefixCacheStore.lookup_full`."""

    layers: Optional[list[Any]]
    matched_tokens: int
    rope_deltas: Optional[mx.array]
    drafter_layers: Optional[list[Any]] = None
    drafter_last_hidden: Optional[mx.array] = None


@dataclass
class _CacheEntry:
    extra_hash: int
    token_ids: tuple[int, ...]
    layers: list[_LayerSnapshot]
    min_match_len: int
    nbytes: int
    rope_deltas: Optional[mx.array] = None
    needs_compact: bool = field(default=False)
    drafter_layers: Optional[list[_LayerSnapshot]] = None
    drafter_token_len: int = 0
    drafter_last_hidden: Optional[mx.array] = None

    @property
    def trimmable_all(self) -> bool:
        return self.min_match_len == 0


def _compute_entries_bytes(entries: Sequence[_CacheEntry]) -> int:
    """Compute physical memory across ``entries``, deduplicating CoW backing slabs."""
    max_by_token: dict[object, int] = {}
    unshared = 0
    for entry in entries:
        unshared += _nbytes(entry.rope_deltas)
        unshared += _nbytes(entry.drafter_last_hidden)
        all_snaps = list(entry.layers)
        if entry.drafter_layers:
            all_snaps.extend(entry.drafter_layers)
        for leaf in _iter_leaf_snapshots(all_snaps):
            lbytes = leaf.logical_bytes
            if leaf.kind in ("kv", "rotating") and leaf.backing_token is not None:
                prev = max_by_token.get(leaf.backing_token, 0)
                if lbytes > prev:
                    max_by_token[leaf.backing_token] = lbytes
            else:
                unshared += lbytes
    return unshared + sum(max_by_token.values())


def _longest_common_prefix(a: tuple[int, ...], b: tuple[int, ...], limit: int) -> int:
    end = min(len(a), len(b), limit)
    n = 0
    while n < end and a[n] == b[n]:
        n += 1
    return n


def _metal_memory_headroom_bytes() -> Optional[int]:
    """Remaining bytes under Metal's max recommended working set, or None."""
    try:
        if not mx.metal.is_available():
            return None
        working_set = int(mx.device_info()["max_recommended_working_set_size"])
        active = int(mx.get_active_memory())
        return working_set - active
    except (AttributeError, RuntimeError, KeyError, TypeError, ValueError):
        return None


class PrefixCacheStore:
    """In-memory CoW prefix cache for single-row MLX inference."""

    def __init__(
        self,
        *,
        memory_max_bytes: int = 0,
        max_entries: int = 32,
        memory_reserve_bytes: int = 0,
        headroom_fn: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        self.memory_max_bytes = max(0, int(memory_max_bytes))
        self.max_entries = max(1, int(max_entries))
        self.memory_reserve_bytes = max(0, int(memory_reserve_bytes))
        self._headroom_fn = (
            _metal_memory_headroom_bytes if headroom_fn is None else headroom_fn
        )
        self._lock = threading.RLock()
        self._backing_tracker = _BackingTracker()
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

    def _recompute_total_bytes_locked(self) -> None:
        self._total_bytes = _compute_entries_bytes(list(self._entries.values()))

    def _evict_key_locked(self, key: tuple[int, tuple[int, ...]]) -> Optional[_CacheEntry]:
        removed = self._entries.pop(key, None)
        if removed is not None:
            self._recompute_total_bytes_locked()
        return removed

    def _make_room_locked(
        self, needed_bytes: int, candidate_entry: Optional[_CacheEntry] = None
    ) -> None:
        while len(self._entries) > self.max_entries:
            oldest_key, _ = next(iter(self._entries.items()))
            self._evict_key_locked(oldest_key)
        if self.memory_max_bytes > 0:
            while self._entries:
                projected = (
                    _compute_entries_bytes([*self._entries.values(), candidate_entry])
                    if candidate_entry is not None
                    else self._total_bytes + needed_bytes
                )
                if projected <= self.memory_max_bytes:
                    break
                oldest_key, _ = next(iter(self._entries.items()))
                self._evict_key_locked(oldest_key)
                self.memory_evictions += 1
        if self.memory_reserve_bytes > 0 and self._entries:
            headroom = self._headroom_fn()
            if headroom is not None and headroom - needed_bytes < self.memory_reserve_bytes:
                deficit = self.memory_reserve_bytes - (headroom - needed_bytes)
                freed = 0
                while self._entries and freed < deficit:
                    before_total = self._total_bytes
                    oldest_key, _ = next(iter(self._entries.items()))
                    self._evict_key_locked(oldest_key)
                    self.memory_evictions += 1
                    freed += max(0, before_total - self._total_bytes)

    def lookup_full(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
        suffix_is_text_only: Optional[Callable[[int], bool]] = None,
        use_buffered_rotating: bool = False,
        buffer_size: int = 64,
    ) -> CacheLookupResult:
        token_tuple = tuple(int(t) for t in token_ids)
        n = len(token_tuple)
        max_len = n - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= max(0, int(min_prefix_tokens)):
            return CacheLookupResult(None, 0, None, None)

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
                return CacheLookupResult(None, 0, None, None)

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
            drafter_restored: Optional[list[Any]] = None
            if best_entry.drafter_layers is not None:
                if best_len == len(best_entry.token_ids):
                    d_match = best_entry.drafter_token_len
                else:
                    drop_tail = len(best_entry.token_ids) - best_len
                    d_match = best_entry.drafter_token_len - drop_tail
                    if d_match <= 0 or any(
                        not dl.trimmable for dl in best_entry.drafter_layers
                    ):
                        d_match = 0
                if d_match > 0:
                    drafter_restored = [
                        _restore_layer(
                            dl_snap,
                            match_len=(
                                min(d_match, dl_snap.offset)
                                if dl_snap.kind in ("kv", "rotating")
                                else d_match
                            ),
                            use_buffered_rotating=use_buffered_rotating,
                            buffer_size=buffer_size,
                        )
                        for dl_snap in best_entry.drafter_layers
                    ]
            drafter_last_hidden = (
                _array_view(best_entry.drafter_last_hidden)
                if drafter_restored is not None and best_len == len(best_entry.token_ids)
                else None
            )
            rd = _array_view(best_entry.rope_deltas)
            return CacheLookupResult(
                restored,
                best_len,
                rd,
                drafter_restored,
                drafter_last_hidden=drafter_last_hidden,
            )

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
        res = self.lookup_full(
            token_ids,
            extra_hash=extra_hash,
            max_prefix_tokens=max_prefix_tokens,
            min_prefix_tokens=min_prefix_tokens,
            suffix_is_text_only=suffix_is_text_only,
            use_buffered_rotating=use_buffered_rotating,
            buffer_size=buffer_size,
        )
        return res.layers, res.matched_tokens, res.rope_deltas

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
        allow_kv_compact: bool = True,
        drafter_cache: Optional[Sequence[Any]] = None,
        drafter_trim_tail: int = 0,
        drafter_target_len: Optional[int] = None,
        drafter_last_hidden: Optional[mx.array] = None,
    ) -> bool:
        target_len = len(token_ids) - int(trim_tail)
        kv_trim = int(trim_tail) if cache_trim_tail is None else int(cache_trim_tail)
        if target_len <= 0 or kv_trim < 0 or not prompt_cache:
            return False

        layers: list[_LayerSnapshot] = []
        for idx, c in enumerate(prompt_cache):
            snap = _snapshot_layer(
                c,
                trim_tail=kv_trim,
                target_len=target_len,
                allow_kv_compact=allow_kv_compact,
                backing_tracker=self._backing_tracker,
            )
            if snap is None and layer_fallbacks is not None:
                fb = layer_fallbacks.get(idx)
                if fb is not None and (fb.offset == target_len or fb.kind == "arrays"):
                    snap = fb
            if snap is None:
                return False
            layers.append(snap)

        drafter_layers: Optional[list[_LayerSnapshot]] = None
        d_target = 0
        if drafter_cache:
            d_raw_target = (
                target_len if drafter_target_len is None else int(drafter_target_len)
            )
            if int(drafter_trim_tail) >= 0 and d_raw_target > 0:
                d_snaps: list[_LayerSnapshot] = []
                d_ok = True
                for dc in drafter_cache:
                    d_snap = _snapshot_layer(
                        dc,
                        trim_tail=int(drafter_trim_tail),
                        target_len=d_raw_target,
                        allow_kv_compact=allow_kv_compact,
                        backing_tracker=self._backing_tracker,
                    )
                    if d_snap is None:
                        d_ok = False
                        break
                    d_snaps.append(d_snap)
                if d_ok and d_snaps:
                    drafter_layers = d_snaps
                    if drafter_target_len is None:
                        d_offsets = [
                            s.offset
                            for s in _iter_leaf_snapshots(d_snaps)
                            if s.kind in ("kv", "rotating")
                        ]
                        if d_offsets and len(set(d_offsets)) == 1:
                            d_target = d_offsets[0]
                        else:
                            d_target = d_raw_target
                    else:
                        d_target = d_raw_target

        rd_view = _array_view(rope_deltas)
        dh_view = (
            mx.contiguous(drafter_last_hidden)
            if drafter_layers is not None and drafter_last_hidden is not None
            else None
        )
        detach_for_cache(layers, drafter_layers, rd_view, dh_view)
        all_snaps = list(layers)
        if drafter_layers:
            all_snaps.extend(drafter_layers)

        key = (extra_hash, tuple(int(t) for t in token_ids[:target_len]))
        entry = _CacheEntry(
            extra_hash=extra_hash,
            token_ids=key[1],
            layers=layers,
            min_match_len=max((s.min_offset for s in layers), default=0),
            nbytes=0,
            rope_deltas=rd_view,
            needs_compact=any(s.needs_compact for s in all_snaps),
            drafter_layers=drafter_layers,
            drafter_token_len=d_target,
            drafter_last_hidden=dh_view,
        )
        entry.nbytes = _compute_entries_bytes([entry])
        if self.memory_max_bytes > 0 and entry.nbytes > self.memory_max_bytes:
            return False

        with self._lock:
            self._evict_key_locked(key)
            projected = _compute_entries_bytes([*self._entries.values(), entry])
            needed_bytes = max(0, projected - self._total_bytes)
            self._make_room_locked(needed_bytes, candidate_entry=entry)
            if len(self._entries) >= self.max_entries:
                oldest_key, _ = next(iter(self._entries.items()))
                self._evict_key_locked(oldest_key)
            self._entries[key] = entry
            self._recompute_total_bytes_locked()
            self.stored += 1
            return True

    def compact_pending(self) -> int:
        """Detach oversized slice views post-generation without duplicating shared KV slabs."""
        with self._lock:
            if not any(e.needs_compact for e in self._entries.values()):
                return 0

            all_snaps: list[_LayerSnapshot] = []
            for entry in self._entries.values():
                all_snaps.extend(entry.layers)
                if entry.drafter_layers:
                    all_snaps.extend(entry.drafter_layers)

            kv_by_token: dict[object, list[_LayerSnapshot]] = {}
            kv_unshared: list[_LayerSnapshot] = []
            rotating_pending: list[tuple[_LayerSnapshot, mx.array, mx.array]] = []
            targets: list[mx.array] = []

            for leaf in _iter_leaf_snapshots(all_snaps):
                if leaf.kind == "kv":
                    if leaf.backing_token is not None:
                        kv_by_token.setdefault(leaf.backing_token, []).append(leaf)
                    elif leaf.needs_compact:
                        kv_unshared.append(leaf)
                elif leaf.kind == "rotating" and leaf.needs_compact:
                    if leaf.keys is not None and leaf.values is not None:
                        kc = mx.contiguous(leaf.keys)
                        vc = mx.contiguous(leaf.values)
                        targets.extend((kc, vc))
                        rotating_pending.append((leaf, kc, vc))
                    else:
                        leaf.needs_compact = False

            kv_groups: list[tuple[list[_LayerSnapshot], int, mx.array, mx.array]] = []
            for group in list(kv_by_token.values()) + [[s] for s in kv_unshared]:
                if not any(s.needs_compact for s in group):
                    continue
                widest = max(group, key=lambda s: s.offset)
                if (
                    widest.keys is None
                    or widest.values is None
                    or widest.backing_len <= widest.offset + 256
                ):
                    for s in group:
                        s.needs_compact = False
                    continue
                kc = mx.contiguous(widest.keys)
                vc = mx.contiguous(widest.values)
                targets.extend((kc, vc))
                kv_groups.append((group, widest.offset, kc, vc))

            if not targets:
                for entry in self._entries.values():
                    e_snaps = list(entry.layers) + list(entry.drafter_layers or ())
                    entry.needs_compact = any(_refresh_compact_flag(s) for s in e_snaps)
                return 0

            detach_for_cache(targets)

            resliced: list[mx.array] = []
            for group, widest_off, kc, vc in kv_groups:
                new_tok = object()
                for s in group:
                    if s.offset == widest_off:
                        s.keys = kc
                        s.values = vc
                    else:
                        s.keys = kc[..., : s.offset, :]
                        s.values = vc[..., : s.offset, :]
                        resliced.extend((s.keys, s.values))
                    s.backing_token = new_tok
                    s.backing_len = widest_off
                    s.needs_compact = False

            for s, kc, vc in rotating_pending:
                s.keys = kc
                s.values = vc
                s.backing_token = object()
                s.backing_len = s.idx
                s.needs_compact = False

            if resliced:
                detach_for_cache(resliced)

            for entry in self._entries.values():
                e_snaps = list(entry.layers) + list(entry.drafter_layers or ())
                entry.needs_compact = any(_refresh_compact_flag(s) for s in e_snaps)
                entry.nbytes = _compute_entries_bytes([entry])
            self._recompute_total_bytes_locked()
            return len(kv_groups) + len(rotating_pending)
