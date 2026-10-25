"""Model-agnostic asymmetric pipeline parallelism for MLX / mlx-vlm models.

Combines the proven distributed mechanisms from upstream `mlx` and `mlx-lm`
(`muse_spark.py`, `qwen3_5.py`, `pipeline.py`) with non-invasive in-place layer
wrapping so any `mlx-vlm` model (text or vision, full-attention or hybrid
Gated DeltaNet) can split its decoder layers asymmetrically across two or more
Macs (e.g. 80/20 across a 192 GB M2 Ultra and a 48 GB M4 Pro):

1. `MLX_METAL_FAST_SYNCH=1` (`enable_fast_metal_synch`): switches Metal
   cross-stream fences (`mlx/backend/metal/fence.cpp`) from `MTLSharedEvent`
   command-buffer splits to 4-byte shared-memory atomics + inline Metal kernels
   for low-latency GPU <-> CPU `mx.distributed` handoffs.
2. Pre-eval layer pruning + selective shard loading (`selective_pipeline_load`):
   filters `.safetensors` shards and tensor keys before `sanitize_weights` /
   FP8 transforms run, and replaces off-rank layers in `inner.layers` with
   zero-parameter `_PassThroughLayer` instances *before* `mx.eval` runs, so
   off-rank weights never materialize in unified memory.
3. Preserved layer-list length (`len(inner.layers) == total_layers`) and
   batch/APC-compatible `_OffsetStubCache`: every model's `fa_idx`/`swa_idx`
   mask probes, M-RoPE position IDs, and `deepstack_visual_embeds` indexing
   continue to work unchanged while off-rank cache slots hold 0 bytes and
   integrate cleanly with `BatchGenerator._make_cache` and `APCCoordinator`.
4. `attach_send_dependency` (`mx.depends`): ties `mx.distributed.send(h)` into
   the active layer's cache state (both standard KV `.keys` and hybrid
   `ArraysCache` `.state` lists) so chunked prefill (`mx.eval([c.state ...])`)
   never dead-code-eliminates the network send.
5. Lockstep Rank 0 -> Rank 1 Step Synchronizer (`PipelineCoordinator` and
   `run_pipeline_worker_loop`): Rank 0 broadcasts a compact 8-int32 control
   header before each `language_model` forward pass (plus `position_ids` /
   `rope_deltas` when present), so Rank 1 steps its decoder slice in lockstep
   using dummy `inputs_embeds` without needing `vision_tower`, `embed_tokens`,
   or `lm_head`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

_log = logging.getLogger(__name__)

_LAYER_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# Control opcodes broadcast from Rank 0 to worker ranks in lockstep mode
CMD_FORWARD = 1
CMD_RESET_CACHE = 2
CMD_SHUTDOWN = 99

# Dtype table for transmitting hidden states / position_ids metadata
_DTYPE_TO_CODE: dict[Any, int] = {
    mx.bfloat16: 1,
    mx.float16: 2,
    mx.float32: 3,
}
_CODE_TO_DTYPE: dict[int, Any] = {v: k for k, v in _DTYPE_TO_CODE.items()}


@dataclass(frozen=True)
class PipelineAssignment:
    """Contiguous layer slice assigned to one rank in an asymmetric pipeline."""

    rank: int
    world_size: int
    start_layer: int
    end_layer: int
    total_layers: int

    @property
    def num_local_layers(self) -> int:
        return self.end_layer - self.start_layer

    def owns_layer(self, layer_idx: int) -> bool:
        return self.start_layer <= layer_idx < self.end_layer


def enable_fast_metal_synch(residency_refresh_ms: Optional[int] = None) -> dict[str, str]:
    """Enable MLX C++ fast Metal synchronization (`MLX_METAL_FAST_SYNCH=1`)
    and optional background residency refresh (`MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`).

    Must be called before the first Metal fence / residency set is constructed
    in the process to take full effect.
    """
    applied: dict[str, str] = {}
    os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
    applied["MLX_METAL_FAST_SYNCH"] = os.environ["MLX_METAL_FAST_SYNCH"]
    if residency_refresh_ms is not None and residency_refresh_ms > 0:
        os.environ.setdefault(
            "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", str(int(residency_refresh_ms))
        )
        applied["MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"] = os.environ[
            "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"
        ]
    return applied


def configure_manual_ring(
    peers: Sequence[str] | str,
    rank: int = 0,
    base_port: int = 32323,
) -> str:
    """Populate `MLX_HOSTFILE` and `MLX_RANK` for SSH-free multi-Mac ring launch.

    `peers` may be a comma-separated string (`"<ip0>,<ip1>"`) or a sequence of
    host strings (`["<ip0>", "<ip1>"]` or `["<ip0>:32323", "<ip1>:32324"]`).
    Returns the path to the generated JSON hostfile.
    """
    if isinstance(peers, str):
        items = [p.strip() for p in peers.split(",") if p.strip()]
    else:
        items = [str(p).strip() for p in peers if str(p).strip()]
    if len(items) < 2:
        raise ValueError(
            f"pipeline_peers requires at least 2 hosts, got {peers!r}"
        )
    if not (0 <= rank < len(items)):
        raise ValueError(
            f"rank {rank} out of range for {len(items)} peers"
        )

    ring_hosts: list[list[str]] = []
    for idx, item in enumerate(items):
        if ":" in item:
            ring_hosts.append([item])
        else:
            ring_hosts.append([f"{item}:{base_port + idx}"])

    fd, path = tempfile.mkstemp(prefix=f"mlx_ring_rank{rank}_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(ring_hosts, f)
    os.environ["MLX_HOSTFILE"] = path
    os.environ["MLX_RANK"] = str(rank)
    return path


def init_pipeline_group(
    peers: Optional[Sequence[str] | str] = None,
    rank: int = 0,
    backend: str = "any",
    max_wait_s: Optional[float] = None,
) -> Any:
    """Initialize and return the `mlx.distributed` group for pipeline execution.

    When `peers` (or `HEYLOOK_MLX_PEERS`) is provided, configures `MLX_HOSTFILE`
    and `MLX_RANK` for SSH-free multi-Mac TCP ring connection and retries
    `mx.distributed.init(strict=True, backend="ring")` until the peer connects
    (or `max_wait_s` expires).
    """
    effective_peers = peers if peers is not None else os.environ.get("HEYLOOK_MLX_PEERS")
    if effective_peers:
        env_rank = os.environ.get("HEYLOOK_MLX_RANK")
        effective_rank = int(env_rank) if env_rank is not None and rank == 0 else int(rank)
        configure_manual_ring(effective_peers, rank=effective_rank)
        ring_backend = "ring" if backend == "any" else backend
        _log.info(
            "Waiting for MLX ring peers (%s) as rank=%d...",
            effective_peers,
            effective_rank,
        )
        t0 = time.monotonic()
        while True:
            try:
                return mx.distributed.init(strict=True, backend=ring_backend)
            except RuntimeError as exc:
                if max_wait_s is not None and (time.monotonic() - t0) >= max_wait_s:
                    raise
                _log.info("Still waiting for MLX ring peer (%s); retrying...", exc)
                time.sleep(1.0)
    return mx.distributed.init(strict=False, backend=backend)


def compute_asymmetric_layer_split(
    total_layers: int, ratios: Sequence[float] = (0.8, 0.2)
) -> list[tuple[int, int]]:
    """Partition `total_layers` into `len(ratios)` contiguous `(start, end)`
    intervals proportional to `ratios`.

    Every rank receives at least 1 layer when `total_layers >= len(ratios)`,
    and the intervals form an exact partition of `range(total_layers)`.
    """
    n_ranks = len(ratios)
    if n_ranks < 1:
        raise ValueError("ratios must contain at least one element")
    if total_layers < n_ranks:
        raise ValueError(
            f"total_layers ({total_layers}) must be >= number of ranks ({n_ranks})"
        )
    if any(r <= 0 for r in ratios):
        raise ValueError(f"all ratios must be positive, got {list(ratios)}")

    total_w = float(sum(ratios))
    norm = [float(r) / total_w for r in ratios]

    raw = [w * total_layers for w in norm]
    counts = [max(1, int(x)) for x in raw]
    diff = total_layers - sum(counts)

    if diff > 0:
        order = sorted(range(n_ranks), key=lambda i: (raw[i] - counts[i]), reverse=True)
        for idx in order[:diff]:
            counts[idx] += 1
    elif diff < 0:
        order = sorted(range(n_ranks), key=lambda i: (counts[i] - raw[i]), reverse=True)
        for idx in order:
            while diff < 0 and counts[idx] > 1:
                counts[idx] -= 1
                diff += 1

    spans: list[tuple[int, int]] = []
    cursor = 0
    for c in counts:
        spans.append((cursor, cursor + c))
        cursor += c
    return spans


def should_keep_weight_key(tensor_name: str, assignment: PipelineAssignment) -> bool:
    """Return True if `tensor_name` belongs to `assignment.rank`.

    - Any key matching `layers.<idx>.*` is kept iff `assignment.owns_layer(idx)`.
    - Non-layer keys (`vision_tower`, `embed_tokens`, `norm`, `lm_head`) are kept
      on Rank 0 (`assignment.rank == 0`) and dropped on worker ranks (`rank > 0`).
    """
    m = _LAYER_KEY_RE.search(tensor_name)
    if m is not None:
        return assignment.owns_layer(int(m.group(1)))
    return assignment.rank == 0


def filter_safetensors_for_rank(
    model_path: Path,
    assignment: PipelineAssignment,
) -> list[Path]:
    """Return the subset of `.safetensors` files in `model_path` needed by
    `assignment.rank`.

    When `model.safetensors.index.json` is present:
    - Rank 0 loads non-layer weights (`embed_tokens`, `vision_tower`, `norm`,
      `lm_head`) plus shards holding `layers.{start_layer..end_layer-1}`.
    - Worker ranks (`rank > 0`) load ONLY shards holding
      `layers.{start_layer..end_layer-1}`.
    """
    model_dir = Path(model_path).expanduser()
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.is_file():
        return sorted(model_dir.glob("*.safetensors"))

    try:
        weight_map: dict[str, str] = json.loads(index_file.read_text()).get(
            "weight_map", {}
        )
    except (OSError, json.JSONDecodeError):
        return sorted(model_dir.glob("*.safetensors"))

    if not weight_map:
        return sorted(model_dir.glob("*.safetensors"))

    needed_files: set[str] = set()
    for tensor_name, filename in weight_map.items():
        if should_keep_weight_key(tensor_name, assignment):
            needed_files.add(filename)

    return [model_dir / fname for fname in sorted(needed_files)]


def _extract_seq_len(x: Any) -> int:
    """Infer sequence length `L` from a hidden state tensor `(B, L, D)` or `(L, D)`."""
    shape = getattr(x, "shape", None)
    if not shape:
        return 1
    if len(shape) >= 2:
        return int(shape[1])
    return int(shape[0])


class _OffsetStubCache:
    """Zero-byte cache placeholder for off-rank layers that satisfies both
    `BatchGenerator._make_cache` (`to_batch`, `extract`, `filter`, `extend`,
    `merge`) and `mlx_vlm.apc_adapters` (`prefix_cache_snapshot`,
    `prefix_cache_restore`, `prefix_cache_merge`, `nbytes = 0`).

    Tracks `.offset` (and `._idx` / `.left_padding` when batched) so `mlx-vlm`
    mask builders (`create_attention_mask`, `fa_idx`, `swa_idx`) see the exact
    same sequence offset across all layers while allocating 0 bytes of GPU memory.
    """

    def __init__(
        self,
        left_padding: Optional[Sequence[int]] = None,
        offset: int = 0,
    ) -> None:
        self._idx: int = int(offset)
        self.keys: Any = None
        self.values: Any = None
        self._right_padding: Optional[mx.array] = None
        if left_padding is not None:
            pads = [int(p) for p in left_padding]
            self.left_padding: Optional[mx.array] = mx.array(pads, dtype=mx.int32)
            self.offset: Any = mx.array([self._idx - p for p in pads], dtype=mx.int32)
        else:
            self.left_padding = None
            self.offset = int(offset)

    @property
    def nbytes(self) -> int:
        return 0

    def size(self) -> int:
        return self._idx

    def advance(self, n_tokens: int) -> None:
        step = int(n_tokens)
        self._idx += step
        self.offset = self.offset + step

    def empty(self) -> bool:
        return self._idx == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        trimmed = min(self._idx, int(n))
        self._idx -= trimmed
        self.offset = self.offset - trimmed
        return trimmed

    def make_mask(self, N: int, return_array: bool = False, **kwargs: Any) -> Any:
        from mlx_vlm.models.cache import create_attention_mask, create_causal_mask

        if self.left_padding is not None:
            return create_causal_mask(
                N, offset=self._idx, left_padding=self.left_padding, **kwargs
            )
        window_size = kwargs.get("window_size")
        return create_attention_mask(N, self._idx, return_array, window_size)

    @property
    def state(self) -> list[Any]:
        return []

    @state.setter
    def state(self, v: Any) -> None:
        pass

    @property
    def meta_state(self) -> str:
        return str(self._idx)

    @meta_state.setter
    def meta_state(self, v: Any) -> None:
        self._idx = int(v) if v else 0
        if self.left_padding is None:
            self.offset = self._idx

    def prefix_cache_snapshot(self) -> dict[str, Any]:
        return {
            "state": [],
            "meta_state": str(self._idx),
            "offset": (
                int(self.offset.item())
                if isinstance(self.offset, mx.array) and self.offset.size == 1
                else (int(self._idx) if isinstance(self.offset, int) else self._idx)
            ),
        }

    def prefix_cache_restore(self, snapshot: dict[str, Any]) -> None:
        restored = int(snapshot.get("offset", snapshot.get("meta_state", 0)) or 0)
        self._idx = restored
        if self.left_padding is not None:
            self.offset = mx.array([restored], dtype=mx.int32) - self.left_padding
        else:
            self.offset = restored

    def prefix_cache_merge(
        self, rows: Sequence["_OffsetStubCache"], prefix_lens: Sequence[int]
    ) -> Optional["_OffsetStubCache"]:
        if not rows or not all(isinstance(r, _OffsetStubCache) for r in rows):
            return None
        return self.merge(rows)

    def to_batch(self, left_padding: Sequence[int]) -> "_OffsetStubCache":
        batched = _OffsetStubCache(left_padding=left_padding)
        if self._idx > 0:
            batched.advance(self._idx)
        return batched

    @property
    def batch_size(self) -> int:
        if self.left_padding is not None:
            return int(self.left_padding.shape[0])
        return 1

    def is_single_row(self) -> bool:
        return self.batch_size == 1

    def prepare(
        self,
        *,
        left_padding: Optional[Sequence[int]] = None,
        lengths: Any = None,
        right_padding: Optional[Sequence[int]] = None,
    ) -> None:
        del lengths
        if left_padding is not None:
            lp = mx.array([int(p) for p in left_padding], dtype=mx.int32)
            if self.left_padding is None:
                self.left_padding = lp
                self.offset = mx.array([self._idx] * int(lp.shape[0]), dtype=mx.int32) - lp
            else:
                self.left_padding = self.left_padding + lp
                self.offset = self.offset - lp
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array([int(p) for p in right_padding], dtype=mx.int32)

    def finalize(self) -> None:
        if self._right_padding is not None:
            self.offset = self.offset - self._right_padding
            if self.left_padding is not None:
                self.left_padding = self.left_padding + self._right_padding
            self._right_padding = None

    def filter(self, batch_indices: Any) -> None:
        if isinstance(self.offset, mx.array) and self.offset.ndim > 0:
            self.offset = self.offset[batch_indices]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
            min_left_pad = int(self.left_padding.min().item())
            if min_left_pad > 0:
                self._idx = max(0, self._idx - min_left_pad)
                self.left_padding = self.left_padding - min_left_pad
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]

    def extend(self, other: "_OffsetStubCache") -> None:
        max_idx = max(self._idx, other._idx)
        self_lp = (
            self.left_padding
            if self.left_padding is not None
            else mx.zeros((1,), dtype=mx.int32)
        )
        other_lp = (
            other.left_padding
            if other.left_padding is not None
            else mx.zeros((1,), dtype=mx.int32)
        )
        self_off = (
            self.offset
            if isinstance(self.offset, mx.array) and self.offset.ndim > 0
            else mx.array([int(self.offset)], dtype=mx.int32)
        )
        other_off = (
            other.offset
            if isinstance(other.offset, mx.array) and other.offset.ndim > 0
            else mx.array([int(other.offset)], dtype=mx.int32)
        )
        self.left_padding = mx.concatenate(
            [self_lp + (max_idx - self._idx), other_lp + (max_idx - other._idx)]
        )
        self.offset = mx.concatenate([self_off, other_off])
        self._idx = max_idx

    def extract(self, idx: int) -> "_OffsetStubCache":
        single = _OffsetStubCache()
        if self.left_padding is not None:
            pad = int(self.left_padding[idx].item())
            single._idx = max(0, self._idx - pad)
        else:
            single._idx = self._idx
        single.offset = single._idx
        return single

    @classmethod
    def merge(cls, caches: Sequence["_OffsetStubCache"]) -> "_OffsetStubCache":
        lengths = [c.size() for c in caches]
        max_length = max(lengths) if lengths else 0
        padding = [max_length - length for length in lengths]
        merged = cls(left_padding=padding)
        merged._idx = max_length
        merged.offset = merged.offset + max_length
        return merged


class _StubSelfAttn:
    """Zero-parameter holder for ``rotary_emb`` on off-rank stub layers so
    models like ``Qwen3_5Model`` that inspect ``layer.self_attn.rotary_emb`` on
    the first non-linear layer work when that layer is off-rank.
    """

    def __init__(self, rotary_emb: Any) -> None:
        self.rotary_emb = rotary_emb


class _IdentityEmbed(nn.Module):
    """Zero-parameter replacement for ``embed_tokens`` / ``lm_head`` on worker
    ranks (`rank > 0`) where Rank 0 owns embedding lookup and final logits."""

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return x

    def as_linear(self, x: Any) -> Any:
        return x


def _copy_layer_traits(src: Any, dst: Any) -> None:
    """Copy decoder-layer metadata attributes onto a wrapper or stub layer so
    model-specific mask and KV-sharing loops (`qwen3_5`, `gemma4`, `mistral3`,
    etc.) see identical attributes.
    """
    for attr in ("use_sliding", "is_linear", "layer_type", "layer_idx"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    self_attn = getattr(src, "self_attn", None)
    rotary_emb = getattr(self_attn, "rotary_emb", None) if self_attn is not None else None
    if rotary_emb is not None and not hasattr(dst, "self_attn"):
        setattr(dst, "self_attn", _StubSelfAttn(rotary_emb))


class _PassThroughLayer(nn.Module):
    """Zero-parameter replacement for off-rank transformer blocks."""

    def __init__(
        self,
        use_sliding: bool = False,
        is_linear: bool = False,
        layer_type: Optional[str] = None,
        returns_gemma4_tuple: bool = False,
    ) -> None:
        super().__init__()
        self.use_sliding = use_sliding
        self.is_linear = is_linear
        if layer_type is not None:
            self.layer_type = layer_type
        self._returns_gemma4_tuple = returns_gemma4_tuple

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        cache = _extract_cache_arg(args, kwargs)
        if isinstance(cache, _OffsetStubCache):
            cache.advance(_extract_seq_len(x))
        if self._returns_gemma4_tuple or "shared_kv" in kwargs or "per_layer_input" in kwargs:
            return x, kwargs.get("shared_kv"), kwargs.get("offset")
        return x


def attach_send_dependency(cache_entry: Any, sent_tensor: Any) -> None:
    """Tie `sent_tensor` (`mx.distributed.send` output) into `cache_entry`'s
    state so chunked prefill (`mx.eval([c.state for c in prompt_cache])`)
    cannot dead-code-eliminate the network send (`muse_spark.py` idiom).
    """
    if cache_entry is None:
        return
    if getattr(cache_entry, "keys", None) is not None:
        cache_entry.keys = mx.depends(cache_entry.keys, sent_tensor)
        return
    # Hybrid recurrent / Gated DeltaNet caches (`ArraysCache` / `CacheList`)
    # store a list of arrays in `.state` without a `.keys` attribute.
    state = getattr(cache_entry, "state", None)
    if isinstance(state, list):
        for i, arr in enumerate(state):
            if arr is not None:
                state[i] = mx.depends(arr, sent_tensor)
                return


def _extract_cache_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "cache" in kwargs:
        return kwargs["cache"]
    if len(args) >= 2:
        return args[1]
    return None


class _BoundarySendWrapper(nn.Module):
    """Wraps the last layer of Rank `r`'s slice (`end_layer - 1`): executes the
    layer, sends the hidden state `h` to `dst_rank`, and records a graph
    dependency on the layer's cache state.
    """

    def __init__(self, layer: nn.Module, dst_rank: int, group: Any = None) -> None:
        super().__init__()
        self.layer = layer
        self.dst_rank = dst_rank
        self.group = group
        _copy_layer_traits(layer, self)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name != "layer":
                layer = dict.get(self, "layer")
                if layer is not None:
                    return getattr(layer, name)
            raise

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(x, *args, **kwargs)
        if isinstance(out, tuple):
            h = mx.distributed.send(out[0], self.dst_rank, group=self.group)
            attach_send_dependency(_extract_cache_arg(args, kwargs), h)
            return (h, *out[1:])
        h = mx.distributed.send(out, self.dst_rank, group=self.group)
        attach_send_dependency(_extract_cache_arg(args, kwargs), h)
        return h


class _BoundaryRecvWrapper(nn.Module):
    """Wraps the first layer of Rank `r`'s slice (`start_layer` for `r > 0`):
    receives the hidden state `x` from `src_rank` via `recv_like` and executes
    the wrapped layer.
    """

    def __init__(self, layer: nn.Module, src_rank: int, group: Any = None) -> None:
        super().__init__()
        self.layer = layer
        self.src_rank = src_rank
        self.group = group
        _copy_layer_traits(layer, self)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name != "layer":
                layer = dict.get(self, "layer")
                if layer is not None:
                    return getattr(layer, name)
            raise

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        x_in = mx.distributed.recv_like(x, self.src_rank, group=self.group)
        return self.layer(x_in, *args, **kwargs)


class _FinalStageRecvWrapper(nn.Module):
    """Placed at `layers[total_layers - 1]` on Rank 0 when `world_size > 1`:
    advances the stub cache and receives the final hidden state `h` back from
    the last rank (`world_size - 1`) so Rank 0 can run `norm` + `lm_head`.
    """

    def __init__(
        self,
        src_rank: int,
        group: Any = None,
        use_sliding: bool = False,
        is_linear: bool = False,
        layer_type: Optional[str] = None,
        returns_gemma4_tuple: bool = False,
    ) -> None:
        super().__init__()
        self.src_rank = src_rank
        self.group = group
        self.use_sliding = use_sliding
        self.is_linear = is_linear
        if layer_type is not None:
            self.layer_type = layer_type
        self._returns_gemma4_tuple = returns_gemma4_tuple

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        cache = _extract_cache_arg(args, kwargs)
        if isinstance(cache, _OffsetStubCache):
            cache.advance(_extract_seq_len(x))
        # CRITICAL: in mlx/distributed/ops.cpp, `recv_like(x, src)` constructs a
        # `Recv` primitive with an empty input list (`std::vector<array>{}`), so
        # `h` has NO graph dependency on `x` (the `Send` tensor from Rank 0's
        # boundary layer) unless tied with `mx.depends(h, x)`. Without this DAG
        # edge, evaluating `logits` on Rank 0 schedules `Recv` before `Send` and
        # deadlocks against Rank 1.
        h = mx.distributed.recv_like(x, self.src_rank, group=self.group)
        h = mx.depends(h, x)
        attach_send_dependency(cache, h)
        if self._returns_gemma4_tuple or "shared_kv" in kwargs or "per_layer_input" in kwargs:
            return h, kwargs.get("shared_kv"), kwargs.get("offset")
        return h


def resolve_inner_decoder(model: Any) -> tuple[Any, list[Any]]:
    """Locate the inner transformer decoder and its mutable `layers` list on
    any `mlx-vlm` VLM or text-only model.
    """
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, list):
        raise ValueError(
            f"Could not find a mutable `layers` list on {type(model).__name__} "
            f"(resolved inner={type(inner).__name__})"
        )
    return inner, layers


def _repoint_model_probe_indices(inner: Any, assignment: PipelineAssignment) -> None:
    """Ensure model-level cache/mask probe indices (`fa_idx`, `swa_idx`,
    `ssm_idx`) point at a layer owned by `assignment.rank` whenever one exists.

    For example, `Qwen3_5Model` sets `self.ssm_idx = 0` and `self.fa_idx = 3`,
    and reads `cache[self.fa_idx]` and `cache[self.ssm_idx]` to construct masks
    and check `_is_single_row_batch_cache`. On Rank 1 (which owns layers
    `K..L-1`), repointing `fa_idx` and `ssm_idx` to the first full-attention and
    linear-attention layers in `range(start_layer, end_layer)` ensures Rank 1
    inspects its own real `KVCache` / `ArraysCache` instances.
    """
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, list):
        return
    owned = range(assignment.start_layer, assignment.end_layer)

    if hasattr(inner, "fa_idx") and not assignment.owns_layer(int(inner.fa_idx)):
        for i in owned:
            layer = layers[i]
            if not getattr(layer, "is_linear", False) and not getattr(
                layer, "use_sliding", False
            ):
                inner.fa_idx = i
                break

    if hasattr(inner, "ssm_idx") and not assignment.owns_layer(int(inner.ssm_idx)):
        for i in owned:
            if getattr(layers[i], "is_linear", False):
                inner.ssm_idx = i
                break

    if hasattr(inner, "swa_idx") and not assignment.owns_layer(int(inner.swa_idx)):
        for i in owned:
            if getattr(layers[i], "use_sliding", False):
                inner.swa_idx = i
                break


def apply_asymmetric_pipeline(
    model: Any,
    group: Any = None,
    ratios: Sequence[float] = (0.8, 0.2),
) -> PipelineAssignment:
    """Partition `model`'s decoder layers in-place across `group` according to
    `ratios` (e.g. `(0.8, 0.2)` for 192 GB M2 Ultra + 48 GB M4 Pro).

    Call this *before* `mx.eval(model.parameters())` when loading lazily so
    off-rank layer weights are dereferenced prior to Metal buffer evaluation.
    Also installs a pipeline-aware `make_cache` wrapper so off-rank layer slots
    receive zero-byte `_OffsetStubCache` instances.
    """
    group = group or mx.distributed.init()
    rank = int(group.rank())
    world_size = int(group.size())

    inner, layers = resolve_inner_decoder(model)
    total_layers = len(layers)
    if world_size <= 1:
        return PipelineAssignment(
            rank=0,
            world_size=1,
            start_layer=0,
            end_layer=total_layers,
            total_layers=total_layers,
        )

    if len(ratios) != world_size:
        raise ValueError(
            f"len(ratios) ({len(ratios)}) must match distributed group size ({world_size})"
        )

    spans = compute_asymmetric_layer_split(total_layers, ratios)
    start_layer, end_layer = spans[rank]
    assignment = PipelineAssignment(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        total_layers=total_layers,
    )

    is_gemma4 = hasattr(inner, "previous_kvs") and hasattr(
        inner, "first_kv_shared_layer_idx"
    )

    for i in range(total_layers):
        orig = layers[i]
        use_sliding = bool(getattr(orig, "use_sliding", False))
        is_linear = bool(getattr(orig, "is_linear", False))
        layer_type = getattr(orig, "layer_type", None)
        if not assignment.owns_layer(i):
            if rank == 0 and i == total_layers - 1:
                stub: nn.Module = _FinalStageRecvWrapper(
                    src_rank=world_size - 1,
                    group=group,
                    use_sliding=use_sliding,
                    is_linear=is_linear,
                    layer_type=layer_type,
                    returns_gemma4_tuple=is_gemma4,
                )
            else:
                stub = _PassThroughLayer(
                    use_sliding=use_sliding,
                    is_linear=is_linear,
                    layer_type=layer_type,
                    returns_gemma4_tuple=is_gemma4,
                )
            _copy_layer_traits(orig, stub)
            layers[i] = stub
            continue

        wrapped: nn.Module = orig
        if rank > 0 and i == start_layer:
            wrapped = _BoundaryRecvWrapper(wrapped, src_rank=rank - 1, group=group)
        if i == end_layer - 1:
            next_rank = (rank + 1) % world_size
            wrapped = _BoundarySendWrapper(wrapped, dst_rank=next_rank, group=group)
        layers[i] = wrapped

    _repoint_model_probe_indices(inner, assignment)

    lm = getattr(model, "language_model", model)
    if hasattr(lm, "_supports_batch_invariant_decode"):
        setattr(lm, "_supports_batch_invariant_decode", lambda: False)

    # Worker ranks do not run the vision tower, embedding table, or final LM
    # head when Rank 0 owns early layers + post-vision embedding merge + final
    # norm/lm_head. Replacing embed_tokens, norm, and lm_head with zero-param
    # _IdentityEmbed drops their uninitialized parameters before mx.eval and
    # ensures lm(...) on Rank 1 returns the exact sent tensor from layer L-1.
    if rank > 0:
        for attr in ("vision_tower", "visual", "multi_modal_projector", "audio_tower"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        for target_obj in (inner, lm, model):
            for attr in ("embed_tokens", "norm", "lm_head"):
                if hasattr(target_obj, attr) and getattr(target_obj, attr) is not None:
                    setattr(target_obj, attr, _IdentityEmbed())
    if not hasattr(lm, "make_cache"):
        from mlx_vlm.models.cache import KVCache

        def _default_make_cache(_lm=lm) -> list[Any]:
            return [KVCache() for _ in range(len(_lm.layers))]

        setattr(lm, "make_cache", _default_make_cache)

    for target in ([model, lm] if lm is not model else [model]):
        orig_make_cache = getattr(target, "make_cache", None)
        if callable(orig_make_cache) and not getattr(
            orig_make_cache, "_heylook_pipeline_wrapped", False
        ):

            def _make_pipeline_cache(_orig=orig_make_cache) -> list[Any]:
                caches = list(_orig())
                for idx in range(len(caches)):
                    if not assignment.owns_layer(idx):
                        caches[idx] = _OffsetStubCache()
                return caches

            setattr(_make_pipeline_cache, "_heylook_pipeline_wrapped", True)
            setattr(target, "make_cache", _make_pipeline_cache)

    setattr(model, "_pipeline_assignment", assignment)
    return assignment


def read_total_layers_from_config(model_path: Path) -> int:
    """Read `num_hidden_layers` from `<model_path>/config.json`."""
    cfg_path = Path(model_path).expanduser() / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config") or cfg.get("llm_config") or cfg
    num_layers = text_cfg.get("num_hidden_layers") or cfg.get("num_hidden_layers")
    if not num_layers:
        raise ValueError(f"Could not determine num_hidden_layers in {cfg_path}")
    return int(num_layers)


@contextmanager
def selective_safetensors_load_scope(
    model_path: Path,
    assignment: PipelineAssignment,
) -> Iterator[None]:
    """Context manager that intercepts `mlx.core.load` during `mlx_vlm.utils.load`
    so off-rank `.safetensors` shards and off-rank weight keys are dropped
    *before* `sanitize_weights` or FP8 transforms run.
    """
    if assignment.world_size <= 1:
        yield
        return

    needed_paths = {p.resolve() for p in filter_safetensors_for_rank(model_path, assignment)}
    orig_mx_load = mx.load

    def _filtered_mx_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        file_str = str(file)
        if file_str.endswith(".safetensors"):
            resolved = Path(file_str).expanduser().resolve()
            if needed_paths and resolved not in needed_paths:
                return {}
            raw = orig_mx_load(file, *args, **kwargs)
            if isinstance(raw, dict):
                return {
                    k: v
                    for k, v in raw.items()
                    if should_keep_weight_key(k, assignment)
                }
            return raw
        return orig_mx_load(file, *args, **kwargs)

    with patch.object(mx, "load", side_effect=_filtered_mx_load):
        yield


def selective_pipeline_load(
    model_path: str | Path,
    *,
    ratios: Sequence[float] = (0.8, 0.2),
    group: Any = None,
    peers: Optional[Sequence[str] | str] = None,
    rank: int = 0,
    skip_audio: bool = True,
) -> tuple[Any, Any]:
    """Load an `mlx-vlm` model with selective per-rank `.safetensors` loading
    and asymmetric layer pipeline sharding.

    1. Resolves `PipelineAssignment` from `config.json` before reading weights.
    2. Runs `mlx_vlm.utils.load(..., lazy=True, strict=False)` inside
       `selective_safetensors_load_scope` so Rank 1 never opens or sanitizes
       Rank 0's shards.
    3. Applies `apply_asymmetric_pipeline` to replace off-rank layers with
       `_PassThroughLayer`s before calling `mx.eval(model.parameters())`.
    """
    from mlx_vlm.utils import get_model_path, load as vlm_load

    enable_fast_metal_synch()
    if group is None:
        group = init_pipeline_group(peers=peers, rank=rank)
    rank = int(group.rank())
    world_size = int(group.size())

    resolved_dir = get_model_path(str(model_path))
    total_layers = read_total_layers_from_config(resolved_dir)
    if world_size <= 1:
        model, processor = vlm_load(str(resolved_dir), skip_audio=skip_audio)
        assignment = PipelineAssignment(
            rank=0,
            world_size=1,
            start_layer=0,
            end_layer=total_layers,
            total_layers=total_layers,
        )
        setattr(model, "_pipeline_assignment", assignment)
        setattr(model, "_pipeline_group", group)
        return model, processor

    spans = compute_asymmetric_layer_split(total_layers, ratios)
    start_layer, end_layer = spans[rank]
    assignment = PipelineAssignment(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        total_layers=total_layers,
    )

    _log.info(
        "Loading pipeline rank %d/%d (layers %d..%d of %d) from %s",
        rank,
        world_size,
        start_layer,
        end_layer - 1,
        total_layers,
        resolved_dir,
    )
    with selective_safetensors_load_scope(resolved_dir, assignment):
        model, processor = vlm_load(
            str(resolved_dir),
            lazy=True,
            strict=False,
            skip_audio=skip_audio,
        )

    apply_asymmetric_pipeline(model, group=group, ratios=ratios)
    setattr(model, "_pipeline_group", group)
    mx.eval(model.parameters())
    model.eval()
    # Barrier across ranks after materialization
    mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32), group=group))
    return model, processor


def _read_cache_offset(cache: Any) -> int:
    if not cache:
        return 0
    for c in cache:
        if c is None:
            continue
        if hasattr(c, "_idx"):
            return int(c._idx)
        off = getattr(c, "offset", None)
        if isinstance(off, mx.array):
            return int(mx.max(off).item()) if off.size > 0 else 0
        if isinstance(off, int):
            return off
    return 0


def _sync_worker_cache_offset(cache: list[Any], target_offset: int) -> None:
    """Align `cache` on a worker rank to `target_offset` when Rank 0 started a
    new conversation (`target_offset == 0`) or trimmed/reused a prefix."""
    current = _read_cache_offset(cache)
    if target_offset == current:
        return
    if target_offset == 0:
        for c in cache:
            if isinstance(c, _OffsetStubCache):
                c._idx = 0
                c.offset = 0
            elif hasattr(c, "keys"):
                c.keys = None
                c.values = None
                c.offset = 0
                if hasattr(c, "_idx"):
                    c._idx = 0
            elif hasattr(c, "cache") and isinstance(c.cache, list):
                c.cache = [None] * len(c.cache)
        return
    if target_offset < current:
        trim_amount = current - target_offset
        for c in cache:
            if hasattr(c, "trim") and callable(c.trim):
                c.trim(trim_amount)


def install_rank0_step_coordinator(
    model: Any,
    group: Any = None,
) -> None:
    """Wrap `model.language_model.__call__` on Rank 0 so every forward pass
    broadcasts an 8-int32 control header to Rank 1 before executing layers:
    `[CMD_FORWARD, B, L, cache_offset, dtype_code, pos_ndim, pos_d0, has_rope_deltas]`.

    If `position_ids` or `rope_deltas` are present, Rank 0 sends them to Rank 1
    so M-RoPE multimodal positions match across ranks without Rank 1 needing
    `image_grid_thw` or `get_rope_index`.
    """
    group = group or mx.distributed.init()
    if int(group.size()) <= 1 or int(group.rank()) != 0:
        return

    lm = getattr(model, "language_model", model)
    if getattr(lm.__call__, "_heylook_coordinator_wrapped", False):
        return

    orig_call = lm.__call__

    def _coordinated_call(
        inputs: Optional[mx.array] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            B, L = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
            dtype_code = _DTYPE_TO_CODE.get(inputs_embeds.dtype, 1)
        elif inputs is not None:
            B = int(inputs.shape[0])
            L = int(inputs.shape[1]) if inputs.ndim >= 2 else 1
            dtype_code = 1
        else:
            raise ValueError("Either inputs or inputs_embeds must be provided")

        cache = kwargs.get("cache")
        if cache is None and len(args) >= 2:
            cache = args[1]
        cache_offset = _read_cache_offset(cache)

        position_ids = kwargs.get("position_ids")
        rope_deltas = kwargs.get("rope_deltas")
        pos_ndim = int(position_ids.ndim) if isinstance(position_ids, mx.array) else 0
        pos_d0 = int(position_ids.shape[0]) if pos_ndim > 0 else 0
        has_rope_deltas = 1 if isinstance(rope_deltas, mx.array) else 0

        header = mx.array(
            [
                CMD_FORWARD,
                B,
                L,
                cache_offset,
                dtype_code,
                pos_ndim,
                pos_d0,
                has_rope_deltas,
            ],
            dtype=mx.int32,
        )
        mx.eval(mx.distributed.send(header, 1, group=group))

        if pos_ndim > 0 and isinstance(position_ids, mx.array):
            pos_Len = int(position_ids.shape[-1])
            if pos_Len > L and cache_offset + L <= pos_Len:
                pos_slice = position_ids[..., cache_offset : cache_offset + L]
            else:
                pos_slice = position_ids
            mx.eval(
                mx.distributed.send(
                    mx.contiguous(pos_slice.astype(mx.int32)), 1, group=group
                )
            )

        if has_rope_deltas and isinstance(rope_deltas, mx.array):
            mx.eval(
                mx.distributed.send(
                    mx.contiguous(rope_deltas.astype(mx.int32)), 1, group=group
                )
            )

        return orig_call(inputs, *args, **kwargs)

    setattr(_coordinated_call, "_heylook_coordinator_wrapped", True)
    # Bind on the class or instance depending on how Python resolves __call__
    lm_cls = type(lm)
    if not getattr(lm_cls.__call__, "_heylook_coordinator_wrapped", False):
        orig_cls_call = lm_cls.__call__

        def _cls_call(self_lm: Any, *a: Any, **kw: Any) -> Any:
            if self_lm is lm:
                return _coordinated_call(*a, **kw)
            return orig_cls_call(self_lm, *a, **kw)

        setattr(_cls_call, "_heylook_coordinator_wrapped", True)
        lm_cls.__call__ = _cls_call  # type: ignore[method-assign]


def send_pipeline_shutdown(group: Any = None) -> None:
    """Send `CMD_SHUTDOWN` from Rank 0 to Rank 1 so the worker loop exits cleanly."""
    group = group or mx.distributed.init()
    if int(group.size()) <= 1 or int(group.rank()) != 0:
        return
    header = mx.array([CMD_SHUTDOWN, 0, 0, 0, 0, 0, 0, 0], dtype=mx.int32)
    mx.eval(mx.distributed.send(header, 1, group=group))


def run_pipeline_worker_loop(
    model: Any,
    group: Any = None,
) -> int:
    """Run the lockstep worker loop on Rank 1 (`rank > 0`).

    Receives the 8-int32 control header from Rank 0 before each prefill chunk
    or decode step, aligns the worker's local cache offset, constructs a dummy
    `inputs_embeds` tensor of shape `(B, L, hidden_size)`, and runs
    `model.language_model` (where `_BoundaryRecvWrapper` at `start_layer`
    replaces the dummy tensor with the real hidden state from Rank 0, executes
    layers `start_layer..end_layer-1`, and `_BoundarySendWrapper` sends the
    output hidden state back to Rank 0).
    """
    group = group or mx.distributed.init()
    lm = getattr(model, "language_model", model)
    inner, _ = resolve_inner_decoder(model)
    args = getattr(lm, "args", None) or getattr(inner, "args", None) or getattr(inner, "config", None)
    hidden_size = int(getattr(args, "hidden_size", 4096))

    worker_cache = lm.make_cache()
    header_template = mx.zeros((8,), dtype=mx.int32)
    steps = 0
    req_decode_steps = 0
    req_t0 = time.perf_counter()
    assignment: Optional[PipelineAssignment] = getattr(
        model, "_pipeline_assignment", None
    ) or getattr(inner, "_pipeline_assignment", None)
    layer_span = (
        f"layers {assignment.start_layer}..{assignment.end_layer - 1}"
        if assignment is not None
        else "worker slice"
    )

    _log.info(
        "Pipeline worker rank %d entering lockstep loop (%s, hidden_size=%d)",
        group.rank(),
        layer_span,
        hidden_size,
    )
    while True:
        header = mx.distributed.recv_like(header_template, 0, group=group)
        mx.eval(header)
        cmd, B, L, cache_offset, dtype_code, pos_ndim, pos_d0, has_rope_deltas = (
            int(x) for x in header.tolist()
        )
        if cmd == CMD_SHUTDOWN:
            if req_decode_steps > 0:
                elapsed = max(1e-6, time.perf_counter() - req_t0)
                _log.info(
                    "Pipeline worker rank %d finished request (%d decode steps, %.1f tok/s)",
                    group.rank(),
                    req_decode_steps,
                    req_decode_steps / elapsed,
                )
            _log.info("Pipeline worker rank %d received CMD_SHUTDOWN", group.rank())
            break
        if cmd == CMD_RESET_CACHE:
            worker_cache = lm.make_cache()
            req_decode_steps = 0
            continue

        if L > 1 or cache_offset == 0:
            if req_decode_steps > 0:
                elapsed = max(1e-6, time.perf_counter() - req_t0)
                _log.info(
                    "Pipeline worker rank %d finished request (%d decode steps, %.1f tok/s)",
                    group.rank(),
                    req_decode_steps,
                    req_decode_steps / elapsed,
                )
            _log.info(
                "Pipeline worker rank %d executing prefill (%s, tokens=%d, cache_offset=%d)",
                group.rank(),
                layer_span,
                L,
                cache_offset,
            )
            req_decode_steps = 0
            req_t0 = time.perf_counter()
        else:
            req_decode_steps += 1
            if req_decode_steps % 50 == 0:
                elapsed = max(1e-6, time.perf_counter() - req_t0)
                _log.info(
                    "Pipeline worker rank %d decoding (%s, %d tokens, %.1f tok/s)",
                    group.rank(),
                    layer_span,
                    req_decode_steps,
                    req_decode_steps / elapsed,
                )

        _sync_worker_cache_offset(worker_cache, cache_offset)
        dtype = _CODE_TO_DTYPE.get(dtype_code, mx.bfloat16)

        position_ids = None
        if pos_ndim == 2:
            pos_buf = mx.zeros((B, L), dtype=mx.int32)
            position_ids = mx.distributed.recv_like(pos_buf, 0, group=group)
            mx.eval(position_ids)
        elif pos_ndim == 3:
            pos_buf = mx.zeros((pos_d0, B, L), dtype=mx.int32)
            position_ids = mx.distributed.recv_like(pos_buf, 0, group=group)
            mx.eval(position_ids)

        rope_deltas = None
        if has_rope_deltas:
            rd_buf = mx.zeros((B, 1), dtype=mx.int32)
            rope_deltas = mx.distributed.recv_like(rd_buf, 0, group=group)
            mx.eval(rope_deltas)

        dummy_ids = mx.zeros((B, L), dtype=mx.int32)
        dummy_embeds = mx.zeros((B, L, hidden_size), dtype=dtype)
        fwd_kwargs: dict[str, Any] = {
            "inputs_embeds": dummy_embeds,
            "cache": worker_cache,
        }
        if position_ids is not None:
            fwd_kwargs["position_ids"] = position_ids
        if rope_deltas is not None:
            fwd_kwargs["rope_deltas"] = rope_deltas

        out = lm(dummy_ids, **fwd_kwargs)
        logits = out.logits if hasattr(out, "logits") else out
        eval_targets = [logits]
        for c in worker_cache:
            st = getattr(c, "state", None)
            if isinstance(st, (list, tuple)):
                eval_targets.extend(x for x in st if isinstance(x, mx.array))
            elif isinstance(st, mx.array):
                eval_targets.append(st)
        mx.eval(*eval_targets)
        steps += 1
    return steps
