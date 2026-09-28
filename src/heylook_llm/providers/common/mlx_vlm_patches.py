"""Local carries of heylook's own open mlx-vlm PRs, applied at MLX load.

The pin stays on an upstream SHA; a fix we have filed but upstream has not
merged is carried here as a wrapper instead of a fork. Each patch:

- names its PR and keeps the PR's own condition, so the carried behaviour is
  the reviewed diff, not a local variant;
- wraps the upstream function and falls through to it for every other case;
- applies only while the wrapped signature is the one it was written
  against, and logs and stands down otherwise.

Retirement is enforced, not remembered: tests/contract/test_mlxvlm_surface.py
runs the UNPATCHED upstream function and fails once it already behaves like
the patch, naming the patch to delete. Moving the pin past the PR is the
moment that test goes red.
"""

import inspect
import logging

_applied = False

# Blaizzy/mlx-vlm#2356: a single restored row keeps its own warm cache.
# upstream merge_rows merges even one row into BatchKVCache, and qwen3_5's
# single-row shortcut then extract()s and re-merge()s the full KV of every
# full-attention layer on every decode step, so a request restored from the
# prefix cache decodes slower than the same request run cold.
MERGE_ROWS_PARAMS = ("self", "picks", "prefix_lens", "kv_quant_config")


def single_restored_row(picks, kv_quant_config):
    """#2356's condition: one row, restored, with its own warm cache, no KV
    quantization. Block-mode hits carry no warm cache and still merge."""
    return (len(picks) == 1 and picks[0] is not None
            and kv_quant_config is None
            and picks[0].get("warm_cache") is not None)


def _patch_merge_rows(coordinator_cls) -> bool:
    upstream = coordinator_cls.merge_rows
    if getattr(upstream, "_heylook_patch", None) == "mlx-vlm#2356":
        return True
    if tuple(inspect.signature(upstream).parameters) != MERGE_ROWS_PARAMS:
        logging.warning(
            "mlx-vlm#2356 carry not applied: APCCoordinator.merge_rows "
            "signature changed; restored qwen3_5 requests may decode slower "
            "than cold ones")
        return False

    def merge_rows(self, picks, prefix_lens, *, kv_quant_config=None):
        # An upstream rename the signature check cannot see (the pick's keys,
        # the manager's lock or stats) falls through to upstream: slower,
        # never a failed request. Nothing is mutated before the increment.
        try:
            if single_restored_row(picks, kv_quant_config):
                with self.manager.lock:
                    self.manager.stats.restored_tokens += prefix_lens[0]
                return picks[0]["warm_cache"], prefix_lens[0]
        except (AttributeError, KeyError, TypeError) as exc:
            logging.warning("mlx-vlm#2356 carry fell through to upstream: %r", exc)
        return upstream(self, picks, prefix_lens, kv_quant_config=kv_quant_config)

    merge_rows.__doc__ = upstream.__doc__
    setattr(merge_rows, "_heylook_patch", "mlx-vlm#2356")
    setattr(merge_rows, "_heylook_upstream", upstream)
    coordinator_cls.merge_rows = merge_rows
    return True


def apply() -> None:
    """Install every carried patch once per process. Idempotent."""
    global _applied
    if _applied:
        return
    from mlx_vlm.apc_coordinator import APCCoordinator

    _patch_merge_rows(APCCoordinator)
    _applied = True
