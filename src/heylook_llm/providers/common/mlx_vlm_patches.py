"""Local carries of heylook's own open mlx-vlm PRs, applied per generator.

The pin stays on an upstream SHA; a fix we have filed but upstream has not
merged is carried here as a wrapper instead of a fork. Each patch:

- names its PR and keeps the PR's own condition, so the carried behaviour is
  the reviewed diff, not a local variant;
- wraps the upstream method and falls through to it for every other case;
- binds to one request's ``BatchGenerator`` (``vlm_engine.generate`` calls
  ``apply_to`` right after building it), never to the class, so nothing is
  left patched in the process;
- applies only while the wrapped signature is the one it was written
  against, and logs and stands down otherwise.

Retirement is enforced, not remembered: tests/contract/test_mlxvlm_surface.py
runs the UNPATCHED upstream restore path and fails once it already behaves
like the patch, naming the patch to delete. Moving the pin past the PR is the
moment that test goes red.
"""

import inspect
import logging

# Blaizzy/mlx-vlm#2356: a restored single row gets the cache a cold row would.
# mlx-vlm gives a cold single row the model's own caches (PromptProcessingBatch)
# unless it drafts, quantizes KV or the model has no make_cache, but sends a
# restored one through APCCoordinator.merge_rows, which builds BatchKVCache
# even for one row. qwen3_5's single-row shortcut then extract()s and
# re-merge()s the full KV of every full-attention layer on every decode step,
# so a restored request decodes slower than the same request run cold. The PR
# (as updated 2026-09-30) routes that row at the restore site in
# generate/ar.py; this carry reaches the same outcome at merge_rows, the one
# call the restore site makes, gated per generator by the same rule.
PATCH_2356 = "mlx-vlm#2356"
MERGE_ROWS_PARAMS = ("picks", "prefix_lens", "kv_quant_config")  # bound method


def plain_single_row_generator(bg) -> bool:
    """#2356's rule from the generator's side: no drafting, no KV
    quantization, and a model with its own ``make_cache``. A generator with
    a drafter keeps batch caches on restore, as it does cold."""
    drafting = (getattr(bg, "draft_model", None) is not None
                and getattr(bg, "draft_kind", None) is not None)
    return (not drafting and getattr(bg, "kv_bits", None) is None
            and hasattr(getattr(bg, "model", None), "make_cache"))


def single_restored_row(picks, kv_quant_config) -> bool:
    """One row, restored, with its own warm cache (a checkpoint hit), no KV
    quantization. Block-mode hits carry no warm cache and still merge."""
    return (len(picks) == 1 and picks[0] is not None
            and kv_quant_config is None
            and picks[0].get("warm_cache") is not None)


def _patch_merge_rows(coord) -> bool:
    upstream = coord.merge_rows
    if getattr(upstream, "_heylook_patch", None) == PATCH_2356:
        return True
    if tuple(inspect.signature(upstream).parameters) != MERGE_ROWS_PARAMS:
        logging.warning(
            "mlx-vlm#2356 carry not applied: APCCoordinator.merge_rows "
            "signature changed; restored qwen3_5 requests may decode slower "
            "than cold ones")
        return False

    def merge_rows(picks, prefix_lens, *, kv_quant_config=None):
        # An upstream rename the signature check cannot see (the pick's keys,
        # the manager's lock or stats) falls through to upstream: slower,
        # never a failed request. Nothing is mutated before the increment.
        try:
            if single_restored_row(picks, kv_quant_config):
                with coord.manager.lock:
                    coord.manager.stats.restored_tokens += prefix_lens[0]
                return picks[0]["warm_cache"], prefix_lens[0]
        except (AttributeError, KeyError, TypeError) as exc:
            logging.warning("mlx-vlm#2356 carry fell through to upstream: %r", exc)
        return upstream(picks, prefix_lens, kv_quant_config=kv_quant_config)

    merge_rows.__doc__ = upstream.__doc__
    setattr(merge_rows, "_heylook_patch", PATCH_2356)
    setattr(merge_rows, "_heylook_upstream", upstream)
    coord.merge_rows = merge_rows
    return True


def apply_to(bg) -> bool:
    """Install every carried patch on this generator. True when #2356's
    carry is in force for it; False for a generator the rule leaves on
    batch caches, one without a prefix cache, or a stood-down carry."""
    coord = getattr(bg, "apc", None)
    if coord is None or not plain_single_row_generator(bg):
        return False
    return _patch_merge_rows(coord)
