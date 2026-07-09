# Bug: Radix Cache Reuse Crashes with "There is no Stream(gpu, N) in current thread"

Last updated: 2026-07-06

Fixed in v1.31.1.

## Symptoms

`RuntimeError: There is no Stream(gpu, N) in current thread.` (N varied
across occurrences -- 10, 19 seen in one E2E session log; 8 occurrences
total that session). Raised inside `mlx_lm`'s `generate_step`, specifically
at `mx.eval([c.state for c in prompt_cache])`.

Before the error-surfacing fix shipped in the same release, this crash was
delivered to clients as normal assistant content -- the provider yielded
the exception text as a regular generation chunk, so the error string was
rendered (and in some cases persisted) as if it were a model response. See
the "Also fixed alongside" section below.

## Investigation

This is a recurrence of the v1.30.5 bug class ("There is no Stream(gpu, 0)
in current thread"), but in a different spot. The v1.30.5 fix
(`mx.new_thread_local_stream` for the *generation* stream) addressed
`wired_limit`'s synchronization call; this crash was about *cached KV
state*, not the live generation stream.

**Naive repro attempts were vacuous.** Ordinary multi-turn conversation
resends (edit a message, regenerate, continue a conversation) essentially
never hit this, and the reason matters: radix snapshots attach only to the
deepest node whose path represents a **full** 32-token block, and a query
only counts as a "hit" against a stored snapshot if the query's own tokens
fill that same full block identically. Multi-turn conversations almost
always diverge mid-block -- role/template markers at turn boundaries
(`<|im_start|>assistant`, etc.) shift the token stream just enough that
the new request's blocks don't align with a previously-stored full block.
Verified: naive multi-turn repro attempts produced 0 cache hits.

**Deterministic repro**: seed a conversation with `max_tokens=1`. This
keeps the end-of-generation snapshot's total token count within the
*prompt's own* full blocks (no straggling generated tokens push the
snapshot past a block boundary), so a resend of the identical request is
guaranteed to hit that block. Then resend the identical request. 4/4
resends crashed with the exact production error before the fix; 4/4 were
clean after, with radix hits confirmed in the log both ways (A/B proof).

**Root cause pinned by direct probe, not inference from the stack trace
alone.** The relevant MLX stream semantics were confirmed experimentally:

- Spawn a thread, create `mx.new_thread_local_stream(mx.gpu)` on it,
  schedule a lazy array on that stream, join (destroy) the thread, then
  `mx.eval` the lazy array from a different thread -> raises "no
  Stream(gpu, N)".
- Repeat with a CPU stream instead of GPU -> succeeds.
- Repeat with the per-thread *default* stream (not an explicitly created
  thread-local one) -> succeeds.

This establishes: **GPU `ThreadLocalStream`s are destroyed when their
owning thread exits; CPU streams and per-thread default streams are
globally registered and survive thread death.** This is exactly why the
regression test (`tests/unit/test_snapshot_thread_affinity.py`) is
Metal-gated (`pytest.mark.skipif(not mx.metal.is_available(), ...)`) -- a
CPU-only run of this test cannot go red for this bug; it would pass for
the wrong reason and give false confidence.

## Root Cause

`snapshot_kv()` (`src/heylook_llm/providers/common/cache_helpers.py`,
lines 83-110, before the fix) captured each cache layer's `.state`
property directly. For `mlx_lm`'s `KVCache`, `.state` returns a **lazy**
slice (`keys[..., :offset, :]`) scheduled on whatever stream is active on
the *calling* thread at the moment `.state` is read.

Generation runs on a fresh single-worker thread per request
(`streaming_utils.async_generator_with_abort`'s per-request executor, at
the time this bug was live -- see
[mlx_thread_teardown_abort.md](./mlx_thread_teardown_abort.md) for the
later, unrelated fix to that pattern), and `generation_stream` -- both
`heylookitsanllm`'s own and `mlx_lm`'s internal one -- is
`mx.new_thread_local_stream`. So the lazy slice captured by `snapshot_kv`
was scheduled on a GPU thread-local stream tied to the *generating*
thread.

`store_generation_cache` published that lazy node into the shared,
cross-request radix tree, on the (correct-at-the-time, wrong-in-general)
assumption that "capturing these references is cheap" -- true when
everything stays on one thread, false once the value crosses threads.
When a **later** request landed on a **different** thread (a fresh
executor per request) and matched the cached prefix,
`process_prompt_with_cache` (`prompt_cache.py`) restored that snapshot,
and `mlx_lm`'s `generate_step` called `mx.eval([c.state for c in
prompt_cache])` on it. The lazy node still pointed at the original
generating thread's now-dead GPU stream. MLX could not resolve it.

## Resolution

`snapshot_kv()` (`cache_helpers.py`, lines 83-110) now calls `mx.eval()` on
the captured per-layer states before returning them:

```python
snapshots = []
for layer in cache:
    if hasattr(layer, 'state') and not layer.empty():
        snapshots.append(layer.state)
    else:
        snapshots.append(None)
mx.eval([s for s in snapshots if s is not None])
return snapshots
```

This materializes the arrays into real memory **while still running on the
generating thread**, where the referenced streams are still alive. The
snapshot that gets published into the radix tree is therefore a fully
materialized value with no pending dependency on any thread-local stream
-- safe to `mx.eval` again later from any thread, including a dead one's
successor. Cost: one batched `mx.eval` sync at end of generation (this
sync would have happened anyway, just later and on the wrong thread).

`process_prompt_with_cache` and `store_generation_cache` in
`prompt_cache.py` were not changed for this fix -- the trim-to-prefix logic
added for the earlier
hybrid-model bug
(`restore_kv_from_snapshot(trim_to=matched_len)`) already assumed
materialized values and continued to work once `snapshot_kv` started
providing them.

`vision_feature_cache.py` was audited for the same hazard: safe. Its
cached image features are scheduled on a default stream (globally
registered, survives thread death per the probe above), not a
thread-local one.

## Why This Approach

The only alternative considered was not evaluating eagerly at all and
instead keeping the generating thread alive until any snapshot referencing
its stream had been consumed -- rejected as strictly worse: it would
couple cache-tree lifetime to thread lifetime (a snapshot could be read
an arbitrary number of times, by an arbitrary number of future requests,
so "keep the thread alive until nobody needs its snapshots" has no clean
end condition). Materializing the value once, at the point it's produced,
removes the thread dependency entirely and is a bounded, one-time cost.

## What Future Work Must Respect

**Radix snapshots must be `mx.eval`'d on the generating thread before
being published.** Any future change to `store_generation_cache` or
`snapshot_kv` that reintroduces lazy values into the shared tree
reopens this exact bug.

## Related Follow-On: Radix Bypassed for Non-Standard Caches (v1.32.0)

This fix makes published snapshot *values* thread-safe, but it does not by
itself make prefix-trimmed restoration numerically correct for every cache
type. `restore_kv_from_snapshot` trims by slicing
`keys[..., :trim_to, :]` -- correct for plain `KVCache` tensors, wrong for
`QuantizedKVCache` (state is a packed tuple, not a plain tensor) and
impossible for `RotatingKVCache` (fixed-size ring buffer, no stable
absolute offset to trim to). This risk was documented in
`radix_cache.py`'s docstring but not enforced.

As of v1.32.0, `process_prompt_with_cache` sets
`prompt_cache._radix_eligible = (cache_config.get("cache_type", "standard")
== "standard" and not cache_config.get("max_kv_size"))`
(`prompt_cache.py`, lines 255-258), and `store_generation_cache` mirrors
the same flag before storing (lines 312-315). Both the lookup and the
store side of the radix path are now skipped entirely for any model
configured with `cache_type != "standard"` or with `max_kv_size` set --
closing the silent-wrong-output risk at the cost of no prefix reuse for
those models. Tests: `tests/unit/test_prompt_cache_gating.py`.

**This gate is keyed on the `cache_type` config field, not on model
architecture.** It does not, by itself, address the *different* hybrid-cache
risk of hybrid caches (models like Qwen3.5 that mix
`KVCache` and `ArraysCache` layers via a custom `model.make_cache()`) --
a separate, independently-tracked risk.

## Also Fixed Alongside (Same Release, Different Bug)

Shipped in the same v1.31.1 release since the crash above had been
shipping as fake assistant content: `MLXErrorChunk` (module-level,
`is_error = True`) is now surfaced as a real error rather than delivered
as `.text` content. See
[mlx_provider.md](../mlx_provider.md) for the
current error-contract description across the four consumer sites. The
A/B verification run for this bug incidentally verified that fix live too
-- crashes arrived as proper SSE error payloads / HTTP 500s, not garbled
content, in the "fix restored" runs.

## Prevention

`tests/unit/test_snapshot_thread_affinity.py` -- 2 tests, Metal-gated.
Directly exercises the failure mode: build a lazy KV state on a worker
thread under a GPU thread-local stream, join the thread, then `mx.eval`
the snapshot from the test's own thread. Must not raise.
