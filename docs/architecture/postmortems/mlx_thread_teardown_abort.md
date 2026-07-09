# Bug: Process Abort (SIGTRAP) on Generation-Thread Teardown

Last updated: 2026-07-06

Fixed in v1.31.2.

## Symptoms

Server process died outright -- not an exception, not a 500, the whole
process aborted -- after a single streaming chat request from the v3 UI to
`Qwen3-VL-32B-Instruct-8bit`. The only signal was a fatal-error line on the
way down:

```
Fatal Python error: PyThreadState_Get
```

Reproduced with bare `curl` against `/v1/chat/completions`, so it was not
v3-specific: any client streaming a request to this model would have killed
the server the same way.

## Investigation

The macOS crash reporter wrote an `.ips` report for the faulting thread
(reports land under `<HOME>/Library/Logs/DiagnosticReports/` -- referenced
generically here, not as a repo path). The faulting-thread stack was:

```
_pthread_exit
  -> TLS cleanup
  -> mlx::core::detail::CompilerCache::~CompilerCache()
  -> tupledealloc
  -> Py_FatalError
  -> abort
```

That stack, read literally, says: a pthread is exiting, its thread-local
storage destructors are running, one of those destructors belongs to MLX's
`CompilerCache`, and tearing down that cache's contents (a Python tuple)
crashed hard enough to invoke Python's fatal-error path instead of a normal
exception.

Cross-referencing against `src/heylook_llm/streaming_utils.py` as it stood
before the fix: `async_generator_with_abort` created a fresh
`ThreadPoolExecutor(max_workers=1)` for every streaming request (a pattern
introduced in v1.30.5 to pin one generation to one thread -- see
[radix_thread_affinity.md](./radix_thread_affinity.md) for why pinning
exists) and called `shutdown(wait=False)` on it at stream end. Every
streaming request therefore spawned a thread that ran MLX work and then
tore it down. One dying MLX-tainted thread per streaming request.

**Why it took a specific model to surface it**: `mlx_lm`'s sampler
functions are `@partial(mx.compile, ...)`. When a `mx.compile`d *Python*
function runs on a thread, its compiled-graph entry can be cached in MLX's
thread-local `CompilerCache`, and that entry can hold references to Python
objects. `gemma-4-31B` ran 40+ clean streaming requests in the v3 E2E
session the day before without ever hitting this -- its decode path
(`cache_type = "standard"`) apparently never populated the cache with an
entry holding Python objects on the worker thread. `Qwen3-VL-32B` with
`cache_type = "quantized"` crashed on the **first** request -- its
quantized-KV decode path exercises a compiled-sampler variant that does.

**Confirmation this had already been happening, invisibly**: a second
`.ips` report, timestamped 2026-07-05 19:05 (the evening before this was
diagnosed), had an *identical* faulting stack. It fired at `pkill`-time
shutdown of a `gemma` server -- i.e. the same teardown path aborts the
process even on ordinary shutdown once a tainted thread exists, it's just
that a shutdown-time abort looks like "the process exited," not a crash a
user notices. This one bug had two independent trigger paths (a live
streaming request finishing, and process shutdown while a tainted thread
existed) and had likely been present since the per-request-executor
pattern shipped in v1.30.5.

## Root Cause

pthread TLS destructors run during `_pthread_exit`, which happens *after*
the thread's Python-level state (`PyThreadState`) has already been torn
down by the time the `ThreadPoolExecutor` worker function returns and the
thread unwinds. MLX's `CompilerCache` destructor does not know or care
about this ordering -- it just drops its cached Python objects
(`tupledealloc`) when the thread-local storage slot is cleaned up. Doing
that without holding the GIL is exactly the situation `Py_FatalError`
exists to catch: it is not safe, ever, to manipulate a Python object's
refcount without the GIL, and by the time TLS cleanup runs, there is no
valid Python thread state left to acquire it with.

This is a genuine ordering bug in the interaction between pthread TLS
destructors and CPython's thread teardown -- not something the mlx-lm
sampler code or heylookitsanllm's provider code did wrong. It only became
observable because the codebase's own pattern (fresh thread per request,
destroyed at end) forced MLX-tainted threads to exit constantly.

## Resolution

`_PinnedExecutorPool` in `src/heylook_llm/streaming_utils.py` (lines
13-47): a pool that leases persistent single-worker `ThreadPoolExecutor`
instances instead of creating and destroying one per request.
`async_generator_with_abort` (lines 63-155) now calls
`_executor_pool.acquire()` to get a worker and `_executor_pool.release()`
to return it when the stream ends -- the executor itself is **never**
shut down while the process lives. The per-request pinning invariant this
replaces is unchanged: a leased executor still serves exactly one
generation at a time, start to finish, on one thread.

If a worker's generator `close()` call times out (a wedged generation),
that worker is deliberately **not** returned to the pool and not shut
down -- it is retired (leaked) rather than risking another destructive
teardown or handing a wedged thread to the next request.

This also folds in the v1.31.1 review's open follow-up about per-request
thread churn growing MLX's stream registry (each new thread materializes
its own `mx.new_thread_local_stream` instance) -- with the pool, stream
registry growth is bounded by the number of concurrently admitted
requests, not by total request count.

## Why This Approach

**Rejected: a single shared generation thread for all requests.** This
was the first alternative considered, since it would bound thread count to
1 instead of "however many requests are concurrently admitted." It
deadlocks. `GenerationGate.acquire()` (the FIFO admission gate,
`src/heylook_llm/providers/common/generation_gate.py`) is called *inside*
the provider's generation generator, on the worker thread, the first time
the generator's `next()` runs. With one shared thread: request A starts
generating (holds the gate, running on the shared thread). Request B
queues; its generator's first `next()` call -- including its call to
`gate.acquire()` -- would also have to run on that same shared thread,
since it is the only one available. `gate.acquire()` blocks until A
releases. But A's *remaining* `next()` calls also need that same thread to
make progress, and the thread is now stuck inside B's blocking
`acquire()`. A can never finish, so A never releases the gate, so B's
`acquire()` never returns. Deadlock, guaranteed under any concurrency.

**Rejected: clear the compile cache before thread exit.** There is no
public MLX API to selectively clear `CompilerCache` entries, and doing so
would not actually fix the bug -- the destructor-ordering problem (TLS
cleanup after `PyThreadState` teardown) exists independent of what the
cache holds; an empty-but-still-thread-local cache could still be
populated again before the next teardown. The only reliable fix is to
stop tearing down MLX-tainted threads at all.

The chosen fix (persistent pool, lease/reuse) sidesteps the deadlock
because concurrently admitted requests each get their **own** persistent
thread from the pool -- it grows to admitted concurrency (bounded by
`1 + max_queue_depth`, ~9 threads at the default `max_queue_depth=8`) and
never shrinks. No thread that ever ran MLX work is destroyed while the
process lives.

## Invariant Going Forward

**Never destroy a thread that ran MLX work.** Any future change to the
streaming/executor layer must preserve this. If the pool's bound on thread
count (`1 + max_queue_depth`) ever becomes a real resource concern, the
fix is to reduce `max_queue_depth`, not to reintroduce executor teardown.

## Prevention

`tests/unit/test_streaming_executor_pool.py` (5 tests: pinning, sequential
reuse, distinct concurrent leases, release semantics, executor survives
stream end). TDD: red on missing pool, green after.

Live verification: the exact crash repro (same 3-message body, same model)
killed the server on request 1 before the fix; 6/6 requests clean after,
zero fatal errors in the log.

## Open Upstream Issue

This is arguably an MLX bug: `CompilerCache`'s TLS destructor should
acquire the GIL before touching Python objects, or defer the Python
decrefs to a point where the GIL is known to be available (e.g. via
`Py_AddPendingCall` from a signal-safe context, or simply not holding
Python references in a native TLS-cleaned-up structure at all).

**DRAFTED 2026-07-06**: a ready-to-file issue (stacks re-extracted and
verified from both `.ips` reports; minimal-repro attempt documented as
negative) is at `internal/backend/upstream_mlx_compilercache_issue.md`.
Filing is a public action under the owner's account, so it is left as a
30-second owner review-and-paste. Our fix (never destroy MLX-tainted
threads) is correct regardless of whether/when upstream addresses this --
long-lived generation threads are the better shape for this workload
independent of the bug.

## Related

- [radix_thread_affinity.md](./radix_thread_affinity.md) -- the other
  thread-lifecycle bug found in the same window. That fix (materializing
  KV snapshots before publishing them) is independent of and complementary
  to this one: it makes published radix-cache values safe to evaluate from
  *any* thread, including the pool's other worker threads, regardless of
  whether the creating thread is later reused, blocked forever (a retired
  worker), or -- pre-1.31.2 -- destroyed. This fix stops threads from being
  destroyed at all; that fix stops radix-tree values from depending on the
  creating thread staying alive in the first place. Neither makes the
  other redundant.

## Update 2026-07-07: already fixed upstream (unreleased)

Pre-submission verification found ml-explore/mlx#3619 + PR #3628
("Fix threaded compile cache cleanup", commit a8ae6d1d5, merged
2026-06-05, present in coderef/mlx, in NO release tag). Root cause per
the diff: v0.31.2 registered the GIL-holding ThreadCleanup only on the
thread that CALLS mx.compile; the fix registers it in
PyCompiledFun::call_impl so every executing thread gets cleanup. This
also explains our negative minimal repro: single-array outputs hold a
shared sentinel (harmless), tuple outputs hold a fresh GC-tracked tuple
(fatal path) -- upstream's regression test
(test_compile_tuple_output_in_thread) fires with a tuple return. Do not
file a new issue; comment on #3619 instead. SHA-pin past a8ae6d1d5 when
convenient; keep _PinnedExecutorPool regardless (it also bounds
stream-registry growth).
