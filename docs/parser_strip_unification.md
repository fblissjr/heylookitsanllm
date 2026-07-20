# Parser strip/holdback unification -- design note

Last updated: 2026-07-20 (from the /simplify + xhigh review passes; TODO.md
carries the P2 pointer to this file; promoted out of internal/ 2026-07-20)

## Problem inventory (as of v1.34.66 / commit 8492e93)

Four parsers route streamed text into content/thinking. Stripping DECLARED
special tokens (`ModelTemplateInfo.special_tokens`) from routed text is
implemented three different ways, with three different partial-token
holdback behaviors:

| Parser | Strip | Holdback for split specials | Status |
|---|---|---|---|
| `PassThroughParser` (reasoning_parser.py ~51) | per-chunk regex sub | NONE | a special split across two stream chunks leaks both halves |
| `HarmonyChannelParser` (~91) | `_drain` epilogue list-comp | `_safe_prefix_len` sized to `_HARMONY_MAX_TOKEN_LEN` (its own structural tokens, max 10 chars) | a declared special LONGER than 10 chars (e.g. `<|reserved_200000|>`) can straddle an emit boundary and leak |
| `GemmaChannelParser` (~250) | same epilogue pattern | `_safe_prefix_len` sized to `_GEMMA_MAX_TOKEN_LEN` (10) | same gap as harmony |
| `HybridThinkingParser` (thinking_parser.py ~122) | rolling per-kind pend buffer keyed on `max(len(strip_tokens))` | CORRECT | the reference implementation |

Two additional latent defects the unification should fix for free:

1. **Harmony's final-flush partial strip is dead code.** In
   `HarmonyChannelParser._drain(final=True)`, `_safe_prefix_len(final=True)`
   returns the WHOLE buffer, so the loop emits everything (including a
   trailing partial control token) before the `if final and self._buffer:`
   leftover block runs -- that block always sees an empty buffer. An abort
   landing mid-`<|channel|>` therefore flushes literal garbage like
   `<|chan`. Gemma had the identical bug; fixed 2026-07-20 (commit 2c64d72)
   by pre-stripping the buffer AT THE TOP of the final drain
   (`_strip_partial_gemma_control` + regression tests
   `test_abort_mid_close_token_drops_partial` /
   `test_abort_mid_open_token_drops_partial`). Harmony was left as-is
   (partials start `<|` and are held back during NORMAL streaming, so the
   window is abort-at-final only) -- port the same top-of-final-drain
   pre-strip when unifying.
2. **Duplicated free-function machinery**: `_safe_prefix_len` bodies are
   byte-identical between Harmony/Gemma (differ only in the module constant
   they close over); `_strip_partial_control` vs
   `_strip_partial_gemma_control` are two shapes of one idea (gemma's
   `any(t.startswith(tail))` membership check is the better one -- it
   generalizes to any token set).

## Fix design (one refactor, not piecemeal)

Lift `HybridThinkingParser`'s rolling holdback into a shared wrapper that
ANY parser composes:

```python
class StripSpecials:  # reasoning_parser.py
    """Wraps a ReasoningParser; strips declared specials from routed
    deltas with a rolling per-kind holdback so a special split across
    deltas cannot leak. Keyed only on strip_tokens."""
    def __init__(self, inner: ReasoningParser, strip_tokens: frozenset[str]): ...
    # process_chunk/flush/reset delegate to inner, then pass deltas
    # through the holdback (Hybrid's current _cleaned/_flush_pend logic,
    # verbatim -- it is already correct and regression-tested)
```

- `select_reasoning_parser` composes: `StripSpecials(HarmonyChannelParser(),
  strip)` etc. `strip_tokens` leaves the individual parsers entirely.
- Delete: the harmony/gemma `_drain` strip epilogues, Hybrid's private
  `_cleaned`/`_flush_pend`/`_pend*` state (moves into the wrapper),
  PassThrough's inline strip (compose it too -- it gains holdback for free).
- Keep INSIDE the channel parsers: `_safe_prefix_len` (that one is about
  STRUCTURAL token boundaries, a different concern than declared-specials
  stripping) -- but dedupe it to one free function
  `_safe_prefix_len(buffer, max_token_len, final)` used by both, and one
  `_strip_partial_token(text, control_tokens)` (gemma's membership version)
  called at the TOP of both final drains (fixes harmony defect #1).
- `token_id` params stay (protocol compatibility), still ignored.

## Test strategy

Chunk-boundary behavior is pinned by `tests/unit/test_reasoning_parser.py`
(reproducer split-per-character tests, `TestStripTokensDefense`,
`TestImplicitThinkOpen`, gemma abort-partial tests) and
`tests/unit/test_thinking_parser.py`. After the refactor:

- All existing tests must pass UNCHANGED except ones that reach into
  private attrs (`_pend`, `_strip_pattern`) -- rewrite those against
  behavior.
- ADD: harmony abort-mid-token final-flush test (mirror the gemma pair) --
  it FAILS on current main, proving defect #1, then passes.
- ADD: long-declared-special straddling an emit boundary for harmony/gemma
  (e.g. 19-char `<|reserved_200000|>` split at chunk boundary inside a
  message body) -- fails on current main, passes after.
- Live sanity: one gemma-4 + one Qwen3.5 request via tests/eval
  (thinking tasks) on a running server.

## Acceptance

- One strip implementation, one holdback implementation, one partial-token
  helper; `grep -c "_strip_specials\|_safe_prefix_len"` shows single
  definitions.
- The two new failing-first tests pass.
- No behavior change for well-formed streams (existing suites green).
