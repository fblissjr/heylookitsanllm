# Parser strip/holdback unification -- design record

Last updated: 2026-07-23 (IMPLEMENTED in v1.39.12; written 2026-07-20 from
the /simplify + xhigh review passes, promoted out of internal/ the same day)

## Status

Implemented. Declared-specials stripping is one class --
`reasoning_parser.StripSpecials` -- composed by `select_reasoning_parser`
over whichever routing parser it picked. The problem inventory below is
kept as the record of WHY the shape is what it is; the "as built" section
records where it landed differently from the sketch.

## Problem inventory (as of v1.34.66 / commit 8492e93)

Four parsers route streamed text into content/thinking. Stripping DECLARED
special tokens (`ModelTemplateInfo.special_tokens`) from routed text was
implemented three different ways, with three different partial-token
holdback behaviors:

| Parser | Strip | Holdback for split specials | Status |
|---|---|---|---|
| `PassThroughParser` (reasoning_parser.py ~51) | per-chunk regex sub | NONE | a special split across two stream chunks leaks both halves |
| `HarmonyChannelParser` (~91) | `_drain` epilogue list-comp | `_safe_prefix_len` sized to `_HARMONY_MAX_TOKEN_LEN` (its own structural tokens, max 10 chars) | a declared special LONGER than 10 chars (e.g. `<|reserved_200000|>`) can straddle an emit boundary and leak |
| `GemmaChannelParser` (~250) | same epilogue pattern | `_safe_prefix_len` sized to `_GEMMA_MAX_TOKEN_LEN` (10) | same gap as harmony |
| `HybridThinkingParser` (thinking_parser.py ~122) | rolling per-kind pend buffer keyed on `max(len(strip_tokens))` | CORRECT | the reference implementation |

Two additional latent defects the unification fixed for free:

1. **Harmony's final-flush partial strip was dead code.** In
   `HarmonyChannelParser._drain(final=True)`, `_safe_prefix_len(final=True)`
   returns the WHOLE buffer, so the loop emitted everything (including a
   trailing partial control token) before the `if final and self._buffer:`
   leftover block ran -- that block always saw an empty buffer. An abort
   landing mid-`<|channel|>` therefore flushed literal garbage like
   `<|chan`. Gemma had the identical bug; fixed 2026-07-20 (commit 2c64d72)
   by pre-stripping the buffer AT THE TOP of the final drain. Harmony was
   left as-is (partials start `<|` and are held back during NORMAL
   streaming, so the window was abort-at-final only).
2. **Duplicated free-function machinery**: `_safe_prefix_len` bodies were
   byte-identical between Harmony/Gemma (differing only in the module
   constant they closed over); `_strip_partial_control` vs
   `_strip_partial_gemma_control` were two shapes of one idea (gemma's
   `any(t.startswith(tail))` membership check is the better one -- it
   generalizes to any token set).

## As built (v1.39.12)

`StripSpecials(inner, strip_tokens)` in `reasoning_parser.py` wraps any
`ReasoningParser`: `process_chunk`/`flush`/`reset` delegate to the inner
parser, then pass its deltas through the rolling per-kind holdback that
used to live in `HybridThinkingParser`. Composition happens in
`select_reasoning_parser`, and ONLY when the model declares specials -- the
wrapper exists solely to strip, so a model with no strip set gets the bare
parser (the factory's `isinstance` contract for structural tests is
therefore "unwrap `.inner` if wrapped").

Two deviations from the sketch above, both deliberate:

- **The holdback is prefix-set-based, not `rfind("<")`-based.** Hybrid's
  original scan assumed every declared special starts with `<`; Mistral's
  do not (`[INST]`, `[control_768]`), so a `[INST]` split across chunks
  would still have leaked through a verbatim lift. `_partial_prefixes`
  (lru_cached alongside `_compile_strip_pattern`) precomputes every proper
  prefix of every special, and the holdback keeps the longest suffix
  present in that set -- `O(max_token_len)` set lookups per delta, so it
  stays cheap for Mistral-sized token sets.
- **The channel parsers' final-flush leftover blocks are gone**, not just
  their strip epilogues: with `_strip_partial_token` moved to the TOP of
  the final drain, the drain loop always empties the buffer, so those
  blocks were unreachable in both parsers.

Kept inside the channel parsers, as planned: structural-token boundary
logic, now one free function `_safe_prefix_len(buffer, max_token_len,
final)` called with the parser's own constant at each site (the constant is
visible at the call site on purpose -- sizing holdback with the wrong
constant is the bug this whole record is about), and one
`_strip_partial_token(text, control_tokens)` used by both final drains.
`token_id` params stay for protocol compatibility, still ignored.

## Test strategy (as run)

Failing-first, all eight red before the refactor and green after:

- harmony abort-mid-token final flush, in `in_message` and in `preamble`
  (mirrors of the gemma pair) -- proved defect 1;
- a 19-char declared special straddling an emit boundary for harmony, for
  gemma content, and for a gemma `thought` body -- proved the sizing gap;
- the same for pass-through -- proved the missing holdback;
- a `[INST]`-shaped special split across chunks -- proved the `<`-only scan
  gap the verbatim lift would have kept;
- factory composition (no strip set -> no wrapper).

Existing suites (`test_reasoning_parser.py`, `test_thinking_parser.py`,
`test_per_request_parser.py`) pass unchanged except three `isinstance`
assertions that now unwrap `.inner`, and three tests that constructed
parsers with the removed `strip_tokens=` kwarg (rewritten to compose
`StripSpecials` explicitly). Non-boundary behavior is unchanged by design:
well-formed streams produce identical output.

## Acceptance

- One strip implementation, one holdback implementation, one partial-token
  helper, one `_safe_prefix_len` -- verified by grep for single
  definitions.
- The failing-first tests pass; `tests/unit/` + `tests/contract/` green.
- Live sanity on a real thinking model (gemma-4 + Qwen3.5 thinking tasks in
  `tests/eval/`) is the remaining check whenever a server is next up --
  unit tests cannot certify the templates/parsers/stop-token class.
