# v3 design language

Last updated: 2026-07-11

The written form of the design system that previously lived only in `css/app.css`
comments. This is a **seed** (plan Phase 4 item 2): it formalizes what v3 already
does well enough that new UI — starting with the j-space visualizer — can stay
on-system, and records the visualizer's load-bearing paradigm decision. It is not
a completed impeccable pass; run `/impeccable audit` per page before calling the
visual work done.

Product context (users, register, anti-references) lives in the repo-root
`PRODUCT.md` — read it first. The one-line version: warm minimal, thinking space
first, desktop + iPhone Safari co-primary, no SaaS-dashboard grammar.

## 1. Tokens (authoritative values in `css/app.css` `:root`)

All color is **OKLCH**. The palette is a pure-white writing surface with warmth
carried by the honey-bronze brand pair, never by surface tints.

| Role | Token | Note |
|------|-------|------|
| page | `--bg` | pure white |
| panels/sidebars | `--surface`, `--surface-2` | warm paper; `-2` = hover/pressed/code |
| text | `--ink`, `--ink-muted` | muted stays ≥4.5:1 on `--bg` |
| placeholders only | `--ink-faint` | ~3.5:1 — never real text |
| hairlines | `--line`, `--line-strong` | |
| selection glints | `--brand` (honey gold), `--brand-tint` | active nav, chosen rows |
| actions/links/focus | `--accent` (deep bronze), `--accent-hover`, `--on-accent` | |
| destructive | `--danger`, `--danger-tint` | |

Type: system stack (`--font`) + `--mono` for anything numeric, token-literal, or
telemetry. Scale: `--text-sm 0.8125rem / --text-ui 0.875rem / --text-body 1rem /
--text-lg 1.1875rem` — four sizes, no more. Radii: `--r-ctl 6px / --r-card 10px /
--r-big 14px`. Motion: `--t-fast 140ms var(--ease)`; every animation has a
`prefers-reduced-motion` fallback (global kill switch in `app.css`).

## 2. The data-strength color system (chips)

v3 encodes scalar "strength" (probability, confidence, rank) as the **background
hue of a small mono chip**, with lightness and chroma held fixed so hue is the
only channel carrying data:

```
strength t ∈ [0,1]  →  oklch(0.86 0.11 (25 + t·120))     // 25=red … 145=green
```

Rules:

- **Fixed L=0.86, C=0.11.** Hue alone moves. This keeps every chip readable with
  the same ink and keeps a row of chips from strobing in perceived brightness.
- **Chip ink is fixed near-black** (`#1a1a1a`-class), not `--ink`: chip
  backgrounds are data, not theme, and stay light enough for dark ink at L=0.86.
- **What t means is per-surface and must be titled** (a `title` tooltip at
  minimum): explore chips use token probability; the jspace strip uses
  within-layer rank; the jspace heatmap uses normalized inverse entropy
  (low entropy = confident = green); the risk badge uses `1 − risk`.
- Chips are `--mono`, `--text-sm` or smaller, radius 3–6px, whitespace rendered
  as visible glyphs (`·`, `⏎`/`↵`, `⇥`/`→`; empty string = `∅`) so token
  boundaries are honest.
- Bars (explore alternatives, perf usage) use `--brand` at reduced opacity /
  `color-mix`, not the strength ramp — bars encode magnitude by length, so they
  don't need a hue channel.

## 3. Selection & pinning grammar

One selection language everywhere:

- **Selected datum**: 2px accent ring — `box-shadow: 0 0 0 2px var(--accent)`
  (explore `.tok--selected`). Never a fill change: fills carry data.
- **Hover affordance on selectable data**: 1px `--line-strong` ring.
- **Chosen/active item in a list**: `--brand-tint` fill (nav, conv list,
  explore's chosen alternative).
- **Pinned detail readout**: a bordered `--surface` card `aside` beside the data
  (explore's `.explore__detail`), *in the document flow* — not a floating
  overlay. On phone widths it stacks below the data, full width.
- Pin semantics: **single selection**; click to pin, click the same datum or
  press `Escape` to unpin, clicking another datum re-pins. Keyboard nav where it
  exists (explore's arrows) moves the pin.
- **Echo highlighting** (lifted from the jlens-qwen36 reference): when a cell is
  pinned, other cells whose top token matches get a soft secondary marker
  (1px accent ring at reduced alpha) — "where else does this token win" — and it
  clears with the pin.

## 4. J-space visualizer: aggregation vs. matrix (DECIDED)

**Matrix-first, aggregation later as an overlay mode — the two compose.**

- The apply API already returns a small, bounded grid: `band_layers` rows
  (the mid-depth workspace band, ~⅓ of the network) × at most the last N prompt
  positions (server default caps the heatmap width). At this scale a full
  layer×position matrix is *cheap*, already shipped, and the most direct answer
  to "walk the workspace layer by layer" — the reason jlens-qwen36's virtualized
  matrix needs springs and row-windowing (4000+ positions × 65 layers) simply
  does not exist here. No virtualization; if the grid ever grows past viewport
  width it scrolls inside its own `overflow-x: auto` container.
- Orientation stays **rows = layers (deep → shallow, reading down), cols =
  positions**, matching the existing strip and heatmap. (jlens-qwen36 transposes
  this; our grids are wide-and-short, theirs tall-and-narrow.)
- **Neuronpedia's two scalable ideas are the designated growth path**, not the
  starting point: a slot-based layer-range slider (click = one layer, drag =
  contiguous range, hover = live single-layer preview, reset affordance;
  re-scoping is pure client-side filtering — no refetch) and a most-common-token
  aggregation sidebar (count of top-k appearances over the scoped
  positions×layers, sorted desc). These land with sequence item 2 and become the
  default reading mode if/when live streaming (item 3) makes transcripts long.
- The **answer-onset column is the privileged column**: it's where the strip
  reads (`positions=[-1]`), it's the only column with full top-k today, and it's
  the column the hallucination-risk features derive from. It gets a visible
  marker; pinned readouts there show the full top-N. Other columns show what the
  API returns (top-1 + entropy) until the per-cell top-N analyze extension
  lands, and the detail panel says so rather than pretending.
- Anti-goals, per `PRODUCT.md`: no glass/glow aesthetic, no spring physics for
  its own sake, no dashboard-density. The visualizer is a reading surface;
  chrome earns every pixel.

## 5. Honest states

Streaming, busy, empty, and error states are designed, not defaulted:

- Analyze is slow (seconds, serialized behind the generation gate): the page
  disables the trigger, says what it's doing in the mono status line, and
  surfaces failure as text in the same line — never a dead button.
- Empty states explain the *path out* (e.g. jspace's "no lens installed" names
  the directory to install one into).
- Buttons never spin; the status line speaks.

## 6. Special tokens are content, not chrome (SHOW, don't hide)

Chat structure tokens — `<|im_start|>`, `<|im_end|>`, `assistant`, `<bos>`,
`<think>`/`</think>`, role markers — are **load-bearing signal, not noise**. They
say *where in the turn the model is*, which is exactly what an interpretability
surface exists to expose. Hiding them is the opposite of this project's
measure-first ethos, and it's what the reference tool (Neuronpedia's Jacobian
Lens) pointedly does *not* do — it renders `<|im_start|>assistant` in the
transcript on purpose.

**Why (so this isn't re-litigated):** stripping specials doesn't just lose
context, it *manufactures a class of bug*.
1. **Position integrity.** Prefill and activation patching address activations by
   position index. If the UI hides tokens the model actually sees, the token the
   user clicks no longer maps to that index — a silent off-by-N between UI and
   reality, undebuggable because the discrepancy is invisible by construction.
2. **The specials are often the object of study** — the assistant onset, the
   `<think>`/`</think>` boundaries, the token where a refusal fires. That's where
   the interesting disposition lives; hide it and you've hidden the answer.
3. **Template bugs go invisible** — this repo has a documented chat-template
   minefield (doubled BOS, python-vs-jinja templates, list-form templates). A
   stripped view can't show you when the prompt was malformed.

Rule, across **jspace, notebook, and token explorer** (and chat where the
rendering path allows): show special tokens **by default**, rendered as visually
distinct tokens (a dim/outlined chip, `--mono`, whitespace as the honest glyphs
from §2). Collapsing them is an **opt-in toggle, default off** — never the
default, never unconditional stripping.

**Realized as ONE global display toggle**, not a per-page control — it lives in
the shared settings drawer and every token-rendering surface reads it (the
canonical cross-cutting display pref; see the settings taxonomy below). Two
invariants:
- **Display-only.** The toggle changes rendering, never what is sent to the
  model. This keeps it from becoming a second generation-settings path.
- **One preference, two render mechanisms.** Token-array surfaces (token
  explorer, jspace) receive token *ids* and flag/style the special ones;
  decoded-text surfaces (chat, notebook) render a *string*, so "show specials"
  means *not stripping them from the decoded text*. Same switch, two code paths —
  don't ship it as if it were uniform.

The shared settings drawer therefore holds three kinds of thing, and the Phase-2
extraction should model them distinctly: **generation params** (samplers — the
existing `settings.js` store), **global display prefs** (show-special-tokens),
and **per-page extras** (jspace's heatmap/chat toggles, explore's logprobs).

**Editing is raw-token-honest (a hard rule, not subject to the toggle).** Any
surface that lets the user *edit* a message — editing a chat turn, prefilling or
continuing an assistant message — must operate on the **full raw text, including
every special/`<think>`/role token present**. You cannot edit through a stripped
"clean" view: round-tripping a lossy render on save would silently drop or
overwrite specials the user never saw and couldn't have intended to change. So
the display toggle above is a *read*-mode preference only; **edit mode always
exposes raw tokens when they exist**, regardless of the toggle. If a message has
no special tokens, there's nothing to expose and the edit box is just its text.
(This is the same position-integrity concern as #1, sharpened: in a read view a
hidden token is a missing label; in an *edit* view it's a token you can destroy
without knowing it was there.)

Known violation to fix (backend): `jspace/analyze.py` decodes the answer with
`skip_special_tokens=True` and its raw-completion path (`chat=False`) drops the
chat template entirely — so the assistant turn and its markers never reach the
UI. The interpretability default should be the *chat turn with markers shown*;
"raw completion, markers stripped" is at most a secondary mode. This is tracked
in `docs/jspace_integration_plan.md` (Part 2).
