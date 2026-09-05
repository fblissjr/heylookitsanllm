# v3 design language

Last updated: 2026-07-23 (§7 settings entry points: pages may add an in-context
opener — chat's top-bar gear — alongside the two shell gears; §6 settings
taxonomy now names the drawer's page-owned lead sections — the shared preset
bar + system-prompt editor — as a fourth kind alongside samplers/display/extras)

The written form of the design system that previously lived only in `css/app.css`
comments. It formalizes what v3 does so that new UI — starting with the j-space
visualizer — stays on-system, and records the visualizer's load-bearing paradigm
decision. The **impeccable audit + polish pass (plan Phase 4 item 2) ran
2026-07-11** across all six pages + the app shell + the settings drawer; its
accessibility + mobile-parity rules are §7 below (they are as load-bearing as the
token system — new UI must honor them). The slop-detector is clean; the technical
score was 17/20 (the pass fixed the mobile + a11y cluster that cost the points).

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
| degraded-but-working | `--warn` (amber), `--warn-tint` | fit meter WARN; never for refusals (that's danger) |

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
canonical cross-cutting display pref; see the settings taxonomy below). Wired
for the decoded-text surfaces in v1.79.6: chat and notebook send
`show_special_tokens` on the wire (`displayWireFields()`, settings.js) and the
server skips its declared-specials strip for that request. Two invariants:
- **Display-only.** The toggle changes rendering, never what is sent to the
  model. This keeps it from becoming a second generation-settings path.
- **One preference, two render mechanisms.** Token-array surfaces (token
  explorer, jspace) receive token *ids* and flag/style the special ones;
  decoded-text surfaces (chat, notebook) render a *string*, so "show specials"
  means *not stripping them from the decoded text*. Same switch, two code paths —
  don't ship it as if it were uniform. Only the decoded-text half is wired
  today; explore and jspace still ignore the pref — and say so structurally: a
  page DECLARES the display prefs it honors (`displayPrefs` on its drawer
  contribution) and passes that same array to `displayWireFields()`, so the
  drawer offers exactly the controls that page acts on. One list, both uses —
  declaring a pref and sending it cannot come apart, and there is no second
  registry-level flag to disagree with it.
- **It is an MLX-only lever, and the help text says so.** The strip lives in
  heylook's own reasoning parser, and the gguf provider routes to a
  pass-through (`template_info()` is None: llama-server pre-splits reasoning
  and re-parsing another engine's output is the documented trap), so heylook
  never strips a gguf reply. Verified live on the *same model* in both
  packagings, gemma-4-E4B MLX vs its q4_0 GGUF: with the pref off, MLX returned
  `STARTEND` where gguf returned `START<mask>END`. Default-on hides the
  asymmetry — both show — so it only surfaces if someone turns it off and
  expects stripping. Not worth "fixing" by teaching heylook to strip gguf
  output: that would add hiding where none exists, against this section.
- **Kept markers must never re-enter a prompt.** The store IS the request on
  chat, so an unstripped assistant row would be replayed into the next turn --
  and a fast tokenizer matches a declared special's *string* and encodes the
  real control token, putting a turn boundary inside prior assistant content
  (worst case: `continue`, whose prefill would end on one). The server therefore
  strips declared specials out of replayed ASSISTANT text
  (`_strip_history_specials`), which is what makes the "display-only" invariant
  above true rather than aspirational. User-authored text is left alone.
  Notebook is the deliberate exception and differs in kind: the reply lands in
  the document body, which is raw, visible, editable text the user is composing,
  and *what you see is what is sent* is that surface's contract — mangling it
  silently would be the bigger lie. Caveat stated in the drawer help text.
- **On the decoded-text surfaces it is a GENERATION-time switch, and the help
  text says so.** The strip runs server-side before the text is streamed *and*
  before it is persisted, so there is no stripped-vs-raw copy to toggle between
  afterwards: a reply generated with the pref on keeps its specials in the
  stored row forever, and one generated with it off never recorded them. That is
  the honest shape (the row holds what the model produced), and it is what makes
  the edit rule below reachable at all — but it means flipping the toggle
  changes the *next* reply, never the thread already on screen.

The shared settings drawer therefore holds four kinds of thing, and the drawer's
`registerSettings(contribution)` contract (`settings-drawer.js`) models them
distinctly: **page-owned lead sections** (`sections()` — chat and notebook each
contribute the shared preset bar, `preset-bar.js`, plus their own system-prompt
editor), **generation params** (samplers — the existing `settings.js` store),
**global display prefs** (show-special-tokens), and **per-page extras**
(`extras()` — jspace's heatmap/chat toggles, explore's logprobs note). Sections
render first, extras last; both are page-owned, but a section composes the
document (prompt/preset), while an extra is a toggle or note that doesn't.

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

**Tag-shaped text is content too (chat's markdown path, v1.79.5).** The same
rule reaches the one surface that renders model text as HTML. marked passes raw
HTML through and DOMPurify then deletes any tag outside its allowlist while
keeping the tag's *content*, so a reply containing `<d>tag</d>` rendered as
"tag" and the markers vanished with no trace — Copy, reading the stored text,
still showed them. `renderMarkdown` (`markdown.js`, still the only sanctioned
text→HTML path) now **escapes raw HTML instead of rendering it**: what the model
wrote is what the message shows, for every tag rather than only the ones
DOMPurify happens to drop. A model that wants HTML *rendered* fences it; fenced
and inline code were always escaped and are unchanged. This is the honesty rule
above, not a security one — DOMPurify stays as the backstop for what marked's
other renderers emit (link hrefs, image srcs).

Known violation to fix (backend), and the reason the pref is only half wired:
`jspace/analyze.py` decodes the answer with
`skip_special_tokens=True` and its raw-completion path (`chat=False`) drops the
chat template entirely — so the assistant turn and its markers never reach the
UI. The interpretability default should be the *chat turn with markers shown*;
"raw completion, markers stripped" is at most a secondary mode. This is tracked
in `docs/jspace_integration_plan.md` (Part 2).

## 7. Accessibility & mobile parity (impeccable pass, 2026-07-11)

The pragmatic-a11y floor (`PRODUCT.md`) and the "one design, two screens"
principle are enforced by concrete rules, not aspiration. New UI must follow
them or it silently regresses the phone surface (a co-primary target, per the
owner: "equally well on desktop web and iPhone 17 Pro Safari").

- **Hover-revealed actions need a touch fallback.** Any control hidden until
  `:hover` (row delete/rename, message actions) MUST also reveal under
  `@media (hover: none)` — touch has no hover, so a hover-only affordance is
  *unreachable* on the phone (this exact bug hid conversation/notebook delete
  on iOS). `.message__actions` was the reference; `.conv-item__delete/__edit`
  and `.notebook-item__delete` now match it. Prefer this to `dblclick`, which
  iOS Safari maps to zoom (conversation rename kept a reveal-on-touch "Ren"
  button for this reason).
- **A pointer-only affordance is allowed only when it duplicates a path that
  already works on touch.** Chat's drag-and-drop attach (v1.72.0) has no touch
  equivalent and needs none: the attach button (which opens the iPhone photo
  library) and paste both remain, and dropping does nothing they cannot. This
  is not an exception to the hover rule above — that rule is about an
  affordance being *unreachable* on the phone, and nothing here is reachable
  only by dragging. Before adding another pointer-only interaction, name the
  phone path that already reaches the same outcome; if there isn't one, the
  interaction is not ready.
- **The settings entry point is device-specific, one drawer.** Desktop = a gear
  `nav-item` at the foot of the sidebar rail; phone = a trailing `⚙` item in
  `#bottom-nav` (`.drawer-gear-bottom`). **Not a floating FAB** — a page with a
  *bottom* composer (chat) leaves no bottom-right corner free, so a FAB collides
  with Send. The bottom-nav gear rides `#bottom-nav`'s own `<=767px` show/hide
  and, being inside `#app`, is sealed by the drawer's focus-trap for free.
  A page whose drawer contribution is a primary workflow (chat: system prompt +
  presets) may additionally mount an **in-context opener in its own toolbar**
  (chat's `.chat__settings-btn` gear, via `drawer.openSettings(btn)`) — same
  singleton drawer, focus restored to the opener on close. In-context openers
  live in a bar, never floating.
- **The phone chat chrome is tiered, not wrapped.** Chat's top bar is two
  tiers under 768px: tier one is what must stay one tap away (Chats, the model
  select stretched to the rest of the line, the gear); tier two is the detail
  group (`.chat__bar-detail`: context size, Load, preset + system-prompt
  chips) on its own line as a non-wrapping horizontal strip with momentum
  scrolling (`overflow-x: auto`), so it never wraps into 3+ vertical rows or
  crushes the message viewport. The composer is two rows: the field spans the
  width and the tool buttons plus Send sit beneath it. Both replace an
  unconstrained wrapping flex row, which broke wherever the widths fell — at
  402pt the chips took line two, the gear stranded alone on a third, and
  three 44px icon buttons beside the field left it 147px wide. Safe-area top
  insets ensure dynamic island / notch clearance without wasting vertical
  space. Desktop is untouched — the wrapper is `display:contents` there and the
  bar and composer are the flat rows they were.
- **The drawer is a real modal dialog.** `role="dialog"` + `aria-modal`,
  `inert` while closed; on open it seals `#app` with `inert` (Tab can't escape),
  moves focus to Close, and on close restores focus to the opener. Escape and a
  backdrop click both close.
- **Honest states are announced, not just shown.** Status lines carry
  `role="status"` (polite: streaming, "server busy — retrying", token counts,
  the preset bar's drift line `.preset-drift`); error surfaces (`.error-note`,
  router mount-failure) carry `role="alert"` (assertive). `setStatus` writing
  `textContent` into a live region is what makes streaming/error legible to a
  screen reader.
- **Form controls are programmatically labeled.** Sampler + display inputs use
  `<label for>`/`id` (the drawer renders one panel at a time, so ids are safe).
  A visible label beside an unassociated input is not a label.
- **Data-color is never the only channel.** Every strength/probability chip
  carries a `title` (explore = `p N%`, jspace = logit/entropy) so the value is
  available without color — this is the §2 rule, now enforced on the explore
  strip too.
- **Active nav carries `aria-current="page"`.** Set in the router alongside the
  `--active` class.
- **Touch targets:** comfortable on phone — bottom-nav items ≥44px, models-page
  actions bump to 44px, the attach-thumb remove control is ≥24px. `btn--sm`
  (26px) is acceptable for dense desktop-hover action clusters (≥ the 24px AA
  floor) but never the sole phone affordance.
- **`aria-pressed` is the toggle-button style hook, not a class.** Icon-only
  toggle buttons (`.btn--icon`, e.g. the chat composer's thinking toggle) set
  `aria-pressed="true"/"false"` and style state off that attribute
  (`.btn--icon[aria-pressed="true"] { ... }` in `app.css`) instead of a
  `.active`/`.is-on` class. One state, expressed once, correct for assistive
  tech by construction instead of by convention. Use this pattern for any new
  binary toggle rendered as an icon button.

- **A long message list must be reconciled, not rebuilt.** `replaceChildren`-ing
  the list drops every open editor and its unsaved draft, and re-runs layout
  for the whole thread. Historically it was worse: under
  `content-visibility: auto` a row knew only its `contain-intrinsic-size`
  estimate until laid out once, that measurement lived on the *node*, and a
  rebuild collapsed `scrollHeight` to a fraction of the truth for the rest of
  the tick — which dumped chat near the top on every send, edit and delete.
  That feature was REMOVED in v1.79.18 (it moved `scrollTop` behind the app's
  back; see css/app.css), so this rule no longer rests on it — but reconciling
  is still how the list is rendered, and the E2E check still pins it. `renderMessages` therefore keys nodes by message id, reuses
  any whose render signature is unchanged, and places them with a reconcile
  that **removes departing children first** — placing before removing walks a
  stale node down the list and re-detaches the whole tail, which has the same
  effect as a rebuild. A forced scroll-to-bottom also re-aims on the next
  animation frame, because a row added this tick is still an estimate.

Verify phone behavior at an iPhone-class viewport with **touch media emulated**
(`hover:none`/`pointer:coarse`) — desktop Chrome reports `hover:hover`, so it
never exercises the touch-reveal rules above. claude-in-chrome refuses
localhost; drive it with puppeteer + system Chrome (as `tests/e2e/` does) and
force the media features over raw CDP (`Emulation.setEmulatedMedia`), since
puppeteer-core whitelists `emulateMediaFeatures` and rejects `hover`/`pointer`.
