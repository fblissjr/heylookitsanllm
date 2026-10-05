# Plan: dark mode, Lamplight to Workbench

Last updated: 2026-10-05 (Phases 1 and 2 on main; open questions 1 to 4 settled; Phase 2 snippets corrected as built; follow-ups listed; Phase 3b, the preset manager, added and approved)

How to use this file: one phase (or one Phase 4 step) per session, on a branch. The values and rules here become code in `frontend/css/app.css` and rules in `frontend/DESIGN.md` as each phase lands; once landed, those files are the source of truth, not this plan. Open decisions are at the end; settle the first two before Phase 1.

## Status

| Phase | State | Where |
| --- | --- | --- |
| 1 Tokens | On main as v2.0.191 (merged 2026-10-05); light control borders raised to 3:1 in v2.0.192 | open questions 1 and 2 settled (below); page suites under dark green at v2.0.193; the iPhone check not yet run |
| 2 Safari fixes | On main as v2.0.193 (merged 2026-10-05, built by session mrgreen) | six fixes; two of this file's snippets were wrong and are corrected below; Chrome suites green in both themes (page suites under dark: chat 53/53, pages 32/32); device checks not yet run; four follow-ups listed under Phase 2 |
| 3 Platform | Items 1, 2, 3 and 5 on main (v2.0.194 to v2.0.196); item 4 not applicable (no boolean control exists) | the Home Screen, Increase Contrast and desktop-picker checks are the owner's |
| 3b Preset manager | Approved (owner, 2026-10-05, mock on the plan doc); next build | right after the on-device pass |
| 4 Toward Workbench | Not started | gated on A being live and measured |

Phase 1 checks run: backend suite, `E2E_COLOR_SCHEME=dark bun run e2e:render`, screenshots of every page in both themes. Not yet run: the page suites under dark, the iPhone check.

Reference (not needed to do the work): the owner keeps a living copy of this plan with a task board, a design system derived from app.css, and the comparison of the three dark directions as private documents; the links are in local notes (`internal/`), not here.

## Summary

heylook gets a dark theme in three CSS-first phases (Lamplight, direction A), then moves toward Workbench (direction B) in seven small steps, each one reversible and measured on an iPhone 17 Pro.

- **Phases 1 and 2 change `app.css` and `index.html` only.** Phase 3 adds two files, one server route and one attribute in `settings.js`. No layout change on desktop. Each phase ships on its own.
- **Phase 4 is the path to B.** It changes the phone chat chrome and, last, the palette. Each step has a gate: the E2E suites stay green and the measured message viewport goes up.
- **Decided:** follow the system theme (no toggle yet), keep system fonts, keep every rule in DESIGN.md sections 2 to 7, keep the data-strength chips light on dark (they are data, not theme).

```
Lamplight (A), each phase ships on its own
  Phase 1  Tokens          light-dark() pairs, two literals tokenised
  Phase 2  Safari fixes    focus zoom, bar tint, safe areas, scroll owner
  Phase 3  Platform        Home Screen app, contrast, switches, themed picker
        |
        v  once A is live and measured
Toward Workbench (B), a step that misses its gate is reverted
  1 Spacing (no change) > 2 Text size (17pt base) > 3 Top nav (+54pt)
  > 4 Capsule (field >= 260pt) > 5 Readout (chips -> line)
  > 6 Scroller (test first) > 7 Palette (graphite)
```

## What iOS 27 and Safari change for heylook

Nine platform facts drive the plan; the last column says where each one lands.

| Platform fact | Since | What it means here | Phase |
| --- | --- | --- | --- |
| Safari ignores `theme-color` and tints its bars from a fixed or sticky element touching an edge and at least 80% wide, else from the body background ([Fiquitiva](https://jahir.dev/blog/safari-toolbar), [Larionov](https://1ar.io/updates/safari-26-liquid-glass-web/)) | Safari 26 | `#bottom-nav`, the phone list panes and the open drawer all qualify, so they must be opaque tokens; `html` and `body` need explicit dark backgrounds | 2 |
| Hidden overlays can still be sampled while they stay in the render tree ([Larionov](https://1ar.io/updates/safari-26-liquid-glass-web/)) | Safari 26 | The closed drawer and its full-screen backdrop are only `visibility: hidden`; use `display: none` when closed | 2 |
| Liquid Glass gains a transparency slider (clear to fully tinted) and darker edges ([MacRumors](https://www.macrumors.com/2026/06/10/how-liquid-glass-is-changing-in-ios-27/)) | iOS 27 | Whatever sits under Safari's bar must read at both extremes: flat, opaque grounds | 1 |
| Scroll anchoring: the engine moves `scrollTop` when content above changes ([WebKit](https://webkit.org/blog/17967/news-from-wwdc26-webkit-in-safari-27-beta/)) | Safari 27 | chat.js owns the message list's scroll position; an engine moving it silently is what DESIGN.md section 7 removed `content-visibility` for | 2 |
| Customizable `<select>` (`appearance: base-select`) ([WebKit](https://webkit.org/blog/18325/webkit-features-for-safari-27-0/)) | Safari 27 | The model select can show residency as a real mark on desktop; the phone keeps the native picker | 3 |
| Every site added to the Home Screen opens as a web app, manifest or not ([Tsai](https://mjtsai.com/blog/2025/10/03/web-apps-in-ios-26/)) | iOS 26 | Opened from the Home Screen, heylook has no Safari bar at all, which removes the bottom stacking problem for free | 3 |
| Inputs under 16px zoom the page on focus | long-standing | Most heylook controls are 14px (`--text-ui`) or 13px, so tapping a setting or the chat-bar select zooms the page | 2 |
| `light-dark()`, `@starting-style`, `transition-behavior: allow-discrete`, the checkbox `switch` attribute | Safari 17.4–17.5 | One token list for both themes; a drawer that animates to `display: none`; native iOS switches for boolean settings | 1–3 |
| `text-wrap: pretty`, anchor positioning, scroll-driven animations | Safari 26 | Better rag in long answers now; a header that compacts on scroll later, on the path to B | 3–4 |

## Phase 1: Lamplight tokens

Every colour token becomes one `light-dark()` pair in the existing `:root` block, so light and dark live on the same line and cannot drift apart. The two literals (drawer shadow, backdrop scrim) become tokens. Nothing else in `app.css` changes, because every other colour already reads a variable.

```css
:root {
  color-scheme: light dark;   /* follow the system; a future toggle sets light or dark here */
  /*                   light                         dark (Lamplight) */
  --bg:          light-dark(oklch(1 0 0),              oklch(0.17 0.008 78));
  --surface:     light-dark(oklch(0.974 0.007 91),     oklch(0.205 0.009 80));
  --surface-2:   light-dark(oklch(0.948 0.010 91),     oklch(0.25 0.010 82));
  --ink:         light-dark(oklch(0.28 0.02 78),       oklch(0.93 0.012 85));
  --ink-muted:   light-dark(oklch(0.49 0.02 82),       oklch(0.74 0.015 85));
  --ink-faint:   light-dark(oklch(0.60 0.018 85),      oklch(0.58 0.014 85));
  --line:        light-dark(oklch(0.905 0.012 88),     oklch(0.29 0.010 82));
  --line-strong: light-dark(oklch(0.83 0.015 88),      oklch(0.52 0.014 85));
  --brand:       oklch(0.842 0.165 91.3);              /* the seed, same in both */
  --brand-tint:  light-dark(oklch(0.962 0.05 95),      oklch(0.30 0.055 88));
  --accent:      light-dark(oklch(0.47 0.10 78),       oklch(0.80 0.115 84));
  --accent-hover:light-dark(oklch(0.40 0.10 78),       oklch(0.86 0.10 86));
  --accent-tint: light-dark(oklch(0.962 0.03 78),      oklch(0.27 0.04 84));
  --on-accent:   light-dark(oklch(1 0 0),              oklch(0.20 0.03 78));
  --danger:      light-dark(oklch(0.50 0.18 29),       oklch(0.72 0.15 29));
  --danger-tint: light-dark(oklch(0.962 0.03 29),      oklch(0.27 0.05 29));
  --warn:        light-dark(oklch(0.52 0.13 60),       oklch(0.78 0.13 62));   /* light value moved: see below */
  --warn-tint:   light-dark(oklch(0.962 0.04 70),      oklch(0.27 0.045 62));
  --scrim:       light-dark(oklch(0.28 0.02 78 / 0.28), oklch(0 0 0 / 0.55));
  --shadow-ink:  light-dark(oklch(0.28 0.02 78 / 0.12), oklch(0 0 0 / 0.5));
}
html { background: var(--bg); }                        /* overscroll and Safari's fallback tint */
.drawer { box-shadow: -4px 0 24px var(--shadow-ink); }
.drawer-backdrop { background: var(--scrim); }
```

In dark, `accent` lifts from bronze to a lit honey, so `on-accent` turns dark: a primary button is honey with dark brown text. Hover goes lighter, not darker. The light `warn` moves from hue 75 to 60 for two reasons: it no longer reads as the bronze action colour, and warn on warn-tint rises from 4.43:1 to 5.05:1 (pending open question 2).

Measured contrast for the dark theme (WCAG 2, from the values above):

| Pair | Dark ratio | Floor |
| --- | --- | --- |
| ink on bg / surface / surface-2 | 15.6 / 14.6 / 13.0 | 4.5 |
| ink-muted on bg / surface-2 | 8.3 / 6.9 | 4.5 |
| accent on bg / brand-tint | 10.2 / 7.3 | 4.5 |
| on-accent on accent / danger (armed) | 9.7 / 6.9 | 4.5 |
| danger on danger-tint | 5.8 | 4.5 |
| warn on warn-tint | 7.4 | 4.5 |
| line-strong on bg / surface | 3.5 / 3.3 | 3.0 |
| ink-faint on bg (placeholders) | 4.5 | 3.0 |

The light theme's control borders were raised to 3:1 in v2.0.192 (open question 1, settled); the light `warn` stays as built (open question 2, settled).

## Phase 2: Safari fixes

Six fixes, all CSS plus three `<head>` lines. Each closes a way the phone or the shell currently misbehaves or could, in either theme.

**1. No dark flash, correct bars.** Declare the scheme before the stylesheet loads, and give browsers that still read `theme-color` (Chrome, older Safari) the surface colour of the bottom nav.

```html
<meta name="color-scheme" content="light dark">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#f8f6f1">
<meta name="theme-color" media="(prefers-color-scheme: dark)"  content="#191713">
```

**2. No zoom when a field is focused.** iOS zooms the page when a focused control's text is under 16px. Place this at the end of `app.css` so it beats the phone bar's own sizes at equal specificity:

```css
@media (pointer: coarse) {
  .input:not(.notebook__title), select, textarea, .sysprompt-input,
  .chat__bar > select, .chat__load-select, .chat__ctx-select, .chat__ctx-custom,
  .cfg-tmpl__body {
    font-size: 16px;
  }
}
```

`maximum-scale=1` would also stop the zoom, but it blocks pinch-zoom in Chrome on Android; 16px fixes the cause and reads better on the phone anyway. As built (v2.0.193), two corrections to the first draft of this snippet: a bare `.input` pulled the notebook title down from its larger size (equal specificity, later rule wins), so the title is excluded; and the models page's template editor has its own class and size, which beats a bare `textarea`, so it is named. `tests/e2e/render.mjs` holds the property (no text field in the chat bar or drawer under 16px on touch).

**3. Landscape safe areas.** At 874pt wide the landscape iPhone gets the desktop layout, with the Dynamic Island over the left of the rail.

```css
#app { padding-inline: env(safe-area-inset-left, 0px) env(safe-area-inset-right, 0px); }
@media (max-width: 767px) {
  #bottom-nav { padding-inline: env(safe-area-inset-left, 0px) env(safe-area-inset-right, 0px); }
}
```

**4. Closed overlays leave the render tree.** The closed drawer and its full-screen backdrop are fixed, edge-touching and only `visibility: hidden`, which is the shape Safari samples for its bar tint. `display: none` with a discrete transition keeps the slide and removes the question:

```css
.drawer:not(.drawer--open),
.drawer-backdrop:not(.drawer-backdrop--open) { display: none; }
.drawer          { transition: transform var(--t-fast) var(--ease), display var(--t-fast) allow-discrete; }
.drawer-backdrop { transition: opacity   var(--t-fast) var(--ease), display var(--t-fast) allow-discrete; }
@starting-style {
  .drawer--open          { transform: translateX(100%); }
  .drawer-backdrop--open { opacity: 0; }
}
```

The `visibility` rules on both go. The reduced-motion kill switch still collapses the slide to a snap. `settings-drawer.js` adds `drawer--open` and then focuses Close, which works because `focus()` flushes style.

**5. One owner for the chat scroll position.** Safari 27 adds scroll anchoring, which Chrome already has. chat.js derives every scroll decision from `scrollTop` and `scrollHeight`; an engine that moves them silently is the failure DESIGN.md section 7 records. Opt the list out on every engine:

```css
.chat__messages { overflow-anchor: none; }   /* chat.js owns scrollTop */
```

This follows VISION principle 1: nothing happens where it cannot be seen.

**6. The nav gear is a nav item, not a browser button.** The sidebar's "⚙ Settings" (`.drawer-gear`) sets no background or border, so it wears the browser's default button chrome: a grey pill with a dark border in light, and in dark (found in the Phase 1 screenshots) a mid-grey pill on near-black. `.drawer-gear-bottom` already resets both. Give `.drawer-gear` the same reset so it reads like the nav items around it in both themes; Phase 4 step 3 moves the gear beside the top nav and inherits the fix.

```css
:where(.drawer-gear) { background: none; border: none; }
```

As built (v2.0.193): the reset sits in `:where()` so it carries no specificity. The plain class form, written first, landed after the `.nav-item` hover rule at equal specificity and removed the gear's hover tint (measured: gear transparent on hover while its sibling links tinted).

**Phase 2 follow-ups, seen while building and not fixed** (fold into Phase 3 or Phase 4 step 3, where the gear moves):

- The gear's label uses the browser's button font (Arial in Chrome) while sibling nav links use the system font, so it still does not fully read as a nav item.
- `.drawer-gear-bottom` has the plain-class reset shape that defeated hover on the desktop gear; by cascade reasoning only, not measured; it is a touch control.
- `.message-edit__thinking`'s smaller font-size is dead: `.message-edit textarea` outranks it, so the thinking editor already computes 16px.
- Fix 3 pads `#app` and `#bottom-nav` only. The drawer (fixed, right edge) and the phone list panes (fixed) get no side inset in landscape. Not verified on a device.

## Phase 3: platform enhancements

Five additions that make A feel native on the phone and finished on desktop. The first is the largest win and the only one that touches the server.

**1. A Home Screen web app.** Since iOS 26, a site added to the Home Screen opens as its own app with no Safari bar, which removes the bottom stacking problem. Four pieces make it look right:

- `frontend/apple-touch-icon.png`, 180 × 180: the honey disc on the DARK `surface`, the owner's launch colour (iOS fills a transparent icon with black). iOS asks for this path by convention. Generated from the `:root` token values, never a retyped hex.
- `frontend/manifest.json`: `{"name":"heylook","start_url":"/#/chat","display":"standalone","background_color":"#191713","theme_color":"#191713","icons":[{"src":"/apple-touch-icon.png","sizes":"180x180","type":"image/png"}]}`
- `index.html`: `<link rel="manifest" href="manifest.json">`, `<link rel="apple-touch-icon" href="apple-touch-icon.png">`, `<meta name="apple-mobile-web-app-title">`. As built (v2.0.194) WITHOUT `apple-mobile-web-app-status-bar-style=black-translucent`: on iOS 26 that style makes `env(safe-area-inset-top)` read 0 and the page runs under the status bar (reported widely; the owner saw the cut-off top once on 2026-10-05 before a reload). The default style starts the web view below an opaque status bar tinted from `theme-color`, and the top bars' inset padding then adds nothing, which is correct.
- `src/heylook_llm/frontend_static.py`: two routes beside `/icon.svg`, because `mount_frontend` registers only the tree's real shape and has no catch-all.

A Home Screen app keeps its own browser storage, so per-browser preferences start fresh there; conversations live on the server and are unaffected.

**2. Increase Contrast.** iOS's Increase Contrast setting reaches the page as `prefers-contrast: more`. Raise the three quiet tokens in both themes:

```css
@media (prefers-contrast: more) {
  :root {
    --ink-muted:   light-dark(oklch(0.40 0.02 82),   oklch(0.84 0.012 85));
    --line:        light-dark(oklch(0.80 0.012 88),  oklch(0.40 0.010 82));
    --line-strong: light-dark(oklch(0.62 0.015 88),  oklch(0.66 0.014 85));
  }
}
```

**3. Better rag in long answers.** Safari 26 and Chrome both support `text-wrap: pretty`, which avoids one-word last lines in paragraphs:

```css
.message-content :is(p, li) { text-wrap: pretty; }
h1, h2, h3 { text-wrap: balance; }
```

**4. Native switches for on/off settings.** NOT APPLICABLE as of 2026-10-05: the only boolean sampler control became a three-state select in v1.79.62 (model default / on / off), and no `type: 'checkbox'` control is built today. If a boolean control returns, give it the `switch` attribute and `accent-color: var(--accent)`; iOS renders its own switch, with haptics, other browsers a checkbox.

**5. A themed model picker on desktop.** Safari 27 and current Chrome support `appearance: base-select`, so the open picker can use heylook's tokens instead of the system menu. Pointer devices only; the phone keeps the native wheel.

```css
@supports (appearance: base-select) {
  @media (pointer: fine) {
    .chat__bar > select, .chat__bar > select::picker(select) { appearance: base-select; }
    .chat__bar > select::picker(select) {
      background: var(--surface); border: 1px solid var(--line-strong); border-radius: var(--r-card);
    }
    .chat__bar option:checked { background: var(--brand-tint); }
  }
}
```

The option labels keep their ● / ○ residency text, so nothing depends on the new styling. As built (v2.0.196): the block above plus option padding, a hover fill and a rounded option; verified in Chrome only, since the open picker cannot be screenshotted headless; Safari 27 is the owner's check.

## Phase 3b: presets get their own manager

Owner, 2026-10-05: the settings drawer mixes the conversation's own settings with preset management, so it is hard to tell which system prompt is in force and what Apply, Save and Save as new would each do. The mock on the plan doc's task board is approved. This phase comes right after the on-device pass, before Phase 3's polish; Phase 4 step 5 (the readout line) should show the provenance line this phase introduces.

**What changes.**

- The drawer keeps only what the conversation runs on: the system prompt box, the samplers, and one provenance line in place of today's drift line: "From preset X", "X, modified: prompt and two knobs", or "No preset". Two actions: **Apply preset**, through a picker that shows each preset's own prompt before it is applied, armed ("Replace prompt?") only when it would replace a non-empty differing prompt; and **Save as new preset**, which always creates and never overwrites.
- A **Presets** surface of its own (a page beside Models, so the drawer shrinks on the phone; the owner may prefer a drawer tab) lists presets and, per preset, edits the prompt and knobs in place, renames, duplicates and deletes. It is the only place a preset is overwritten, and editing one never touches a conversation.
- Removed from the drawer: the preset's read-only preview block, the Save versus Save as new split, the drift line. The shared sections stay shared by chat and notebook (`preset-bar.js`, `prompt-section.js` are rewritten, not forked per page).

**What it keeps** (owner rules with incidents behind them, `sharp_edges.md` "Presets and the system prompt"): what the model reads is one bag, the conversation's own prompt and samplers, stored, shown and sent; a preset is a copy, never a link; an empty preset prompt makes no claim. `applied_preset_id` keeps meaning "this document was stamped by an explicit Apply or Save as new"; the provenance line derives "modified" by comparing the document's bag to the stamped preset, the way the drift line does today.

**Gates.**

- The question "which prompt is this conversation using, and is it the preset's?" is answerable from the drawer at a glance, without opening anything.
- No path in the drawer can overwrite a stored preset.
- The chat and notebook e2e preset checks are rewritten with the new sections and pass live on a model; `e2e:render` stays green in both themes.
- `docs/frontend_v3_user_guide.md` section 2 (Presets) is rewritten to the new model and its rough-edges list gains the closed entry.
- DESIGN.md section 6's settings taxonomy names the Presets surface and the provenance line.

## Phase 4: the path to Workbench

Seven steps take A to B, ordered so the cheapest and most reversible land first and the palette changes last. Each has a gate; a step that misses its gate is reverted, not patched.

1. **Spacing tokens, no visual change.** Name the ten steps app.css already uses as `--space-*` with their current values and replace the literals. B's 4px grid then becomes a change to token values only.
   - Gate: `e2e:render` and page screenshots identical before and after.
2. **iOS text size on touch.** Under `@media (pointer: coarse)`, set `html { font: -apple-system-body; font-family: var(--font); }`. Body becomes 17pt (iOS's own default) and every rem follows the user's Text Size setting.
   - Gate: the phone chat bar and composer fit at the default and at the largest non-accessibility size.
3. **Page navigation moves to the top on the phone.** A segmented control (Chat, Notes, Models, Perf) in the app shell's header replaces `#bottom-nav`; the gear moves beside it. DESIGN.md section 7's settings-entry rule is rewritten in the same change.
   - Gate: measured message viewport on the phone rises by at least 54pt (the bottom nav's height).
4. **Capsule composer.** The field and its tool buttons share one rounded border; Send becomes an icon button with `aria-label="Send"`. Touch targets stay 44pt.
   - Gate: the field is at least 260pt wide at 402pt, and the chat E2E suite passes.
5. **A readout line.** Tier two of the phone chat bar becomes one mono line: context, cache reuse, tok/s, preset and system-prompt state. The system-prompt state stays visible, as DESIGN.md requires.
   - Gate: every fact the chips showed is still on screen without a tap.
6. **The document as the phone scroller.** Let the page scroll instead of `.chat__messages`, so Safari can retract its toolbar. Reports disagree on whether iOS retracts it for inner scrollers ([root scroller only](https://github.com/aparte-luxurious-homes/landing-page/pull/143), [inner elements too](https://developer.apple.com/forums/thread/690835)), so test first on iOS 27; skip this step if it already retracts.
   - Gate: toolbar retracts on device and tail-follow passes the chat E2E suite.
7. **The Workbench palette and shape.** Swap the dark half of each `light-dark()` pair to graphite with honey as the only accent, and set radii to 9, 16 and 22px. The light theme keeps Lamplight's values unless decided otherwise (open question 5).
   - Gate: the contrast table passes in both themes.

## Verification

Each phase is proved on the device it is for; desktop Chrome cannot show Safari's toolbar, focus zoom or Home Screen behaviour.

| Phase | Check | Where |
| --- | --- | --- |
| 1 | Contrast for every text and border pair, both themes, computed from the token values | script over app.css |
| 1 | `e2e:render` and the page suites with `prefers-color-scheme: dark` emulated (`E2E_COLOR_SCHEME`; a run without it follows the OS theme, so on a dark Mac a plain run is a second dark run: set `light` explicitly) | desktop Chrome via puppeteer |
| 1 | Each page and the drawer, light and dark | iPhone 17 Pro and desktop |
| 2 | Tap every input, select and textarea: the page does not zoom | iPhone 17 Pro, iOS 27 |
| 2 | Open and close the drawer: Safari's bar returns to the bottom-nav colour | iPhone 17 Pro, iOS 27 |
| 2 | Landscape: the rail clears the Dynamic Island | iPhone 17 Pro |
| 2 | Long streamed reply with an image above: tail-follow holds | chat E2E (Chrome) + Safari 27 by hand |
| 3 | Add to Home Screen: own icon, no Safari bar, status bar clear of content | iPhone 17 Pro |
| 3 | Increase Contrast on: muted text and borders strengthen | iPhone 17 Pro |
| 4 | Each step's gate, with the message viewport read in Web Inspector as `document.querySelector('.chat__messages').clientHeight` | iPhone 17 Pro + chat E2E |

Per AGENTS.md's done rules, each phase lands with its docs: DESIGN.md section 1 gains a dark column, section 7 gains the touch-size, closed-overlay and scroll-owner rules, and CHANGELOG plus `__version__` move together (proposed in the report when the work is on a branch).

## Open questions

1. SETTLED 2026-10-05 (owner): light `line-strong` raised to `oklch(0.64 0.015 88)` (v2.0.192): 3:1 on white and on `surface` with margin; the plan's 0.66 cleared white only, which the contrast test caught. A control is identified by its edge, not only its text.
2. SETTLED 2026-10-05 (owner): the light `warn` stays at hue 75 as built; its 4.43:1 on `warn-tint` is within rounding of the floor and dark already separates warn from accent.
3. SETTLED 2026-10-05 (owner, simplicity): system theme only, no toggle. `color-scheme` on `:root` stays the single switch.
4. SETTLED 2026-10-05 (owner): the dark surface (`#191713`). The phone is used mostly at night; the per-scheme `theme-color` metas cover the page itself in daytime.
5. OPEN, by choice: decide at step 7 after living with dark Lamplight. Nothing seen so far argues either way.
6. SETTLED 2026-10-05 (owner): the phone is used BOTH as a Home Screen app and in a WebKit browser (Orion), so Phase 3's Home Screen work and Phase 4 step 3's top nav are both needed and the order stands; neither may add much complexity. Open question 2's light `warn` stays as built for the same reason it was settled: light is the daytime Mac theme, and at night the phone is in dark, where warn already sits at hue 62 (Night Shift's warm cast makes amber and bronze converge, which is one more reason dark keeps them apart).

## Sources

- [MacRumors: How Liquid Glass is changing in iOS 27](https://www.macrumors.com/2026/06/10/how-liquid-glass-is-changing-in-ios-27/)
- [WebKit: Features for Safari 27.0](https://webkit.org/blog/18325/webkit-features-for-safari-27-0/)
- [WebKit: News from WWDC26, Safari 27 beta](https://webkit.org/blog/17967/news-from-wwdc26-webkit-in-safari-27-beta/)
- [Jahir Fiquitiva: Tinting Safari's toolbar in iOS 26](https://jahir.dev/blog/safari-toolbar)
- [Pavel Larionov: Safari 26 Liquid Glass for the web](https://1ar.io/updates/safari-26-liquid-glass-web/)
- [Michael Tsai: Web apps in iOS 26](https://mjtsai.com/blog/2025/10/03/web-apps-in-ios-26/)
- [aparte landing page PR #143: toolbars retract only for the root scroller](https://github.com/aparte-luxurious-homes/landing-page/pull/143)
- [Apple Developer Forums: iOS 15 toolbar hides when scrolling within an element](https://developer.apple.com/forums/thread/690835)
