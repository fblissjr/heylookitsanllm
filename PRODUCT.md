# Product

Last updated: 2026-07-20 (Product Purpose: added the j-space interpretability
page, shipped after this doc was first written)

Design context for the heylookitsanllm frontends (current build target: frontend v3,
`docs/frontend_v3_spec.md`). Read by impeccable commands before any design work.

## Register

product

## Users

One user: the server's owner, a technical practitioner running a personal MLX inference
server on their own Apple Silicon machine. Context of use is exploratory and unhurried --
chatting with local models, writing in notebooks, inspecting token probabilities, checking
what's loaded and how the machine is doing. Sessions happen on a desktop browser at a desk
and on an iPhone (Safari, iPhone 17 Pro class) away from it. There is no onboarding problem,
no persuasion problem, and no second user to design for.

## Product Purpose

A personal frontend for a local LLM server: conversations, a plain-text notebook with
generate-at-cursor, model load/unload/import administration, on-demand system metrics, a
token-probability explorer, and a j-space page for reading a model's internal layer-by-layer
workspace (Jacobian-lens interpretability -- an introspection surface, not a chat feature).
Success is the owner reaching for it daily because it is faster, calmer, and more pleasant
than any hosted alternative -- and because the whole thing stays simple enough to hold in
one head (vanilla JS, no framework, no build step).

## Brand Personality

Warm minimal. Soft, unhurried, writing-focused: the chat and notebook are thinking spaces
first, control panels second. Quiet confidence over density; the interface recedes while
model output and the user's own words carry the page.

## Anti-references

- Desktop-only layouts. Anything that assumes a wide viewport, hover, or a keyboard is a
  failure: every page must genuinely work for exploration on iPhone Safari, not merely
  "not break" there.
- Generic SaaS dashboard grammar: identical card grids, hero metrics, gradient accents,
  icon-rail chrome.
- The default centered-column AI-chat-wrapper look; this should not read as a ChatGPT skin.
- Control-panel density creep: admin affordances (models, metrics) stay quiet and out of
  the writing surfaces.

## Design Principles

- Thinking space first: chat and notebook are for reading and writing; chrome earns every
  pixel it takes from them.
- One design, two screens: desktop and iPhone Safari are co-primary. Design the phone
  layout deliberately, not as a squeezed desktop.
- Warmth through type and rhythm, not decoration: the calm comes from typography, spacing,
  and restraint -- not tints, cards, or ornament.
- Simplicity is the feature: if a design choice needs framework-grade machinery to
  implement, it is the wrong choice for this codebase.
- Honest states: streaming, busy (503 backpressure), empty, and error states are designed,
  not defaulted.

## Accessibility & Inclusion

Pragmatic floor for a single-user tool: body text at >=4.5:1 contrast, reduced-motion
alternatives for every animation, keyboard navigation preserved where it exists (token
explorer), comfortable touch targets on mobile. No formal WCAG audit.
