# Plan: server-side conversation orchestration (chat reliability)

Status: APPROVED by the owner 2026-08-13 (direction confirmed; Phase 1
contract details still open below). Companion to `plan_2026-07.md` -- this
plan IS the design vehicle for its Wave 3 / Phase 3b chat half.

## The problem, stated once

Chat mutations intermittently misbehave: edits that don't take, regenerate
doing nothing, messages hidden after save. The owner's observed profile
(2026-08-13): usually iOS Safari; editing an ASSISTANT message then Save or
Save & Regenerate; sometimes after stopping a generation mid-stream and
editing the partial; no error shown; the message row is *hidden, not gone* --
it reappears on entering edit mode. Recreating the DuckDB store makes things
look better temporarily and then the drift returns.

Diagnosis (code-read 2026-08-13, chat.js + conversation_api.py + db.py +
streaming.js): **the store is fine; the protocol is the defect class.** The
backend has no concept of "generate into this conversation" -- it offers a
CRUD store (`/v1/conversations`) and a stateless generation endpoint
(`/v1/chat/completions`), and the browser is the only thing connecting them.
Every user action is a client-orchestrated saga of 2-4 independent HTTP
calls (e.g. Save & Regenerate = PUT message -> DELETE ?after=P -> stream ->
POST assistant message). Each call is atomic; the saga is not. The client
keeps a full mirror (`s.messages`) with no versioning, no conflict
detection, and no reconciliation -- `selectConversation` deliberately skips
re-fetching the active conversation, so once the mirror and the store
diverge they stay diverged until a full reload. Known live instances of the
class:

- Silent no-op guards: Edit / Regenerate / Delete / Save & Regenerate all
  `if (s.stream) return` with no message. Any live stream object --
  including the whole pre-first-token window on a cold load -- makes those
  buttons dead and mute.
- The unsaved fallback row (`{id:null, position:guessed}`) after a failed
  assistant save: Edit on it silently no-ops, and the guessed position can
  collide with the next server-assigned one, after which position-anchored
  truncation deletes the wrong tail SERVER-side.
- Stop-then-act: `releaseStream` nulls `s.stream` before the partial-save
  POST resolves, so an immediate Regenerate interleaves truncation with the
  in-flight save -- ghost rows with interleaved positions.
- Two devices (the product pillar is iPhone+desktop co-primary): two
  independent mirrors, zero invalidation, destructive position-anchored ops
  from whichever is stale.

The fix history confirms it's a class, not a list: v1.61.1 closed two Save
& Regenerate races, v1.62.4-ish closed the null-editingId edit bug, the
row-reuse work closed the rebuild-scroll class -- and new instances keep
surfacing. Per-instance fixes cannot close a class the protocol permits.

**Separate second track:** the "message hidden until I click Edit" symptom
is not a saga bug -- data survives; the ROW stops painting. `.message`
carries `content-visibility: auto` + `contain-intrinsic-size: auto 3rem`
(`apps/heylook-frontend-v3/css/app.css` ~532) and save/regenerate swaps
nodes via the reconciler; the hidden-until-interaction phenomenology on
WebKit points at a skipped-content invalidation bug there. Needs a live iOS
repro, not more code-reading (Phase 0.5 below).

## What we are NOT doing

- Not replacing DuckDB. Single serialized writer, transactional ops,
  server-assigned positions -- the store is the most solid layer. Any store
  behind this protocol would show the same symptoms.
- Not adding a frontend state framework. v2 had one; it had its own version
  of these problems. The mirror isn't badly managed -- it shouldn't be
  load-bearing at all.
- Not bolting sync machinery (etags/If-Match/409 + re-fetch) onto the
  current shape as the PRIMARY fix. That keeps two sources of truth and
  turns every new feature into distributed-systems whack-a-mole. (A light
  guard of this kind survives in Phase 2 for the CRUD edits that remain.)

## Phase 0 -- immediate hardening (cheap, diagnostic, ships now)

STATUS: SHIPPED v1.64.0 (all four items + a fifth found during
implementation: a `pendingSave` latch in finishStream, because
`releaseStream` nulls `s.stream` before the partial save resolves, so the
stream-guard alone left the Stop-then-act window open). Verified red-first
via `e2e:render` (23 checks incl. an iPhone-emulation boot). Real-device
iOS remains Phase 0.5.

Independent of the endpoint; each item kills a live confusion source.

1. **Loud guards.** Every silent `if (s.stream) return` on a user action
   speaks in the status line ("Finish or Stop the current response first").
   Cost: minutes. Also diagnostic: if the owner sees this line when nothing
   is visibly streaming, we've caught a leaked stream object in the act.
2. **Unsaved-row honesty.** The `id:null` fallback row renders with an
   explicit "not saved -- Retry" chip; while one exists, position-anchored
   destructive ops (edit-save-regenerate, regenerate, delete) on ANY row
   are blocked loudly, because the client's positions are known-divergent.
   Retry re-POSTs and re-syncs.
3. **Reconcile at saga end.** On stream completion (and on any mutation
   failure), re-GET the conversation and adopt the server's rows wholesale.
   Localhost round-trip is cheap; the reconciler already reuses unchanged
   nodes, so this is not a repaint. This is the honest stopgap for the
   mirror until Phase 2 removes its authority.
4. **Thinking-block editing** (owner ask 2026-08-13). Backend already
   supports it (`MessageUpdate.thinking`, `_UPDATABLE_MESSAGE_FIELDS`);
   the editor simply never offers it. `buildEditEl` grows a second
   textarea (visible when the message has `thinking`), saved through the
   same PUT. Small, independent, immediately felt.

## Phase 0.5 -- the hidden-row bug (needs a device, not a diff)

Reproduce on real iOS Safari with the same build: edit assistant message ->
Save (and Stop-mid-gen -> edit -> Save & Regenerate). Confirm or refute the
`content-visibility` skip-state hypothesis (inspect whether the row's
containment state is stuck skipped while on-screen). Candidate fixes, in
preference order: force an invalidation after reconcile placement (e.g.
temporarily clear `content-visibility` on the swapped-in node for one
frame); exclude the just-edited row from `content-visibility: auto`; drop
the property on WebKit only (`@supports` guard) if the bug is wholesale.
Extend `e2e:render` with the edit-save reconcile sequence so the Chrome
half of the behavior is pinned; iOS itself stays a manual check
(documented limitation -- no suite can drive real WebKit here).

## Phase 1 -- the missing layer: conversation-scoped generation

One new endpoint family, designed once with the Messages migration
(plan_2026-07 Wave 3 / Phase 3b) so the SSE grammar and extension
namespace are decided a single time:

    POST /v1/conversations/{conv_id}/generate
      body: { mode: "append" | "regenerate" | "continue",
              message_id?: str,     # regenerate/continue anchor
              user_content?: ...,   # append mode: the new user turn
                                    # (content blocks), persisted atomically
              overrides?: {...} }   # optional one-shot sampler overrides

Contract points:

- **The store is the request.** The server builds the wire request from the
  conversation row itself: system_prompt, params (the sampler bag),
  model_id, messages as content blocks. The client sends intent, never the
  transcript. Capability-gated media dropping (the G1 drop-with-disclosure
  rule) moves server-side; the disclosure data rides the stream so the
  transcript marker stays honest.
- **Truncation is server-side and anchored by message id,** not by a
  client-remembered position. `regenerate` truncates after the anchor's
  predecessor; `continue` truncates after the anchor and prefills with it
  (same MLX-only user-role rule as today, enforced where the provider
  knowledge lives).
- **Persistence is server-owned.** The assistant row (or continuation
  merge) is written by the server -- on completion AND on client
  disconnect/abort (partial-save keeps today's abort-as-completion
  contract, now enforceable even if the phone locks mid-stream, which is
  an iOS reality the current design cannot survive).
- **The final SSE frame carries the authoritative rows** (saved message(s)
  + conversation updated_at). The client's post-stream state is assignment,
  not arithmetic.
- **One active generation per conversation,** arbitrated server-side: a
  second generate on the same conversation gets 409 (or cancels-and-
  replaces -- owner call below). This is what makes two devices safe: the
  server is finally in a position to notice.
- **Wire shape: Messages-style SSE** with the namespaced extensions
  (logprobs, timing/usage telemetry) ported per the Phase 3b order-critical
  rule -- this endpoint becomes 3b's first consumer instead of a second
  migration. `/v1/chat/completions` stays, stateless, for API clients.
  Spec §4 updates land in the same commits (standing rule).

## Phase 2 -- the client thins down

chat.js drops `buildRequestBody`/`toWireContent`/position math for
generation; `s.messages` becomes a render cache of the last server
response, refreshed from generate-stream final frames and mutation
responses. The CRUD edits that remain client-initiated (message text PUT,
title rename) carry `If-Match: <updated_at>` -> 409 -> re-fetch, as cheap
insurance for the two-device case. Notebook stays on its current shape
(single-document, no positions -- the class mostly doesn't apply) and
migrates only if it starts growing message-like structure.

## Test strategy

- Contract tests for `/generate`: all three modes, anchor-id validation,
  409 arbitration, disconnect-persists-partial (the load-bearing one),
  capability drop disclosure.
- `tests/e2e` chat suite grows the flows that actually hurt: edit -> Save &
  Regenerate, Stop -> edit partial -> Save & Continue, rapid Stop ->
  Regenerate.
- `e2e:render` grows the edit-save reconcile check (Phase 0.5).
- Each Phase 0 guard gets shown red first per the checks-and-assertions
  rule (e.g. loud-guard test drives the action during a stubbed stream and
  asserts the status line).

## Open questions (owner)

1. 409 vs cancel-and-replace when a second generate hits a busy
   conversation? (Recommend 409 + explicit Stop affordance -- only-loss-
   gates says a cancel is a loss and shouldn't be implicit.)
2. Does Stop become `DELETE /generate` (server aborts + finalizes partial)
   or stay client-side abort? (Recommend the former -- it makes
   phone-locks-mid-stream and Stop identical, one persistence path.)
3. Should `append` mode also adopt the pre-create flow (no conversation
   yet -> create + first message + generate in one call)?
