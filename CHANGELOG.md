# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.79.48]

Model load stops being an admin operation.

### Changed

- **`POST /v1/admin/models/{id}/load` moved to `POST /v1/models/{id}/load`,
  gated on `require_api_key` instead of `require_admin_token`.** The admin
  gate protected nothing: `/v1/messages` and `/v1/chat/completions` both call
  the same `router.get_provider(model_id)` on the way in, so any client that
  can generate already triggers a multi-GB load and, at
  `max_loaded_models=1`, an eviction — just by naming a model in the body.
  The token only stopped a client from doing EXPLICITLY and OBSERVABLY what
  it could already do implicitly. Whoever may generate may load.
- **Why a client wants it:** the load runs BEFORE the response begins, so
  during a cold load nothing is on the connection at all — no headers, no
  `message_start`, no keepalive — on either wire. Streaming does not cover
  it, because the stream has not started. A non-streaming consumer sees one
  opaque POST and cannot distinguish a loading model from a hung server.
  Calling `/load` first relocates that wait into a request the client can
  label, and adds no work: it is the same `get_provider` call.
- **A MOVE, not an alias.** Two URLs for one operation is the duplication
  this repo keeps paying for, so the admin path is gone and every in-repo
  caller moved with it: v3's `api.js`, `scripts/dev_server.sh`,
  `tests/e2e/lib/server.mjs`, `tests/smoke/run.py`, `tests/eval/lifecycle.py`,
  plus the five docs that memorize the URL as the canonical readiness call
  (CLAUDE.md, the v3 spec, the e2e README, the dev-server skill, and the
  OpenAPI narrative). `test_the_old_admin_path_is_gone` pins it at 405 — the
  admin catch-all still owns that URL for other verbs — asserted as 405
  rather than loosened to `in (404, 405)`, which would pass if the route
  came back.
- **`unload` and `reload` stay admin** (they stop a model out from under
  other clients, including SIGTERM to a gguf subprocess), and so does
  `GET /v1/admin/models`, which discloses `model_path` and full per-model
  config. The shared `load_and_warm` body moved to the new
  `model_ops_api.py`; admin's `/reload` imports it, so the warm contract
  still cannot fork.

### Fixed

- **The contract suite's `MockRouter` was missing `stale_reload_fields`**, so
  `GET /v1/admin/models` answered 500 for any LOADED row. Unreachable while
  the load tests lived inside `test_admin.py` and ran after the list tests;
  newly reachable the moment they moved to their own file. A fake that only
  answers the calls one file ordering happens to make is not a fake. Also
  added `unload_all`, whose absence had been logging a teardown traceback on
  every contract run.
- **The load tests now leave the router as they found it.** The `client`
  fixture is session-scoped, so a model loaded in one file stayed loaded for
  every later one — and `test_admin.py` has a case whose whole subject is an
  UNLOADED model. Green in collection order, reverse order, and alone.

### Added

- **`docs/api_integration.md` §3 "Paying the model load up front"** — the
  invisible cold-load window, the unconditional pre-flight, why `?warm=true`
  is wrong per-request (it takes the generation gate), and the eviction
  caveat. Written for a non-streaming consumer, which is the case the wire
  serves worst.
- **§6 now dates the cancel endpoint.** It is v1.79.44 on BOTH wires;
  `/v1/chat/completions` read `X-Request-ID` earlier but only for log
  correlation, so "we already send the id" was not evidence of having had
  cancellation. Reported by the consuming-side twin skill, which nearly
  shipped the same wrong inference.

## [1.79.47]

An audit of `/openapi.json` and `docs/api_integration.md` against the .43-.46
wire. The document held up; the generated schema had drifted in four places,
all of them prose a client reads and none of them derived.

### Fixed

- **`/v1/messages` carried its tag twice.** `["Messages API", "Messages API"]`
  in the generated JSON: the router already declares the tag and the route
  decorator declared it again. Harmless to Swagger, not harmless to a client
  generated from the schema.
- **The `Config` tag told clients operational settings resolve `env > DB >
  default`.** There is deliberately no env layer -- `settings.py` says so in
  its module docstring and the `GET /v1/admin/config` description says so in
  the same schema, so the document contradicted itself for anyone reading the
  tag first. Three copies of the wrong precedence (the tag, the startup log
  line, the `include_router` comment); all now say `DB > default` and why.
- **The schema header did not mention cancellation.** A whole router shipped
  in .44 and the narrative an integrating client reads first still said
  `X-Request-ID` was for log correlation. It now names `DELETE
  /v1/requests/{id}`, that the header is the precondition on the
  non-streaming path, and that responses echo it on both wires. `422` added
  to the header's error list, which had 400/500/503 and the in-band SSE case
  but not the one a malformed body actually produces.

### Changed

- **`FastAPI(description=...)` was dead code.** `custom_openapi()` replaces it
  wholesale, so the one-line summary sitting there could never reach a client
  -- a trap for whoever edits it next expecting a change. Replaced with a
  pointer to the real source.
- **The `/v1/messages` route description named only `text` and `image`
  blocks.** Audio blocks exist on the gguf arm (MLX answers 400).

### Verified unchanged

- `docs/api_integration.md` is level with the code: its knob list is exactly
  the 20 non-`model`/`messages`/`system` fields of `MessageCreateRequest`, the
  "thirteen extensions" count is thirteen, `samplers.available` is the real
  path, `stop_sequence` is a response `Literal` member with no request field
  and no `message_delta` field, and the `/v1/models` row shape is the one the
  route builds. The .44 cancellation section and the .43 vision-resolver
  paragraph were already written.
- The consuming-side twin (`heylook-provider` in the owner's marketplace) is
  NOT level -- see `docs/project/TODO.md`. It lives in another repo, so it is
  recorded here rather than fixed here.

## [1.79.46]

Acting on the review of .43-.45. The theme is that this branch shipped the same defect three times and left a second copy of a fix it had just made.

### Fixed

- **`GET /` reported 12 of 48 endpoints — the exact defect .45 fixed in the startup banner, left standing in the second copy.** Two route walks over the same route table; one was fixed and the other, on the surface a client reaches FIRST, was not. It omitted `/v1/messages`, every conversation/preset/notebook route, all 19 admin routes, and the cancel endpoint added one release earlier. Both now read the OpenAPI schema. Verified live: `_get_api_endpoints()` returned 12 while `server.get_api_endpoints()` returned 48.
- **A cancelled non-streaming `/v1/chat/completions` reported `finish_reason: "stop"`** — the third occurrence of one rule broken the same way. .40 fixed it on `conversation_generate_api`, .44 fixed it on `/v1/messages`, and .44 *introduced* it here, in the very commit that made this path cancellable: the abort just breaks the decode loop, no chunk carries a finish_reason, and the fallback asserts the model finished. Reproduced against a running server (cancel at 3.0s → returned at 3.0s with `finish_reason: "stop"`) before being fixed. Now `length`, guarded on the default so a real engine reason keeps priority.
- **`resolve_request_id` accepted a trailing newline.** `re.match` with `$` — Python's `$` also matches *before* a trailing newline, so `"abc\n"` passed the guard whose stated job is stopping a forged log line. `fullmatch` now. The parametrized test covered an *interior* newline and was green throughout; trailing cases are added.
- **`tracked_stream` unregistered while the wrapped generator was still live.** Starlette never calls `aclose` on a response body iterator, and an `async for` interrupted by `GeneratorExit` does not close its iterable — so on a hang-up the id vanished while the real generator stayed suspended holding the generation gate, and a DELETE in that window answered 404 on a demonstrably running generation. The wrapper now closes what it wraps in a `finally`.
- **`_build_args` could raise `AttributeError` and could read the process CWD.** The sidecar skip-warning dereferenced `self.model_id`, which `__new__`-constructed instances (the argv/metadata drift test) do not have; and an empty or bare-filename `model_path` made `.parent` the CWD, so a stray `chat_template.jinja` where the server happened to start could become a model's prompt format — the same CWD-relative trap that made file logging opt-in. Discovery now requires the **weights file itself** to exist, which closes both and makes the drift test's `/tmp/model.gguf` a clean miss regardless of what is in `/tmp`.
- **The two halves of the abort rule disagreed inside one file.** `_non_stream_messages` overwrote the engine's `finish_reason` unconditionally while `_stream_messages` guarded on the default. Both guard now.
- **The non-streaming `/v1/messages` response did not echo `X-Request-ID`** — the one path the cancel endpoint exists for, and the one where a client whose header was rejected had no way to learn it before its DELETE 404'd.
- **A cosmetic banner could stop the server.** `.45` moved endpoint discovery to `app.openapi()` and called it from `main()`, so any schema-generation error would raise before `uvicorn.run` instead of surfacing as a 500 on `/openapi.json` while inference kept serving. Wrapped; degrades to no list.

### Changed

- **Uncontrolled performance figures removed from five tracked files.** CLAUDE.md's rule is that no performance numbers live in tracked docs, and .44 hand-copied a consuming client's timings into `requests_api.py`, `request_registry.py`, `test_request_registry.py`, `TODO.md` and the changelog — one commit after this same branch retracted a different number from that same reporter for that same reason. The mechanism is what justifies the endpoint and it does not need them. Includes editing the .44 entry, noted here rather than done quietly.
- **`_mlx_serves_vision`'s fail-open docstring was false for the common case.** It claimed an unreadable `config.json` keeps the capability; in fact `_resolve_modalities` derives modalities at validation and falls back to `["text"]`, so a THIN entry loses `vision` before the router's fail-open branch is reached. The behaviour is right — an unreadable checkpoint is not evidence of a vision tower — but only an explicit-`modalities` entry ever exercised the path the test certified. Docstring corrected, and the common shape now has its own test.
- **`docs/api_integration.md` documents cancellation.** The endpoint was built for external clients and they were the one audience not told it exists: the id is the one they send, sending it is the precondition on the non-streaming path, 404 means already-finished, and there is no distinct cancellation value on the wire — a cancelled run is indistinguishable from budget exhaustion, so track your own.

### Added

- **`tests/unit/test_abort_stop_reason.py` — behavioural, not another meta-test.** .40's answer to this defect was `TestStopReasonHasOneMapper`, which checks *how* `stop_reason` is written and stayed green through both later regressions: a wrong value written the right way satisfies it, and the non-streaming overrides set `finish_reason`, outside its regex entirely. Deleting either override left the whole suite green. These drive an aborted generation on both wires and read what a client receives.
- `test_a_finished_request_becomes_unknown` asserted against a fresh `RequestRegistry()` that `track_request` never writes to, so it passed even with `unregister` deleted. It now exercises the module registry it names.

## [1.79.45]

Follow-through on the v3 side of the last two releases, plus a startup banner that had been lying since routers arrived.

### Fixed

- **The startup banner listed 12 of 48 endpoints and looked complete.** `get_api_endpoints` walked `app.routes` keeping anything with a `.path`, but a router mounted via `include_router` appears there as an `_IncludedRouter` carrying neither `.path` nor a `.routes` list to recurse into — so every endpoint behind a router was invisible. It omitted `/v1/messages` (the wire this project's own frontend speaks), all of `/v1/conversations`, `/v1/presets` and `/v1/admin`, and the `/v1/requests` route added one release earlier. Same shape as the `GET /` staleness fixed in 1.79.40: the first thing an operator reads, confidently describing a server that is not this one. Derived from the OpenAPI schema now — the surface this repo already treats as authoritative — and the line prints a count plus the `/docs` link rather than 48 comma-separated paths, since a banner that long buries the two lines above it. Pinned by a property test (whatever the app serves under `/v1` is what the banner reports), not a count that would need editing per route.

### Changed

- **`docs/frontend_v3_spec.md` §4 brought level with 1.79.43-.44.** The spec is the authoritative backend contract for v3 and the rule is that it moves in the same commit as any contract change; two releases went out without it. Records where the MLX `vision` capability now comes from (the loader router, so the advertised capability and the provider's 400 cannot disagree — and `modalities` is deliberately unchanged, which is why chat's history-media disclosure reads capabilities and not modalities), that both Messages-grammar routes now report `max_tokens` on an aborted run, and what `X-Request-ID` and `DELETE /v1/requests/{id}` mean for v3 — namely nothing, stated explicitly so nobody wires it up. Every v3 generation streams, and streaming has always been cancellable by hanging up.

### Verified (no code change needed)

- **The v3 frontend needs no changes for 1.79.43-.44.** Checked each surface rather than assumed: v3 reads `end_reason`, never `stop_reason`, so the abort-vocabulary fix does not reach it; chat cancels through the conversation route and notebook/explore through `controller.abort()`, so the new cancel endpoint is redundant here; and the models page is schema-driven off `/v1/admin/model-options`, where `use_sidecar_chat_template` already publishes as a boolean with `default: true` and renders through the existing tri-state control.
- **Exactly one served model's capabilities changed.** `Qwen3.5-0.8B-MLX-8bit-textonly` went `vision` → no `vision` (18 MLX models examined, it is the only one). Its twin at the same path keeps `vision`, which is the point — the entry differs only by `loader = "mlx-lm"`. In v3 this means the attach button, the paste/drop staging gate and the drop overlay now correctly refuse that model instead of accepting an image the server would 400. The frontend was already right; the server had been lying to it.

## [1.79.44]

Cancellation for non-streaming requests, and the 1.79.43 review acted on -- including one finding that would have silently broken image input on a served vision model.

### Added

- **`DELETE /v1/requests/{request_id}` cancels an in-flight generation.** A STREAMING request has always been cancellable by hanging up, because the server is writing chunks and notices the peer is gone; a NON-STREAMING one writes nothing until it finishes and so never notices. A consuming client observed an aborted non-streaming run holding the GPU for its full remaining length while the next request waited behind it. (Their timings are not reproduced here — uncontrolled for model, quant, context and machine; this repo keeps performance numbers out of tracked files, and the mechanism is what justifies the endpoint.) The abort plumbing was already there (`AbortEvent` reaches both engines; `generation_core.py` breaks its decode loop on it) — what was missing was a way to name a running request from outside it. Scope stated plainly: this makes an abandoned run **stoppable, not self-stopping**. A client that hangs up without calling DELETE still leaves the generation running; noticing that is disconnect polling, a separate mechanism deliberately not built (owner call — an explicit endpoint cannot mistake a proxy hiccup for a departed client and kill a live generation).
- **`/v1/messages` honours the client's `X-Request-ID`.** It previously always generated its own, so the id a client sends — and which the docs tell it to send — named nothing the server could find. Cancellation is by that exact value, so a rewritten id would be uncancellable. Ids are bounded and charset-restricted through one shared resolver before use, because they reach logs and the JSONL telemetry streams and a header carrying a newline could forge a log line.
- Cancellation is keyed on a **set** per id, not one signal. The id is client-supplied, so uniqueness is not the server's to assume — a retry or a shared correlation id can put two live requests under one name, and a single-slot map would let the second registration orphan the first, leaving a running generation nothing can name. `cancelled` reports how many were signalled; an unknown id is a 404, so a client that was too late learns it was too late rather than believing it stopped something.

### Fixed

- **A sidecar chat template may no longer cost a model its projector.** The 1.79.43 default would have promoted a `chat_template.jinja` unconditionally — and all three sidecar-carrying gguf models here are multimodal. A media-blind sidecar would have loaded the vision tower and then rendered prompts that can never reference an image: a vision model quietly answering as a text one, which nothing downstream flags. The guard detects media handling STRUCTURALLY (a jinja branch on a content part's type) rather than by a token allowlist, because the vocabularies do not agree — measured across the three sidecars, Qwen3.8 emits `<|vision_start|><|image_pad|>` and Muse-Glimmer emits `<|patch|>`, sharing no tokens while both spell the branch `part['type'] == 'image'`. A token list scores the second as media-blind. Failure direction is deliberate: an unrecognised template declines the promotion and leaves the embedded template in force, because a false refusal is cosmetic and a false acceptance is silent.
  - **Correction to the review that found this:** the finding asserted that model's sidecar "contains none of the vision control tokens the model needs", derived by grepping for Qwen-family markers. That is wrong — the template handles images via `<|patch|>`. The first verification here reproduced the error by reusing the same marker list, which is what a check that shares the claim's method always does. The risk class is real and the guard stays; the named instance was not an instance.
- **`/v1/models` moved off the event loop.** Deriving capabilities reads each model dir's `config.json`, and 1.79.43 added the loader-router resolution to that path. CLAUDE.md's rule — that the read routes building a model response are plain `def`, not `async def`, precisely because of this cost — was stated and violated in the same changeset.
- **The admin row builder resolved the loader up to three times per mlx row.** `effective_capabilities` now accepts an already-resolved loader. The comment directly above that code explains the route was moved off the event loop *because* its per-row cost is real.
- **argv and the spawn log could disagree about the template.** Both probed the filesystem ~85 lines apart, so a sidecar created or removed in between made the log describe a command line never issued — and the log is the only record, since llama-server's stdout goes to DEVNULL at the default observability level. Resolved once in `load_model` and handed down.
- **The provider hand-copied the new field's default.** `cfg.get("use_sidecar_chat_template", True)` meant flipping the field would leave every raw-dict caller (the drift test, the provider unit tests) on the old behaviour with the suite green. Read off `model_fields` now.
- **The `chat_template_path` pre-flight error gave advice that stopped being true.** It told operators to remove the field "to use the template embedded in the GGUF"; removing it now falls back to the normal ladder, which may find a sidecar.
- **Three documents described the pre-1.79.43 rule**, one of them 60 lines above the field that contradicts it: `GGUFModelConfig`'s class docstring, `README.md`, and `docs/api_integration.md` — the last still documenting the vision over-report fixed in 1.79.43 as current behaviour, with the now-fixed model as its worked example. It now says the MLX guarantee holds and why a client should still handle the 400 anyway (gguf forwards to llama-server unchecked, and a hand-written `capabilities` override is honoured verbatim).
- **`tests/unit/test_thinking_capability.py` had silently acquired a dependency on mlx-vlm being installed.** Deriving vision from the loader router made a `loader=auto` fixture depend on the registry probe, so the assertion would go red under the mock tree for a reason unrelated to what it tests. The loader is pinned now.
- **A cancelled generation reported `stop_reason: end_turn`** on both `/v1/messages` paths — found by live-testing the new endpoint, not by a test. An aborted run stops between tokens, so the last chunk carries no `finish_reason` and the value fell through to its default, which positively asserts the model finished its turn. This is the same defect v1.79.40 fixed on the conversation-generate route, reappearing the moment this path became cancellable: per-path behavioural tests are structurally blind to it, which is why that release added a meta-test rather than a behavioural one. Both paths now report `max_tokens` (Anthropic has no cancellation value — there, cancellation is a dropped connection rather than an end state), and the non-streaming path sets it through `finish_reason` so it still goes through the one shared mapper.
- **`TestStopReasonHasOneMapper` read a `==` comparison as a write.** Guarding an override by first reading the field is the natural way to say "only when the provider said nothing", and the check forbade that shape rather than the defect. Comparisons are excluded now; every actual write is still checked.
- **`scripts/dev_server.sh` could not size a DISCOVERED model**, so `--model <discovered-id>` refused to start on a model the server serves — `scripts/ram_report.py` read `models.toml` alone, which has been override-only since v1.69.0. It now resolves through the same `discover()`/`merge_discovered()` the router uses. The second bug found on the way is the more valuable one: `ram_fit` returns 0.0 GiB for a path it cannot read, and 0 GiB clears every ceiling, so an entry pointing at a missing file printed `RAM pre-flight OK: ~0 GiB` and exited 0 — the gate waving through exactly the case it exists to refuse. Unsizeable is now exit 2 with a reason, distinct from a memory refusal. The same zero-size hole remains open in `/v1/admin/{id}/fit` and is recorded rather than fixed.
- **The smoke suite's vision skip branch explained a mechanism that no longer exists.** Reaching it used to mean the ordinary over-report; it now means either a hand-written `capabilities` override or a regression in the shared derivation. Kept and kept loud, because the second reading is the one worth catching.

## [1.79.43]

Three of the four open handoff items closed against a running server, plus a
gguf template default. The gguf smoke arm went from UNCOVERED to green without
a code change -- the binary had already been rebuilt.

### Added

- **A `chat_template.jinja` sitting beside a .gguf is now used by default** (`use_sidecar_chat_template`, on). llama-server otherwise reads the template EMBEDDED in the GGUF, which is whatever the quantizer baked in -- and this repo has measured two publishers shipping materially different templates for identical weights. A sidecar jinja is the readable, diffable, editable answer, so it wins over the embedded one. Precedence is a three-way ladder: an explicit `chat_template_path` (someone naming a file on purpose) still outranks a file that merely happens to be in the directory, and the embedded template stays reachable via `use_sidecar_chat_template = false` -- a downloaded snapshot dir is not somewhere to have to vandalize to get the documented default back. Discovery is scoped to the model file's own directory and degrades to a clean miss on anything unreadable, so `_build_args` still works with paths that do not exist (its drift test relies on that). Bounded in practice: 3 of the 14 served gguf models carry such a file today. Every spawn now LOGS which of the three sources is in force, because a template can now change without models.toml changing -- dropping a file next to the weights is enough -- and an unannounced prompt-format change is the kind that resurfaces later as "the model got worse" with nothing to point at.

### Fixed

- **`/v1/models` advertised `vision` on a model the provider then refused images for.** `Qwen3.5-0.8B-MLX-8bit-textonly` sets `loader = "mlx-lm"` on a checkpoint whose directory declares a vision tower; the capability read the DECLARATION while `MLXProvider`'s guard reads `is_vlm`. Both were reading something real, and a client doing exactly what the API docs say -- gate on `capabilities` -- got a 400 anyway. The capability is now derived through `effective_loader_for_config`, the same resolver `MLXProvider.__init__` calls, so the two agree by construction rather than by two rules kept in step by hand. It also picks up a case nobody had reported: an explicit `loader = "mlx-lm"` on a genuinely dual-capable VLM refuses images too, and used to advertise them. `modalities` is unchanged -- the checkpoint still declares what it declares; description and served capability are different fields on purpose. Fails OPEN, inheriting the router's own direction: an unreadable `config.json` keeps the capability, so only positive non-support drops it.

### Verified

- **The gguf smoke arm is covered, and needed no fix.** The Aug-29 failure was `llama-server exited with code 1`; the model's architecture is `qwen4exp`, and the canonical build has since been rebuilt from a checkout that supports it. Both conformance rows are now covered live, across two models because no single served model exercises both: `google_gemma-4-E4B-it-qat-q4_0-gguf` 30/30 (audio accepted; thinking block UNCOVERED -- that model returned none), `unsloth_Qwen3.8-27B-UD-Q8_K_XL` 30/30 (thinking block carries `thinking`; audio UNCOVERED -- no audio modality). The capture path in the handoff was confirmed working: raising `observability_level` above `off` did produce the llama-server log that the failure message said was missing.
- **`bun run e2e:chat` was run, and is still 33/46.** The v1.79.41 repair fixed two real selector rots and the claim that they were "the whole static gap" was wrong. The deeper cause is architectural, not a selector: `tests/e2e/lib/browser.mjs` seeds sampler settings into `localStorage` and expects the chat settings panel to reflect them, but since v1.65-66 chat hydrates that panel from the DOCUMENT (`hydrateDocParams` -> `applySettings(doc.params)`), so selecting a conversation overwrites the seed. `seeded max_tokens is reflected in the settings panel` fails with the document's value, and the preset and system-prompt checks rest on the same stale model of where chat state lives. Recorded rather than repaired: this is a suite rewrite, not a patch, and it stays out of gate status until it is done.

## [1.79.42]

A second review of the 1.79.41 fixes. The headline is that 1.79.41 fixed a presence-vs-value bug by making the same mistake one level up, and that both versions of it were sitting on top of a silent drop that predates all of this.

### Fixed

- **A media block that declares a source type and carries no payload is now a 422 instead of vanishing.** `_require_source_type` validated only the discriminator, so `{"type":"image","source_type":"base64"}` with no `data` (and the nested `{"source":{"type":"base64","media_type":...}}` spelling of the same thing) validated clean — and then `converters.py`, which requires `source_type == "base64" and block.data` else `block.url`, hit its `continue` and dropped the block. The request answered **200**, the text parts survived, and the model never saw the image: a confident answer about a picture that was never sent. This predates the nested-source work entirely. A 422 naming the missing field is strictly better on a vision request, and it is the failure the "make the nested spelling work" releases were about in the first place.
- **Filling from the nested `source` is PER FIELD, which is what the contract already promised.** 1.79.41 fixed the presence-testing *gate* and left the identical mistake in the whole-block early return: any set flat field suppressed filling of all the others, so a client that sets `source_type` and leaves the payload to `source` had its type resolved and its image dropped. `docs/frontend_v3_spec.md` §4 and `api_integration.md` both described per-field behaviour that did not exist. The one case the nested object is still ignored wholesale is a genuine disagreement — flat `source_type: "url"` against nested `type: "base64"` — where merging would build a block the caller never described.
- **`tests/e2e/suites/pages.mjs` carried the same `Save` rot just fixed in `chat.mjs`.** The v1.79.41 audit was scoped to the suite that was red and missed its sibling, which `bun run e2e` also runs — the notebook preset check clicks `.preset-section button` by the exact text `Save`, replaced by `Update` / `Save as new` in v1.79.26. (Checked the rest: `pages.mjs`'s two `.model-config` `Save` clicks are correct — that button really does read `Save`.)
- **`TestStopReasonHasOneMapper` enumerated its two routes by hand and its regex skipped annotated assignment.** The class's whole premise is that routes must not diverge, so a hand-written list is the one shape guaranteed not to notice a third route — the membership rule is now "drives `StreamingEventTranslator`", derived at import. The regex also missed `self.stop_reason: str = ...`, which `messages_api.py:87` already spells that way, so respelling a bad default as an annotated assignment evaded the check entirely.
- **`schema/streaming.py` still advertised an `"error"` stop reason** in `MessageDeltaEvent`'s field description, removed from the vocabulary in 1.79.40. These models are documentary and `MessageDeltaEvent` is not in `components.schemas`, so nothing served it and no client branched on it — the reader being misled was a code reader, which is who that file is for.
- **`api_integration.md` named only one of the two shapes a capability refusal takes.** It said an over-reported capability is "refused at generation time with a 400". Non-streaming that is right; streaming, the provider guard fires at the first token — after headers flush, so the status is already 200 — and the refusal arrives in-band as an `error` event typed `invalid_request_error` (`messages_api.py:596`). §5 documented the in-band shape correctly and §1 never connected them, so a streaming client following the capability advice renders a refusal as a hang or as assistant text. The MLX/gguf asymmetry is now stated too: only MLX refuses, the gguf path forwards to `llama-server` unchecked, and whether an over-reported gguf model refuses at all depends on that subprocess. Found by the consuming-side skill's own review, which hit the same gap one repo over.
- **The Knobs list in `api_integration.md` promised a cascade for two fields that have none.** `show_special_tokens` and `include_performance` are plain booleans defaulting to `false`; "absent means the server's cascade decides" is the wrong model for both, and `metadata` was missing from the list entirely.

### Changed

- **The Clone re-entry guard is page-level, not per-conversation-id.** A per-id guard closed the double-tap and left the race its own comment named: Clone on row A then row B inside the request window is still two clones and two racing `selectConversation` awaits deciding the active document.
- **`tests/e2e/render.mjs` now covers Clone** (102 checks). Clone needs no model and no real generation, so the model-free harness is the right home; the stub holds the clone response back so both taps land in flight. The check also pins that the guard RELEASES — a one-shot latch would satisfy the double-tap assertion and silently break cloning for the rest of the session. Shown red against the guard removed.

## [1.79.41]

A review of 1.79.37-.40. Five findings, all verified against running code before being touched -- and the first one's obvious one-line fix was not sufficient, which is the entry worth reading.

### Fixed

- **The nested `source` spelling still 422'd for exactly the clients it was added for.** `_flatten_source` gated on `"source_type" in values` -- key PRESENCE, not value. `source_type` is Optional in the published schema, so a client generated from `/openapi.json`, or any pydantic client doing `model_dump()` without `exclude_none`, sends `{"type":"image","source_type":null,"data":null,"source":{"type":"base64",...}}`. The gate short-circuited, nothing flattened, and the block was rejected as "requires `source_type`" -- on the exact spelling `docs/api_integration.md` tells integrators to prefer. Serializing absent optionals as nulls is ordinary generated-client behaviour, not a malformed request. **The gate was only half of it:** the `setdefault` calls below it test key presence too, so fixing the gate alone moved the failure from the discriminator to the data (`source_type` resolved, `data` stayed null). Both now treat null as absent. A null flat field is filled from the nested object; a flat field actually set still wins; a genuinely absent source is still a 422.
- **`TestStopReasonHasOneMapper` exempted every string literal, including the one it exists to catch.** The test's stated purpose is that the two Messages-grammar routes cannot disagree again, but `if rhs.startswith('"'): continue` waved through any quoted value -- so `stop_reason = "length"`, OpenAI's vocabulary reaching the Messages wire verbatim and the precise 1.79.39 defect, passed green as long as it was spelled as a literal. Literals are now checked for membership in `StopReason`.
- **Dropping the Clone confirm removed the only thing coalescing a double-tap.** `armedConfirm` required arm-then-fire, so two fast taps produced one clone; after 1.79.37 they produce two conversations plus two racing `selectConversation` awaits, whose interleaving decides which is active. On a phone, on a small button, double-tap is the norm. `cloneConversation` now guards re-entry per conversation id. This is not a resurrected confirm -- the owner's ruling stands, nothing is armed and nothing asks twice; it is the re-entry protection the arm had been providing as a side effect.
- **`docs/api_integration.md` documented the pre-1.79.40 `content_block_start`.** It showed `content_block:{type}` while the emitter had moved to always including `text: ""` (plus `thinking: ""` on thinking blocks) -- which was the point of that change. `frontend_v3_spec.md` was updated for it and this file, the one external integrators are pointed at from the OpenAPI header, was not. Same hand-copied-description drift the release it documents was about.
- **The `_flatten_source` comment justified the gate with a case that can no longer occur.** It cited a flat block carrying an unrelated dict-valued `source` key "previously ignored as an extra field" -- but 1.79.39 made `source` a declared `MediaSource`, so such a block now fails at field validation regardless of the gate. The comment now explains the real reason.

### Changed

- `docs/frontend_v3_spec.md` §4 records the null-means-absent rule, in the same commit as the behaviour.
- **`docs/api_integration.md` brought level with the wire**, by diffing it against the generated schema rather than re-reading it: the null-beside-nested payload and the `exclude_none` workaround for older servers, `stop_sequences` (which Anthropic takes on the request and this server ignores rather than honours — a port relying on it generates past its stop with no error), `include_performance` in the knobs list, and an Extensions bullet that named four of thirteen extension fields.
- **`tests/e2e/suites/chat.mjs` unrotted.** It asserted `body[data-page]`, an attribute that exists nowhere in the frontend, so the check failed on a healthy page and the suite cascaded behind it; and it clicked `.preset-row button` by the exact text `Save`, which v1.79.26 replaced with `Update` / `Save as new`. The first now pins the router's real observable (`aria-current="page"` on the nav link). Every other selector and clicked label in the suite was audited against the frontend and resolves. STILL NOT RUN GREEN — the suite spawns a server and loads a model, so a live run is owed before it counts as a gate again.

## [1.79.40]

Code review of the 1.79.38-39 work. Fourteen findings acted on; the pattern in most of them is that a conformance pass fixed one copy of a shape and left a second copy behind.

### Fixed

- **The `error` stop reason did not exist and four documents said it did.** 1.79.39 added `error` to the `StopReason` literal, claiming `api.py` set it on the non-streaming failure path. It does not: `MessageResponse` has exactly ONE construction site, which takes `stop_reason` only from `to_stop_reason()`, and that function cannot return `error`; `api.py`'s `stop_reason="error"` is a kwarg to `_maybe_log_request_event` -- a JSONL telemetry field, not a response -- and a non-streaming failure RAISES `HTTPException`, so no `MessageResponse` is built at all. The member is removed. An unreachable enum member is worse than no member: an integrator writes and tests a branch the server cannot enter. The claim had been copied into `responses.py`, two places in `api_integration.md`, and the changelog before anyone traced it, which is the review's own summary of the release: the mechanism was asserted, not verified.
- **`schema/streaming.py` still described the pre-change payloads.** `ThinkingDelta` declared `text` only and `ContentBlockInfo` declared only `type`, while the emitter had moved to `{thinking, text}` and `{type, text}`; the module header's examples were stale too. These models are documentary -- nothing builds or validates an event through them -- so nothing failed. That is precisely the defect CLAUDE.md names: 1.79.39 fixed one hand-copied description of the wire and drifted a second in the same commit.
- **The nested `source` form was invisible to machine consumers.** It was accepted only by a `model_validator(mode="before")`, which contributes nothing to the generated JSON Schema, so `/openapi.json` advertised `source_type` as required and had no `source` property -- while the new guide told integrators the schema is authoritative AND that new code should prefer the nested form. A client generated from the schema, or a schema-validating proxy, would have rejected the recommended spelling. `source` is now a declared `MediaSource` field and `source_type` is optional in the schema, with a `mode="after"` check keeping it mandatory in fact.
- **`_flatten_source` stripped any dict-valued `source` key** and read its `type` as a discriminator even on a block already in the flat spelling. It survived only because the flat fields were present so every `setdefault` no-opped -- an accidental guard. Now gated on `source_type` being absent.
- **`ThinkingBlock` lost its non-empty guarantee.** `text: str` became `text: str = ""` so the constructor could take either spelling, which also made `ThinkingBlock()` validate into a block carrying nothing. A `mode="after"` check now requires one of the two.
- **An aborted generation claimed the model finished.** `conversation_generate_api` set `end_reason="aborted"` but left `stop_reason` at `end_turn` -- on the same stream. A consumer keying on the shared Messages grammar could not tell a cancelled turn from a completed one, and now got a spec-defined value that positively asserted the wrong thing. Aborts report `max_tokens`; Anthropic has no cancellation value, so that is the nearest honest one.
- **`GET /` -- the first surface a client reaches -- still described a one-provider server.** `model_providers: ["MLX (Apple Silicon)"]` (there are three), `model_caching: "LRU (max 2 models)"` (the default is 1), and a quick-start naming only `/v1/chat/completions`, the wire the new guide tells clients not to default to. The provider list now derives from `PROVIDER_CONFIG_CLASSES`. Same staleness the release fixed on `/openapi.json` and `/v1/capabilities`, left standing one room over.
- **The "closed list" of differences from Anthropic was not closed.** It omitted that thinking blocks carry no `signature`, that there is no `stop_sequence` field, that `message_start` omits both, that `logprobs` can be a non-streaming content block, and that `message_start.usage.input_tokens` is structurally 0 (the event is emitted before the first chunk is absorbed). All five are now listed, and the section no longer instructs readers to assume Anthropic's spec for anything absent -- it says the list is hand-maintained, has been wrong, and loses to the source.
- **The Gemini migration table taught the flat image spelling** the same document tells readers not to use -- for exactly the reader that table exists for.
- **`tests/e2e/render.mjs` was updated for `stop_reason` but not for `content_block_start`**, so the only automated consumer of this grammar stopped mirroring the wire in the same commit that changed it.
- **`TestStopReasonHasOneMapper` was partly vacuous and CWD-dependent.** Its import arm asserted only that the substring appeared somewhere in the file -- which the call site guarantees even with the import deleted -- and both arms resolved paths against the process CWD, so pytest from any other directory raised `FileNotFoundError`. Paths are now repo-root relative and the import arm checks the import.

### Changed

- **`docs/frontend_v3_spec.md` §4 now records the contract change**, which CLAUDE.md requires in the same commit and 1.79.38-39 did not do across five commits: the accepted media-block spellings, the `thinking_delta` payload, the `content_block_start` payload, and the `stop_reason` vocabulary on BOTH Messages-grammar routes.
- **Version bumped for a wire change that shipped without one.** `adc8096` changed `conversation_generate_api`'s emitted `stop_reason` while only appending to the existing 1.79.39 changelog entry, so two builds both reported 1.79.39 and emitted different values on the conversation route. Sharper than usual because the immediately preceding commit's rationale for `version=__version__` was that `/openapi.json`'s version is the first field an integrating client reads.

### Coverage

This release changes `conversation_generate_api`, so the Phase 4 release standard applies: every engine arm runs, and an arm that does not is NAMED here rather than passed over.

- **mlx-lm 26/26, mlx-vlm 31/31**, including the three conformance rows added this release, which had never executed before the run. The nested-image-source row passed live on mlx-vlm -- the arm that regressed hardest.
- **gguf is UNCOVERED.** `Qwen3.8-Flash-Next-UD-Q5_K_XL` 500s at load with "llama-server exited with code 1 -- output not captured (file logging was off when this model spawned)", so the whole gguf engine is unverified for this conformance work. Diagnosing it needs `observability_level` raised above `off` and a reload BEFORE the load attempt, because the provider decides capture at spawn.
- **Standing UNCOVERED, by design:** thinking DEPTH on both MLX arms (the only served MLX model advertising `reasoning_effort` is `gpt-oss-120b`, so covering it costs a 120B load), and the vision spelling on the text-only model.
- `bun run e2e:chat` is red at 33/46 and is NOT a signal here: the suite has bit-rotted against v1.79.26's preset-bar rename and asserts a `body[data-page]` attribute the frontend does not have. Tracked in `docs/project/TODO.md`.

## [1.79.39]

`/v1/messages` now actually speaks the Messages API it advertises.

### Fixed

- **The image block was flat where Anthropic nests it under `source`.** An Anthropic SDK, or anyone following Anthropic's docs, sent `{"type":"image","source":{"type":"base64",...}}` and got a **422** from an endpoint whose whole premise is being Messages-shaped. Both spellings are now accepted and normalize to one internal shape in the schema (`_flatten_source`), so nothing downstream knows which arrived. The flat form stays valid -- every existing client sends it.
- **The thinking block and its delta named the field `text`; Anthropic names it `thinking`.** A conformant reader found the block and nothing in it. Both fields are now populated on `ThinkingBlock` and on `thinking_delta`. `text` is kept deliberately: v3's `streaming.js` reads it in two places, and dropping it would blank the reasoning pane on notebook and explore. `thinking` is the conformant name and what new clients should read.
- **`stop_reason` carried the provider's OpenAI `finish_reason` vocabulary onto the wire** -- `"stop"` / `"length"`, neither of which the Messages spec defines. Now `end_turn` / `max_tokens` / `stop_sequence`. (This entry originally added a fourth member, `error`, on the stated grounds that `api.py` set it on the non-streaming failure path. That was false and is corrected in 1.79.40 -- see there.) The mapping is ONE table, `STOP_REASON_FROM_FINISH_REASON` in `converters.py`: it previously existed as an inline `if/elif` in the non-streaming converter AND as a raw passthrough in the streaming path, which is the hand-copied-constant defect this repo keeps re-learning -- the streaming arm had no mapping at all, so the two arms disagreed about the same generation. A comment above the non-streaming branch even claimed a mapping happened that did not.
- **`content_block_start` now opens with its content field present and empty**, as Anthropic's does, so a client accumulating into `content_block` has something to accumulate into.

- **The same `stop_reason` passthrough existed a SECOND time**, in `conversation_generate_api.py`, which shares `StreamingEventTranslator` and therefore speaks the same Messages grammar. Fixing only `/v1/messages` left the busiest route -- v3's chat page -- emitting `"length"` while its sibling emitted `"max_tokens"`, for exactly one commit. Both now call the shared mapper. `TestStopReasonHasOneMapper` asserts that every write to `translator.stop_reason` in either module goes through `to_stop_reason()`: per-path behavioural tests structurally cannot see two paths disagreeing, which is why 1700 of them stayed green while this was broken. The block deltas never diverged -- they come from the shared translator.

### Changed

- **The integration guide stops documenting traps and starts documenting differences.** Conformance deletes prose: the flat-image explanation existed in five places because it was a surprise that had to be re-explained everywhere, and it is now one line saying the older spelling still works. What remains is a short, closed list of deliberate divergences (`max_tokens` optional, `thinking` is a bool rather than Anthropic's config object, no tools, the `error` stop reason, the heylook extensions, no server-side resize, install-local model ids). Everything not on that list defers to Anthropic's published spec -- a source of truth this repo does not maintain.
- `ImageBlock` / `AudioBlock` docstrings now state that the nested form is accepted, because a `model_validator(mode="before")` is invisible in the generated JSON Schema -- an agent reading `/openapi.json` would otherwise still conclude nested `source` was invalid.

## [1.79.38]

External-integration docs, and the OpenAPI header they tell people to trust.

### Added

- **`docs/api_integration.md`** -- the contract for an application adding this server as an inference provider, scoped to what such a client needs (discovery, one wire, images, SSE, errors, auth) and pointing at `frontend_v3_spec.md` §4 as authoritative for the traps rather than forking its list. It leads with the one that costs an afternoon: `/v1/messages` LOOKS Anthropic-shaped and its image block is **flat** (`source_type`/`media_type`/`data` on the block), so Anthropic's nested `source` object matches no member of the block union and comes back 422. Confirmed by constructing both spellings through `MessageCreateRequest`, not read off a docstring. Also stated: `/v1/messages` deliberately has no server-side resize (clients downscale, as `image-prep.js` does) while `/v1/chat/completions` keeps `resize_max` and friends, and the stream ends at `message_stop` with no `[DONE]` on the Messages wire but with one on the OpenAI wire.

### Fixed

- **`/openapi.json` said the server was version 1.20.0.** Hardcoded in the `FastAPI(...)` call and ~60 releases stale, on the first field an integrating client reads. Now `version=__version__`, which also makes the version-sync hook's "no other files carry the version" true.
- **Every tag description was silently dropped from the served schema.** `custom_openapi()` called `get_openapi()` without `tags=`, so `openapi_tags` -- ten curated descriptions -- reached nothing, and `/docs` grouped routes under bare names. Passing `tags=app.openapi_tags` restores them. The Messages API had no entry at all and now has one.
- **The schema's narrative header described a server that no longer exists.** It documented MLX as the only backend (there are three providers), never mentioned `/v1/messages`, and its copy-pasteable examples named models retired long ago -- `llava-1.5-7b-hf-4bit`, `qwen2.5-coder-1.5b-instruct-4bit`. Rewritten to cover both wires and their divergences, the error taxonomy, the auth gates, and that startup loads nothing. Example model ids are now **placeholders on purpose**: a real id is install-local, since the registry serves whatever sits under the scanned folders, so a concrete one rots into a 400 for every reader.
- **`/v1/capabilities` named only `/v1/chat/completions` under vision**, telling a client discovering the server that vision was that endpoint's alone -- the opposite of where integrators are pointed. Both wires are now listed with their `server_side_image_resize` difference explicit.

## [1.79.37]

Both open items from the 1.79.36 review ruled on by the owner.

### Removed

- **The Clone button's armed confirm.** Only LOSS gates: a clone destroys nothing and one Del undoes it, so the second tap bought inconvenience rather than safety -- and it sat immediately beside Del, whose arm does guard a loss, teaching that these buttons ask twice as a matter of course. The outcome is disclosed instead, which is what the rule prescribes here ("Conversation cloned." already existed). Clone stays a one-tap action that stops event propagation, which is what `armedConfirm` was doing for it besides.
- **`scripts/migrate_conversations.py`** and its `scripts/README.md` section. It was the hand-run escape hatch for carrying a conversation store across a schema bump; the rule it sat against ("NEVER write migration code") stands unamended, and the store's own policy is what it always was -- `db.Store` recreates on a version mismatch. **Consequence, stated rather than left implicit:** opening an older store with newer code now DROPS `conversations`, `messages`, `media_blobs` and `notebooks` with no prompt, no backup and no escape hatch. That is the intended posture for this deploy (solo, no conversation data to preserve); it is only a loss if that posture changes.

## [1.79.36]

Code review of the whole `frontend` branch. Twelve findings acted on; two are left open as owner decisions (below). Each was re-verified against the code before being touched -- a subagent report is not evidence.

### Fixed

- **The idle-unload re-check let a gguf generation take a SIGTERM.** `unload_idle_models` computes candidates, releases the lock, then `_unload_idle` re-acquires and pops -- so anything becoming true in that window must be caught by the re-check. It tested only `generation_queue_stats()`, which is `None` for a provider that queues its own requests. gguf is exactly that, so a run starting inside the window was popped and llama-server SIGTERMed under an open stream: the "gguf had strictly less protection" hole `_is_generating` exists to close, still open on the path that does the popping. `_pinned` was missing from the re-check too.
- **A busy model was unloaded on the very next tick after its run ended.** `_last_used_ts` is stamped at REQUEST time, so a generation longer than the idle threshold is already stale for its whole second half -- routine since runs began outliving their response. Skipping without re-stamping only deferred the unload by one tick, tearing the model down the instant the reply landed. All three refusals mean IN USE and all three now stamp.
- **A finished run could pop a NEWER generation's claim.** The stream generator's `finally` did a bare `_ACTIVE.pop(conv_id, None)`, contradicting the invariant stated where the claim is installed -- and it is the release most able to break it, running on a detached task that outlives the response. If anything released run A while A was alive, a second POST claims as B and A's finally pops B: B is unstoppable (`DELETE` 404s), reports idle so the composer offers Send, and a third POST passes the 409 gate and writes into the same conversation.
- **The `/v3` gzip cache held exactly one entry.** It cleared itself on every miss, so each asset of a ~20-file page load evicted the previous one; nothing was ever a hit and the level-6 compression it exists to keep off the event loop ran on every load -- the same loop delivering SSE tokens. Eviction is per-path now.
- **`MODEL_BUSY` told the user the wrong thing.** It has two causes -- the generation queue being full, and eviction BLOCKED because every loaded model is generating -- and all three endpoints that map it to 503 replaced the raised message with a fixed sentence about the queue. The eviction raise names the models and the remedy; that now survives. One speller for all three (`busy_response.py`); the wire shape is deliberately unchanged, because v3 retries on exactly `(503, model_overloaded)` and changing the code would silently disable that.
- **`markdown-stream`'s boundary rule was violated for CommonMark HTML blocks of types 1-5** (`<pre>`, `<script>`, `<style>`, `<textarea>`, `<!--`, `<?`, `<!X`, `<![CDATA[`). They end on a closing condition, not a blank line, so they reach forward past the only thing the scanner treats as a block break -- and a committed prefix is never revisited, so the seam is permanent. Marked `unsafe`, the same treatment a link-reference definition gets.
- **Three textareas lost their height caps on Chrome/Edge.** `autoGrow` returns early where `field-sizing: content` is supported, handing sizing to CSS *including the cap*, and `.sysprompt-input`, `.message-edit textarea` and `.message-edit__thinking` had no CSS `max-height`. A long system prompt pushed the drawer off-screen; editing a long message put Save/Cancel below the fold.
- **`cloneConversation` unshifted the full clone response into the list-shaped `s.conversations`** -- pinning the cloned thread in memory for the session and carrying no `generating` key, so `refreshAfterResume`'s membership test answered differently for that one row.
- `explore`'s `renderStrip` ran a `querySelector` over the whole strip on every append; `s.chipEls.length === 0` answers the same question in O(1).
- `MockProvider` called neither `super().__init__()` nor anything installing `BaseProvider`'s state, so the teardown guard above was untestable against it -- the trap `FakeProvider` documents avoiding, in the other mock.

### Removed

- `cachetools` and `python-multipart` as declared dependencies: nothing in `src/` imports either (the only `cachetools` user was the deleted `fast_image.py`; no route uses `Form`/`File`/`UploadFile`), and the `cachetools` comment named the deleted module. The OpenAPI description also advertised it as a live feature.

### Changed

- `admin_api`'s comments about the unload drain were made false by the generating-refusal: the drain is no longer what covers an active generation on the explicit path (it covers gate waiters and `force=True`), and the 409 now has two causes rather than one. A reload issued during any generation refuses rather than waiting the model out -- including a detached run the requester cannot see. Behaviour intended; the comments were not.

### Owner decisions (both ruled in 1.79.37)

- The Clone button's `armedConfirm`. Cloning loses nothing, so by the standing "only LOSS gates" rule this is a click-through-training confirm -- and it sits beside Del, which does guard a loss. Left as-is pending a ruling.
- `scripts/migrate_conversations.py` against the "NEVER write migration code" rule. Its docstring argues it is a one-off outside the app, which is defensible, but it imports `db._SCHEMA_SQL` so a schema change now has a second file to keep working. Wants an explicit ruling that amends the rule text rather than a carve-out living in the script.

### Not a finding

- The review said v3 auto-retries a blocked `MODEL_BUSY` "every second for the whole duration of a run". It does not: `MAX_BUSY_RETRIES` is 3, so the client gives up after about three seconds. The message was wrong; the backpressure was not.

## [1.79.35]

### Fixed

- **`tests/smoke/` picked its arms' models by `len(id)`, which is not a cost signal.** Unnarrowed on this machine it chose gpt-oss-120b for mlx-lm and 27Bs for the other two arms -- so the Phase 4 release standard shipped one commit earlier ("green on all three arms") was something nobody would run, which is the failure the whole plan is about, one level up. Selection is now resident-first, then **smallest by measured weight size**: `POST /v1/admin/models/{id}/fit` sizes from file stats with no load, and takes ~1s for a 30-model registry. A model whose size cannot be determined sorts LAST -- an unknown size must not win a contest about smallness. The old heuristic's own docstring already said "a short name is not a small model"; it was right and the code kept doing it anyway.

### Verified

- The unnarrowed run -- the invocation the Phase 4 standard actually mandates, and the one every earlier check in this arc had skipped by passing `--arm` -- is green: **52/52 across all three engines**, picking ~1GB, ~1GB and an E4B gguf. Four honest skips (thinking depth on both MLX arms; the Stop path on both, where the model finished inside the window).

## [1.79.34]

### Added

- **Engine coverage Phase 3 -- the same-feature-two-mechanisms checks** (`tests/smoke/`). Four rows, each a feature with two implementations split by engine, which is exactly where reasoning from the provider Literal hides the second one.
  - **Audio**: MLX must refuse with a **400** whose text names gguf as the way to run audio; gguf accepts it. Two details decide whether this check runs the guard at all. It goes to `POST /v1/messages`, not the conversation generate endpoint -- that path drops media the current model cannot take AT THE WIRE (the mid-conversation model-switch rule), so the provider guard is unreachable there by design and a check aimed at it would have passed without ever running it. And it is **non-streaming**, because a provider's request guards fire at the first `next()`, which on the streaming path is after the headers flush -- the same refusal then arrives as a 200 carrying an in-band `invalid_request_error` event.
  - **Thinking capability** and **thinking depth**: a model that ADVERTISES the capability must ACCEPT the corresponding request. Depth sends `medium`, the one value in both published sets (Qwen3.8 takes xhigh|medium|low, harmony low|medium|high), which keeps the check about the mechanism rather than one model's vocabulary -- and on gguf a value the template rejects is a raised jinja exception that llama-server returns as a 500.
  - **Chat-template source**: model-free, in the contract half, so `--contract-only` covers it in seconds. `chat_template_source` must be offered for mlx and not gguf; `chat_template_path` for gguf and not mlx. The failure mode is silence -- setting the wrong one does nothing at all and the publisher's embedded template goes on picking your prompt format.
- **Engine coverage Phase 4**: the release standard is written into `CLAUDE.md`'s test section. Before a release touching provider, loader, template or lifecycle code, `tests/smoke/` runs green on all three arms, and an uncovered arm -- or an uncovered Phase 3 mechanism -- is named in the changelog rather than passed over. Deliberately a documented standard, not a CI gate.
- A stdlib WAV fixture in `tests/smoke/run.py`, same rule as the PNG beside it (no deps). Well-formed on purpose: a malformed part would be rejected by the schema and that 400 reads identically to the provider's.

### UNCOVERED

- **Thinking depth on both MLX arms.** The only served MLX model advertising `reasoning_effort` is `gpt-oss-120b-MXFP4-Q8-mlx`, so covering it costs a 120B load. Reported as uncovered on every run rather than quietly skipped -- the rule this plan exists to enforce, applied to itself. The gguf half is covered.

### Verified

- All three arms green live: mlx-lm 23/23, mlx-vlm 24/24, gguf 27/27, contract 11/11. The mlx-lm arm is cheap for the first time -- a `loader = "mlx-lm"` entry on a small model, the fix the plan recommended for its own v1.79.31 finding.

## [1.79.33]

### Fixed

- **`deleteConversation` asserted an outcome it never waited for.** It fired `stopGenerate` and immediately claimed, via `ABANDON.DELETE`, that the run had "genuinely ended server-side" -- on a request whose status it discarded. It awaits now: 200 (stopped) and 404 (nothing active, so it already finished) both mean genuinely ended; `null` -- the DELETE never arrived -- claims nothing, which is the weakest reason in the set. `stopRemote` has modelled this for the same endpoint since v1.79.29 and the two must not diverge. Awaiting also removes the race the 409-then-retry branch below it was covering for: the conversation DELETE used to go out concurrently with the stop it depends on. Pinned by a render check that holds the stop's ANSWER back 400ms -- dispatch order was always stop-then-delete, so only a delayed answer can show whether the second waited.
- **`setRemoteGenerating` tested for a live stream's EXISTENCE where it meant OWNERSHIP**, and `releaseStream` restored the composer to a literal `'Send'` rather than to the conversation's state. Together: switching from one generating conversation into another could leave the composer reading Send for a run that is going, with no way to stop it. Stated precisely, because it matters -- **no reachable symptom was found**. The abort unwind is microtasks and the switch awaits a GET, so the unwind always wins; that was measured, not assumed. What is fixed is that the invariant no longer depends on that ordering: a stream aimed at another conversation owns THAT conversation's button, and a released stream restores what the current document actually is. The two had to change together -- correcting the guard alone converts a timing-dependent bug into a certain one, since the abandoned stream's unwind is guaranteed to land last.

### Changed

- The composer button's two rest states have one speller (`setSendButton`). Text and title were written from two places, which is how "Stop" came to mean two different things with only a tooltip between them.

## [1.79.32]

### Fixed

- **An abort reason could be silently downgraded by a second abort.** `abortStream` was last-writer-wins on `abandonReason`, and `deleteConversation` calls it twice for one stream -- once as `delete` (the run genuinely ended server-side), then again as `switch-conversation` (the opposite claim: it detaches and finishes) when it selects the next conversation. The reasons are ranked now and the strongest wins; equal ranks still take the later value, since two switches are both "abandoned" and the newer one describes where the user actually is. No consumer reads `delete` today, which is precisely why the downgrade was invisible -- a reason is a claim, and one a later weaker call can overwrite is not one the next consumer can trust.

## [1.79.31]

### Added

- **`effective_loader` on `GET /v1/admin/models`** (and the single-model read): `"mlx-lm" | "mlx-vlm"` on an mlx row, `null` on every other provider. Provider `mlx` is TWO upstream repos on separate release trains, so `provider` does not name an engine and `config.loader` is only a routing hint (`auto` degrades vision→mlx-lm when mlx-vlm does not register the `model_type`). DERIVED from the config plus the model dir's `model_type` — deliberately **not** read off a loaded provider, because `MLXProvider.effective_loader` is null for every model that is not resident, which is precisely the set a live harness has to pick its arms from. Read-only, so it sits top-level beside `provider` and never inside `config`, where the schema-driven editor would render it as a settable field.
- **`tests/helpers/engines.py`** — one engine classifier, imported by both `tests/smoke/run.py` and `tests/eval/run.py`. `tests/eval`'s `fetch_models` is gone rather than duplicated; it now gets capabilities from the same call. A model classifies as an arm, as *excluded* (embeddings, by owner call), or as *unclassified* — and unclassified prints as a coverage hole with no name, which is different from a deliberate exclusion.
- **Engine-coverage reporting in both live harnesses.** Per arm: `covered`, or `UNCOVERED` distinguishing "no model of this engine is served" from "models are served and this run touched none of them". Smoke prints it BEFORE the arms run. Eval prints it after its summary, plus the count it most needed: **how many tasks ran on NO model**, by category — the bank's `required_capabilities <= model_caps` filter was silent, so a text-only `--models` list ran zero vision tasks under a full-width green.

### Fixed

- `tests/smoke/run.py` read `got["params"]` off a conversation GET without a status guard, so a failed GET raised out of the whole contract half — a harness that reports a server problem as a traceback reports nothing at all. Body reads now survive the three shapes `call` can return (dict, `None`, raw error string).
- `tests/e2e/render.mjs` aimed Apply and Update with positional `nth-of-type` selectors, which silently re-aim at whatever button lands in that slot; Update is the destructive one. They anchor on the stable `title` attributes now (`armedConfirm` relabels `textContent` when armed and never touches `title`, which is also why a text lookup is not the answer).

### Changed

- `GET /v1/admin/models` and `GET /v1/admin/models/{id}` are plain `def` (FastAPI threadpool) rather than `async def`: the derivation stats each served model's `config.json`, and that is disk work, per request, which an `async` handler would do on the event loop mid-stream.
- `mlx_vlm_supports` is `lru_cache`d and the vision→mlx-lm degradation logs once per `model_type` per process. Both were free when the probe ran once per LOAD; it now runs per row of every admin list.

### Found

- **This machine serves exactly one mlx-lm model, `gpt-oss-120b-MXFP4-Q8-mlx`.** The cheap text arm `docs/project/plan_engine_coverage.md` assumed — `Qwen3.5-0.8B-MLX-8bit` — is **mlx-vlm**: Qwen3.5 declares vision and mlx-vlm registers `qwen3_5`, so it positively routes there. The mlx-lm arm therefore costs a 120B load today, which means it will not get run. Recorded in the plan with the cheapest fix (pin `loader = "mlx-lm"` on a small model; an explicit loader forces the engine and the new field reports it). Nothing could have surfaced this before — which is the argument for the field.
- Verified live against a real server: for all 17 served mlx models the value on the wire equals what `MLXProvider.__init__` computes for the same id. Same rule, same two inputs; the check is that the two call sites feed it the same thing.

## [1.79.30]

### Fixed

- **The lifecycle outcome is now the SERVER's answer, not three client-side proxies.** `finishGenerate` already awaited `resyncMessages` — which returns the store's own `generating` flag — and discarded it, then inferred the same fact from `aborted`, `stream.userStopped` and `stream.abandonReason`. Every lifecycle bug in v1.79.26–.29 was a symptom of that: each new abort path had to be hand-classified, and the classification was silently wrong the moment it drifted. The rule is now what it always should have been — `heylook_saved` present means the run ENDED and `end_reason` says how; absent means the GET decides. The proxies survive only where the server cannot answer.
- **`resyncMessages` returned `undefined` on a failed GET**, which the recovery loop read as "the server is done" — so an offline retry burned its attempts and then announced "Recovered what the server saved." having never reached the server. It returns `'generating' | 'idle' | 'unknown'`; `unknown` is the absence of an answer, not a negative one.
- **A stop that never reached the server said nothing.** `stopGenerate` resolving null left the stream painting and the button reading Stop, inviting a second press that would do as little. It now discloses, matching `stopRemote`, which already handled the identical case.
- **An abandon note could never be cleared.** `setRemoteGenerating`'s off-branch only cleared lines starting "Still generating", which the model-switch wording does not — so the page went on claiming a reply was coming for an idle conversation, permanently. One shared prefix and one predicate now; a wording cannot come apart from its own eraser.
- **An async abandon note destroyed an on-screen switch warning and its buttons**, leaving the select on an unconfirmed model with no way to confirm or cancel — and the next send going to that model. Status writes now yield to a live switch warning.
- **A mid-stream model switch silently dropped the thinking-loss disclosure**, reintroducing on v1.79.26's path the exact capability loss v1.79.27 closed. The note is computed before the abort and carried on the stream.
- Smaller, same family: `abandonNote` named `modelSelect.value`, which can be an unconfirmed target, rather than the committed model; an untitled conversation silenced the leaving-disclosure entirely (`?? null` does not normalise `''`).

- **The preset write path.** `updateSelected` — the one destructive write — dropped the pre-write `refresh()` that `save()` had, while the harmless `saveAsNew` kept it, so it decided *and* executed against a cache refreshed only at drawer-open. It now refetches, and blocks on **"did it change since you looked"** (comparing the fresh row against what the preview rendered), not on "is this still destructive" — that question is still true a click later and would refuse every confirmed update. The stored value is also part of `armedConfirm`'s `target`, so an in-process refresh between arm and confirm re-arms. It no longer re-sends a cached `name` (which reverted a concurrent rename, or 409'd the confirmed write away), and a missing selection now says so instead of silently doing nothing.
- **A name-in-use refusal no longer eats the name you typed.** It forced a drawer rebuild, replacing the section — and on a phone closing the keyboard — at the moment you were told to go do something else with that name.

### Changed

- `tests/smoke/run.py`, after review: `stream_until` parses the deltas it saw, so the walk-away check compares the persisted answer against that prefix — the comparison that distinguishes a detached run from a **truncation**. It previously asserted only that a non-empty assistant row existed, which a truncation also satisfies, and the CHANGELOG claimed a property the code could not observe. Also: `ended_complete` (a failed run ends cleanly too, so "it finished" was true of a dead engine), `warmed` checked on load, every user-turn POST checked, a 404 stop no longer accepted as success (and a run that finished first *skips* rather than passes), the walk-away skip waits the run out instead of cascading into later checks, HTTP errors reported rather than raised, and an all-skipped run exits **2** instead of printing PASS.
- `classify()` no longer claims an engine it cannot confirm: an explicit `loader` is believed, and an auto-routed vision model is taken as mlx-vlm but reported as **not confirmed**, because mlx-vlm degrades to mlx-lm for an unregistered `model_type` and nothing served can see it. Model auto-selection prefers a RESIDENT model — sorting by name length picked gpt-oss-120b for a "cheap" smoke run.
- Constants that had grown copies: the four abort reasons are a frozen set, and the `cold` residency derivation is one `isCold()` rather than four.
- Stale prose removed and replaced with what is now true: `preset-bar.js`'s header grammar and drift tooltip, `frontend_v3.md`, `frontend_v3_spec.md`, `CURRENT.md`, `CLAUDE.md` (including a pronoun that made the smoke suite read as the stubbed one, and a rule about Enter that the split made obsolete — Save as new has no arm to route around), and the user guide's orientation table, which still named a **Save** button this work deleted.

## [1.79.29]

### Fixed

- **A completed generation reported "Stopped."** — the v1.79.26 lifecycle regression, found independently by two reviews, and a re-opening of one a 2026-08-13 review had already closed from the other side. `stopStream` set `stream.userStopped` at the top, before knowing whether the server had stopped anything. Two of the four `stopGenerate` outcomes then lied: a **404 with events** means the server already finished and `heylook_saved` is still in flight (the stream completes with the whole answer), and a **null status** means the DELETE never arrived and the run is still going. Both showed "Stopped -- partial response saved." over a live or complete response, and the 404-with-events case also lost the token/timing line. The flag now lives in the one branch where a local abort really does mean stopped — `404 && !sawEvent`, beside the abort it explains. A 200 needs no flag: `end_reason: 'aborted'` already carries it.
- **A failed run that was then stopped reported a clean stop.** The error branch sat third in the chain, so "Stopped." overwrote the red `Generation failed:` line in normal colour and the failure left no report at all. Order is now failure first.
- **The abandon note could claim a reply was still coming over a finished one.** It was gated on `aborted` alone, but `heylook_saved` is always the last event (spec §4) — its presence means the run ENDED whatever the fetch did. Now `aborted && !saved`.

### Changed

- `abortStream`'s comment no longer claims teardown as a call site. `page.js`'s `unmount()` aborts `ctx.signal` and `linkedController` chains the stream's controller off it, so leaving the page never comes through `abortStream` at all — the `teardown` default is the safe value, not a documented path. Correcting a claim the review showed was false.
- `s.lastCommittedModelId` is a local again. It was set and read ten lines apart inside `commitModelSwitch` and nowhere else, while reading as page state meaningful elsewhere — and `switchWarnings(ctx, from, to)` immediately above already takes `from` as a parameter.
- `tests/smoke/run.py`'s stream tail is RETURNED rather than stashed in a module global. The stop check runs `stream_until` on a worker thread, and a global cleared-then-appended from a worker is a race waiting for the first person to drive two streams at once.
- A render-suite check pins the regression: the stub already answers the stop DELETE with 404, which is exactly the real shape, so pressing Stop mid-stream and letting the run finish reproduces it directly.

## [1.79.28]

### Added

- **`tests/smoke/` — a live smoke test, opt-in, one arm per ENGINE.** `tests/e2e/render.mjs` drives the real `/v3` page against a *stubbed* `/v1`, so everything the client's preset and lifecycle work rests on — a store that refuses a duplicate preset name, params round-tripping onto a conversation, a run that detaches and finishes after the reader walks away — was unverified. This is that half. Never spawns a server (same rule as `tests/eval/`); `--contract-only` runs the storeside checks in seconds with no model load.
  - **Arms are engines, not providers.** The provider `Literal` has three values but they are not three engines: `"mlx"` routes to *two separate upstream repos* (mlx-lm for text, mlx-vlm for vision, on separate release trains) via `effective_loader`, while `"gguf"` is one llama-server binary. "We covered mlx" is a claim about a config value, not about code. The harness runs a text arm and a vision arm separately and reports a missing arm as **uncovered**, never as green. `mlx_embedding` is out of scope (owner call).
  - Verified against a real server on all three arms: disconnect mid-stream, then confirm the run committed and the conversation went idle. (As shipped this asserted only that a non-empty assistant row existed, which a truncation would also satisfy — v1.79.30 made it compare against the prefix the harness actually saw, which is what distinguishes a detached run from a truncated one.)
  - The vision arm failed on its first run and the cause was the fixture: a 1×1 PNG dies in gemma's aspect-ratio-preserving resize (PIL refuses a degenerate `(1,1,1)` array) before the model is reached. It now generates a 64×64 PNG with stdlib `zlib`/`struct`. A smoke test must go red when the engine is broken, not when a fixture is degenerate.
  - A no-delta ending now quotes the stream's own trailing bytes. The first failure reported only `saw_delta=False`, and the reason was reachable only by digging through the server log.

## [1.79.27]

### Fixed

- **Losing thinking on a model switch was announced only when media was also being dropped.** `switchWarnings` gated the note on `lines.length`, so it could only ever ride an existing image/audio warning — in a text-only conversation, switching from a thinking model to a plain one said nothing at all and the toggle simply vanished. Found while closing the `Save & Continue` edge below. The fact now has its own path: it rides the blocking warning when one exists, and otherwise goes to the status line beside the load-cost note. Deliberately **not** promoted to a blocking warning of its own — a capability going away destroys nothing, and only-loss-gates says disclose rather than interrupt a routine switch.

### Changed

- **`Save & Continue` is disabled with a reason instead of absent.** Continuing your own message is MLX-only (llama-server prefills assistant turns and has no user-turn spelling), and the button used to disappear — including while the provider was still unknown — so it read as arbitrary at exactly the moment you looked for it. It is still fail-closed: disabled means it cannot fire, which matters because the continuation discards everything after the message and that truncate would land before the failure. The render suite's check moved from "is absent" to the stronger "is present, disabled, and gives a reason".
- **`Thinking depth` says its accepted values are per-model.** The control offers the union across model families, so a value one model takes another rejects — for a gguf model a raised jinja exception comes back as a 500. The backend cannot narrow this (the accepted set lives in the chat template, and for gguf inside the GGUF's metadata), so the control states the caveat and points at `auto`, which always works. `PARAM_META` grew an optional `note`, rendered inside the field's label so `.settings-row` stays a two-child flex.
- **An armed confirm now holds for eight seconds, not three.** On a phone, tap → read "Overwrite prompt?" → pause → tap landed as a fresh *first* tap, which reads as "the button didn't work" — the cost fell on the careful readers the confirmation exists for. Safety never rested on the timer: `target` already makes a stale arm refuse to fire, so it only clears a visually stale label.
- **"Still generating — this reply was started elsewhere"** no longer claims an origin it cannot know. Since abandoning a run is routine and disclosed as of v1.79.26, the commonest way to see that line is a run *this* tab started and walked away from. It now reads "Still generating on the server — Stop ends it and keeps the partial." Send/Stop also carries a tooltip separating "stop what you are watching" from "stop the run finishing on the server".
- `.gitignore` covers `*.duckdb-*`. A hand-made copy of the store lands as `conversations.duckdb-2`, which `*.duckdb` does not match — runtime data that was one `git add` away from being committed.

## [1.79.26]

### Fixed

- **The client reported "Stopped." for a generation that was still running.** A run outlives the response that started it, so aborting the fetch ends the *subscription* while the run detaches, finishes, and commits the whole answer. Switching models mid-stream aborts locally into the same conversation, which meant `finishGenerate` reached its abort branch with the conversation still active and announced a stop that had not happened — a false statement about the server, and the opposite of what the reader needed. `abortStream` now carries WHY it let go, `stopStream` marks a genuine user stop, and only a server-side abort, an explicit Stop, or a conversation delete may claim "Stopped".

### Added

- **The app says a generation survives you leaving**, at the moment you leave. Switching conversations names the one being left ("…keeps generating — it will finish on the server and be there when you come back"); switching models says the reply in flight finishes on the previous model and saves to this conversation, and carries the load-cost clause for the new one. This is the app's best behaviour and was undiscoverable: the only place it had ever been mentioned was a status line you saw if you happened to return mid-run. Page navigation is NOT covered and cannot be — the chat status line is gone by then.

### Changed

- **Overwriting a preset is no longer something you can do by typing.** The select pre-filled the save-as name, so browsing a preset silently aimed Save at it, and a typed name that happened to match overwrote — the shape that cost a 35k-character prompt. Which preset you write is now decided by which button you press:
  - **Update** overwrites the preset in the dropdown, the one the preview directly above it is showing. The only path to an overwrite, and still armed by the same three questions (nothing to lose / would this blank it / is the document already running it).
  - **Save as new** creates under the typed name and REFUSES a name in use, naming Update as the way to overwrite. Never armed, because it cannot destroy anything.
  - The guard simplifies with the design: it resolved a typed name to a row, which is what let a write aim at a preset nothing on screen was describing. That case is gone rather than guarded, and its check is replaced by one asserting the refusal.
- `.preset-row` wraps and its select keeps a `flex-basis` floor. Update joined a row that already held a select, Apply and Del, and an armed button relabels itself to "Overwrite prompt?" — at phone width a nowrap row would overflow or crush the select, the one control naming what Update would hit. Guarded by a new check in the render suite's phone boot, measured armed and unarmed.
- The user guide describes both changes and moves two more rough edges into its Closed ledger. Three remain.

## [1.79.25]

### Changed

- **The sampler panel now says what it applies to.** It was simultaneously the open document's stored params, the seed for the next new document, and a panel that silently refilled itself on every document switch -- with nothing on screen saying which. That ambiguity is upstream of the preset machinery and accounts for most of the confusion the guards in v1.79.20-.22 were catching downstream of.
  - A note under the *Sampling* heading: "Applies to this conversation — changes save as you make them.", or "Defaults for new conversations." when none is open. Composed in one place (`settings.documentScopeNote`) so chat and notebook cannot drift apart on wording; explore, which binds no document, states its own different fact. Reaches the panel as a new optional `scope()` on the drawer contribution.
  - **"Reset to defaults" is now "Clear all overrides".** The behaviour was always right -- every value to null hands it back to the backend cascade -- but "defaults" read as something global while the button in fact rewrites the open document's params. No confirm: sampler values are trivially recoverable, and one there would train click-through past the confirms that protect work.

- **A preset carrying no system prompt is labelled where you choose it**, as "<name> — settings only". Such a preset is inert toward the prompt by design (the override-box rule), but it looked identical to every other entry in the dropdown, so applying one and watching the prompt not change is the state that reads as "my preset disappeared". The preview said so; the dropdown, which is where the choice happens, did not. Options carry `data-name` with the raw name, so the label is display-only and nothing that treats a name as an identity reads it back.

- **The drift line names which half drifted** -- "Prompt differs", "Settings differ", or "Prompt and settings differ", against one shared suffix. A single binary string covered both a nudged temperature and a rewritten prompt, so it read as alarming after a trivial edit and identical after a total rewrite. `matchesState()` now derives from the same `driftParts()` the line uses rather than keeping a second copy of the comparison, and a promptless preset scores no prompt drift by construction.

- `docs/frontend_v3_user_guide.md` describes the new labels and moves these three from "known rough edges" to a "Closed" ledger beneath it. The five remaining edges are unchanged.

## [1.79.24]

### Added

- **[docs/frontend_v3_user_guide.md](./docs/frontend_v3_user_guide.md)** -- how the UI behaves, written for the person using it. Covers the settings state model, presets end to end (what one holds, what Apply copies, what Save overwrites, what happens when you change a knob and do neither), conversation management and per-conversation models, the generation lifecycle, message editing including thinking blocks, and attachments. Linked from `docs/README.md`, `docs/frontend_v3.md` and `CLAUDE.md`.
  - It opens with the thing that explains most of the confusion: sampler settings exist in **three** places at once -- browser defaults, the conversation, the preset -- and the screen never says which one you are looking at. The system prompt has the same three. Everything else in the guide hangs off that.
  - It ends with a **known rough edges** section rather than an appendix of caveats: places where the interface does not say enough for the behaviour to be guessable. The sampler panel not naming whose settings it shows; browse and save-destination being the same control; the drift line not saying what drifted; nothing disclosing that a generation survives you leaving; Send/Stop being one button that can mean a run you did not start; "Save & Continue" appearing and vanishing with the provider; thinking depth offering values a given model rejects; a prompt-less preset looking healthy.

### Fixed

- **A stale comment in `chat.js` described walking away from a generation as lossy.** `abortStream` said the server "persists the partial" on disconnect. It has not worked that way since generation outlived its response: aborting the fetch ends the *subscription*, the run detaches, finishes, and commits the **whole** answer. Only an explicit Stop keeps a partial. The comment was believed while drafting the user guide and only caught by reading `conversation_generate_api._Run` -- so the correction says what the behaviour is and names the wrong version, since that is what a future reader will otherwise re-derive.

## [1.79.23]

### Removed

- **The "prove it red before you trust it green" rule, from the repo entirely.** Owner call, asked for twice (2026-08-17 and again 2026-08-28: "it seems to be creating havoc"), and this time with no carve-outs -- the parser-invariant exemption the first relaxation preserved is gone too. Deleted from `CLAUDE.md` (the markdown-stream boundary rule and the parser-invariant line), `tests/e2e/README.md` and `render.mjs` ("a check that has never been seen failing is decoration"), `plan_chat_orchestration.md` ("per the checks-and-assertions rule"), `jspace_integration_plan.md` and `TODO.md` (a planned gate that had "mutation-check it" written into the step).
  - Statements of what was actually done ("shown red against the pre-fix tree") are history and stay; only the standing instructions were removed. Two surviving mentions describe the `jlens-mlx` sibling repo's own practice, not this one's.
  - The first ask was about an undocumented habit. Between the two asks the habit was written INTO the repo as policy and then justified itself on every read, which is what turned a corrected reflex into a standing instruction. `CLAUDE.md`'s test section now says so, so it does not grow back a third time.
  - A deliberate red remains available as a tool -- `E2E_V3_ROOT` still points the render suite at another copy of the frontend -- for the case where you personally doubt a specific check can fail. It is not a step and not something to narrate.

## [1.79.22]

### Fixed

- **An armed confirm never re-checked what it was confirming, so the blanking guard added in v1.79.21 was bypassable in two clicks.** `armedConfirm` fired unconditionally on the second click: arm Save on a preset ("Overwrite prompt?"), clear the system-prompt box sitting below it in the same drawer, click again -- the preset was blanked with no confirmation and question (2) of the guard was never evaluated. No amount of `disarm()` wiring in the preset bar could have caught it: Save's payload is the *document's* prompt, edited in a different section the bar gets no events from.
  - `armedConfirm` now takes an optional `target()` returning destination + payload. It is captured when the button arms and re-read on the confirming click; if it moved, the click re-arms instead of firing. The check belongs in the primitive, not in each consumer's event wiring -- the same reasoning the drawer's hide-flush rule already uses.
  - `disarm()` stays, for a different job: `target` makes a stale arm *harmless*, `disarm` makes it *visibly gone*. A button still reading "Overwrite prompt?" while aimed elsewhere is a lie even once clicking it is safe.

- **The name box disarmed Apply, which it does not aim.** Typing a save-as name while Apply sat armed silently cancelled it, so the next Apply click only re-armed and the apply never happened. Each control now disarms only the buttons it re-aims: the select moves all three, the name box moves Save alone.

- **A hand-typed save-as name could earn the iterate-loop exemption against a preset nothing on screen was describing.** Selecting one preset and typing another's name aimed Save at the second while the preview and drift line described the first; if that name matched the document's stamp, the exemption applied and it overwrote in one click. The exemption now requires the loop in full -- the document runs the preset *and* the select is showing it.

- **Corrected a claim that had stopped being true.** "A stale list can only mis-arm, never mis-save" held for the pure difference test, but the exemption resolves a *name to an id*, so a list stale about which row owns a name can grant it against the wrong row. Documented at the function, in the file header, and no longer restated in `CLAUDE.md`, which now points at the function rather than carrying a second copy of the branch order.

- The exemption's known boundary is now written down rather than left implicit: a stamp is cleared only by delete, so a document whose prompt is later replaced wholesale still counts as running its preset. That is the trade the exemption exists to make; the note says what to do instead if it ever needs closing (clear the stamp, never add a drift percentage).

### Changed

- `tests/e2e/render.mjs`: `clickSave` counts **POST as well as PUT**, so a "zero requests" assertion can no longer be blind to a save that landed as a create. Preset controls resolve through one selector map instead of three ad-hoc spellings. The disarm check waits out the arm and asserts a known start state rather than inheriting an armed button from its predecessor, and the fixture comment claiming a preset was never written to -- falsified by the loop check -- was replaced with a fixture nothing writes to.
- Three more checks: the payload-change bypass, a save that changes no prompt at all (the previously uncovered branch, which made v1.79.21's "every branch has a check" claim false), and the hand-typed-name divergence. Each was shown red against a mutation of the exact line it guards. Suite is 87/87.

## [1.79.21]

### Fixed

- **The preset guard from v1.79.20 charged the iterate loop for the accident it was catching.** Apply a preset, edit the prompt, Save it back is the bar's whole purpose, and it armed every time. An arm met dozens of times a day is an arm nobody reads, which is what makes the one that matters worthless -- the failure mode the only-loss rule exists to prevent, reintroduced by the fix for it. `wouldOverwritePresetPrompt` now asks three questions in order: is there anything to lose (a new name, or a preset carrying no prompt, has nothing at stake); would this **blank** it (always arms -- clearing the box is not editing it, and a NULL write leaves an override-box preset present but inert); is the document already **running** this preset (then it is the iterate loop, and stays one click). Saving onto a preset the document is *not* running -- the shape that caused the loss -- still arms.

- **A pending confirm survived a change of target.** Arming Apply or Save on one preset and then picking another left the next click confirming an action never previewed: the same accident in two steps. `armedConfirm` grew a `disarm()`, and the preset bar calls it from both the select and the name box, since either re-aims what the buttons would hit. This was found by a test that failed for the right reason while checking something else.

- `preset-bar.js`'s header comment was rewritten to match: the select is inert toward the document but READS its stamp, both buttons are armed, re-aiming disarms, and the arm (unlike the save) is decided against the local list.

- Four more wire-level checks in `tests/e2e/render.mjs`, all shown red against deliberate mutants: neutering `disarm`, widening the running-preset exemption to any stamped document, dropping the blanking guard, and removing the exemption altogether. Each branch of the new logic has a check that fails when that branch alone is broken. Suite is 84/84.

## [1.79.20]

### Fixed

- **Preset Save could overwrite a stored preset's system prompt on one unconfirmed click.** The preset `<select>` pre-fills the save-as name box, so picking a preset to *look* at it left that preset armed as Save's target; Save then wrote the **document's** prompt over it with an `UPDATE` that keeps no history. Live loss on 2026-08-28: one preset's stored prompt is now byte-for-byte the open conversation's prompt.
  - The guard was on the wrong direction. `Apply` was armed ("Replace prompt?") -- it overwrites the *document*, which is recoverable by re-applying the preset. `Save` was a bare click listener and overwrites the *preset*, which is not recoverable. Per the only-loss-gates rule, Save is the one that gates: it now arms whenever a preset of that name exists and Save would not leave its prompt as it is -- replacing it with different text, or (the quieter loss) writing `NULL` over it from a document with no prompt, which leaves an override-box preset present but inert.
  - Enter in the name box routes through the button rather than calling `save()` directly; a second entry point past the arm is the same hole with a keyboard on it.
  - No arm when nothing is at stake (a new name, or a preset that carries no prompt) -- a confirm that fires with nothing to lose only trains click-through.

- **Every preset appeared to hold the same system prompt.** The drawer renders the preset section directly above the per-document prompt box, and that box shows the **document's** prompt regardless of the selection (the select is inert by design). Labelled only "System prompt", it read as the selected preset's, so clicking through the dropdown showed one unchanging block of text -- and a Save "to keep what I'm looking at" is exactly the click above.
  - The preset section now carries a collapsed, read-only view of the **selected preset's own** prompt (or says the preset carries none), so a preset can be inspected without applying it.
  - The document's box names its owner: "System prompt for this conversation" / "…this notebook", via a new optional `label` on `createPromptSection`'s adapter.

- **The preset select showed nothing selected on a conversation that was running a preset.** It initialised from an explicit pick only, never from the document's `applied_preset_id`, so the one control that could have answered "which preset is this?" said "Presets…" -- which is what sent the user clicking through the dropdown in the first place. It now follows the document's stamp until an explicit pick is made, and an explicit pick is remembered against the document it was made on, so switching documents falls back to the new document's stamp instead of carrying the previous document's selection.

- Nothing above changes the API. `tests/e2e/render.mjs` grows five checks that assert **on the wire** -- the `PUT` that must not be sent -- because the arm relabels the button, so a DOM assertion would pass whether or not the request was held back. All five were shown red against a pre-fix copy of the tree via `E2E_V3_ROOT`, and the first one fails there with the overwrite payload in the message. The stub `/v1` store grew a preset `PUT` handler and seedable presets/stamp for them.

## [1.79.19]

### Added

- **`scripts/migrate_conversations.py`** -- copy a conversation store into a fresh one on the current schema. `db.Store` recreates on version mismatch rather than migrating (deliberate policy), which means opening an older store with newer code drops `conversations`, `messages`, `media_blobs` and `notebooks` with no prompt and no backup. This is the escape hatch, run by hand; it is not migration code in the app.
  - Opens the source **read-only** and never writes to it; refuses an existing destination; prints only counts and column names, never stored content.
  - **Fails closed on renames** (exit 2). A renamed column is indistinguishable from a dropped one plus a defaulted new one, so a name-matching copy would silently discard the data. `--rename [TABLE.]OLD=NEW` carries it across; `--accept-drops` confirms the loss is intended; `--dry-run` just reports.
  - Derives the target schema from `db._SCHEMA_SQL` rather than restating it, so it cannot drift from the app.
  - The result contains the current schema and nothing else, stamped with the current version so the app opens it without dropping anything.
  - Verified end to end against a synthetic v6 store (a renamed column, a stale column, real rows): the rename carries its value, the stale column is gone, the source is left untouched, and `db.get_connection()` opens the result with the rows intact.

## [1.79.18]

### Removed

- **`content-visibility: auto` on `.message`.** It was in the original v3 scaffold as a generic "make long lists fast" pattern -- not a response to any measured problem -- and it never paid for itself. A skipped row reports its `contain-intrinsic-size` estimate instead of its real height until the engine lazily decides it is relevant, so `scrollHeight` lurches by thousands of pixels and **the engine moves `scrollTop` on its own** (clamping, then scroll anchoring). Every scroll decision in `chat.js` derives from those two values.
  - Measured on Chrome: a conversation opened **14,380px above its end** (800x600) / 23,659px (390x844) and stayed there; the streaming tail-follow failed in roughly three generations out of four.
  - v1.79.1 gated it behind `@supports (overflow-anchor: auto)` after it broke iOS, on the stated belief that "Chrome/Firefox compensate with scroll anchoring, so the list stays put". They do not compensate -- scroll anchoring is one of the things *moving* `scrollTop`. The gate was protecting the optimization in the one engine where it was also broken, just less visibly.
  - **What it bought, measured** (CDP `LayoutDuration`, initial render, median of 6): flat layout as the thread grows -- 18ms vs 120ms at 1000 messages, 17ms vs 45ms at 300. One-time, on open, desktop Chrome only (Safari never ran it). It is also the smallest of the three long-thread costs: the conversation fetch returns every message and `renderMessages` builds a node for every one. If long threads ever need work, pagination and windowing are where it goes.
  - The re-aim `requestAnimationFrame` in `scrollMessages` went with it -- it existed only so a row still reporting a 3rem estimate could be re-aimed once the browser laid it out.
  - Every comment asserting the feature as a live fact was corrected in the same commit (`chat.js`, `DESIGN.md`, `CLAUDE.md`, `frontend_v3.md`, `frontend_v3_spec.md`, `tests/e2e/README.md`, the render-suite header).

### Fixed

- **`followingTail`'s `pinnedTop` discriminator stays, and now works.** Removing `content-visibility` was the root cause but not the whole fix: a viewport resize shrinks `clientHeight` by a few hundred px in one step, opening a gap far past `STICK_SLACK_PX` while the reader has not moved at all. Growth and resize move `scrollHeight`/`clientHeight`; only the reader moves `scrollTop`. Previously the discriminator was defeated by the engine's own `scrollTop` writes -- which came from `content-visibility`, so with that gone it is reliable.

### Tests

- **The mid-stream-resize check was weak in a way that hid all of this.** `serveV3`'s stub default `drip.position = 2` makes `adoptSavedRows` truncate the mirror from 32 rows to 3 at stream end; the resulting collapse is a harness artifact that masks the real strand. Measured: the bug is caught ~17% of the time at position 2 and ~79% at a realistic append. The check now overrides `drip.position = 31` for its duration. Red against HEAD at 1355-1383px, green three runs after the fix.

## [1.79.17]

### Tests

- **`tests/eval/lifecycle.py`** -- generation-lifecycle checks that need a real model actually decoding. Sibling of `run.py` with the same cost profile (opt-in, points at an already-running server, never spawns one, not in `/test-suite`) and a different kind: `run.py` judges what the model says, this asserts the plumbing around it.
  - Per model: a generating model cannot be unloaded (409, naming it); Stop returns 200 and the run clears; **a stopped generation persists exactly what the wire delivered, byte for byte**; and at most one llama-server subprocess.
  - Takes an MLX model **and** a gguf one deliberately: the two fail differently when torn down mid-generation, and gguf is the one that had no protection.
  - It replaces mock-based unit tests that were deleted rather than kept, because they certified a guard that did not work: the guard reads a provider's in-flight count, MagicMock reports whatever it is told, and the suite was green while the guard returned False for every gguf model. Shown red by removing the gguf counter -- the unload goes 409 to 200, and Stop then 404s because nothing is left alive.

## [1.79.16]

### Fixed

- **v3 assets**: gzip output is memoised on the same `(mtime, size)` the etag derives from -- every cache-missing load was re-compressing ~20 assets at level 6 on the event loop, the same loop delivering SSE tokens. The gzip and identity representations now carry **distinct etags** (RFC 9110 requires one per content-coding; a shared validator lets a cache answer an identity request with a gzip body), and `Vary: accept-encoding` is set on all three exits rather than only the compressed one.
- **Attachments encode in parallel at send.** Eight staged photos ran eight full FileReader round trips in series at exactly the moment the user is waiting; `Promise.all` costs the slowest read instead of their sum.
- `optimizations/__all__` still named the deleted `fast_image`, so `from ... import *` would have raised.

### Tests

- The send-refusal assertion in the remote-generating check was **vacuous**: it set the textarea through `page.evaluate` (which never focuses it) and then pressed Enter, so the key reached `document.body` and the assertion held whether or not the guard existed. It now focuses the field and types for real, and additionally asserts the refusal is *loud*. Shown red against a mutation that removes the guard -- which makes v1.79.12's "every new guarantee was shown red" claim true rather than aspirational.

### Docs

- `docs/project/plan_2026-07.md` still listed `/v1/chat/completions/multipart` as ACTIVE for an endpoint deleted in v1.79.11. CLAUDE.md names that file as the first thing a new session reads, so a stale entry there misdirects the next agent.
- `docs/frontend_v3.md` was stamped v1.79.9 at v1.79.15 and missing `image-prep.js` / `document-writer.js`, plus the backend coupling changes it exists to record (`generating` added, `system_prompt`/`params` dropped from the list). Its performance figures moved to the CHANGELOG per CLAUDE.md's rule against numbers in tracked docs -- the superlinear-to-linear *shape* is the durable claim.

### Note

- The review finding that `delete_conversation` and `clone_conversation` lacked an active-run guard was **incorrect** -- both call `_refuse_while_generating` as their first statement. Verified before acting; no change made.

## [1.79.15]

### Fixed

- **The teardown guard now actually covers gguf.** v1.79.14's `_is_generating` read `generation_queue_stats()`, which `LlamaServerProvider` does not implement -- it inherits the base's `None`, deliberately, because llama-server queues its own requests and there is no MLX-style FIFO gate to report. So the guard returned False for every gguf model, always: it protected only the provider that already had a 30s wait, and did nothing for the one with no protection at all. The unit tests missed it because they used mocks that reported whatever they were told to.
  - `_active_generations` / `_active_lock` moved from `MLXProvider` to `BaseProvider`, with a `generation_active()` context manager and an `active_generations` property. The gguf provider wraps its yielding in it, so the count follows the generator's real lifetime (exhaustion, `close()`, or an abort that closes it). The router asks one question of any provider.
  - **Verified live, both providers, real models** (`Qwen3.5-0.8B-MLX-8bit` and `google_gemma-4-E4B-it-qat-q4_0-gguf`): an unload attempt during generation returns 409 naming the model; a Stop returns 200 and clears; exactly one llama-server process throughout. Shown RED by removing the gguf counter and repeating: the same unload returns **200**, SIGTERMing llama-server out from under an open stream.
  - Also verified live, and previously only assumed: **a stopped generation persists exactly what the wire delivered** -- byte-identical on both providers, so the abort lands on the token it stopped at and the reasoning parser's rolling holdback drops nothing. That is what makes a stopped reply editable afterwards.

- **The generation task no longer swallows failures.** `_pump` caught every exception and emitted only the end-of-stream sentinel, handing a still-attached client a clean 200 and a well-formed but truncated stream -- which the client's recovery path reads as "the transport died, the server is committing", announcing a FAILED generation as a recovered one. It now emits a typed `error` event first. This is the common path on gguf, where a killed llama-server surfaces as a connection error rather than a `GenerationFailed` the saga can shape.
- **Stop reports what actually happened.** `stopRemote` treated all three outcomes of `stopGenerate` as success; a network failure now says so and leaves the run marked generating, and the settle-check polls (bounded, backoff, only after an explicit Stop) until the server confirms the run ended, so the partial is not invisible until the next reselect.
- **Resume adopts `generating` from the conversation list**, not only from a body refetch. Reaching it solely through `adoptConversationMeta` -- which runs only when `updated_at` moved -- meant a run that generated nothing left the composer reading "Stop" and refusing every send until the user navigated away and back.

## [1.79.14]

### Fixed

- **A model with an active generation is no longer torn down.** `router._is_generating()` now guards all three teardown paths (LRU evict, admin unload, idle unload) the same way `_pinned` already did — a generating model is pinned, by definition, so no new concept was needed.
  - The guard belongs at the ROUTER because the hazard differs per provider and gguf had **less** protection than MLX, not more: `MLXProvider.unload` waits for `_active_generations` but force-unloads after 30s, freeing weights under a live Metal command buffer (its own docstring says that crashes); `LlamaServerProvider.unload` SIGTERMs the llama-server subprocess with **no wait for actives at all**, because `_active_generations` is MLX-local and appears nowhere in `router.py` or `base.py`.
  - The race is pre-existing (`get_provider` resolves, and can evict, before the generation gate is acquired), but v1.79.12 widened the window from "while a client is watching" to the full length of every abandoned run, which with `max_loaded_models=1` is the common case.
  - Refusal reuses the existing backpressure vocabulary rather than inventing one: `ModelBusyError` → the saga's existing `MODEL_BUSY` branch → 503 + Retry-After, which v3 already retries; admin unload/reload → the 409 the pinned path already returns, carrying a detail that names the generating model.
  - **Known cost:** the generation gate is a process-global singleton, so the predicate is conservative *across* models — with `max_loaded_models > 1` a generation on one model blocks evicting any other. No effect at the default of 1. Failing toward "do not tear down" is the safe direction for a teardown decision.
  - `_is_generating` type-checks `generation_queue_stats()` against its documented `Optional[Dict]` rather than truth-testing it: a bare Mock is truthy and its `.get()` returns another truthy Mock, which would report every mocked provider as permanently generating and un-unloadable.

## [1.79.13]

### Changed

- **One document write path instead of two** (`document-writer.js`, new). `putSystemPrompt` and `setAppliedPreset` were byte-identical in `chat.js` and `notebook.js` apart from which api function they called -- and notebook's copy said "same shape as chat's putSystemPrompt" in a comment, which is the tell. What the copies duplicated is not boilerplate: it is the keepalive ordering rule (a hide-time flush must be DISPATCHED rather than queued behind an in-flight PUT, because a request still queued when the page unloads is never sent), an invariant hand-maintained in two files and guarded in neither. The PUT chain now lives in the writer's closure rather than on each page's state. Each page keeps only what is genuinely its own: which api function, and the pre-create stamp guard.
- **`paintPresetChip` moved into `preset-bar.js`**, next to the `onIndicator` callback that drives it. Both pages had identical copies, both commented "the bar chip's ONE renderer".

### Tests

- Two render-suite checks now pin the ordering rule in both directions -- a keepalive write is dispatched ahead of an in-flight PUT, and ordinary writes stay serialised so an older value cannot land after a newer one. The first was shown red against a mutation that queues the keepalive write. Neither behaviour had any coverage while it lived in two copies.
- The mid-stream viewport check is condition-based rather than a single post-stream read: the final rate-limited paint and WebKit's own post-resize scroll restore both land near that boundary, which made it flaky. Verified over repeated runs.

## [1.79.12]

### Changed

- **A generation now outlives the HTTP response that started it.** Switching tabs or conversations used to KILL it: the client aborted its fetch, the server saw the disconnect, set the abort event and committed whatever partial had accumulated. The work is the expensive half and the server already owns persistence, so the response is now a **subscriber** to a run that finishes and commits either way -- walk away mid-generation and the whole answer is there when you come back.
  - `_Run` (`conversation_generate_api.py`) owns the abort event, an event queue and the task. `_pump` drives the saga to completion regardless of subscribers; `_subscribe` feeds one HTTP response. A subscriber leaving sets `detached`, after which the pump keeps generating and stops enqueueing, so an abandoned run cannot grow a queue nobody drains.
  - `async_generator_with_abort` gained `abort_on_disconnect`. It stays **true** for the stateless wires (`/v1/messages`, `/v1/chat/completions`): nothing is persisted there, so a client that leaves means the work has nowhere to go. It is **false** for the conversation saga alone. Keepalives are emitted either way.
  - `_ACTIVE` holds a `_Run` rather than a bare `AbortEvent`; every identity guard compares the run. The response's `BackgroundTask` no longer releases the claim unconditionally -- that now fires while the run may still be going, and popping the claim there would let a second POST start a rival generation into the same conversation. The task's own `finally` owns the release; the background task and the 60s watchdog are the belt for a run that never took its first step.
  - **No pinning was moved because this path never pinned.** `pin_model` is used only by `rlm.py` and `jspace_api.py`; what protects a running generation from an LRU evict is the provider's `_active_generations` counter, which `unload()` waits on and which a detached run keeps raised.
- **The runaway is visible and stoppable.** A run nobody is subscribed to would otherwise have no off switch from a client that navigated away. `GET /v1/conversations` and `GET /v1/conversations/{id}` now carry `generating` (in-process, read off the same dict the 409 check and Stop use). v3's composer has three states rather than two: with a remote run it reads **Stop** and the status line says the reply is finishing on the server; a send is refused loudly instead of racing it; Stop sends the same `DELETE` the local button does.

### Tests

- `TestRunOutlivesResponse` drives the pump and subscriber **directly**, and the reason is worth keeping: httpx's `ASGITransport` runs the app to completion before yielding the first line, so a test that opens a stream, breaks out and calls that a disconnect never disconnects anything -- the run has already finished. The first version of these tests did exactly that and passed while proving nothing. `TestDisconnectPolicy` pins `abort_on_disconnect` both ways so the stateless wires cannot silently inherit the new behaviour.
- Every new guarantee was shown red against a mutation of the branch it guards: the pump returning instead of continuing when detached, the disconnect probe ignoring the flag, and (in the render suite) adoption ignoring the `generating` field.

## [1.79.11]

### Removed

- **Unreachable surface** (each removal is a search that came back empty): `api_multipart.py` and `/v1/chat/completions/multipart` (no in-repo client; it also carried its own copy of the resize logic, so `utils_resize.py` is now the single implementation, shared by the OpenAI JSON path and batch-labeler); `optimizations/fast_image.py` (its `ImageCache` had one construction site, behind a `hasattr(app.state, 'image_cache')` nothing ever set); the empty `middleware/` package; `/v1/capabilities`' `fast_vision` block (advertised the deleted endpoint, plus "57ms faster per image" -- a performance number in a shipped surface), its `use_multipart` recommendation, its invented `limits` block (nothing enforces any of those), and `cache_size: 2` (contradicted the `max_loaded_models=1` default). The `ChatRequest` resize fields are NOT dead and stay.
- **v3 dead code**: the `onDisplayChange` listener mechanism (no subscriber ever, so `setDisplayPref`'s notify loop always ran over an empty set while its comment promised live re-render), `api.capabilities`, write-only `s.scanConfig`, `--z-toast`, `dataset.tok`, `document.body.dataset.page`.

### Fixed

- **`memory.py`'s `sampler_summary_from_request` now DERIVES from `REQUEST_SAMPLER_FIELDS`.** It had drifted while it was a hand-copy: it carried the retired `preset` (which `config.py` 422s, so the getattr was permanently `None`), lacked the current `sampler`, and never picked up `vision_tokens`. Its twin in `conversation_generate_api.py` was already fixed by derivation; this sibling was missed, which is exactly the failure mode CLAUDE.md names.
- **`test_request_schema_parity.py` exempted five fields citing `/v1/messages/multipart`** -- an endpoint that has never existed. The exemption is legitimate (server-side downscale knobs; Messages clients resize before sending) but a green test was certifying a reason that was fiction.

### Optimized

- **v3 assets: revalidate, but never resend.** `no-cache` is correct and stays -- v3 is no-build with unhashed URLs, so revalidation is the only thing that can invalidate a cached module. What it *cost* was the problem: starlette's `FileResponse` sets an etag but has no conditional branch (only `StaticFiles` does), so every revalidation was answered with the whole file. The handler now answers `If-None-Match` with a 304, and gzips text assets above 1KB. Measured: a cold load goes 427 KB -> 138 KB (68% off), and a revalidating load goes to zero body. The in-code note used to call the full re-send "free for a localhost frontend" -- true until the client is a phone, where iOS discards backgrounded tabs and reloads, making it a half-megabyte transfer per wake.
  - Compression is in the v3 handler deliberately, **not** `GZipMiddleware`: that wraps every response including the generate endpoint's SSE, where buffering to a minimum size would sit on the first token. A contract test pins that the middleware is absent.
  - `index.html` now preloads the modules chat.js pulls in (streaming, markdown, markdown-stream, image-prep, preset-bar, prompt-section, and both vendor modules) -- the two vendor files are the largest assets after chat.js and were three hops deep.

## [1.79.10]

### Optimized

- **Frontend-v3 attached images (upload size and heap)**: Measured through the real staging path before this change: eight 3MB camera-roll photos left the phone as a **single 32MB POST** (base64 is 1.33x, exactly), and nothing between the file picker and the request reduced anything.
  - **Resolution cap at staging (`image-prep.js`, new)**: `prepareImage()` decodes each staged image honouring EXIF orientation (phone cameras store a rotation flag; a canvas draw that ignores it silently rotates the photo) and reduces it to `MAX_EDGE_PX` (2048) on its longest edge. An image already within the cap is passed through **untouched** rather than re-encoded -- a lossy round-trip that saves nothing is worse than doing nothing -- as is one whose re-encode came out larger. PNG stays PNG (a re-encoded screenshot shows JPEG ringing on text); everything else, including HEIC/HEIF off an iPhone, becomes JPEG at q0.85. Every failure path returns the original file: a downscale bug must not be able to lose a picture.
  - **Base64 is minted at send, not at staging (`chat.js`)**: a staged attachment now holds a Blob plus an object URL for its thumbnail. Previously each one retained a base64 data URL for as long as it sat in the composer, `buildContentBlocks` took a second full copy via `.slice()`, and `JSON.stringify` made a third -- all three live simultaneously, at full camera-roll resolution. `buildContentBlocks` is now async and encodes from the (already capped) blob.
  - **Thumbnails read the capped blob** and carry `decoding="async"`, as do message images. `loading="lazy"` is deliberately NOT used: a lazily-loaded image has no height until it arrives and WebKit has no scroll anchoring to absorb the shift -- the same reason `content-visibility` is gated off for `.message`. Revisit only alongside stored image dimensions to reserve the box.
  - The cap is **disclosed, not confirmed** (owner rule: loss gates a switch, cost gets disclosed) -- one status line per staging batch, only when something was actually scaled.
  - `MAX_EDGE_PX` is a deliberate default, not a derived one. How much resolution a vision model can use is a model question: the Qwen-VL family in this repo's modelzoo declares dynamic resolution (a pixel budget of ~4096x4096), so it consumes what it is given and pays in vision tokens and prefill, while a fixed-input tower discards the surplus.

### Tests

- **`tests/e2e/render.mjs`**: `dropRealImage` stages a genuinely decodable JPEG of chosen dimensions -- the existing 4-byte placeholder cannot exercise a path that decodes what it is handed. New checks: an oversized image is capped before it reaches the wire (asserting both that the POST is smaller than the source and that the *pixels* shrank, not merely that the JPEG re-compressed), and an image already within the cap is passed through un-re-encoded. The first was shown red against the pre-fix tree.
- **Fixed a flaky check**: `a send during the attachment read does not lose it silently` polled the status line for a transient value. The status line is a sequence of announcements, not a state to sample -- the send that races the attachment overwrites it within milliseconds, so the check was passing on a window a few ms wide and went red when an unrelated change to the staging path moved it. It now records every value the line takes (`watchStatus`/`statusLog`) and asserts the announcement was *written*. The underlying behaviour was verified correct throughout.

## [1.79.9]

### Optimized

- **Frontend-v3 streaming render (the battery cost of a long generation)**:
  - **Incremental markdown (`markdown-stream.js`, new)**: A streaming message used to be re-parsed WHOLE through `marked` + DOMPurify and assigned to `innerHTML` on every animation frame, destroying and rebuilding the message's entire subtree each time. `marked`'s parse is superlinear in document length (measured on a non-repeating prose/list/code document: doubling the length costs ~3.2x), so the per-frame cost grew faster than the response did. `MarkdownStream` splits the accumulated text at the latest boundary no markdown construct can span, renders each new segment ONCE into a committed prefix whose DOM nodes are then left untouched, and re-renders only the tail. Per-paint cost is now bounded by the largest single block (one paragraph, table or fenced code block) rather than by the response. A link-reference or footnote definition disables splitting for that message (it can be referenced arbitrarily far below), falling back to whole-document rendering.
  - **Rate-limited painters (`utils.js`, `page.js`, `chat.js`, `notebook.js`)**: Added `throttleToInterval` and `ctx.throttleTime`. Chat and notebook painted once per animation frame -- up to 120/s on a ProMotion display, for text nobody reads at that rate. Both now paint at ~15/s, still frame-aligned. `throttleToFrame` remains correct for cheap work (explore's token strip, j-space's detail panel) and is unchanged.
  - **Scroll follow (`chat.js`)**: "Is the reader at the tail?" was measured inside every paint -- a read/write/read on scroll metrics plus a second `requestAnimationFrame` that read and wrote again, forcing a full-list layout per frame. It is now measured once per paint at the TOP of the painter, before anything mutates the DOM: there the reads are cache hits (layout is still clean from the previous paint) rather than a forced re-layout, and the gap describes content the reader has actually seen. Measuring after the write would flip it false the moment one paint appends more than the slack (a code block, a table) and strand the view for the rest of the generation. `scrollMessages` keeps its read-back and re-aim for STRUCTURAL changes, where a row added this tick is still at its `contain-intrinsic-size` estimate. This mattered most on iOS, where `content-visibility` is deliberately gated off (Safari has no scroll anchoring), so off-screen rows are never skipped. A cached flag fed by scroll events was implemented first and rejected: pinning coalesces scroll events to a handful across a whole generation, so the flag went stale whenever the viewport changed without a scroll -- a phone keyboard opening, every time -- and stranded the view above the tail.
  - **Thinking boxes (`chat.js`, `explore.js`)**: Both rewrote the whole accumulated thinking string on every delta. `appendPlainText` appends only the delta. Explore's was not throttled at all, so this was O(thinking so far) per token there.
  - **Measured** (real Chrome, real modules, one 4000-token response): render work fell from ~4.3s to ~80ms, and its growth with response length went from superlinear to near-linear.

### Tests

- **`tests/e2e/render.mjs`**: Added a drip-fed `/generate` SSE route to the harness's own static server -- puppeteer request interception can only answer in one shot, so no existing stub could show a painter repainting over time. New checks: a growing message re-renders the tail rather than being rebuilt (the largest single node removal separates the two, independent of timing); the repaint rate is bounded by elapsed time rather than by delta count; the streamed render matches a whole-document render; and a property check that grows generated documents one chunk at a time through `MarkdownStream` and diffs against a whole-document render, covering loose lists, fences with blank lines inside, unclosed and nested fences, indented code across blank lines, setext headings, tables, quotes and reference/footnote definitions. The two cost checks were shown red against the pre-fix tree; the property check was shown red against three deliberate mutations of the boundary rules (list-marker guard removed, fence tracking disabled, reference-definition escape hatch removed).

## [1.79.8]

### Added

- **Conversation Cloning & Mobile Safeguards**:
  - **Storage layer (`db.py`)**: Added `clone_conversation()` to atomically duplicate conversations in DuckDB. Creates a fresh conversation record with new unique ID and timestamps, clones `model_id`, `system_prompt`, `params`, and `applied_preset_id`, generates fresh message IDs for all turns while preserving position/role/thinking/model attribution, duplicates content-addressed `media_blobs`, and rewrites internal media URL sources in `content_blocks` to keep media lifecycles isolated.
  - **REST API (`conversation_api.py`)**: Added `POST /v1/conversations/{conv_id}/clone` (201 Created) endpoint with optional title override support and generation-lock protection (409 Conflict while streaming).
  - **Frontend API (`api.js`)**: Added `cloneConversation(id, body)` route mapping.
  - **Frontend-v3 UI & Mobile Touch Safeguard (`chat.js`, `app.css`)**: Added a two-tap armed confirmation (`armedConfirm`) "Copy" action to conversation list items. On mobile/touch screens (`hover: none`), single taps or accidental scroll swipes only arm the button into "Copy?" for 3 seconds before safely resetting, preventing unintended duplicates while scrolling. Styled `.conv-item__clone.btn--armed` with the brand accent color (`--accent`) for visual clarity. Automatically opens and hydrates the cloned conversation on creation.
  - **Unit & Endpoint Tests (`test_conversation_api.py`)**: Added comprehensive tests for default/custom titles, parameter/message preservation, media blob cloning, isolation after parent deletion, and REST endpoint contracts.

## [1.79.7]

### Optimized

- **Frontend-v3 rendering & memory performance**:
  - **Token Explorer (`explore.js`)**: Eliminated $O(N^2)$ chip recreations during logprob streaming via incremental `DocumentFragment` appending and in-place `.tok--selected` class toggling on keyboard/click selection, reducing DOM operations per stream by over 98%.
  - **J-Space stability & matrix marking (`jspace.js`)**: Replaced array spreading in `Math.min(...flat)` / `Math.max(...flat)` with a single-pass iterative loop, preventing call stack overflow exceptions on large matrices ($>65\text{k}$ elements). Optimized matrix cell lookup in hover and marking loops.
  - **Fast HTML escaping (`markdown.js`)**: Replaced DOM-based HTML escaping (`document.createElement('div')`) with zero-allocation compiled regex replacement.
  - **Image & audio preview memory lifecycle (`chat.js`)**: Staged thumbnails now use `URL.createObjectURL(file)` rather than multi-megabyte base64 strings in the thumbnail DOM, with deterministic URL revocation on remove, composer clear, send, and page teardown.
  - **Layout thrashing decoupling & modern CSS (`app.css`, `utils.js`)**: Added native `@supports (field-sizing: content)` to composer textareas across chat, explore, jspace, and notebook; `autoGrow()` bypasses forced layout measurements when native sizing is supported.
  - **CSS Design Tokens (`app.css`)**: Replaced undefined CSS variables (`--border`, `--font-mono`, `--space-*`) with canonical design system tokens.
  - **Module preloading (`index.html`)**: Added `<link rel="modulepreload">` for core shared modules.
  - **Notebook resume sync (`notebook.js`)**: Ported `refreshAfterResume` / `ctx.onResume` store re-adoption to the Notebook page, syncing edits made in background/mobile tabs upon resume.
  - **Media block preservation on `retrySave` (`chat.js`)**: Ensured unsaved rows with image/audio `content_blocks` re-POST all media attachments instead of only flattened text.
  - **Model-switch lock during load (`chat.js`)**: Disabled the model select dropdown while `loadModelNow` is actively loading a model into memory, eliminating switch race conditions.
  - **Preset inheritance on first send (`chat.js`)**: Unified conversation creation to inherit active preset stamps (`applied_preset_id`) and prompts on first send without requiring explicit "+ New" clicks.

## [1.79.6]

### Added

- **Request-schema parity guard** (`tests/unit/test_request_schema_parity.py`).
  `ChatRequest` (the OpenAI wire, and the model providers are driven with) and
  `MessageCreateRequest` (the Messages wire) share 19 fields, so every knob is a
  decision made twice and getting it wrong is silent -- the field just does not
  exist on the other surface. The guard derives the sampler roster from
  `REQUEST_SAMPLER_FIELDS` (never a hand-list), requires every asymmetry between
  the two schemas to be DECLARED with a reason, catches a declaration that has
  gone stale, and pins that shared fields agree on type and default (a knob with
  different bounds on each wire is a value one API accepts and the other 422s --
  the reason the `reasoning_effort` Literal is shared). Each assertion was seen
  failing against a deliberate mutation.


### Added

- **"Show special tokens" is wired** (v3 settings drawer, DESIGN.md §6's
  toggle -- it has been in the store with `wired: false` since the pref was
  designed). The server strips a model's DECLARED specials (`special: true` in
  its tokenizer files) out of the text it returns, as a guard against
  fast-detokenizer leaks; that guard also deletes a special the model wrote
  deliberately, and those say where in the turn the model is.

  New opt-IN request field `show_special_tokens` on `POST /v1/messages` and
  `POST /v1/conversations/{id}/generate`: when set, the declared-specials
  filter is not composed and those specials arrive as the model emitted them.
  Scoped to that filter -- a trailing PARTIAL structural token is still
  trimmed at the final drain (`_strip_partial_token`, a deliberate guess about
  text the model never finished), and routing still consumes its own
  markers. Absent
  or false is the previous behavior, so `/v1/chat/completions` and every
  existing consumer are unchanged. It never reaches the model and is never
  merged into the sampler bag.

  v3's chat and notebook send it from the global display pref, which defaults
  to ON (honesty-first, per §6). It is a GENERATION-time switch on these
  surfaces, and the drawer's help text says so: a reply is persisted exactly
  as it was parsed, so the toggle decides what the next reply records rather
  than how an existing one renders. Explore and jspace are the token-ARRAY
  half of the same pref and still ignore it.

  Routing is unaffected: a parser still consumes its own structural tokens
  (`<think>`, harmony/gemma channel markers) -- the switch governs only the
  declared-specials filter over the text those parsers route.

  Kept markers never re-enter a prompt: the conversation surface replays stored
  rows as the next turn's messages, so the server strips declared specials out
  of replayed ASSISTANT text before the request reaches the provider -- without
  that, turn 2 of every chat fed the model its own control-token string (a fast
  tokenizer encodes it as the real token), and `continue` would prefill ending
  on a turn boundary. User-authored text is untouched. Notebook differs by
  design: the reply becomes visible, editable document text, and that document
  is what gets sent.

  The drawer's Display panel is now offered only on pages that declare they
  honor a pref (`displayPrefs` in the drawer contribution) -- explore and jspace
  read token ids, not this, and a checkbox that silently does nothing on the
  page you are looking at is the thing the pref's `wired` gate exists to
  prevent.

  MLX-only in practice: the strip lives in heylook's parser and the gguf
  provider routes to a pass-through, so a gguf reply is never stripped and the
  toggle is a no-op there. Verified live on gemma-4-E4B in both packagings --
  with the pref off, MLX returned `STARTEND` where the GGUF returned
  `START<mask>END`.

## [1.79.5]

### Fixed

- v3 chat silently deleted tag-shaped text from a model's reply. marked passes
  raw HTML through and DOMPurify then removes any tag outside its allowlist
  while KEEPING the tag's content, so a reply containing `<d>tag</d>` rendered
  as "tag" with the markers gone -- while Copy, which reads the stored text,
  still showed them. Same root cause mangled ordinary prose: `a <b and c> d`
  parsed as an inline `<b>` and re-emitted as markup, and an HTML comment
  removed its content outright.

  Raw HTML in model text is now escaped at the renderer, so the rendered
  message matches the stored text for EVERY tag rather than only the ones
  DOMPurify happens to drop -- fixing the class, not the instance. Fenced and
  inline code (already escaped), autolinks, and markdown structure are
  unchanged; DOMPurify stays as the backstop for what marked's other renderers
  emit. Nothing else in the pipeline touches tag-shaped text: the backend
  removes only tokens a model's own tokenizer files mark `special: true`.

  Guarded by four checks in the model-free render suite (`bun run e2e:render`),
  each seen failing against a pre-fix copy of the frontend.

## [1.79.4]

### Fixed

- Code-review follow-through on the resume/hide work (all introduced in
  1.79.2-1.79.3, none shipped past this repo):
  - notebook's prompt flush-on-hide disabled itself after the first hide
    with the drawer closed (the shared section treated "detached" as
    "replaced"); the hook now lives as long as the section and chat, which
    builds one per drawer render, explicitly `release()`s the one it replaces.
  - the resume sync committed the new `updated_at` before adoption succeeded,
    so one failed body fetch (or one deferred adoption) became a permanent
    "unchanged"; the stamp is now committed only once everything it covers
    is adopted, and a render check pins the retry.
  - the resume focus guard skipped ALL meta adoption for any focused drawer
    field; it now skips only the prompt, and only while the prompt box itself
    is being typed in.
  - hide-time flushes send with `keepalive` and are dispatched ahead of the
    PUT chain, so they survive an unload instead of waiting behind an
    in-flight write that never completes; `bindDocumentParams` owns its hide
    flush (and its teardown now flushes rather than cancels), and notebook's
    content autosave flushes on hide too.
  - a resume no longer swaps the sidebar under an in-progress rename, no
    longer duplicates the user row when it lands mid-send, and keeps an open
    editor's draft (with a status line) when its row was deleted elsewhere.
  - `selectConversation`'s failure branch resets the sampler panel too
    (it kept the previous conversation's params under the failed id).
- Backend test now asserts a message write strictly bumps the conversation's
  `updated_at` (spec §4 states the guarantee the resume sync relies on).

## [1.79.3]

### Changed

- v3 page lifecycle grew `ctx.onHide(fn)` / `ctx.onResume(fn)` (`page.js`),
  each registering both spellings of its edge (`visibilitychange` +
  `pagehide` / `pageshow.persisted`) so a consumer cannot wire one and
  forget the other; chat's resume sync and the prompt editor's flush ride
  them, and the sampler-params binder now flushes on hide too. The resume
  sync fetches the conversation body only when the list's `updated_at`
  for the active conversation moved (every message write touches it), and
  repaints the sidebar/drawer only on an actual change. Its "is the user
  typing in the drawer" test is the drawer's own exported predicate
  (`isEditingInDrawer`), not a re-derivation with different semantics.
  `render.mjs` gained a check that an unchanged resume never downloads the
  body.

## [1.79.2]

### Fixed

- v3 chat re-adopts the store when the tab comes back (`visibilitychange`
  to visible, bfcache `pageshow`): conversation list, preset list, and the
  active conversation's prompt, stamp, params and rows. The page mirrors
  the store with no other invalidation (nothing polls; the select guard
  skips re-fetching the active conversation), and iOS Safari resumes a
  backgrounded tab with the heap it had -- so every whole-value write
  from that tab (prompt keystroke, sampler PUT, preset Save) re-played
  hours-old state over whatever happened since. A live stream's rows and
  a prompt being typed in the drawer are left alone (the textarea's own
  debounce writes it). The shared system-prompt editor also flushes its
  pending debounced write when the page is hidden or unloading, the
  phone's equivalent of the drawer-close window it already covered.
  Guarded by three `e2e:render` checks, each shown red against the
  previous tree (and the focus guard against a mutated one).

## [1.79.1]

### Fixed

- v3 chat on iOS Safari: opening a conversation landed far above its end,
  and a message edited on the phone slid under the composer after Save (or
  Cancel) until something nudged the scroll -- the long-standing
  "message disappears while I edit it" report. Cause, measured on an
  iOS 26.5 simulator with a control run: `.message` rows used
  `content-visibility: auto`, whose skipped rows report a 3rem estimate
  until WebKit lazily decides they are relevant, and WebKit has no scroll
  anchoring to absorb the re-layout shift (Chrome does, which is why the
  desktop never showed it). The optimization is now gated on
  `@supports (overflow-anchor: auto)` -- the precondition itself, so
  Chrome/Firefox keep it byte-for-byte and WebKit gets plain rows -- and
  closing a message editor re-aims the scroll at its row (`block: 'nearest'`,
  a no-op when the row is already visible) to absorb the keyboard-dismissal
  drift that remains even without content-visibility.

## [1.79.0]

### Changed

- The server's default port is now **8000** (`--port` overrides, as ever).
  Anything that relied on the old default -- shell aliases, service installs,
  client base URLs -- needs either the new port or an explicit `--port`.
  Reinstall the background service (`heylookllm service install`) to pick up
  the new default there.
- README rewritten: shorter, with a "Things to know" section for
  heylook-specific behavior (override-only `models.toml`, single resident
  model, GGUF binary/template/`max_tokens` gotchas, opt-in auth and
  telemetry). Stale claims fixed along the way (resident-model default,
  llama-server binary provenance).

### Security

- `HEYLOOK_ADMIN_TOKEN` now also gates `/v1/admin/config` -- it was the one
  `/v1/admin/*` router without the dependency, so a set token left runtime
  settings (observability level, MLX cache cap) writable by any client that
  could reach the port. Unset token remains a no-op, as everywhere else.

### Fixed

- Raising `observability_level` mid-run via `PUT /v1/admin/config` (or a
  reset that lands on a non-off default) now emits the one-shot startup
  record, so streams enabled at runtime still carry the hardware/config
  header. Previously only a boot-time level produced it.

## [1.78.2]

### Fixed

Two test-infrastructure defects from the ordering review (all 100 test
files were run in isolation as a census; every directory-level invocation
shape is green):

- **test_router_pinning's import-scoped sys.modules mock was an active
  eviction bug.** patch.dict restores by clear+update, so modules
  FIRST-imported inside its window (the whole provider chain, plus real
  numpy) were evicted from sys.modules on exit. Whenever that file led a
  per-file invocation (IDE run-this-file, changed-files CI, sharding),
  later re-imports failed -- 11 failures across the router files, and a
  reproducible hard SEGFAULT paired with test_generation_core. Full runs
  were healed purely by collection luck. The router now imports REAL,
  with importorskip on non-Apple; those tests never touched MLX at
  runtime anyway.
- **The mock tree's TokenizerWrapper was a MagicMock INSTANCE**, so
  `isinstance(tok, TokenizerWrapper)` raised TypeError -- the reason
  test_generation_core.py was the single file (of 100) failing alone. It
  is a real class now.

The review's other verdicts: the unit `mock_mlx` fixture stays
unconditional on purpose (its oracles assert against the mocks); ordering
plugins are the wrong tool (both hazards were collection-time, and the
real exposure is per-file invocation). Full report:
test_ordering_review (internal, local-only).

## [1.78.1]

### Fixed

Ten code-review findings against v1.78.0, two of them real shipped bugs:

- **The mRoPE gate's first cut went dark on telemetry.** Nulling the cache
  model id skipped working-cache registration entirely, so gated models
  vanished from context-usage metrics, /v1/cache info, clear_cache, and --
  worse -- the byte-budget/memory-pressure reclamation paths (a gemma slot
  could stay wired forever while a gated model re-prefilled under
  pressure). Eligibility now lives INSIDE process_prompt_with_cache (path
  enforcement: every present and future caller inherits the config gate,
  the mRoPE gate, and the spec-decode allow_reuse together), and the
  working cache always registers. Verified live: the gated model appears
  in /v1/cache/list with real token counts while staying re-prefill-only.
- **_reset_vlm_positions was silently poisoning warm mRoPE models --
  pre-existing, empirically confirmed.** Its object.__setattr__ bypassed
  mlx Module.__setattr__, creating a permanent instance-dict None shadow
  over the module dict: after the second request, the model could never
  see its own _rope_deltas/_position_ids writes again, so every warm
  generation recomputed positions from the current input. Fixed
  (delattr-the-shadow + plain setattr), pinned by a two-cycle regression
  test on a real nn.Module. NB the chained-restore empty-output bug
  SURVIVED this fix (re-verified live with the gate bypassed), so the
  mRoPE gate is its own necessity, not a symptom patch.

Also from the review: the unwrap walk + mRoPE attribute names are now ONE
shared spelling in model_wrappers (the reset and the gate inspect the same
object by construction, per the no-hand-copied-lists rule); the reuse
verdict log latch lives on the cache manager and clears with
invalidate/clear, so a model reloaded under the same alias re-announces
its (possibly different) verdict; the draft-model reuse-disable is logged
instead of silent; vision requests on a draft-configured model WARN and
drop the draft rather than silently discarding the vision prefill KV; the
misleading "qwen3_5 restores correctly" comment is gone (it is gated, as
the CHANGELOG said); and run_generation-level wiring pins make the gate
undeletable-while-green. Remaining review item tracked in TODO: the gate
fails open on renamed upstream attrs and there is no per-model
cache-reuse config escape hatch yet.

## [1.78.0]

### Fixed

**mRoPE models are excluded from prompt-cache reuse (silent-empty-output
class closed).** Chained cache restores on Qwen3-VL-style models produced
deterministic EMPTY responses (greedy: fresh multi-turn = "Rome",
cache-reused = "") -- their language models keep _position_ids/_rope_deltas
as INSTANCE state maintained across decode steps, and KV written under that
bookkeeping is not reconstructible from a restored cache. Verified latent,
not new: identical on mlx-vlm 0.6.13 and 0.6.15, and the radix era shared
the exposure. The gate is an attribute check on the unwrapped language
model (the same walk _reset_vlm_positions does), logged once per model; a
behavioral logits-equivalence probe was built first and REJECTED with
evidence -- single-hop synthetic restores pass on the very model whose
chained server restores fail, so only the real generate->store->restore
chain discriminates. gemma-4 (incl. sliding-window layers) keeps its
verified-working reuse; qwen3_5 hybrids are gated too, consistent with
their documented mRoPE state (their effective reuse was already nil
through trim refusals).

**Latent spec-decode cache landmine defused.** mlx-lm's speculative path
slices a provided prompt_cache into target+draft halves; heylook always
passed a target-only list, which would hand any future MLX drafter an
empty cache (no MLX entry ships one today -- all three drafters are gguf).
With a draft model, mlx-lm now builds its own paired caches and
cross-request reuse is skipped.

### Changed

mlx 0.32.0 -> 0.32.1 and mlx-vlm 0.6.13 -> 0.6.15 (floors follow the
resolved-and-tested rule; mlx-lm 0.31.3 already IS the latest published
release). The mlx 0.32.1 bump surfaced one real strictness change --
overwriting a safetensors file whose lazily-loaded (mmap-backed) arrays
are the data source now fails at save; the one test doing that
materializes first (the repo's documented mx.load laziness gotcha, now
enforced by the library).

## [1.77.1]

### Fixed

The backend suite no longer depends on invocation order. `tests/contract/`
applied a SESSION-scoped `sys.modules` MLX mock, and a session fixture tears
down only when the whole pytest run ends -- so `pytest tests/contract/
tests/unit/` ran every unit test against MagicMock arrays: 57 failures and 8
collection errors, all of which passed in isolation. The mock exists only so
imports succeed where MLX is absent (contract tests drive a FakeProvider and
never touch a generation path), so it is now conditional -- skipped entirely
when real MLX imports. Scope was the wrong lever: a heylook module first
imported under the patch binds MagicMocks into its own namespace for the life
of the process, which no fixture teardown undoes.

Two consequences of that mock shadowing a working MLX install go away with it.
`tests/contract/test_capabilities.py` run alone errored on "No module named
'mlx_lm.tokenizer_utils'" -- the mock tree never listed that path, which
`generation_core` imports; the entry is added for the non-Apple path, along
with a note that only MODULE-LEVEL import paths belong there (mocking
`mlx_vlm.generate.diffusion`, which product code probes optionally, makes
`_detect_diffusion`'s absent-dependency branch untestable). And contract-only
runs no longer abort at interpreter teardown with `gilstate_tss_set: failed to
set current tstate`, which reproduces in a bare interpreter as `import
heylook_llm.api` under the mock tree and not with real MLX.

Separately, `test_sanitize_handles_rotary_and_vision_keys` saved a
safetensors file back over the path it had just `mx.load`ed without
materializing the lazy mmap-backed arrays first. Latent until mlx 0.32.1 began
raising "[read] Unable to read from file" on it -- which then poisoned an
unrelated `test_jspace.py` case in the same process.

## [1.74.1]

### Fixed

`/v1/models` no longer costs ~1.65s per call on a 29-model registry. The
`reasoning_effort` template probe (v1.71.0) shipped uncached while its
thinking-probe sibling was lru_cached, so every call re-read and re-parsed
every MLX model's template files -- delaying every v3 page's first
paint-to-usable window (this is what the e2e suites' early checks were
racing), and taxing every generation start with the single-model slice via
`effective_capabilities`. One decorator: 1650ms -> ~1ms measured live.
Cache-property tests now pin BOTH probes behaviorally (repeated calls hit
the filesystem once; shown red against the uncached version), so a third
probe added beside them without the cache gets caught.

## [1.77.0]

### Removed

**The v2 frontend is deleted (cutover complete -- owner call).**
`apps/heylook-frontend-v2/` and its `/v2` mount in api.py are gone; v3 at
`/v3` is the only frontend, as it has been in practice. A contract test now
pins `/v2` as 404 -- the deletion is the contract. Both retired frontends
live on in git history. Rides along: the root `/` info endpoint reports the
real `__version__` instead of a hardcoded "1.0.1" that had been stale since
v1.0.

## [1.76.0]

### Changed

**Dependency hygiene: xxhash and PyTurboJPEG removed.** Both were imported
only to flip a status flag -- no hash call and no decode call existed
anywhere in the repo, and PyTurboJPEG additionally demanded a system
libjpeg-turbo it never used. If multipart JPEG decode is ever really wired
to TurboJPEG, the dependency returns together with its call site, never
ahead of it. Consequences on the wire: `/v1/capabilities`
`optimizations.image` no longer reports `xxhash_available` /
`turbojpeg_available` / speedup strings (no consumer read them --
frontend, tests and spec all swept), and
`recommendations.vision_models.use_multipart` is now unconditionally true
(the multipart win is bandwidth, which holds regardless of what is
installed). `build` and `twine` left the dev group (`uv build` /
`uv publish` cover both). Dependency floors refreshed to the versions
this repo actually resolves and tests against, with the torch requirement
documented in place: it is NOT derivable from dependency metadata
(`import mlx_vlm` reaches a bare `import torch` through transformers'
generation modules), so it must never be "cleaned up" on `uv tree`
evidence.

(This work was found uncommitted from an interrupted session; audited and
landed: zero surviving consumers of anything removed, `uv lock --check`
clean, removed packages absent from the lock, and the entire day's suite
runs and live servers already ran on this dependency state.)

## [1.75.0]

### Changed

**The radix prompt cache is deleted (plan Q7, decided 2026-07-06, executed
now).** Its replacement is a per-model SINGLE SLOT holding immutable
per-layer (state, meta_state) snapshots of the last completed generation,
registered with the token count the cache actually covers (read off the KV
offset -- the final sampled token never passes through the model). Reuse
has exactly two shapes: EXTENSION (multi-turn: reconstruct fresh cache
objects from the snapshots and continue -- valid for every cache type) and
TRIM to the common prefix via mlx-lm's own trim_prompt_cache (per-layer
honest: hybrid ArraysCache recurrent state and rotated sliding windows
REFUSE and re-prefill instead of being sliced). Deleted with the tree:
segment eviction, the byte-budget partial trims (slots evict whole, LRU),
`_find_system_boundary` and its extra per-request tokenization.

Two failure modes were caught LIVE during the rewrite and shaped the final
design -- both are now pinned by tests:

- The radix's restore-time `keys[..., :N, :]` slicing was documented as
  "technically incorrect but slight" for hybrid models; under exact prefix
  matching it produced constant greedy garbage on Qwen3.5-0.8B
  (fresh="Paris.", restored="\n\n"). Hybrids now refuse partial restores
  outright -- correct, not "limited correctness".
- An intermediate design stored LIVE cache objects; MLX arrays are
  immutable but cache objects are not, and this server quarantines wedged
  generator threads ALIVE -- a zombie generation kept mutating the shared
  objects and poisoned the process (garbage on cache MISSES until
  restart). Slots therefore store snapshots only; reuse reconstructs.
- The snapshot store must mx.eval the EXACT capture it registers: each
  .state access builds fresh lazy slice objects, so eval-one-capture/
  store-another shipped lazy state that crashed the NEXT request's thread
  ("There is no Stream(gpu, N)" -- the radix_thread_affinity postmortem
  class, re-caught by the live chat e2e). A two-thread unit test now pins
  the store path directly (shown red against the buggy split).

Verification: suite green; greedy live probes (the instrument that caught
both failed designs) green on the hybrid 0.8B and the MoE incl. repeat,
multi-turn and deep-trim shapes; eval bank diffed per-task against a
stashed radix baseline (24/26 outcome-identical, two known-flaky tasks
moved in opposite directions); live chat e2e. NB afternoon bank re-runs
on THIS machine hit an intermittent Metal command-buffer fault cascade --
reproduced on the stashed BASELINE code, so it is a pre-existing
machine/driver-state class (tracked in TODO), not a property of this
change. Accepted Q7 trade: switching conversations re-prefills.

## [1.74.0]

### Changed

**Phase 3b consumer migration: no v3 page speaks /v1/chat/completions
anymore.** Notebook and explore now stream `POST /v1/messages` (chat has
used its conversation-scoped sibling since v1.66). The client grammar
parser is ONE shared core in streaming.js (`streamTypedSSE`) under both
`streamMessages` and `streamGenerate`; `streamChat` is deleted from v3
(the backend endpoint stays for external consumers). The sampler bag's
Messages spelling is DERIVED (`messagesParams()`: `enable_thinking` ->
`thinking`), never a second hand-written copy.

**/v1/messages extension namespace** (what the migration rode in on):

- Streaming logprobs: `event: heylook_logprobs`, one per token when
  `logprobs: true`, entries in the SAME shape as the OpenAI wire's
  `logprobs.content` ({token, logprob, top_logprobs}) so a migrating
  consumer keeps its parser. Non-streaming responses now really include
  the logprobs content block the endpoint docstring always promised.
- Timing/KV telemetry rides `message_stop.performance`: `peak_memory_gb`,
  `kv_cache_bytes`, `queue_wait_ms`, `draft_acceptance` (the
  heylook_saved.timing vocabulary; absent telemetry is omitted, never
  null).
- `sampler` (named SamplerRegistry bundle) and `vision_tokens` accepted
  with ChatRequest semantics.
- `max_tokens` is now OPTIONAL (deliberately unlike Anthropic's): absent =
  the server-side sampler cascade's default. The old hard 1024 schema
  default silently overrode the cascade for every client that omitted the
  field -- exactly the knob-loss the migration guards against.

### Added

Per-message model attribution in v3 chat: a muted label under assistant
rows naming the producing model, rendered only while the thread MIXES
models (a single-model thread would just restate the header). Uses the
v1.73.0 `model_id` column.

Live-suite coverage: the long-open disconnect-persistence gap is closed --
the chat suite reloads mid-stream (no Stop) and asserts the server's
detached task persisted the partial. The suite's early one-shot checks now
poll (they raced the ~1.7s first `/v1/models` on a 29-model registry and
took six checks down as "No models available"), the delete check pins
single-row semantics, and the models-page failure-restore retries a click
the puppeteer interception-teardown race can silently drop. Full live run
green: chat 46/46, pages 43/43.

### Fixed

Review findings on v1.73.0: one shared media-type->capability mapping for
the generate saga's prefetch filter and wire build (a hand-copied second
spelling would 500 with a false "store corruption" when they drift); the
media endpoint serves `Cache-Control: private` + `X-Content-Type-Options:
nosniff`, and the write boundary constrains a blob's media_type to the
block's own family (an image block claiming text/html stored as
octet-stream -- stored-XSS shape closed); chat's scroll re-aim
distinguishes "reader scrolled away" from "row grew underneath" by
scrollTop movement instead of a distance threshold, so a row growing
more than 100px past its content-visibility estimate no longer strands
the view above the response tail.

## [1.73.0]

### Changed

**Media by reference (schema v7).** Base64 media no longer persists inside
message rows: every message write relocates it into a per-conversation blob
store (content-addressed on the bytes, so a re-pasted image stores once)
and the stored block carries a url source pointing at the new
`GET /v1/conversations/{id}/media/{media_id}` endpoint (immutable,
browser-cacheable). This is a relocation, not new storage -- the same
user-attached bytes the messages table already held, moved so that a
conversation read is text-sized instead of shipping every image inline on
every select, resync, and edit response. The generate saga resolves blob
references back to inline bytes at wire-build time (providers never see
the internal URLs; a missing referenced blob is a loud 500 before any
stream starts, never a silent drop). Blobs are garbage-collected when the
last referencing message goes -- the reference check's only false-positive
direction is retention, never deletion -- and deleting a conversation (or
Clear all data) deletes its blobs. Clients keep SENDING base64 exactly as
before; only the stored/returned shape changed. Schema v7 drops and
recreates the store per the no-migration policy (presets and settings
survive as always).

**Delete means delete.** The per-message Delete button deleted the clicked
message AND everything after it -- the only server spelling was positional
truncation, and the armed confirm never said so. New
`DELETE /v1/conversations/{id}/messages/{msg_id}` removes exactly one row
(later rows keep their positions; gaps are fine, nothing assumes density),
and the v3 button now calls it. The `?after` truncation endpoint stays for
API users; the client no longer calls it anywhere -- regenerate/continue
truncation has been server-owned since v1.65.

### Added

Message rows carry `model_id` -- which model produced an assistant row,
stamped by the generate saga's fresh-row commits (null on user turns; a
continuation keeps the anchor's original stamp rather than misattributing
a co-written row). Metadata only, no content; rides the same schema bump.
No UI yet -- the column is the deferred G5 attribution substrate.

Tests: blob lifecycle (dedup, GC direction, cross-conversation media_id
stripping, round-trip honesty), single-delete neighbor survival (unit +
render suite, the render check shown red against the truncation-era
client), wire inlining for blob-backed media, and cap-dropped media
counted-not-fetched.

## [1.72.3]

### Fixed

`scripts/build_llama.py` can no longer hand over a mislabeled llama-server.
The canonical checkout had been cloned shallow by hand before the script
existed, and llama.cpp stamps its build number as `git rev-list --count
HEAD` -- so a correctly built b10472 binary introduced itself as "build
151", making every "am I on the latest?" glance read as a failed update.
Two self-checks keep the build honest now: a checkout that has gone shallow
is repaired before building (`fetch --unshallow`; cheap, the blob:none
filter still keeps blobs out), and after the build the binary's
self-reported version must match the rev that was built -- commit prefix
always, build number for `b<N>` releases. Numbers only, never the string's
layout, which upstream has already reformatted once between releases. Both
refusals were shown red against the real mislabeled binary before the
repair; the manifest is written before the check so `--status` describes
the binary that exists even when it is refused.

## [1.72.2]

### Fixed

Stream end no longer blanks the response and jumps the view. The chat client
removed the streamed message node and then awaited a wholesale conversation
GET before rendering the saved row -- for that whole round-trip (which
re-downloads every base64 image block in the conversation) the response was
absent from the DOM, so scrollHeight collapsed and the viewport clamped
upward, reappearing only when the fetch landed. Post-stream state is now
ASSIGNED synchronously from the rows the stream's own `heylook_saved` event
already carries (the spec §4 contract as written), and the same reconcile
pass that inserts them retires the streaming node -- no frame without the
response, no network on the happy path. The GET survives only as the
fallback for endings that carry no usable rows: transport death, and the
failed/empty generation whose hidden tail resync restores.

Related hardening in the same pass:

- Message-list DOM structure is now mutated ONLY by `renderMessages`.
  `startStream` and the stream error path no longer hand-place or
  hand-remove nodes; the streaming placeholder (and a continuation's
  anchor-row replacement) go through the same reconcile as everything else.
- `scrollMessages` re-aims after layout on the non-forced path too (a row
  added this tick still reports its `content-visibility` height estimate),
  re-checking near-bottom in the frame so a reader parked in scrollback is
  left alone.
- Edit Save writes the server's response onto the row the LIST currently
  holds, not only the editor's captured object -- a resync landing between
  opening an editor and saving it used to repaint the pre-edit text.
- New render-suite check (`tests/e2e/render.mjs`) pins the swap: removing
  the streaming node while the saved reply is not on screen fails, as does
  any happy-path conversation re-fetch. Both shown red against the pre-fix
  tree via `E2E_V3_ROOT`.

## [1.72.1]

### Fixed

Five review findings against v1.72.0. One was a functional bug in the headline
change, and its check passed anyway:

- **Paste after clicking in the thread still did nothing** -- the case v1.72.0
  claimed to fix. The listener sat on the chat root, but clicking a message
  leaves focus on `document.body`, which is an *ancestor* of that root, so the
  event never reached it. Root scope only ever worked when a field or a
  selection inside the thread held focus, which is what already worked. Moved
  to `document` (with `ctx.signal` for teardown); the guard that leaves other
  editables alone now also protects the drawer's system-prompt box, a body
  child outside `#app`.
- **The check that covered it was vacuous.** It dispatched the synthetic
  `ClipboardEvent` at `.chat__messages`, proving only that *some* ancestor
  listened. It now takes a real mouse click first and dispatches at
  `document.activeElement`, asserting that target is outside the chat root --
  so it cannot silently degrade back to testing the wrong node.
- **The drop overlay described the previous conversation's model.**
  `selectConversation` refreshed the capability-gated chrome *before* moving
  the model select, and both read `currentCaps()` off it. Switching from a
  vision conversation to a text-only one left the overlay promising "Drop
  images to attach" over a model that refuses the drop. The ordering bug was
  pre-existing (it already mis-set the attach button and the picker's accept
  list) but harmless while nothing visible rode it.
- **A refused paste ate the text half of the clipboard.** `preventDefault` ran
  as soon as attachable files were found, before the capability check. A
  payload carrying both text and an image -- what Excel, Word and most web
  pages put on the clipboard -- lost its text at a text-only model, leaving an
  error and an empty composer. The paste is now cancelled only when something
  will actually stage.
- **An attachment could vanish silently mid-read.** `addPendingFiles` captured
  the pending array before awaiting the `FileReader`, but `send()` and
  `clearPendingAttachments` *replace* those arrays rather than emptying them.
  Sending (or switching conversations) during the read left the result pushed
  into an orphan and rendered from the new one. Drag-and-drop makes
  multi-megabyte reads routine, so the window stopped being theoretical. Now
  detected by identity and disclosed.
- **A drop carrying nothing attachable said nothing.** A PDF, a `.txt` or a
  folder (whose `File.type` is empty) matched no kind and disappeared, looking
  exactly like a broken drop target.

### Testing

- `tests/e2e/render.mjs` is at 36 checks. Four of the five fixes have a check
  shown red against the v1.72.0 tree via `E2E_V3_ROOT`, each failing with the
  message that names the bug. The fifth (the swallowed text half) passes
  vacuously there -- the pre-fix listener never ran at all -- so it was
  additionally shown red under a targeted mutation that restores the
  unconditional `preventDefault`.
- The race check is deterministic rather than timed: `addPendingFiles` awaits a
  `FileReader`, so a send dispatched in the same synchronous block always wins.
  Confirmed stable over repeated runs.

## [1.72.0]

### Added

- **Drag-and-drop image (and audio) attach in v3 chat.** Dropping files onto the
  message thread stages them exactly as the attach button does. While a file
  drag is over the thread, a dashed overlay names what the *current* model
  accepts ("Drop images to attach" / "Drop images or audio to attach" / "This
  model takes text only"), so the answer arrives before the drop, not after.
  Three things that break naive drop handling are handled deliberately and
  commented: `dragenter`/`dragover` both `preventDefault` (otherwise the drop
  never fires and the browser navigates away to the file, taking the page and
  any unsent message with it); `dragleave` is depth-counted, since it also
  fires when the pointer crosses into a child element and a boolean flag makes
  the overlay flicker; and the handlers gate on `dataTransfer.types` carrying
  `Files`, so dragging selected text inside the page does not light up the
  overlay. A drop landing just *outside* the thread is swallowed at the window
  for the same navigate-away reason.

### Changed

- **Picker, paste and drop now share one staging routine.** All three funnel
  through `addFiles` -> `addPendingFiles`, which owns the capability gate, the
  count cap and the aria-live announcement. Paste had its own copy and had
  drifted: it staged images only, so pasting an audio clip into an
  audio-capable gguf model silently did nothing.
- **Paste is scoped to the chat page rather than the composer textarea.** A
  paste after clicking anywhere in the thread previously landed nowhere. It
  still never intercepts a paste aimed at another field, and still only calls
  `preventDefault` once it has actually taken files, so ordinary text paste is
  untouched.
- **Paste reads `clipboardData.files` with an `items` fallback.** Safari has
  historically populated one and not the other; reading both removes the guess.
- **Capabilities are checked at staging time, not only at send.** Dropping or
  pasting an image onto a model without the vision cap now refuses immediately
  with a status line and stages nothing. The send-side block remains, but its
  case is now only "staged on a capable model, then switched away" — which is
  what its comment always described. Staging a blob the user must later hunt
  down and clear is worse than a straight no.

### Testing

- Six checks added to `tests/e2e/render.mjs` (the model-free, server-free
  render suite): the overlay appears and clears, its label tracks the model,
  drop and paste each stage a file, a `text/plain` drop stages nothing, and a
  drop onto a text-only model refuses loudly. Drop and paste are event-only
  affordances that no click can reach, so these are the only automated checks
  that they reach the staging path at all. Each helper asserts the synthetic
  event really carried a file, so a check cannot pass because the handler
  bailed early. All were shown red against the pre-change tree via
  `E2E_V3_ROOT`; the MIME-filter check was additionally shown red under a
  deliberate mutation of the split it guards.

## [1.71.2]

### Fixed

Thirteen review findings against v1.71.0/v1.71.1. The knob shipped
substantially unreachable:

- **v3 chat dropped it entirely.** `_SAMPLER_KEYS` in
  `conversation_generate_api` was a hand-maintained copy of
  `samplers.REQUEST_SAMPLER_FIELDS`, and only the cascade was updated -- so the
  one surface that generates server-side accepted the setting, stored it on the
  conversation, sent it, and discarded it. Now derived from the shared tuple.
- **The harmony family could never receive it.** Both providers gated on
  `enable_thinking`, on the assumption that every template reads
  reasoning_effort inside a thinking branch. gpt-oss refutes it: its template
  reads reasoning_effort unconditionally and has no enable_thinking at all.
  Sent whenever set now.
- **The MLX vision path never forwarded it** -- depth worked on a text turn and
  silently reverted to the template default the moment an image was attached.
- **`reasoning_effort` is now its own capability**, not `thinking`. The two
  come apart in both directions (Qwen3.5 reads enable_thinking only, gpt-oss
  reasoning_effort only). MLX probes the template precisely; gguf rides
  `supports_thinking` because the template lives inside GGUF metadata.
- The Messages API could not express it (`MessageCreateRequest` +
  `to_chat_request`), which Phase 3b would have propagated.
- `sampler_summary_from_request` omitted it, so two requests at different
  depths logged byte-identical sampler summaries.
- The `Literal` union was spelled three times; now one `ReasoningEffort` alias.
- MLX had zero tests for any of this; added, including the TypeError-retry
  interaction the call-site comment reasons about.

Two regressions from the v1.71.1 build-script fix, both self-inflicted:

- `--rebuild` routed the manifest's symbolic `rev` through the new branch
  resolution, so rebuilding a `master` build compiled NEWER upstream source
  instead of re-linking the same source. It uses the recorded `sha` now.
- `--rev HEAD` resolved to `origin/HEAD` -- every clone has that ref -- turning
  "build what is checked out" into "build the remote default branch tip".
  HEAD is excluded by name.

Also: `heylookllm import --help` lost its paragraph break to argparse's
default reflow (needs `RawDescriptionHelpFormatter`), and the v3 spec stamped
`reasoning_effort` as v1.70.0 while everything else said 1.71.0.

## [1.71.1]

### Fixed

- `scripts/build_llama.py --rev <branch>` built the LOCAL branch, which a
  fetch never advances -- so `--rev master` compiled whatever the clone held
  at clone time, forever, while the manifest recorded `rev: "master"` with no
  hint of the gap. The 2026-08-14 llama-server build was made this way and
  compiled 4-day-old source. A branch name now resolves to its remote-tracking
  ref and the substitution is announced with both SHAs; tags, SHAs and an
  already-qualified `origin/<branch>` pass through untouched (`origin/b10472`
  does not exist, so the lookup simply fails). The default no-`--rev` path was
  never affected -- it passes a concrete tag from ls-remote.

## [1.71.0]

### Added

- `reasoning_effort` (`low|medium|high|xhigh`) -- thinking DEPTH, reachable
  per request, per preset, and as a per-model default on both providers.
  Qwen3.8 introduced the knob and its template defaults to `xhigh`, so until
  now every request to it ran at maximum reasoning depth with no way down.
  It is a chat-template variable, not a sampler knob: it rides
  `chat_template_kwargs` (gguf) / apply_chat_template kwargs (MLX) beside
  `enable_thinking`, and is dropped when thinking is off because every
  template that reads it reads it inside its thinking branch. The accepted
  set is MODEL-SPECIFIC, so the schema Literal is the union of the Qwen and
  harmony vocabularies -- a typo is a 422 here rather than a template
  exception llama-server would surface as a 500. v3 renders it as a
  `thinking`-gated select; `samplerParams(caps)` drops it at the wire for
  models without the capability.

### Changed

- `heylookllm import` now says in its own `--help` that it is usually
  unnecessary: anything under a `[scan].folders` watch folder is served with
  no entry. The command stays -- it is still the route for a model outside a
  watched tree, and for pinning an entry you intend to hand-edit.

## [1.70.0]

### Added

- v3 can tell a written-down model from a discovered one, and manage where
  models come from:
  - `AdminModelResponse.source` (`"config"` | `"discovered"`). Not derivable
    from `config`: a discovered model's config is NOT empty -- it carries what
    the scanner assigned, so on the wire it reads like a hand-written entry
    storing those keys. The difference is that those values are re-derived
    every load, and that the first save materializes an entry. v3 shows it as
    a `no entry` token on the row plus a disclosure above the config editor.
  - `ScannedModelResponse.served`. `already_configured` stopped being enough
    at v1.69.0: a model under a watch folder reports it as false while the
    router already serves it, so v3's scan panel offered Import for models
    running in the list above it. The Import filter now gates on `served`.
  - `GET|PUT /v1/admin/models/scan-config` -- the `[scan]` watch folders,
    editable. PUT writes models.toml (comments carried), reloads the router,
    and answers with `models_served` so the UI names the consequence instead
    of saying "Saved". v3's Models page replaces its localStorage-only scan
    path box with this; the one-off scan stays, relabelled, for folders you do
    not want watched.
  - `docs/frontend_v3_spec.md` §4 updated for all three.

### Fixed

- `_write_toml` rejected a models.toml with no `[[models]]` at all, because it
  validated the raw parse and `AppConfig.models` is required. Since v1.69.0
  that file is a legitimate config (everything discovered) -- so saving watch
  folders on a fresh install failed validation. Found by the new tests, and
  the same class as the startup bug fixed in 6d557ec.

## [1.69.1]

### Fixed

- The mutating admin routes no longer run discovery ON the event loop. Each of
  them does a models.toml read/modify/write plus `model_registry.discover` --
  a recursive walk of the `[scan]` folders with GGUF header reads, unbounded
  in principle -- and they were `async def` handlers that never await, so that
  work sat on the loop and froze every in-flight SSE generation stream for its
  duration. One mutation runs the walk twice -- once in `ModelService`
  (materializing an entry, or the delete guard), once in the reload -- so this
  was not theoretical. `update_model_config`,
  `toggle_model`, `add_model_config`, `_import_models`,
  `_bulk_set_default_sampler`, `_scan_for_models` and `reload_models` are now
  plain `def` (FastAPI runs them in its threadpool); `remove_model_config` has
  to await its unload, so its blocking tail goes to `asyncio.to_thread`
  instead. Same reasoning, same scan, as
  `MemoryManager._maybe_rescan_models`, which already pushed it to an
  executor. Guarded by `tests/unit/test_admin_offloop.py`, which counts how
  many times an idle task gets to run during each request -- with a control
  test proving the heartbeat can starve.

  The read path is unchanged and stays as it was: `list_configs`/`get_config`
  are models.toml-only, and the admin list route reads the router's merged
  snapshot rather than rescanning.

- `DELETE /v1/admin/models/{id}` returns 409 instead of 500 when the entry is
  a disabled override for a file discovery still finds. The refusal exists to
  explain that deleting the entry would silently re-enable the model, and an
  uncaught `ValueError` turned that explanation into "Internal Server Error".

## [1.69.0]

### Added

- **Discovery-as-registry** (`model_registry.py`): models found under
  `[scan].folders` are served without an import, a symlink, or an edit.
  models.toml is never written -- the merge happens at load, in
  `ModelRouter._load_config`, so both startup and `reload_config` get it.
  Entries in models.toml are served exactly as written and always win;
  discovery can only ADD. That makes models.toml override-only: write an entry
  when you want to change something (a hand-chosen id, `chat_template_path`,
  `spec_type`, `enabled = false`, a comment recording a trap).

  Matching is on the RESOLVED `model_path` (`.resolve()` follows symlinks),
  which is Phase 6 item 1's rule. Id matching is what fails: an id is derived
  from the directory name, so a hand-renamed entry stops matching itself, and
  `modelzoo/<vendor>` symlinks make one file reachable by two spellings that
  share no prefix.

  Discovery is best-effort and isolated per folder -- a scan that raises is
  logged and skipped for that source only, and the server comes up on
  models.toml alone. It runs at router load (startup and `reload_config`) and
  on the rare admin write that materializes an entry; the read surfaces serve
  from the router's snapshot rather than rescanning, which is what keeps a
  recursive filesystem walk off the event loop. `scan_interval_seconds = 0`
  disables it, matching what `ScanConfig` documents and `MemoryManager`
  already honored.

- Admin edits materialize on write: `update_config`, `toggle_enabled`, and
  `bulk_set_default_sampler` create a real entry when handed a
  discovery-only model, since editing one is by definition the moment it
  stops being default-shaped. Reads never materialize, so browsing the models
  page does not grow the file. `remove_config` deliberately does NOT
  materialize -- removing a discovered model's entry would be a no-op that
  reads as success, because the next scan serves it straight back.

### Fixed

- `heylookllm import` no longer re-adds a model that is already configured
  under a different id. Dedup was by derived id alone, so a hand-renamed entry
  was invisible to it and a rescan appended a second entry for the same
  weights -- observed 2026-08-17, where the duplicate also carried a wrong
  `supports_thinking`. Now the resolved `model_path` is compared too, the same
  rule the registry merge uses. A re-import with nothing new leaves the file
  byte-identical.

## [1.68.0]

### Added

- gguf: `chat_template_path` overrides the GGUF-embedded chat template
  (llama-server `--chat-template-file`). Until now the template baked in at
  quantization time was the only one reachable, so choosing a quant publisher
  silently chose the prompt format -- and publishers differ materially on the
  same weights. Verified live on Qwen3.8-27B against both templates: ggml-org
  embeds Qwen's official template byte-identically (8952 bytes); unsloth
  embeds a patched one (9993) adding a `developer` role and merging up to two
  leading system messages. Two leading system messages render under unsloth
  and are a 500 under official ("System message must be at the beginning"),
  since a raised jinja exception surfaces as a 500 from llama-server. Both
  still reject a system message appearing mid-conversation.

  Classified `requires_reload`, which is the honest class: llama-server takes
  the template at spawn, so there is no per-request or per-preset form of it.
  Being schema-derived, the field reaches the reload set, the import
  allowlist, and v3's models-page config editor with no further wiring. A
  configured-but-unreadable template fails at load with a named error rather
  than degrading to the embedded one -- silently serving a different prompt
  format is the failure this field exists to prevent, and at
  `observability_level=off` (the default) the subprocess log is DEVNULL, so
  the spawn failure would otherwise leave no diagnostic.

## [1.67.1]

### Fixed

Ten findings from the second review pass (scope: everything after the first
review's snapshot -- i.e. the first review's own fixes plus the conversion
tooling):

- The generate route claims the conversation BEFORE snapshotting its rows:
  snapshot-before-claim left an interleaving where a message written
  between the two was destroyed by the positional commit -- the exact
  phone+desktop hole the CRUD gate was built to close. With claim-first,
  the store's single FIFO writer guarantees every pre-claim write is in
  the snapshot and every post-claim write 409s.
- The never-started-stream claim leak now has a deterministic release (a
  StreamingResponse BackgroundTask on Starlette's cleanup path, identity-
  guarded) with the 60s watchdog demoted to a belt -- a held claim now
  freezes message CRUD too, so the window mattered much more than before.
- send() restores the composer (text + staged attachments) when the
  user-message POST fails -- the new mid-generation 409 was destroying
  typed content client-side.
- Deleting the actively-streaming conversation works again: the call site
  stops the generation first and retries once past the claim's release.
- Stop no longer aborts a completed stream mid-delivery of its saved rows:
  the 404-means-abort-locally rule now applies only before any event has
  arrived (the retry/dispatch window it was built for).
- The pending sampler-params PUT is FLUSHED before every generate:
  overrides carry set values past the debounce window, but a CLEARED
  panel value is expressed by absence, which overrides cannot spell.
- The no-heylook_saved recovery retries with backoff (the detached
  disconnect-persist can lose a DB-writer race beyond any fixed delay)
  and closes its status line instead of leaving an error stranded.
- The keepalive marker's wire spelling lives once in streaming_utils
  (keepalive_sse), consumed by all three SSE loops.
- convert_gguf.py resolves its DEFAULT dest (the converter subprocess runs
  with cwd=checkout, so the relative default silently wrote the multi-GB
  output into the llama.cpp tree while printing the repo-side path) and
  imports build_llama's checkout resolver instead of carrying a copy.

## [1.67.0]

### Fixed

Ten findings from the day's adversarial code review of the chat-orchestration
work (9 confirmed, 1 hardened), each red-first where a unit seam existed:

- Two `_ACTIVE`-claim leaks that could 409-lock a conversation until
  restart: the MODEL_BUSY 503 path RETURNS (skipping the exception
  release) — now pops explicitly, test-pinned; and a stream generator
  cancelled before its first step runs no code at all — a 60s
  identity-guarded watchdog releases a claim whose stream never started.
- The generate loop (and `/v1/messages`' loop — same pre-existing hole)
  crashed on the keepalive sentinel the abort wrapper yields during long
  prefills or FIFO-gate waits; both now emit `: keepalive` comments like
  the OpenAI route always did.
- Message CRUD now 409s while a generation streams into the conversation:
  its atomic commit truncates by position, and rows appended mid-stream by
  a second client were silently destroyed at commit (the phone+desktop
  case). Metadata PUTs stay open.
- v3 sends `overrides: {model, ...panel snapshot}` on every generate — the
  panel's writes to the store are debounced/fire-and-forget, so a fast
  Send could generate with stale params or the previous model.
- `send()` re-checks the active conversation after its user-message POST —
  a switch mid-flight pushed the row into the other conversation's mirror
  and generated there.
- Stop now works during the 503-busy retry window: DELETE answering 404
  (nothing active server-side) makes the client abort locally instead of
  letting the retry launch a generation the user stopped.
- A stream ending without `heylook_saved` is no longer treated as success:
  the client says so and re-adopts after the server's disconnect persist
  has had time to commit.
- The llama-server no-log message no longer claims `observability_level=off`
  as the DB truth (during pre-warm the in-process cache is still the
  default); gguf-probe now always captures llama-server logs.
- The server-side cap gate (`_CAP_GATED`) got its missing unit tests —
  stored params AND overrides, capable and incapable models.

## [1.66.1]

### Changed

- **One llama-server build, no silent shadowing** (owner rule): the
  canonical build (`scripts/build_llama.py`'s fixed home-dir output) is the
  single intended binary source. The `server_binary` / `$HEYLOOK_LLAMA_SERVER`
  overrides remain as escape hatches but now WARN at every spawn, naming
  their source and the canonical build they shadow -- a stale exported env
  var silently beat a freshly built canonical binary and the first symptom
  was a load failure on a new model architecture. Pinned by
  TestBinaryResolution (shown red first).

## [1.66.0]

### Changed

- **v3 chat generates through the conversation endpoint** (plan Phase 2, the
  client half of 1.65.0): send/regenerate/edit-regenerate/continue now call
  `POST /v1/conversations/{id}/generate` (Messages SSE grammar parsed by the
  new `streamGenerate` in streaming.js) instead of client-orchestrated
  truncate -> `/v1/chat/completions` -> persist. The client no longer builds
  request bodies, converts stored blocks to wire parts, or does position
  arithmetic for generation -- the mirror hides tails visually and adopts
  the store at stream end, so a failed or empty generation gets the rows
  back. Stop = `DELETE .../generate` (server aborts + persists the partial;
  the stream still delivers the saved rows); teardown and switches abort
  the fetch and lean on the server's disconnect persistence. The v1.64
  pendingSave latch is gone (no client-save window exists anymore).
  Explore/notebook stay on `/v1/chat/completions`. E2E: render suite green
  through the cutover unchanged (23/23 -- the Phase 0 net held); live chat
  suite 45/45 with three checks rewritten off the old wire (sampler state
  is asserted in the conversation's params + a sampler-free generate wire,
  and the stop test raises its budget through the panel so it reaches the
  store). Live gguf matrix verified on DeepSeek V4 Flash (append /
  regenerate / continue-in-place / user-role-continue typed 400 / abort
  persists partial / pre-split thinking persisted).

## [1.65.0]

### Added

- **Conversation-scoped generation** (`POST /v1/conversations/{id}/generate`,
  plan_chat_orchestration.md Phase 1): the server-side saga. The server
  builds the provider request from the stored conversation (system prompt,
  sampler params with cap gating, model, rows with media dropped-and-counted
  for the target model's capabilities), anchors truncation by message id,
  streams the Messages SSE grammar (this endpoint is the Phase 3b
  migration's first consumer), owns persistence — completion, abort, and
  client disconnect all persist, via a detached task in the disconnect case
  — and emits the authoritative stored rows as a final `heylook_saved`
  event. Three modes: append (optionally persisting a new user turn first),
  regenerate, continue (prefill; merges onto the anchor row). Destructive
  truncation COMMITS only together with the row it produced
  (`db.replace_tail_with_message` / `replace_tail_with_update`, one
  transaction each), so a failed or empty generation leaves the thread
  untouched — an invariant the client-orchestrated flow could not offer.
  One active generation per conversation (409 otherwise);
  `DELETE .../generate` aborts it (partial persists). Spec §4 updated;
  17 new contract tests. The v3 client cutover (Phase 2) follows.

## [1.64.0]

### Added

- **v3 chat: thinking blocks are editable.** The message editor grows a
  second, labelled textarea when the message carries thinking; Save PUTs
  both (empty clears the column). The backend always accepted it -- only
  the editor was missing.
- **v3 chat: unsaved-row honesty.** When persisting a response fails, the
  kept-on-screen row now says so (always-visible note, warn-tinted) and
  offers exactly two exits: Retry save (re-POSTs, row becomes real) and
  Discard (armed, local). While one exists, send and every
  position-anchored op (edit-regenerate, regenerate, delete) refuse loudly
  -- the client's positions are known-divergent and a send would mint a
  colliding server position.

### Fixed

- **v3 chat: refusals are loud.** Edit/Regenerate/Delete/Save &
  Regenerate/Enter-to-send during a live stream said nothing and did
  nothing -- indistinguishable from a broken button, worst during the
  pre-first-token wait of a cold load. They now state why in the status
  line.
- **v3 chat: the Stop-then-act window is closed.** The stream releases
  before its partial save lands; a Regenerate clicked in that gap truncated
  the thread and the late POST appended a ghost row. A pendingSave latch
  keeps destructive ops loudly blocked until persistence settles.
- **v3 chat: the mirror reconciles.** On stream completion (and after any
  failed edit/regenerate/delete saga) the client re-fetches the
  conversation and adopts the server's rows -- divergence dies at the saga
  boundary instead of compounding into the next position-anchored
  truncation. Unchanged rows keep their DOM nodes; unsaved rows survive
  adoption.
- `tests/e2e` render suite: stub /v1 became a stateful mini-store and the
  suite grew from 8 to 23 checks (all new ones shown red against a pre-fix
  tree first), including an iPhone-emulation boot (viewport + touch +
  hover:none via CDP) pinning touch reachability of the new affordances,
  and a Chrome-side pin for the iOS hidden-row report (plan Phase 0.5).

## [1.63.0]

### Changed

- **File logging is off by default.** `observability_level` now defaults to
  `off` (was `minimal`): a fresh server writes nothing under `logs/` until
  the level is raised via `/v1/admin/config`. The knob was already the
  master switch for the spine and memory.py's streams; the llama-server
  subprocess log (`logs/llama_server_<id>.log`) was the one writer that
  ignored it and is now gated by the same switch (checked at spawn --
  raise the level and reload the model to capture output; load-failure
  messages say so instead of pointing at a file that was never written).
  An unrecognized level value now degrades to `off`, not `minimal`, so a
  bad stored value can never turn file logging on by accident. Rationale:
  `logs/` resolves relative to the working directory, so an on-by-default
  level sprinkled log dirs wherever the server happened to be started.

## [1.62.7]

### Fixed

- Two import-merge hardenings from the targeted `write_merged` review:
  deriving `default_model` now skips id-less entries instead of raising a
  bare `KeyError` (such an entry is already invalid to the server, but the
  import must not crash on it), and top-level array-of-tables keys keep
  their `[[section]]` form on rewrite instead of being re-rendered as an
  inline array (values were always preserved; the shape change orphaned any
  comments on the old sections). Both pinned by tests.

## [1.62.6]

### Changed

- **Dependency management is plain uv now.** `pyproject.toml` is a
  hand-maintained manifest of published releases ("current as of" floors;
  `uv.lock` pins the exact versions), and the channel/pin updater
  (`scripts/update_deps.py`, with its `[tool.heylook.deps]` /
  `[tool.uv.sources]` machinery) is gone -- updates are
  `uv lock --upgrade[-package X] && uv sync`. Running an upstream's git
  commit is a machine-local working-tree experiment (a `[tool.uv.sources]`
  entry, or a one-off `uv run --with`), never committed: uv honors no
  gitignored home for source pins (`override-dependencies` in `uv.toml` is
  silently ignored; verified on uv 0.11.32) and pins propagate into
  `uv.lock`, so the new pre-commit guard blocks committing either file while
  it carries one.
- **`heylookllm import` merges by default.** An existing output file keeps
  every entry and top-level setting verbatim (comments re-injected via the
  same toml_comments machinery as admin writes); scans only append newly
  found models, an already-configured id can never re-enter through a scan,
  and an unparseable existing file refuses instead of clobbering. Hand edits
  like a per-model `server_binary` now survive reimport. `--fresh` restores
  the wholesale rewrite; `--merge` (which only printed advice) is gone.
- Dependency floors raised to the releases in use (`mlx>=0.32.0`,
  `mlx-lm>=0.31.3`, `mlx-vlm>=0.6.13`); `torch` relaxed from `==2.13.0` to
  `>=2.13.0` (uv.lock still carries the exact version).

### Added

- `scripts/build_llama.py`: clones and builds llama-server at the newest
  `b<N>` release tag by default (llama.cpp's releases ARE those tags; no
  semver), `--rev` for any other tag/branch/SHA, plus `--status`,
  `--rebuild`, `--clean`, and `--openmp` with the resolved-flag readback
  that refuses a silently-downgraded OpenMP build. Writes the
  `heylook-build.json` manifest; never touches pyproject/uv.lock. Replaces
  the build half of the retired updater; the provider's zero-config fallback
  location is unchanged and now pinned by a literal in its test.
- `scripts/guard_stable_channel.sh`, wired into the pre-commit hook: blocks
  committing `pyproject.toml`/`uv.lock` while they carry a git pin. Checks
  staged blobs (a clean commit from a pinned worktree passes), fails closed
  (a broken read blocks the commit rather than silently passing), and
  matches with bash rather than PATH grep. `HEYLOOK_ALLOW_CHANNEL_COMMIT=1`
  is the deliberate-exception escape hatch.

## [1.62.5]

### Fixed

- **Chat no longer jumps to the top of the thread on send, edit or delete.**
  `renderMessages` rebuilt every row with `replaceChildren`. `.message` carries
  `content-visibility: auto`, so an off-screen row only knows its
  `contain-intrinsic-size` estimate (3rem) until it has been laid out once --
  and that measurement lives on the node. Throwing every node away collapsed
  the list's `scrollHeight` to a fraction of the real one for the rest of the
  tick, so every pixel-based scroll after it aimed at a thread that was about
  to grow underneath: a forced `scrollTop = scrollHeight` after Send landed
  near the *top* of a long conversation (measured: 658px into a list whose
  bottom was 9.8k). Rows are now keyed by message id and reused unless their
  render signature changed (role, position, content, thinking, media blocks,
  editing state, and the current model's caps -- which own the drop
  disclosure), and the list is reconciled in place. Two corollaries the fix
  needed: the reconcile removes departing children *before* placing the rest
  (placing first walks the stale node down the list and re-detaches the whole
  tail -- an edit-cancel slammed the view to the bottom), and a forced
  scroll-to-bottom re-aims on the next animation frame, since a row added this
  tick is still an estimate. Reuse also means a re-render no longer destroys
  the row you were interacting with, so focus and armed Delete buttons survive
  it, and images no longer re-decode. A live stream's row is carried through
  the reconcile as well, instead of being detached mid-paint by an unrelated
  re-render (a model switch mid-stream). Design rule: `DESIGN.md` §7.

- **Row reuse, second pass (code review).** Three corrections to the above.
  Model-dependent chrome is now scoped to the rows that carry it -- caps reach
  a row only if it HAS media (they decide the drop disclosure), provider only
  an open editor (Save & Continue is MLX-only). Keying every row on the current
  model instead invalidated the whole list the first time anything re-rendered
  after the residency fetch landed, rebuilding every node and reproducing the
  original jump on the first send after load (measured: 1446px into a list
  whose bottom was 15.7k). Second, `editingId === msg.id` matched null to null,
  so an id-less row -- the shape `finishStream` pushes when saving the reply
  fails -- rendered as an open editor whose Save would PUT to
  `/messages/null`; the test is now `msg.id != null && ...` (the bug predates
  the rewrite, which inherited it at both sites). Third, the signature joins on
  NUL rather than a space, so a field's tail can't read as the next field's
  head.

### Added

- **`bun run e2e:render` -- a model-free render suite** (`tests/e2e/render.mjs`,
  opt-in, seconds). It drives the real `/v3` chat page against a stubbed `/v1`,
  so it needs no server, no model, no Metal and no DB -- which is why it is a
  separate entry point and not part of `bun run e2e`, whose prerequisites it
  does not share. Nine checks over the class of bug above: a long thread must
  not move when the reader did nothing, Send must land at the bottom, edit and
  cancel must hold position, an id-less row must render as a message, a
  background residency refresh must reuse every unchanged row, and the render
  that repairs an open editor must not eat the text in it. Each one was
  observed FAILING against a deliberately broken copy of the frontend (the
  pre-fix renderer, a shared model key, the draft-carry removed, the residency
  re-render removed) -- `E2E_V3_ROOT` exists for exactly that. Nothing else
  automated can see any of this: it is client-side layout, invisible to server
  telemetry, and the model-driven suites never scroll a long thread.

### Changed

- **The residency fetch now re-renders the message list.** `refreshLoadedIds`
  fills `providerById` after first paint, and an editor opened before it landed
  is missing its Save & Continue button (it fails closed while the provider is
  unknown) until some unrelated render happens by. It now catches up on the
  spot. The reason this was not already true is that a rebuild would discard
  whatever was typed in the box, so the rebuild carries the live value and
  caret across (`carryEditorDraft`) -- with the signature scoped as above, the
  open editor is the only row that render touches at all.

## [1.62.4]

### Changed

- **Dependency audit: one dead entry, one misplaced, one undeclared.**
  `pytest-mock` is gone -- nothing imported it and nothing took its `mocker`
  fixture (the suite mocks with `unittest.mock` and the repo's own `mock_mlx`,
  which the MLX no-module-level-`.start()` rule wants anyway). `pyyaml` moved
  from the runtime deps to the `dev` group: nothing under `src/` imports it,
  its one use is `scripts/export_openapi.py`'s `--yaml` writer, and that
  import is already guarded -- so it was a claim every
  `pip install heylookitsanllm` paid for and the server never made. (Presence
  is unchanged either way: uvicorn's `standard` extra pulls PyYAML.) The four
  CLI-only dev deps that an import audit reads as dead -- `build`, `twine`,
  `py-spy`, and `rich` (a lazy import inside `scripts/benchmark.py`) -- now
  say so in place, as does `python-multipart`, which starlette imports on our
  behalf for `api_multipart.py`.

### Fixed

- **A bare `pytest` failed at collection with 4 errors.** Two independent
  causes, both pre-existing. `tests/integration/test_keepalive.py` imported
  `aiohttp`, the one undeclared dependency in the repo, so it could not run
  after a clean `uv sync` -- ported to `httpx`, already a dev dep, which also
  retired its hardcoded `dolphin-mistral` model id (long gone, so the script
  404'd before reaching the keepalive path it exists to watch; it asks the
  server now). And `tests/integration/mlx_perf/` and `tests/unit/mlx_perf/`
  were both packages named `mlx_perf`, which collide under pytest's default
  prepend import mode -- the `__init__.py` files carried nothing but a
  docstring and nothing imports either as a package, so they are gone and the
  test modules (all uniquely named) import cleanly. `testpaths` is now
  `tests/unit tests/contract`, the suite `/test-suite` runs, so a bare
  `pytest` no longer sweeps integration's live-server scripts; running those
  explicitly still works.

- **A mount failure now says when it is a stale browser cache, and that a
  plain reload will not clear it.** Reported as "the bar at the top of chat,
  where the model loader is, doesn't show up" -- and it doesn't: a throw
  anywhere in a page's `setup()` rejects `mount()`, and the router replaces
  all of `#main` with its error note, so the whole chat page goes with it.
  The throw was `s.presetBar.promptState is not a function` from
  `paintSysPromptChip`, the last unguarded call in chat's setup: a new
  `chat.js` running against a `preset-bar.js` cached from before v1.62.3
  added `promptState`. This is exactly the mixed-module-version failure the
  v1.62.2 no-cache headers exist to prevent, but those only bind entries
  cached from then on -- one stored under an earlier server keeps its
  heuristic freshness (~10% of the file's age, days for a long-lived file)
  and is never re-requested. The router now recognises that signature
  ("X is not a function", module-fetch failures) and says to hard-reload,
  because a normal reload revalidates the document and then serves the stale
  SUBRESOURCE right back out of cache. Diagnosis only -- no behaviour
  change on a healthy load.

## [1.62.3]

### Fixed

- **A system prompt typed before any conversation existed was silently lost.**
  It lived in page state alone -- nothing on the server owns a prompt until a
  conversation does -- so a reload, or a trip to another page and back, ate
  it. The sampler params beside it survived, because `settings.js` parks
  those in localStorage, which is exactly why the loss read as "everything
  else loads fine, just not the system prompt". The draft is now parked the
  same way (`heylook.v3.chat.draft-prompt`) until a conversation adopts it;
  both create paths clear it. This also cuts the worse half: with the box
  silently blank, a Save onto an existing preset name stored null OVER a good
  prompt, which is how two presets lost their prompts at once.
- **A preset carrying no system prompt no longer blanks the one in use.** The
  prompt is an OVERRIDE box (owner rule): a preset owns a prompt and carries
  it onto whatever it is applied to, but an empty one makes no claim and
  leaves the conversation's prompt -- or the model's own default -- exactly
  as it was. Empty means "does not speak for the prompt", never "set it to
  empty". Only a carrying preset can arm "Replace prompt?" or count as drift
  (`matchesState` ignores the prompt for a promptless preset).

### Added

- **A system-prompt chip in the chat bar** (`.chat__sysprompt-chip`): what
  prompt is in force, where it came from, and whether it still matches that
  source -- "No system prompt" / "custom" / "&lt;preset&gt;" / "&lt;preset&gt;
  (modified)". Always rendered, including the empty case: a hidden chip is
  indistinguishable from a broken one, and that ambiguity is what let the
  disappearing prompt go unnoticed. The full text rides the tooltip and
  accessible name; clicking any state opens the editor focused. Fed by the
  preset bar's new `promptState()`, kept separate from the whole-document
  `indicatorInfo()` so a chip that claims to track the prompt cannot flip on
  a temperature nudge.
- E2E coverage for all three: the override rule, the chip's states, and its
  click-through (`tests/e2e/suites/chat.mjs`).

### Changed

- **Load cost is disclosed, never confirmed.** Switching to an unloaded model
  no longer raises a Cancel / Switch-anyway gate: choosing a model IS choosing
  to pay for it, so there was no decision to gate, and the confirm fired
  hardest on the emptiest state -- nothing resident, no conversation -- where
  its "may evict the resident model" was also false. Only LOSS gates now
  (history media the target cannot read). In its place: the switch says what
  it costs, and Send names the wait it previously spent in silence --
  `Loading <id>…` on a cold target, `Waiting for the first token…` otherwise,
  cleared by the first content or thinking delta and by stream teardown, so a
  zero-token completion or a Stop mid-load cannot strand it.

## [1.62.2]

### Fixed

- **v3 assets are served `Cache-Control: no-cache`, ending silently
  mixed-version frontends.** The `/v3` route sent no cache directive and
  starlette's `FileResponse` has no 304 path, so browsers fell back to
  HEURISTIC freshness -- roughly 10% of a file's age at cache time, computed
  per file. On a no-build frontend with unhashed URLs that is a version
  skew generator: a file edited every session (`js/pages/chat.js`) earns a
  freshness window of minutes and refetches, while a rarely-touched module
  it imports (`js/preset-bar.js`) keeps a window of hours and is served from
  cache without a request. The new caller then calls into the old module and
  the page dies on `... is not a function` -- observed as "Create failed:
  presetForNewDoc is not a function" on New conversation, with the function
  present in the source all along. Revalidating every v3 asset costs a full
  re-send (no 304s), which is nothing for a ~450KB localhost frontend.
  Pinned by `test_v3_assets_are_revalidated`.

## [1.62.1]

### Fixed

- **All 8 confirmed /code-review findings on v1.62.0, plus the /unload and
  DELETE ride-alongs.** The reload route: every unload now runs via
  `asyncio.to_thread` (the drain can wait up to 30s on active generations
  whose SSE delivery the event loop itself drives -- running it ON the
  loop guaranteed the full freeze and then a force-unload under an active
  Metal command buffer); a pinned model is a 409 with the pin reason
  (was an opaque 500), on /reload, /unload AND DELETE (checked BEFORE the
  config row is deleted); a load already in flight is refused with 409
  instead of silently joined-and-misreported (`router.is_loading`, since
  unload_model cannot distinguish 'not loaded' from 'loading'); and the
  route calls `reload_config()` first so a hand-edited models.toml is
  picked up -- which also let the now-caller-less
  `router.reload_single_model` be deleted. The observed-resident line:
  fetches metrics with `force_refresh` (the 30s cache predates exactly
  the load the line exists to reveal), treats the collector's 0.0
  failure sentinel as no-reading instead of rendering "0.0 GiB
  measured", and is built by the same row() helper as the estimate rows
  so a restyle cannot fork its markup. Plus the contradictory TODO
  sentence.

## [1.62.0]

### Added

- **Server-owned reload** (design ask #4): `POST
  /v1/admin/models/{id}/reload[?warm=true]` runs unload + load(+warm) as
  ONE operation, sharing `/load`'s exact body so the warm contract cannot
  fork; reloading an unloaded model is just a load. v3's "Reload now"
  points at it -- the old browser-driven unload-then-load pair could
  strand a model unloaded if the tab died between the calls. Spec §4 +
  contract tests.
- **Fit meter's observed line** (design §5's closing loop): a LOADED
  model's Memory-fit section shows "Resident now" from `/v1/system/metrics`
  `models[id].memory_mb`, labelled measured-after-load -- the user sees how
  good the sizing was and learns whether to trust it. Best-effort; the fit
  rows stand alone without it.

## [1.61.1]

### Fixed

- **All 8 confirmed /code-review findings on the v1.61.0 continuation
  commit** (8 finder angles, ~35 candidates, 10 verified, 8 confirmed; the
  suspected gguf echo over-strip was REFUTED -- the pinned build renders
  the prefill verbatim, so the positional strip is exact). The three
  destructive ones, two of which were pre-existing races in Save &
  Regenerate that the new Save & Continue inherited: `save()`'s
  truncate-then-stream paths now carry the same `s.stream` guard as
  regenerate/delete; every truncation is anchored to the conversation id
  captured BEFORE the first await (a mid-save conversation switch could
  irreversibly truncate the NEWLY opened conversation); and the user-role
  Save & Continue gate fails CLOSED while the provider is unknown
  (`undefined !== 'gguf'` showed the button on gguf models before the
  admin fetch resolved -- truncation committed, then the 400 landed).
  Contract honesty: streaming guard refusals are now typed
  `invalid_request_error`/`invalid_request` in-band (both APIs; they fire
  after headers flush, so a real 400 is impossible -- previously they read
  as `server_error`, spec §4 updated), and gguf auto-continuation no
  longer 400s an all-text parts-list prefill (the Messages-API block form,
  which has no opt-out field and streamed fine on v1.60) -- the payload's
  copy is flattened with MLX's own ' '-join rule so the echo strip stays
  exact; non-text-part prefills continue unstripped (the v1.60 behavior)
  with a warning instead of a new 400. Completeness: the batch path now
  runs the same continuation resolution as the single-request path (it
  still rendered the closed-turn half-state and silently ignored the
  flag), the VLM text path maps the continuation-template ValueError to
  the same 400 as the pure-text branch (was a 500), and a continuation's
  seeded prior thinking survives the first new thinking delta on screen.

## [1.61.0]

### Added

- **True continuation (prefill) + Save & Continue** (owner request).
  `ChatRequest.continue_final_message`: absent = auto (a trailing assistant
  message is continued -- the long-standing convention, now actually TRUE:
  before this, MLX only suppressed the generation prompt and the turn still
  rendered CLOSED, so the model saw a finished message and nothing to
  continue); `true` = continue the final message whatever its role
  (user-role co-writing is MLX-only); `false` = never continue. MLX passes
  `continue_final_message=True` through the template (transformers trims
  the closing markup; refusals are loud 400s, never a silent closed turn),
  and a continuation disarms the `prefills_thinking` parser assumption --
  there is no generation prompt, so the stream starts inside CONTENT and an
  armed parser would misfile the whole continuation as thinking.
  llama-server (verified live on the pinned build via /apply-template)
  continues a trailing assistant turn natively but ECHOES the prefill back
  as the leading delta(s); the provider strips it positionally (byte
  equality would false-negative -- retokenization attaches whitespace to
  the echoed span), so the response is continuation-only on every provider.
  What llama-server cannot express 400s honestly: user-role continuation,
  and `false` with a trailing assistant message (it ALWAYS continues one).
  v3: the message editor gains **Save & Continue** (both roles; hidden for
  user messages on gguf models): save the edit, truncate everything after,
  and stream the continuation into the SAME message row -- an abort keeps
  the partial, same contract as a normal stream. Not supported with image
  history or diffusion models (400). Tests: template-kwarg shape +
  loud-refusal paths, echo-strip cases, parser edge, flag resolution; E2E
  "save & continue extends the message in place" against the live model.

## [1.60.0]

### Added

- **Fit endpoint + fit meter** (design doc §5 + §9 ask #2 -- "the heart of
  it"). The memory ceilings/sizing/verdict logic moved from
  `scripts/ram_report.py` into `heylook_llm.ram_fit` (the script is now a
  CLI renderer over the same structured report -- one computation, two
  faces, `--quiet` contract for dev_server.sh unchanged), and
  `POST /v1/admin/models/{id}/fit` exposes it: body
  `{config_overrides?, headroom_gb?}` (candidate unsaved edits; null =
  reset-to-default, the PATCH spelling), response = weights (whole shard
  set + sidecars), Metal working set, per-allocation cap, reclaimable RAM
  (total - anonymous - wired, NOT free+inactive), KV headroom, structured
  per-ceiling verdict lines, and `hard_working_set` -- the
  provider-derived engine asymmetry (over the working set is FAIL for MLX,
  which refuses above the recommendation; WARN for gguf, which loads past
  it and degrades into paging). The sysctl hint (`sysctl_suggest_mb`) is
  server-gated: only while `iogpu.wired_limit_mb` is at its OS default.
  The v3 Models-page config editor renders it as a "Memory fit" section:
  measured rows in mono, amber warn / red fail verdict, live recompute per
  field edit (debounced 300 ms, aborted on the next), "fit unavailable" on
  any failure -- the client NEVER computes fit itself -- and a FAIL
  verdict disables the row's Load button with the reason (gguf's warn
  never blocks). All numbers are measured today; the response's
  `estimated` flag is reserved for the day offload deltas make part of it
  an approximation. Tests: unit (engine asymmetry, sizing traps, sysctl
  gating), contract (route shapes, override semantics, read-only
  guarantee), E2E (meter renders + verdict resolves on the real model).

## [1.59.0]

### Added

- **New documents start as the selected preset** (owner decision, option 1
  of the 2026-08-11 preset-semantics review). The preset bar's select was
  fully inert: a preset landed on a document only via explicit Apply, so a
  new conversation opened with an empty prompt while half the settings
  panel (sampler knobs) carried forward anyway -- which read as "my prompt
  wasn't saved". Now the shared bar exposes `presetForNewDoc()` (explicit
  selection, else the active document's durable stamp, so the behavior
  survives a reload) and BOTH pages' New buttons create the document as
  that preset: prompt + params + `applied_preset_id` stamped at birth
  (starting-as is an explicit apply under the stamp rules, not an
  inference). A prompt drafted before any conversation exists still wins
  over the preset; without a preset, the old rules hold verbatim. Wire
  change: `POST /v1/conversations` and `POST /v1/notebooks` accept
  `applied_preset_id` (spec §4 updated; db create paths persist it,
  test-pinned). E2E: new chat check "new conversation starts as the
  selected preset".

### Fixed

- **Chat message editor opens as a slit** (owner report): the edit
  textarea sized itself in a microtask (pre-layout, so `scrollHeight` read
  short), never re-grew while typing, and capped at 400px regardless of
  viewport. Now it sizes on a `requestAnimationFrame` (post-layout), grows
  on input like the composer, and caps at ~60% of the viewport height.

## [1.58.0]

### Added

- **models.toml comments survive admin writes** (`toml_comments.py`).
  Every admin write (import, PATCH, config edit) regenerates the file
  through tomli_w, which emits no comments -- routine editor saves were
  silently wiping every comment. Now `_write_toml` still renders values
  through tomli_w (layout, quoting and key order stay canonical -- the
  old file's formatting is never spliced in), then carries the previous
  file's comments onto the fresh render. A comment is carried only while
  its anchor is unchanged, so a note can never outlive what it describes:
  a top-level key's comment needs that key unchanged; everything inside a
  `[[models]]` entry needs that model's values byte-identical (normalized
  through tomli_w, so hand-formatting doesn't pin anything); a block
  sitting above the next model's header additionally needs that model
  unchanged and still immediately next. Merging is best-effort and gated
  on the merged text parsing to exactly the fresh render's values: any
  parse failure, missing anchor, or value drift degrades to a comment-less
  write -- never a refused or corrupted one. Implementation note pinned in
  the module docstring: tomlkit is used strictly read-only for extraction,
  because mutating any item of a tomlkit-parsed array-of-tables (even
  comment trivia) makes it re-render as a malformed inline array -- the
  failure mode that sank the earlier whole-table-splice attempt against
  `test_import_reimport.py`, which now passes alongside 16 new tests
  (`test_toml_comment_preservation.py`).

## [1.57.1]

### Fixed

- **update_deps.py write path hardened** (the three fixes from the
  KEEP-WITH-FIXES audit verdict): pyproject.toml writes go through ONE
  guarded helper that refuses (nothing written) if the file changed on disk
  since this run loaded it -- parallel sessions are normal here, and the
  llama.cpp path holds the in-memory doc across a minutes-long C++ build,
  so the final write was last-writer-wins over anything another session
  landed meanwhile. A failed `uv lock` after a write now ROLLS THE WRITE
  BACK (both the mid-run stable-channel write and the final one), so the
  030119f class -- an on-disk pyproject every project-scoped uv command
  then chokes on -- is dead structurally, not just for the one trigger
  that fix removed. And the write path has tests at all now
  (`tests/unit/test_update_deps_write_path.py`): the latest-channel
  sources entry is pinned to exactly {git, rev} (uv hard-rejects `branch`
  alongside `rev`), the tomlkit round-trip is byte-identical on the real
  pyproject (the property separating this from the models.toml/tomli_w
  comment loss), and both guard directions fire.

## [1.57.0]

### Added

- **v3 chat: switch models anytime, honestly** -- the §15 arc of the
  load-options design doc, closing the two gaps that made "switch anytime"
  nominal:
  - **G1, history media**: `toWireContent` is caps-gated -- blocks the
    current model cannot take are dropped from the WIRE (never the store;
    switch back and they ride again), with a per-message "N images not sent
    to this model" transcript disclosure. Previously a vision conversation
    switched to a text-only model shipped image parts unconditionally and
    failed raw on send. Staged attachments still BLOCK -- the asymmetry is
    deliberate and commented at both sites.
  - **G3, load cost**: option labels carry residency (● resident / ○ idle,
    plain ids until known -- never guessed; refreshed at mount, after Load,
    after each completed generation, no polling); switching runs a
    pre-switch check whose warnings (dropped media, unloaded target's load
    cost, thinking-toggle note riding along) render inline in the status
    area with Cancel / Switch anyway -- the switch does not commit until
    confirmed, a clean switch commits silently, and Send with the
    unconfirmed target selected commits it. A Load button beside the select
    pays the warm load deliberately while reading.
- E2E: chat suite grows a warning-flow check (warn before commit, Cancel
  reverts) and the capability-gating check now confirms switches like a
  user who means it (41 checks).

## [1.56.0]

/code-review high over the branch: 12 verified findings, 10 confirmed, all
10 applied.

### Added

- **`stale_reload_fields` on admin model responses** -- server-derived list
  of requires_reload keys whose saved value differs from what the LOADED
  process was built with (`router.stale_reload_fields`; `[]` when
  unloaded). Replaces v1.55.0's client-side reload-needed Set, which never
  repainted on save, died on page remount, and mis-cleared when an
  unload-succeeds/load-fails reload left the marker on an unloaded model.
  The v3 row marker and the panel's Reload offer now both render from it.
- `ui:"hidden"` on config fields the schema exposes but no editor should
  offer: gguf host/port/server_binary/startup_timeout_s and mlx's derived
  `vision` mirror. Declared on the field, so the v3 editor, the summary
  chip, and the E2E schema check all read ONE source instead of each
  hand-copying a name list (the E2E mirror had already been born stale).

### Fixed

- **`exclude_unset` was not "stored keys only"**: `_resolve_modalities`
  assigns derived `modalities`/`vision` during validation, which added them
  to `__pydantic_fields_set__` -- every mlx entry reported them as
  explicitly stored, re-creating the every-default-looks-chosen failure for
  validator-assigned fields. The validator now restores the fields-set;
  pinned by a test.
- **The per_request refresh skipped disabled-but-loaded models**
  (`get_model_config` filters on `enabled`, and toggle does not unload) and
  had no provider-class guard (re-import can change an entry's provider
  under a loaded provider -- pushing one class's keys into another's
  snapshot). Both closed; pinned in `test_per_request_refresh.py`.
- **Array config fields no longer corrupt comma-bearing elements**: the
  editor edited arrays as a comma join/split, silently rewriting
  `extra_args = ["--tensor-split", "3,1"]` into a different argv on the
  next edit. Arrays now edit one element per line (mono textarea).
- Editor section partition made exclusive (a load_time_only + ui:advanced
  field would have rendered twice with duplicate ids) and the Advanced
  title only says "(requires reload)" when every field in it does.
- The panel's save-outcome note is written into the live region AFTER
  mount, so it actually announces to assistive tech; the row marker is
  visual-only (a freshly inserted role=status node never announces).
- The mid-stream model-switch comment + CHANGELOG no longer claim an
  attribution guarantee the code does not establish (see the corrected
  v1.55.0 entry).

## [1.55.0]

### Added

- **gguf: the three flag families the backend design doc decided to expose
  and that never landed** (found by a coverage audit of GGUFModelConfig vs
  the pinned llama-server build; every already-declared spelling matched, so
  these are the only gaps): `cache_type_k`/`cache_type_v` (`-ctk`/`-ctv` KV
  cache quantization -- usually the better first lever for a KV headroom
  problem than expert offload), `n_cpu_moe_draft`/`cpu_moe_draft`
  (`-ncmoed`/`-cmoed`, the drafter's half of the residency budget) and
  `spec_draft_n_min` (`--spec-draft-n-min`, third member of the interacting
  spec tuning family). All requires_reload; offload/KV fields ui:advanced;
  argv spellings pinned by the existing metadata test (samples added).
- v3 Models page: a persistent per-row "config changed — reload to apply"
  marker after a reload-required save on a loaded model, cleared only by a
  reload or unload. The panel-local note dies with the panel; this is the
  guard against "I set ctx_size and nothing happened".

### Fixed

- **`per_request` defaults now actually reach a loaded model.** A provider is
  built with a snapshot of its config dict and reads per_request defaults
  (default_sampler, enable_thinking, temperature, vision_tokens, ...) from it
  at request time -- so a PATCH to one of them reported "no reload required"
  while the loaded process kept serving the old default: the stale-snapshot
  lie the effect classification exists to prevent, relocated into the
  per_request bucket, and rendered as a false "Applies immediately" by the
  class's first real consumer (the new editor). `reload_config()` now pushes
  per_request keys from the fresh config into every loaded provider;
  requires_reload keys deliberately stay snapshots (the reported reload is
  their real cost). Pinned by `tests/unit/test_per_request_refresh.py`.
- `ModelUpdateRequest` is now `extra="forbid"`: a config key sent at the top
  level (`{"ctx_size": ...}` without the `config` wrapper) used to validate,
  get silently ignored, and return 200 with nothing changed. Now 422, naming
  the key.
- The editor no longer offers mlx's `vision` -- it is a derived mirror of
  `modalities` that config re-derives at load, so an edit silently reverted:
  a dead knob with a live-looking affordance.
- **Admin model responses now report the STORED config keys only**
  (`exclude_unset`), not the resolved model. The resolved dump made every
  default look deliberately chosen: the new editor rendered `n_gpu_layers`
  as an explicit 999 instead of an honest placeholder, and the non-default
  summary chip fired on every row (an mlx entry chipped
  `default_hidden_layer -2 · ... · +4 more` with nothing set). Absent is how
  a default is spelled in models.toml; the wire now says the same.
- v3 chat: switching models mid-stream now stops the in-flight stream BEFORE
  `model_id` changes hands, so the old model stops generating the moment the
  user switches away. (Correction, same day: this does NOT settle
  attribution -- the partial still persists asynchronously into the
  re-labelled conversation, and messages carry no model column. Honest
  per-message attribution is G5 in the switching design, deferred to the
  next `_SCHEMA_VERSION` bump.)

## [1.54.0]

### Added

- **v3 Models page: per-model config editor** (`js/model-config.js`), the first
  consumer of `GET /v1/admin/model-options` -- and therefore the first thing
  that renders the six effect classes distinctly rather than collapsing them.
  Every control is generated from the option schema (type, bounds, enum,
  default, `arg` spelling), so a new backend config field appears in the UI
  with no frontend change. Grouped by effect: immediate / requires-reload /
  advanced (collapsed `<details>`; `extra_args` now declares `ui:"advanced"`
  at the field, its comment already said a UI must not make it casual) /
  fixed-with-reason (disabled). Booleans and enums render as tri-state selects
  whose first option is `default (X)`; empty input = unset; a cleared field
  PATCHes explicit null (the wire spelling of reset-to-default). The dirty
  note names what a save will do PER CLASS before the click -- never a reload
  cost for a live field, never silence about one that needs it -- and after a
  reload-required save on a loaded model the panel offers an armed-confirmed
  "Reload now" (unload + `load?warm=true`, per-row busy). gguf's
  host/port/server_binary/startup_timeout_s are deliberately not rendered
  (design doc §7: nothing good comes of a per-model binary picker in a web
  form). Rows with non-default load options carry a mono summary chip
  (`ctx_size 65536 · spec_type draft-mtp`) so a differently-configured model
  says so on the list.
- Three E2E checks (`tests/e2e/suites/pages.mjs`): the rendered field set is
  asserted against the live option schema (schema-driven is the feature, so
  schema-driven is the check); the PATCH round-trip is asserted on the wire --
  typed integer on save, explicit null on clear -- with the request
  intercepted, because the E2E server runs on the REAL models.toml and a
  landed PATCH would rewrite it (and drop its comments); the open panel is
  overflow-checked at phone width.
- Design record: the load-options surface, fit-meter arc, and model-switch
  hardening are `internal/research/expert_offload_design_frontend.md`; what
  shipped here is its schema-form half. The fit meter (backend ask #2) and the
  editorial field groups/hints stay open.

## [1.53.0]

### Added

- **`GET /v1/admin/model-options`** -- every settable default per provider
  (mlx 25, mlx_embedding 2, gguf 24), each with the effect class that says WHEN
  a change takes effect, plus type, bounds, enum choices, default, and the
  `arg` spelling. Derived from the provider config classes via
  `model_json_schema()`, so a new field appears without touching the route.
  Deliberately NOT under `/v1/admin/models`, whose `/{model_id:path}` would
  capture it as a model id. This is the first consumer that can distinguish
  all six effect classes: the reload set collapses them to a binary and the
  import allowlist to "not identity", so until something reads `effect` per
  field, a misclassification is invisible.
- `load_time_only` fields now carry a `reason`. They render disabled, and the
  class genuinely cannot imply why -- `max_queue_depth` is fixed because it is
  process-wide, `port` for an unrelated reason.
- **`spec_draft_p_min`** (`--spec-draft-p-min`) on GGUFModelConfig, previously
  unreachable. The two spec levers INTERACT and the interaction inverts, so a
  one-dimensional `n_max` sweep finds a different and wrong optimum -- tune
  them together. Justified because the tuned setting wins or ties everywhere
  measured, but the SIZE of the payoff is prompt-length dependent and the
  numbers only mean anything with that condition attached: on gemma-4 12B MTP,
  tuned beats the shipped default by +14.7 points at a ~30-token prompt. The
  long-context comparisons that would say whether that survives are SUSPENDED
  -- those runs hit a llama-server draft-memory-sizing warning and an
  unexplained 3.4x `--ctx` effect -- and every number here is temp 0, which is
  not a regime anyone serves. So the field is justified because the setting is
  reachable and harmless, NOT because a payoff has been established. No
  defensible global default: its sign differed between the models looked at.
  (Figures deliberately omitted -- every performance number produced that day
  was later withdrawn; conditions live in internal/research/.)
- **Expert offload**: `n_cpu_moe` (`-ncmoe`), `cpu_moe` (`-cmoe`, a bare flag)
  and `override_tensor` (`-ot`). On unified memory these do not shrink RAM --
  they move bytes out of the Metal working set, and that math onto CPU cores.
  Unmeasured here; consider `-ctk/-ctv` KV quantization first for a headroom
  problem. `-ncmoe` past the layer count is a silent no-op, not an error.

## [1.52.0]

### Added

- `tests/unit/test_gguf_argv_matches_metadata.py`: the `arg` spelling a field
  declares must be the flag `_build_args` actually emits. This is the third
  leg of the same drift -- the metadata is what a UI and any derived emitter
  read, the builder is what the process gets, and nothing tied them together.
  It immediately caught one: `draft_model_path` declared `--model-draft` while
  the provider emits the `-md` alias. Same behaviour, different command line,
  so the metadata is now the spelling actually used.
- `heylookllm import` now VALIDATES every entry against the provider config
  class before writing `models.toml`. It validated nothing, so one mistyped
  `--override` (`ctx_sze=8192`) produced a successful-looking import and a
  server that then refused to start -- the failure surfacing at config load,
  far from the command that caused it, with no indication which entry was at
  fault. The error now names the entry, the bad key, and the settable keys for
  that provider (derived from the class, not a second hand-written list).
- llama-server spawn logs a warning when any `LLAMA_ARG_*` is set in the
  environment. Verified against llama.cpp's parser: a CLI arg WINS over its env
  var, so anything heylook passes is safe -- but a flag it does not pass gets
  set silently, and the running process then differs from what models.toml and
  the admin API report. Surfaced rather than stripped, since someone may be
  using it deliberately.

- **Every provider-config field now declares WHEN a change takes effect**, as
  `json_schema_extra={"effect": ...}` on the field itself, in one of six
  classes: `identity`, `requires_reload`, `load_time_only`, `applies_live`,
  `per_request`, `descriptive`. Field-local on purpose -- every drift this
  replaced existed because the fact lived somewhere other than the declaration
  it described. `arg` alongside it carries the llama-server spelling.
- `RELOAD_REQUIRED_FIELDS` and the gguf import allowlist are now DERIVED from
  that metadata rather than hand-maintained, and the reload set is
  **provider-aware** (`reload_required_for(provider)`), which one shared
  frozenset could never be.
- `_validate_effect_declarations()` runs at IMPORT and refuses to start with an
  unclassified or misspelt field, naming it. A test only fires when the suite
  runs; this makes shipping one impossible.
- `tests/unit/test_config_effects.py` + `test_config_effects_adversarial.py`
  (39 tests): completeness, partitioning, and mutation-derived guards.

### Fixed

- **Changing `ctx_size` (or any gguf load-time flag) on a loaded model reported
  "no reload required" and kept serving the old argv.** The single
  `RELOAD_REQUIRED_FIELDS` frozenset was MLX-shaped and named no gguf field at
  all; it also still listed `supports_thinking`, removed from MLXModelConfig in
  v1.46.0. Now derived per provider, so this class of drift is unrepresentable.
- The gguf import path silently dropped `n_gpu_layers_draft`, `cache_ram_mb`,
  `load_mode`, `sleep_idle_seconds` and `enable_thinking` -- its allowlist had
  drifted from the config class.
- **Clearing a config field back to its default 500'd.** `null` is how every
  optional load option spells "inherit the default", but nested nulls survive
  `exclude_none=True`, pass pydantic validation (`Optional[...] = None` is
  legal), and only fail at `tomli_w` with a `TypeError` the route did not
  catch. An explicit null now removes the key -- absent IS how a default is
  spelled on disk -- and a serializer refusal returns 400, not 500.
- `get_field_reload_info` filled one dict from two loops with the
  hand-maintained `RUNTIME_CHANGEABLE_FIELDS` last, so it silently overrode the
  derived answer -- reporting a spawn-time flag as a live knob. The derived set
  now wins. Latent (the sets are disjoint today) and found by an adversarial
  pass, not by inspection.
- A misspelt effect (`"requires-reload"`) used to get its own bucket, leave the
  unclassified set empty, pass every completeness check, and drop the field out
  of the reload set -- reintroducing the exact bug by one character. An
  unrecognised effect is now treated as unclassified, not as a new category.

## [1.51.0]

### Added

- **llama.cpp joins `scripts/update_deps.py`, which is now the one place any
  upstream moves.** It clones (blobless), checks out, builds `llama-server`,
  verifies the binary, and writes a `heylook-build.json` manifest recording
  rev + flags + version. `uv sync` installs the Python packages but cannot
  build C++, so a llama.cpp bump is always an explicit run -- never a side
  effect of sync. The script prints the `HEYLOOK_LLAMA_SERVER` export line and
  warns when that variable already points at a different binary (in which case
  the server keeps using the old one).
- **Upstream channels in `[tool.heylook.deps]`.** Each upstream runs on
  `stable` (PyPI for the Python packages, newest `b<N>` release tag for
  llama.cpp) or `latest` (branch tip, pinned to the exact commit). `channel` is
  the project default, `overrides` is per package, and `--channel` changes and
  persists the decision -- pyproject always states what is actually installed.
  mlx-lm/mlx-vlm are held on `latest` by override, per the release-starvation
  posture in `docs/architecture/ecosystem_strategy.md`. `[tool.heylook.deps.git]`
  keeps the git origins so a stable/latest round trip is one flag.
- New flags: `--all`, `-y/--yes`, `--rebuild` (rebuild the pinned rev),
  `--clean`, `--lto`, `--openmp`, `--ui`, `--jobs`. `--release` is gone,
  replaced by `--channel stable`.
- **`scripts/gguf_probe.py` can attach a LoRA and A/B it.** `--lora` /
  `--lora-scale` ride `extra_args`, the same raw passthrough a models.toml
  gguf entry uses, so a flag proven in the probe transfers to a server config
  verbatim. `--lora-ab` toggles scale over llama-server's
  `POST /lora-adapters` between two otherwise identical runs sharing one model
  load, reporting the tok/s, draft-acceptance and output deltas -- identical
  output proves the adapter did nothing, so repeat at a pinned seed. An
  acceptance collapse means the draft path is unadapted, which is the thing to
  know before pairing a LoRA with spec decode. The probe also now prints the
  llama-server binary it resolved, since more than one build can exist on a
  machine and it silently inherits `$HEYLOOK_LLAMA_SERVER`.

### Security

- **No upstream moves unless you name it.** The script previously had a default
  package set, so a bare run fetched and pinned whatever mlx-lm and mlx-vlm
  HEAD happened to be -- unreviewed upstream code, no naming, no confirmation,
  no visibility into what changed. Now a bare run only *reports* the current
  pins (no network, no writes). Naming packages resolves the targets and prints
  a plan first: old -> new, a GitHub compare link so the diff is one click
  away, and an explicit note when a bump pins unreviewed code. Then it asks.
- `-y/--yes` is **required** when stdin is not a terminal, so an automated
  caller cannot drift the pins as a side effect of running the script.
- All named packages are resolved before any is applied, so a multi-package run
  cannot half-land before the whole plan has been seen.
- The build manifest records a `sha256` of the produced binary, so "is the
  llama-server I am running the one I built" is answerable, plus the
  `effective` cmake cache values (not just the requested args) so it describes
  the binary rather than the wish.
- Writes into a missing `[tool.uv.sources]` table are no longer silently
  discarded. `.get(...)` chaining returned a throwaway dict, so the script
  could report a pin it never wrote and then lock against a floating source --
  reachable because the stable channel deletes entries and can empty the table.

### Fixed

Two pre-existing `scripts/gguf_probe.py` bugs, surfaced by running the new
LoRA A/B against Qwen3.6-27B:

- **The probe aborted on Qwen3.6 before generating.** The apply-template diff
  raised `StopIteration`: `zip(on, off)` stops at the shorter string, so when
  one template is a strict *prefix* of the other (Qwen3.6 appends rather than
  edits) it yields no differing pair and the bare `next()` raised. Falls back
  to the common-prefix length.
- **The draft-acceptance grep reported another model's numbers.** Every run
  reuses one log filename and the provider appends, so a run that generated no
  acceptance lines of its own confidently printed the *previous* probe's. Now
  records the file offset after load and reads only that run's tail. Any
  acceptance figure taken from a session that probed more than one model is
  suspect.

Found by `/code-review high` on this changeset, all pre-release:

- **A fresh clone could never build.** Build was gated on the pin having moved,
  so with pyproject already pinning the newest tag, the command `setup.sh`
  recommends printed "already current" and exited without cloning or building.
  Gated on artifact state now (binary present *and* built from the pinned rev),
  so a missing or stale binary plans a build and says which it is.
- **The apply loop could build when the plan said it would not.** `--all` with
  another package moving ran the llama.cpp build unconditionally. Plan and
  apply now share one `plan_acts()` predicate, so consent matches action.
- **`--openmp` silently produced a non-OpenMP binary** -- ggml warns rather
  than failing when OpenMP is missing, and AppleClang needs explicit flags that
  `-DOpenMP_ROOT=` does not supply. The flag now passes the working incantation
  and verifies `GGML_OPENMP_ENABLED` in the cache, refusing to build otherwise.
  This one mattered: it would have quietly invalidated the expert-offload A/B
  the flag exists for.
- `--all --channel X` set three per-package overrides instead of moving the
  project default, leaving the file claiming a channel nothing used.
- `--branch` was applied to every named package, so `--all --branch main` died
  on llama.cpp (whose branch is `master`) after resolving the others. Now
  restricted to a single named package.
- The dirty-tree guard counted untracked files, so a stray `.DS_Store` blocked
  a checkout with a message insisting it was "real work" that `git stash` would
  not clear. Blocks on tracked modifications only; untracked files are reported
  and carried across.
- `-G Ninja` was injected at an existing build dir configured with another
  generator, a hard cmake error surfacing as a bare "command failed (1)". The
  cached generator is reused when one exists.
- "nothing changed; uv.lock left as-is" could be false after the stable path
  had already run `uv lock`. Lock writes are tracked separately from pyproject
  writes and reported honestly.
- `print_status` never read the build manifest, so it printed the pinned rev
  next to a binary that might have been built from something else. It now flags
  the mismatch and shows the effective OpenMP/LTO state.
- Restored the ability to bump ordinary PyPI dependencies, dropped in the
  rewrite. `transformers` is the live case, and `--pin` now also keeps
  `[tool.uv] override-dependencies` in step with the floor -- pyproject's own
  comment says the two are kept equal, and nothing enforced it.
- Python packages are applied before the llama.cpp build, so a failing build
  cannot strand a half-applied pyproject.

### Changed

- Build settings for llama-server are chosen for a Metal-bound Apple Silicon
  host and documented with their rationale in `scripts/README.md`: static,
  native, Metal with embedded metallib, Accelerate + Apple BLAS, ccache; no
  LTO (reaches only CPU-side glue, costs a multi-minute link, defeats ccache,
  and upstream enables it on no platform), no OpenMP (ggml's own threadpool
  does the same decomposition with the same affinity/priority handling, and
  macOS has no libomp -- so this is the path every upstream mac binary already
  runs), no WebUI (re-provisions from a network bucket on every build; the
  provider drives llama-server over HTTP). `GGML_METAL_NDEBUG` deliberately
  stays OFF: it compiles out the "allocated size is greater than the
  recommended max working set size" warning, which is the ceiling that
  actually refuses loads here.
- **llama.cpp is not vendored and is not a submodule.** No upstream source or
  build output lives in this repo, tracked or untracked, so nothing can be
  committed, packaged into an sdist/wheel, or shipped, and there is no
  submodule to initialise or forget. The checkout goes to a fixed directory
  under the user's home directory (`.heylook/llama.cpp`) -- the same path on
  every machine, so docs and errors can name it. `dir` in
  `[tool.heylook.llama-cpp]` relocates it (relative paths resolve against the
  repo root); `$HEYLOOK_LLAMA_CPP_DIR` overrides both. The plan prints the
  destination before anything is written.
- The script refuses to check out over a dirty tree rather than clobbering
  local work, and discards a build tree that was configured for a different
  source path (moving a checkout otherwise leaves cmake hard-erroring on a
  cached absolute path, with no hint that deleting the tree is the fix).

## [1.50.2]

Post-review fixes (`/code-review high` on v1.50.0-.1). All six findings taken.

### Fixed

- **The Models page has never once shown a load failure.** Every error it can
  raise was painted and then wiped ~200ms later: the handler writes its
  failure, then refetches the model list to update badges, and that refetch
  cleared the status area on success. The new `warm_error` branch inherited
  the same fate, so a model could sit there marked "Loaded" with its warm-up
  generation silently failed. Internal refetches now pass `keepStatus` --
  a refresh succeeding says nothing about the action that triggered it. Its
  OWN failure still reports, since that is news.

  Guarded by an E2E check that asserts the message survives *after* the row
  re-renders (asserting immediately passes even with the bug), and
  mutation-verified: reverting the one-line fix turns it red.
- **Every MLX scan row reported `supports_thinking: null`.** Only the gguf
  entry builder writes that key -- `MLXModelConfig` actively rejects it -- so
  the field's own description ("read from the model's own chat template") was
  false for half the results, and the page disagreed with itself: a Qwen3 MLX
  dir showed no thinking before import and `thinking` in its capabilities
  after. MLX rows now answer from the same template probe the capability
  surface uses.
- **A gguf entry could no longer ask for thinking on by default.** Making
  unset mean OFF (v1.50.0) is right, but `GGUFModelConfig` is `extra="forbid"`
  and had no `enable_thinking` field, so a model that previously inherited its
  template's thinking-ON default had no way to get it back short of
  `default_sampler = "thinking"` -- which drags `presence_penalty = 1.5` along
  with it. The field exists now (`None` = unset = off), distinct from
  `supports_thinking`, which only describes capability.
- Models page empty state pointed at a `"Scan HF cache"` button that this
  release renamed, and omitted the folder scanning it exists to expose.
- `docs/architecture/mlx_provider.md` still described `_resolve_enable_thinking`
  as a request-vs-config resolution -- exactly what v1.50.1 deleted. It now
  records the single-owner rule and why the old shape drifted.
- Duplicate `@pytest.mark.unit` on `TestMLXPromptSideMatchesReportedThinking`.

## [1.50.1]

### Fixed

- **A named sampler that turned thinking on built a thinking prompt and armed
  a content-state parser.** The two readings of one decision had stopped
  taking the same input: the prompt side reads the CASCADE OUTPUT, the parser
  side read the RAW request -- differing by the entire sampler layer. So
  `sampler="thinking"`, or a model whose `default_sampler` is thinking, split
  them. On a `prefills_thinking` template (Qwen3.5 pre-fills an unclosed
  `<think>`) the model's output starts inside the block, so a content-state
  parser routes the whole reasoning trace into `content`. Same failure as
  v1.34.64, reachable again through a different door.

  Measured on `Qwen3.5-0.8B-MLX-8bit`, `sampler="thinking"`, same prompt and
  budget: before, `thinking=0ch content=275ch` with the content literally
  beginning `"Thinking Process:\n\n1. **Ancede..."`; after,
  `thinking=929ch content=0ch`. Confirmed on `Qwen3.5-27B-8bit-mlx`.

  A shared resolver was supposed to prevent exactly this and did not, because
  a shared function cannot fix callers that hand it different arguments. The
  fix is a single OWNER instead: `BaseProvider.effective_thinking(request)`
  derives it from the shared cascade, and nothing re-derives it.
  `api.py` and `messages_api.py` now ask the provider (messages resolves it
  once where the converted ChatRequest still exists, since its handlers only
  carry the MessageCreateRequest, whose raw `thinking` field is missing the
  sampler layer).

  `reasoning_parser.resolve_enable_thinking` and `effective_thinking_flag`
  are DELETED rather than fixed. With the flag resolved once from the cascade
  there is no absent case left to default -- which is what made their
  absent-key fallback (an arbitrary `True`, inconsistent with the cascade's
  own `False`) unanswerable on its own terms. `mlx_provider._resolve_enable_thinking`
  is now a plain read of the effective request.

  Pinned as a PROPERTY over the request/config cross-product (the divergence
  lived in specific combinations, which is what an example-per-case test
  missed), with a guard asserting the matrix is not all one value.

- **The contract tests' fake providers duck-typed `BaseProvider`.** Adding an
  obligation to the provider contract turned every route test into a bare 500
  with no hint of the cause. `FakeProvider`, `PreSplitProvider` and
  `_FailingProvider` now SUBCLASS `BaseProvider`, so a contract addition
  arrives with its default in place and only a deliberate difference has to
  be written down. Same drift class as the fake-provider file deleted in
  v1.32.1.

### Known, pre-existing, unrelated to the above

- `Qwen3.5-27B-8bit-mlx` collapses into repetition on the thinking path at
  large budgets (~13k chars of `ejahterejahter...` at `max_tokens=2500`;
  coherent at 600). Reproduces with a plain `enable_thinking=true`, whose
  code path is byte-identical before and after this change, so it is not
  introduced here -- but it is now VISIBLE in the thinking channel where it
  used to be mislabelled as content. Worth its own look.

## [1.50.0]

Frontend v3 catch-up against the backend changes since 2026-07-26
(v1.45.0-1.49.9). Everything here is live-verified against a real server;
backend suite 1310 green, E2E chat 40/40 + pages 39/39.

### Fixed

- **Thinking could not be turned off on a gguf model.** v3's toggle is binary --
  it sends `enable_thinking: true` or omits the key, never `false` -- so what an
  omitted key resolves to IS the off state. MLX resolved it to an explicit
  `False`; the gguf provider only sends `chat_template_kwargs` for a non-None
  value, so an omitted key handed llama-server's `--jinja` run to the GGUF's own
  template default, which is thinking-ON for gemma-4 / Qwen3.6 / DeepSeek-V4.
  One checkbox, opposite meanings per engine. v1.49.6 made this reachable by
  teaching import to detect the thinking capability, so the control started
  appearing for gguf models -- and lying.

  Fixed in the shared cascade rather than per-provider:
  `samplers.resolve_effective_sampling` now materializes the effective switch
  unconditionally instead of only when the thinking overlay fires, so both
  engines send an explicit bool. The MLX effective request is byte-identical
  before and after (its config always carried the key). Live-verified on
  `google_gemma-4-E4B-it-qat-q4_0-gguf`: omitted and `false` both produce no
  thinking, `true` still does.

  A tri-state (auto/on/off) making the template default reachable again stays
  the deferred Phase-3b design item.
- **The Models page could not see any locally downloaded model.** It posted
  `{scan_hf_cache: true}` with no `paths`, so the entire GGUF import arc
  (v1.44.0 detection through v1.49.6 -- shard pinning, four-family drafter
  pairing, header reads, per-quant variant dirs) targeted folders the UI had no
  way to name. The scan section now takes a folder list (comma/newline
  separated, persisted locally) plus an HF-cache toggle, and refuses a scan with
  both sources off rather than round-tripping a request that can only return
  nothing.
- **Scan results claimed `vision: false` for every model, of both providers.**
  `_raw_to_scanned` read `config["vision"]`, a key only the MLX entry builder
  ever wrote -- and derive-at-load (v1.47.0) stopped writing it there too.
  Modalities are now derived: gguf states them in its entry (read from the
  projector's own header), everything else goes through the same shared detector
  the config validator uses. Deriving is right here specifically because a scan
  result is reporting, not stored config.
- **`GET /v1/admin/models` reported capabilities that were always empty.** It
  read the stored `ModelConfig.capabilities` OVERRIDE rather than deriving, so
  the Models page listed no capabilities for anything while the chat page, one
  endpoint over, gated its whole UI on them. Inference moved to a new
  `capabilities.py` shared by both surfaces, with the explicit-override
  short-circuit stated once.
- **E2E: the thinking check failed on legal model output.** It waited for a
  persisted assistant row, but with thinking on at temperature 1.0 this model
  sometimes spends the whole token budget inside the thinking block and emits no
  content (measured: 1 run in 6, `finish_reason=length`, 1513 chars of thinking
  and 0 of content). `finishStream` deliberately drops an empty completion, so
  nothing persists and the wait timed out -- a red bar for exactly the outcome
  the suite's README says never to flake on.

### Added

- **Scan results report what the importer actually found**: `modalities`,
  `supports_thinking`, a paired `draft_model_path` and the `draft_spec_type` it
  REQUIRES. That last one was written only to the server log, so the one surface
  where someone decides whether to import a model could not see it. It stays
  reported and never applied -- import pairs a drafter's path but leaves
  `spec_type` unset, because whether speculative decoding pays off is a
  per-model measurement.
- **`ScannedModelListResponse` is now the /scan route's `response_model`.** The
  response model sat unreferenced for months while the dataclass grew, which is
  how the declared contract and the shipped payload drifted apart silently.
- **The Models page Load button sends `?warm=true`** -- the server-owned
  readiness call (v1.38.0) that `dev_server.sh` and the E2E harness already use
  -- and reports the warm timing. "Loaded" now means ready, not merely resident.
  A warm failure reports as a note, not an error: the model is loaded either way.

### Known gap (owner action, not code)

- `modalities`/`supports_thinking` are STORED on a gguf entry at import, so
  entries written before v1.49.4/.6 under-report: every local gemma-4 GGUF
  reports `["chat","vision"]` from its entry while a fresh scan of the same files
  derives `["text","vision","audio"]` + thinking. v3 gates modality UI on
  capabilities, so those models show no thinking toggle and no audio attach. A
  stored value is indistinguishable from a deliberate override, so nothing
  rewrites it for you -- re-import the affected entries, or drop those two keys.
  Deriving them at load (as the MLX path does) is the durable fix and belongs
  with the Wave 1 derive-at-load arc.

## [1.49.9]

### Added

- **`gguf_probe --temp`**, and an honest statement of what `--seed` buys.
  v1.49.8 implied a pinned seed made runs comparable. It narrows the gap; it
  does not close it. Speculative decoding changes the verify batch
  composition per eval, which perturbs floating-point reductions, so two
  drafter configurations can diverge in generated text even at a fixed seed
  -- and acceptance rate tracks text. `--temp 0` removes sampling as a
  variable entirely and is the right control for a drafter A/B; without it,
  single runs are samples and conclusions need repeating across seeds (which
  is why the DSpark n-max result was reported as sign-only, over 3 seeds).

  The script's own usage block had also gone stale -- neither `--draft`
  (v1.49.8) nor `--seed` appeared in it, and the `gguf-probe` skill points
  readers at that header.

## [1.49.8]

### Added

- **`gguf_probe --seed`, pinned by default (1234).** The script exists to
  compare runs, and unseeded the noise swamped the signal: each run generates
  different text and draft acceptance tracks content, so two
  nominally-identical DSpark runs came out 11.7 acceptance points apart --
  wider than the Q8_0-vs-BF16 effect being measured. `--seed -1` restores
  random sampling (llama-server's own documented value for it) for variance
  checks. See v1.49.9 for what seeding does NOT fix.
- **`gguf_probe --draft`** to override the paired drafter, for A/B-ing two
  builds of the same speculative module (DeepSeek-V4 ships its dspark module
  as both Q8_0 and BF16, with the BF16 in a `dspark/` subdir that root-level
  sidecar pairing cannot see).

## [1.49.7]

### Fixed

- **`ram_report` refused loads that fit** -- and since v1.49.2 wired it in,
  `dev_server.sh` refused to start them. It gated on `free + inactive`, which
  is the wrong number on macOS: both llama.cpp and MLX load weights through
  **mmap**, so those pages are clean and file-backed and the OS evicts them
  on demand -- but it parks recently-touched file pages in the ACTIVE queue,
  where `free + inactive` cannot see them. Caught in practice this session:
  the check said a 138 GiB load would not fit against "125 GiB available",
  while 154 GiB was file-backed and only 27 GiB anonymous; the load then ran
  with zero swapins and zero swapouts.

  The gate is now `total - anonymous - wired` (`reclaimable_gb`) -- anonymous
  pages have no backing file so they can only be compressed or swapped, never
  dropped, and wired pages cannot even be that; everything else is
  negotiable. The conservative figure is still reported for context, clearly
  labelled as not the gate. Falls back to the old figure when `vm_stat`
  counters are unavailable (i.e. off macOS), and refuses to compute from a
  partial pair rather than guessing the missing term.

  Now covered by `tests/unit/test_ram_report.py` -- the script became a gate
  in v1.49.2 with no tests behind it, which is how this shipped.
- **mmproj lookup now matches the drafter's.** v1.49.5 taught sidecar pairing
  to search the repo root for a per-quant variant folder, but only for the
  drafter. A multimodal model shipped as quant subdirectories keeps its
  projector beside them too, and a dropped projector is the worse failure of
  the two: the model imports text-only and vision simply never works, rather
  than failing loudly.

## [1.49.6]

### Added

- **GGUF thinking capability is now detected, not hand-set.** `supports_thinking`
  carried the note "no cheap GGUF-metadata probe yet" -- reading the header is
  that probe. Import checks the GGUF's OWN embedded `tokenizer.chat_template`
  for `enable_thinking`, reusing `template_info._ENABLE_THINKING_PATTERN` so
  the two engines share one rule rather than drifting apart. Left unset when
  there is no template to judge (an MTP/drafter head has none) rather than
  asserting a false. The flag stays overridable by hand.

  Effect on real entries: gemma-4 GGUFs now infer
  `["chat", "vision", "audio", "thinking"]` where they previously inferred
  `["chat", "vision"]` -- audio from the projector flags (v1.49.4), thinking
  from the template.

## [1.49.5]

### Fixed

- **GGUF import: per-quant subdirectory layouts.** HF repos shipping many
  quants of one large model put each in its own subdirectory, so a
  structure-preserving download nests the weights and leaves sidecars at the
  repo root:

      <repo>/dspark-<model>-Q8_0.gguf
      <repo>/UD-IQ4_XS/<model>-UD-IQ4_XS-00001-of-00004.gguf

  Two things broke there. The entry took its id from the directory, yielding
  `UD-IQ4_XS` -- uninformative, and colliding across every model quantised
  the same way. And the drafter went unpaired, because sidecar lookup only
  ever searched the model's own directory. Both now key off one shared test
  (`_is_variant_dir`): a folder whose name is already spelled out in the
  weight file's name is labelling a quant, not naming a model, so the id
  comes from the file and sidecars are also looked for one level up. The
  test compares against the SHARD-STRIPPED name and requires a PROPER
  substring, so `foo/foo.gguf` and `foo/foo-00001-of-00002.gguf` stay
  ordinary model directories -- otherwise they would adopt an unrelated
  sibling's drafter from the parent. Directory-named repos are unaffected,
  so no id already in a models.toml moves.
- `scripts/ram_report.py`: `--path` now passes the primary to the drafter
  picker (so a variant layout's sidecar is counted), and every printed unit
  is labelled GiB. The arithmetic was always `1024**3`; calling it GB
  understated each figure by ~7% and invited exactly the decimal-vs-binary
  confusion that makes vendor sizing tables look wrong against a Mac.

## [1.49.4]

### Added

- **`gguf_metadata.py` -- read the GGUF header instead of guessing from the
  filename.** A stdlib-only KV-header reader (no new dependency; the gguf
  provider's no-extra-deps property is deliberate). It walks only the header
  and stops as soon as the requested keys are found, skipping values it was
  not asked for -- so a multi-MB tokenizer vocab costs one seek, not a list
  of a hundred thousand Python strings. Verified key-for-key against
  llama.cpp's own `gguf-py` reader on every GGUF in the local modelzoo.

### Fixed

- **GGUF import: omni projectors lost their audio tower.** Modalities were
  inferred from "an mmproj sidecar exists" -> vision, with a standing note in
  `model_importer` that audio "would need reading the GGUF's own metadata
  (out of scope here)". The projector declares the two SEPARATELY
  (`clip.has_vision_encoder` / `clip.has_audio_encoder`), and gemma-4's sets
  both -- so every gemma-4 GGUF imported as vision-only. Detection now reads
  those flags. This also makes `api.py`'s `audio` capability branch reachable
  for auto-imported models; it was correct code that nothing could trigger.
  An unreadable projector still falls back to the old presence heuristic
  rather than stripping vision from a model that plainly has one.
- **GGUF import now reports which `--spec-type` a paired drafter needs.**
  `spec_type` stays deliberately unset (whether speculative decoding pays off
  is a per-model measurement), but WHICH type a drafter requires is a fact
  about the file, and guessing it wrong is a load failure. Derived from the
  prefix llama.cpp itself resolves siblings by -- notably `dspark-` vs
  `dflash-`, which share `general.architecture == "dflash"` and are
  distinguishable only by name.

## [1.49.3]

### Added

- **GGUF provider: memory + lifecycle knobs that were unreachable.** Four
  new `GGUFModelConfig` fields, each reaching `llama-server` argv:
  - `n_gpu_layers_draft` (`-ngld`) -- separate GPU offload for the drafter.
    The pair can exceed the GPU budget when the target alone does not: on a
    192 GiB M2 Ultra the Metal residency recommendation is ~161 GiB, so a
    144 GiB target plus a 10 GiB drafter is over it while either alone is
    under. `0` keeps the drafter off the GPU.
  - `sleep_idle_seconds` (`--sleep-idle-seconds`) -- llama-server's own idle
    sleep, which frees the model and KV cache but KEEPS THE PROCESS and
    reloads on the next task. Strictly cheaper than heylook's idle-unload
    (SIGTERM + respawn), so setting it below the effective
    `idle_unload_seconds` gets the cheap recovery first.
  - `cache_ram_mb` (`-cram`) -- prompt-cache budget; llama-server defaults to
    only 8192 MiB.
  - `load_mode` (`-lm`) -- `mmap`/`mlock`/`mmap+mlock`/`dio`.

  All four use `is not None` rather than truthiness, because `0` (drafter off
  the GPU, cache disabled) and `-1` (unlimited cache) are real settings.

- **Sleeping-server wake budget.** A model slept out by
  `--sleep-idle-seconds` reloads before it emits a single byte, and for a
  large model that reload is minutes -- well past the 120 s socket timeout
  that exists to detect a *wedged* server. The provider now checks
  `/props.is_sleeping` (an endpoint llama-server exempts from counting as a
  task, so asking neither wakes it nor resets its timer) and allows
  `startup_timeout_s` for that request only. Without this, enabling idle
  sleep would turn the first request after any idle gap into a timeout.

## [1.49.2]

### Added

- **`scripts/ram_report.py` -- memory pre-flight.** Reports what is holding
  RAM (RSS rolled up by app) and, more usefully, the ceilings that actually
  refuse a load. Total RAM is the wrong number: on a 192 GB M2 Ultra the
  Metal `max_recommended_working_set_size` is ~161 GB, so a 155 GB model
  fits with ~6 GB for KV cache and a 10 GB drafter beside it does not --
  a refusal that is invisible if you only look at free memory. With
  `--model`/`--path` it sizes a model as it is really loaded (whole shard
  SET plus mmproj/drafter sidecars) and checks it against each ceiling;
  `--quiet` gives one line and an exit status. It also reports whether that
  Metal ceiling is the OS default (`iogpu.wired_limit_mb=0`) or a tuned
  value, and prints the sysctl to raise it -- server.py's
  `mx.set_wired_limit` CONSUMES that budget for MLX, it does not enlarge it.
  The working-set verdict is per ENGINE, because the two treat it
  differently: MLX refuses outright above the recommendation, while
  llama.cpp wires through the same `MTLResidencySet` but checks the
  recommendation only as a debug-build warning -- past the line a GGUF model
  still loads and degrades into paging. Reporting one verdict for both would
  call a performance warning a hard failure.
- `dev_server.sh` now delegates its RAM pre-flight to that script. Its
  inline version sized a GGUF entry as the single shard `model_path`
  names, so it would have cleared a 155 GB model as needing 10 GB.

### Fixed

- **GGUF import: multi-shard models picked an unloadable shard.**
  `_pick_primary_gguf` took the largest `.gguf` in the directory, which for
  a split model is never shard 00001 (that one is a small index shard).
  `llama_model_loader` hard-errors on any other shard -- *"model must be
  loaded with the first split"* -- because it derives its siblings from the
  given file's own split index. Sharded entries are now pinned to shard
  00001, and a candidate's weight for the largest-wins rule is the size of
  its whole shard SET, so a 155 GB split model no longer loses to a
  standalone `.gguf` beside it. Surfaced by importing
  DeepSeek-V4-Flash-0731 (5 shards), which resolved to shard 00004.
- **GGUF import: only `mtp-` drafter sidecars were paired.** llama.cpp
  resolves drafter siblings by one prefix per speculative family
  (`common/download.cpp`): `mtp-`, `dspark-`, `dflash-`, `eagle3-`. Knowing
  only `mtp-` left the other three unpaired *and* let them compete as
  primary-weight candidates. All four are now recognised as drafters.
  `spec_type` stays deliberately unset -- pairing the drafter path is not
  the same decision as turning speculative decoding on.

## [1.49.1]

### Changed

- **/simplify pass on the port-move + example-file commits:**
  - `DEFAULT_PORT` constant in config.py -- server.py's two argparse
    defaults, service_manager's install defaults, and the OpenAPI
    `servers` entry all pull from it (the renumbering commit was itself
    the failure mode scattered literals invite). Docs/scripts/tests keep
    prose literals (a Python constant can't reach markdown).
  - `models.example.toml` is now test-anchored: a unit test round-trips
    every entry through the real Pydantic validators, so a schema change
    breaks CI instead of silently rotting the example.
  - docs/architecture/config.md examples de-fattened to the thin-entry
    format (they still showed `vision = true` / `cache_type = "standard"`
    style entries, contradicting the field table below them and the new
    example file); `gguf` added to its provider table.

## [1.49.0]

### Changed

- **Default port moved 8080 -> 1263.** 8080 is llama-server's default
  port, and llama.cpp-ecosystem clients (including llama-server's own web
  UI) probe `localhost:8080` with `GET /props` -- caught live: a restored
  Chrome tab was hitting heylook's 8080 with the `/v1/models` + `/props`
  pair on every startup. Moving off 8080 removes the collision class.
  Swept across CLI defaults, service manager, benchmark/eval/integration
  harness defaults, and all docs (67 references); historical records
  (CHANGELOG, plan notes, archives) keep their original port. The E2E
  harness defaults to 1264 -- it spawns its OWN server and must not
  collide with the daily instance. External clients (shrug-prompter etc.)
  need repointing, or run the server with `--port 8080`.

## [1.48.1]

### Changed

- **Post-review cleanup of the derive-at-load arc** (/code-review +
  /simplify over v1.45.0-v1.48.0):
  - `modality_detect.read_model_config_json` gained an mtime-keyed parse
    cache -- the reader runs inside config validation, and AppConfig is
    rebuilt on every admin write, so one "toggle model" call was
    re-reading every model's config.json from disk.
  - `cache_defaults.weights_size_gb` caches per (path, dir-mtime), so
    idle-unload/LRU reload cycles skip the repeated rglob+stat pass, and
    it is now the ONE byte-summing implementation: model_service's scan
    path calls it, and the importer's dead `_get_model_size` copy is
    deleted (with its orphaned tests; the gguf-bytes claim retargeted).
  - `loader_routing.read_model_type` delegates to the shared cached
    reader instead of hand-rolling its own config.json parse.
  - `ModelService.get_config` constructs only the requested entry
    instead of materializing all N configs per lookup.
  - Dead imports/exports cleaned out of model_importer (json, re,
    get_smart_defaults/available_samplers re-exports) and the leftover
    admin-import `model_info`/`vision` locals removed.
  - Skipped by judgment (single copies, documented belts): the thinking
    fallback dict in the shared cascade, cache_helpers' None->standard
    belt, a GGUF payload-key partition test, and relocating modality
    detection out of the Pydantic validator (the read cache removes the
    cost that motivated it).

## [1.48.0]

### Changed

- **Derive-at-load complete (Wave 1 / 6a done).** The remaining
  materialized copies are gone:
  - **Cache defaults resolve at model load.** `cache_type` is now
    Optional; None = auto, resolved by the new `cache_defaults.py` from
    ACTUAL weight bytes vs machine RAM at load (the import-time copy froze
    that decision against whatever machine/weights existed at import).
    Explicit values and pinned knobs (`kv_bits`/`kv_group_size`) are never
    overridden. `get_smart_defaults` delegates to the same single
    implementation (survives for the admin what-would-it-be surface).
  - **The /v1/admin import route builds thin entries** matching the CLI
    wizard (no vision/smart-defaults/auto template-source materialization;
    caller-supplied description/tags are kept -- operator intent; auto-text
    defaults are not).
  - **GGUF + embedding CLI entries drop auto description/tags** (GGUF
    keeps `modalities` -- no config.json to probe at load). The dead
    `_detect_tags` helper is deleted.

## [1.47.0]

### Changed

- **Derive-at-load substrate, first slice (Wave 1 / 6a).** MLX models.toml
  entries are THIN now: the importer materializes only `model_path` +
  operator intent (sampler/cache flags, explicit `--chat-template`). What
  it used to copy in is derived where it's needed instead:
  - `modalities`/`vision`: detected at config-load time from the model
    dir's own `config.json` via the new shared `modality_detect.py` (one
    implementation for importer and config; a stored value is an explicit
    override and always wins; no config.json -> legacy vision-bool
    fallback). GGUF entries still materialize modalities -- no config.json
    to probe at load.
  - auto-detected `chat_template_source`: no longer recorded; load-time
    auto resolution (template_info.py) applies the identical policy.
  - auto description/tags text: no longer written (display-only noise).
  Existing entries with materialized fields keep their exact behavior.
- **config_tui retired** (+ the import wizard's `--interactive` per-model
  customization branch, + the `questionary` dependency). Its hand-editing
  role is dead under thin entries; operator intent at import is
  `--sampler`/`--override`; richer editing arrives with the Wave 4 admin
  CRUD.

### Notes

- Phase 1 item 8 (path identity + PUT re-import on the admin path) was
  found ALREADY IMPLEMENTED and test-pinned (`test_import_reimport.py`) --
  the plan carried it as open; Wave 1's fold-in reduces to nothing.
- Remaining 6a (next session): cache/kv smart defaults computed at load
  (`cache_type` becomes Optional=auto), GGUF description/tags parity.

## [1.46.0]

### Changed

- **One sampler cascade for all providers.**
  `samplers.resolve_effective_sampling` is now THE cascade; MLX's
  `_apply_model_defaults` (wraps it: cached vendor-layer read + runtime
  cache/spec fields) and the gguf provider's `_build_payload` (calls it
  directly) no longer carry hand-mirrored copies -- including the two
  duplicated thinking fallbacks v1.45.0 introduced. gguf picks up MLX's
  semantics where the mirrors had drifted: a request's named sampler now
  suppresses the model `default_sampler` layer (was: applied then
  overlaid), and an unknown `default_sampler` logs-and-skips instead of
  raising 400 on every request (models validate at startup; a registry
  miss there is post-startup drift).
- **MLX `supports_thinking` config field removed.** It was a manual flag
  triple-shadowed by derived truth (the `enable_thinking` default-on flag,
  the chat-template probe, and the explicit `ModelConfig.capabilities`
  override) -- the same rot class as the dead thinking layer fixed in
  v1.45.0. An MLX models.toml entry setting it now fails loudly at load
  (`extra="forbid"`). GGUF keeps its flag: the template lives inside GGUF
  metadata, nothing cheap to probe pre-load.

### Docs

- Phase 6 plan refinement: derive-at-load, override-only registry --
  stop materializing derived metadata into models.toml instead of
  building merge machinery to preserve it.

## [1.45.0]

### Added

- **Vendor sampling layer.** The effective-request cascade now reads
  `temperature`/`top_p`/`top_k` from the model dir's own
  `generation_config.json` (`load_vendor_sampling`, read once at first
  request, cached on the provider), directly above the global floor.
  Every MLX model gets its vendor's decode tuning without models.toml
  churn: gemma-4 runs 1.0/64/0.95, Qwen3 thinking models 0.6/20/0.95,
  each from its own file. models.toml fields, samplers, and request
  fields all still override. Best-effort: a missing or malformed file
  contributes nothing and never blocks a load.

### Fixed

- **Thinking mode ran with zero repetition control.** The cascade's
  thinking layer was keyed on the model config's `enable_thinking`, which
  nothing sets -- dead code. A request toggling thinking on (the v3
  drawer checkbox) got floor sampling with no presence penalty, which
  looped a gemma-4 MoE thinking trace over two images within one reply.
  The layer is now keyed on the effective switch (request field when
  present, else model config), and the gguf provider gains the same
  overlay in `_build_payload` (request-keyed only; GGUF config has no
  `enable_thinking` field).

### Changed

- **`thinking` sampler slimmed to loop control** (`presence_penalty 1.5`
  + `enable_thinking`). Its old Qwen-tuned `temperature 0.6 / top_k 20 /
  max_tokens 4096` came from before the vendor layer existed and was
  wrong for every other family whenever thinking was on; per-model decode
  tuning now comes from each model's `generation_config.json`. Explicit
  `sampler: "thinking"` users now get their model's vendor temperature
  instead of Qwen's.

## [1.44.7]

### Fixed

- **llama-server subprocesses no longer outlive the server.** The gguf
  provider spawns with `start_new_session=True` (its own process group, so
  unload can kill the whole tree) -- which also means the terminal's Ctrl-C,
  a SIGINT to the FOREGROUND process group, never reached it. Nothing reaped
  it either: lifespan shutdown closed the DB and left every provider loaded.
  Every exit leaked a multi-GB llama-server (observed: two orphans, ~22GB,
  PPID 1). Three layers now: lifespan shutdown calls the new
  `ModelRouter.unload_all()` (ordered before the DB close and guarded, so no
  later failure strands a child), `unload_all()` unloads every provider
  best-effort, and an `atexit` backstop kills any still-registered process
  group for exits that skip the lifespan. Verified live: SIGINT to a running
  server with a loaded GGUF leaves zero orphans.

### Changed

- **Import no longer picks a `default_model`.** It used to stamp
  `imported[0].id` whenever the field was unset, which silently undid a
  deliberately-cleared default on the next scan and made an arbitrary member
  of the batch the routing target.

## [1.44.6]

### Fixed

- **Only model-RESOLUTION failures are 400; load failures stay 500.** v1.44.5
  caught bare `ValueError` around the whole setup block, but `get_provider`
  re-raises load failures too (mlx-lm/transformers raise plain `ValueError` for
  corrupt weights, an unsupported `model_type`, a malformed config.json) -- so a
  broken model on disk answered 400 `model_not_resolved`, telling clients not to
  retry, and lost the operator's traceback. The router now raises a dedicated
  `ModelNotFound(ValueError)` from its two id-resolution sites and the API
  layers catch only that; everything else keeps its 500 and `exc_info`.
- **`default_model = "none"` is treated as unset.** The importer and
  `model_service` write that literal string when a scan finds no models, and it
  is truthy -- so it was routed to as a real model id and every model-less
  request failed with `Model 'none' not found` instead of the actionable "no
  default configured". Coerced to `None` (along with `""`) in `AppConfig`.
- **A stale `default_model` is reported at startup.** Boot used to validate it
  implicitly by pre-warming it; with preload now opt-in, the router warns (and
  still loads nothing) when the default names no enabled model.

## [1.44.5]

### Fixed

- **Model routing failure is now 400, not 500** (`/v1/chat/completions` and
  `/v1/messages`). `router.get_provider` raises ValueError for an unknown or
  disabled model id -- and, now that `default_model` is optional, for a request
  that names no model with no default configured. Both escaped to the generic
  handler as a 500. Providers signal their own failures with typed
  `GenerationFailed`/`InvalidGenerationRequest`, so a bare ValueError from that
  block is always routing.

## [1.44.4]

### Changed

- **Startup model pre-warm is now opt-in.** `default_model` in `models.toml`
  no longer loads a model at boot; it stays what its name implies for
  ROUTING only -- the fallback for a request that names no model, resolved
  (and loaded) on that first request. Explicit `--model-id` is the one thing
  that still pre-warms at startup. Opening the server no longer pins a
  multi-GB model into RAM nobody asked for.

## [1.44.3]

### Added

- `scripts/gguf_probe.py` (+ local `gguf-probe` skill): direct
  llama-server diagnostics for one GGUF model, the layer below
  `dev_server.sh` -- /props modalities, thinking-template on/off/unset
  diff, one-shot generation with tps + draft acceptance, auto-teardown.
  Sidecar pairing reuses the importer's pickers; llama.cpp flag
  knowledge stays in the provider.
- **`.claude/` repo-contract subset now tracked** via a per-FILE
  gitignore allowlist mirrored by the pre-commit hook's ALLOWED_PATHS:
  the model-delegation rule + its two agents, four skills wrapping
  git-tracked runners (test-suite, dev-server, eval-ab, gguf-probe),
  and four repo-invariant hookify rules. Two-agent review before
  publish: personal framing redacted, the test-suite skill rewritten
  (it still described the frontend suite deleted 2026-07-09), and all
  blanket dir negations tightened (a bare `!.claude/skills/` would
  silently track future files). commit-msg leak gate installed
  (path-privacy 0.10.2, chained over the repo's own hook) and
  verified with leak bait.

### Changed

- CLAUDE.md Tests section corrected: `/test-suite` is backend-only
  (no frontend unit suite exists); documented the two-place rule for
  tracking new `.claude` files; `scripts/README.md` indexes gguf_probe.

## [1.44.2]

### Changed

- /simplify pass over the day's range (4 cleanup reviewers; 7 fixes
  applied, 6 deliberate skips recorded):
  - `merge_presplit_thinking` extracted into `reasoning_parser.py` -- the
    byte-identical merge at both non-streaming consume sites (chat +
    messages) now lives once, next to `parse_reasoning`.
  - api.py streaming grew the same `note_delta` bookkeeping extraction
    messages_api's translator already had -- the pre-split branch and the
    parser-delta loop share one timing/counter routine.
  - chat.js: `addImages`/`addAudio` copy-paste twins collapsed into one
    `addPendingFiles(ctx, files, kind)` driven by an `ATTACH_KINDS`
    table; `hasImageBlocks`/`hasAudioBlocks` collapsed into
    `hasBlocks(msg, type)`.
  - `GenerationChunk.from_draft` dropped -- dead migration carryover
    (nothing read it; the spec-decode signal is the cumulative
    `draft_tokens`/`draft_accepted`).
  - `_truncate_image_url` renamed `_truncate_b64_field` (it truncates
    audio too now); mid-file `GLOBAL_SAMPLER_FLOOR` import hoisted.
  - Skipped on purpose: importer triple-iterdir (explicit-scan-only
    cost), `absorb()` getattr tolerance (load-bearing for test fakes),
    capability-inference if/elif + registry sync points + MLX audio-guard
    placement + truncation table (all judged right-altitude or
    wait-for-third-case by the altitude reviewer).
  - Verified: backend 1187 green; E2E chat 40/40 (attach flow was
    refactored, so the browser suite re-ran).

## [1.44.1]

### Fixed

- Code-review pass over the day's range (5 parallel reviewers; 2 real
  bugs, both new-in-range):
  - **gguf: model-level `max_tokens` was dead config.** The cascade
    guard (`"max_tokens" not in merged`) could never fire because the
    global floor pre-seeds the key -- a `max_tokens` on a GGUF entry
    silently fell back to 4096. Config now overlays the floor
    unconditionally (floor -> model config -> samplers -> request),
    mirroring MLX; pinned by tests.
  - **Messages API under-reported spec decode.** Both
    `messages_api.py` RequestEvent sites omitted
    `draft_tokens`/`draft_accepted`, so `/v1/messages` traffic read as
    "no drafting" in the perf trends. Fields threaded through; a new
    contract test also pins pre-split thinking on the Messages path
    (the same surface where the omission hid).
- Hygiene from the same review: `AudioBlock` fields use keyword-form
  `Field(default=None, ...)` (house pyright rule); `blockSourceUrl`'s
  comment no longer overclaims a render/wire single source for audio
  (the raw-base64 wire divergence is deliberate); four "tagged by
  mlx-lm" telemetry comments now say "tagged by the engine".
- Docs de-staled: `mlx_provider.md` §4.5 rewritten for the v1.33.0
  raised-exception error contract (`MLXErrorChunk` no longer exists);
  `frontend_v3_spec.md` §4 names the `timing.draft_*` fields and drops
  the "audio UI not yet built" claim; README refreshed for the
  multi-provider reality (GGUF/llama-server, audio input, speculative
  decoding, importer coverage).

## [1.44.0]

### Added

- **GGUF import/scan support** (plan Phase 7e detection layer). The scan
  now recognizes GGUF model dirs and builds `provider = "gguf"` entries:
  primary weight file (largest non-sidecar .gguf), mmproj sidecar paired
  by precision preference (F16 > BF16 > F32; matches "mmproj" anywhere in
  the basename -- google names projectors `<model>-mmproj.gguf`),
  root-level `mtp-*.gguf` drafter auto-paired into `draft_model_path`
  with `spec_type` deliberately UNSET (spec decode is measured
  model-dependent; enabling it is an explicit owner choice), modalities
  text(+vision from mmproj; audio needs GGUF-metadata reading, later).
  HF-format Assistant/drafter checkpoints (`architectures[]` contains
  "Assistant") are recognized and REFUSED before the MLX branch could
  misimport them; `imatrix_*.gguf_file` calibration artifacts are never
  treated as models. GGUF dir sizes are computed (were 0 GB);
  `generate_toml` groups entries by provider section;
  `ScannedModelResponse.provider` widened to
  `mlx | mlx_embedding | gguf` (scan already emitted embedding entries);
  `model_service` validators + default-sampler stamping cover gguf, and
  `import_models` forwards only GGUF-legal fields (the mlx-shaped config
  it used to build would have failed `extra="forbid"`). Verified against
  the real driver fleet: 6 correct entries from 8 dirs, both assistant
  checkpoints skipped.
- **Audio eval tasks** (completes plan 7d): `audio_speech_keywords`
  (committed speech fixture; keywords actually said must surface) and
  `audio_tone_vs_speech` (deterministic stdlib-synthesized sine; tone
  vocabulary required) -- both gated on the `audio` capability. Bank now
  15 tasks; 15/15 live-green on gemma-4-E4B-it-qat-gguf.

## [1.43.0]

### Added

- **v3 audio attach/render** (plan Phase 7d frontend half, re-plan
  2026-07-26 v3 mini-workstream). Chat's attach affordance is now
  CAPABILITY-GATED like the thinking toggle: hidden for text-only models,
  the picker's `accept` list built from the model's `vision`/`audio` caps
  (the previously ungated image button rode along to models that would
  reject it). Audio clips stage as named removable chips (2-clip cap,
  aria-live overflow message), render as `<audio controls>` in bubbles,
  store as Messages-style `audio` blocks, and convert to `input_audio`
  wire parts (raw base64) at send. Send refuses loudly if staged
  attachments outlive a model switch that removed the cap. Edit stays
  text-only for any media message.
- **Draft acceptance on the perf page**: `/v1/performance/profile` trends
  now carry hourly token-weighted `draft_acceptance` (None when nothing
  drafted -- distinct from 0); the perf trends table shows a "Draft acc"
  column only for ranges that drafted; chat's post-stream stats line
  appends `draft N%`.
- Decision recorded (docs/frontend_v3.md): v3 gates ALL modality UI off
  `capabilities`; the descriptive `modalities` field stays deliberately
  unread by the frontend.
- `scripts/dev_server.sh` RAM pre-flight now sizes single-FILE model
  paths (gguf) including mmproj/drafter sidecars.
- Verified: E2E chat suite 40/40 (attach cap, paste staging, image
  round-trip all exercise the new gating); backend suite green.

## [1.42.0]

### Added

- **Audio input content parts** (plan Phase 7d, API layers; gguf models
  only). `/v1/chat/completions` accepts OpenAI-wire
  `{type:"input_audio", input_audio:{data:<raw base64>|url, format?}}`
  parts; the Messages API gains the `AudioBlock`
  (`{type:"audio", source_type, media_type?, data?|url?}`) bridged by the
  converters. The llama-server provider forwards audio verbatim
  (WAV/MP3/FLAC, sniffed by magic bytes, resampled to 16 kHz); the MLX
  provider REJECTS audio with a 400 and a pointer to gguf (its audio
  towers are stripped at load -- the old behavior would have silently
  dropped the part). gguf models declaring the `audio` modality now
  advertise the `audio` capability (never MLX). Base64 audio is truncated
  in debug logs like base64 images. Live-verified end-to-end: E4B
  described a 15s WAV through heylook's own API. v3 audio UI + audio eval
  tasks remain 7d follow-ups.
- **Spec-decode acceptance telemetry** (plan Phase 7c closeout).
  `GenerationChunk` carries cumulative `draft_tokens`/`draft_accepted`
  (llama-server: from the final `timings` frame; MLX: running counters
  stamped per chunk); `ChunkTelemetry` latches them; surfaced in the SSE
  usage chunk's `timing` (`draft_tokens`/`draft_accepted`/
  `draft_acceptance`), `RequestEvent` (request_events.jsonl), and the
  metrics.jsonl `request_complete` record.

## [1.41.0]

### Added

- **GGUF provider via llama-server subprocess** (plan Phase 7b,
  `provider = "gguf"`). One managed llama-server process per loaded model:
  spawn in its own process group with a free port, poll `/health` until
  ready (503 = still loading; process exit = load failure), stream the
  subprocess's OpenAI-compat `/v1/chat/completions` and adapt SSE frames to
  `GenerationChunk` -- `reasoning_content` deltas become `chunk.thinking`
  (llama-server pre-splits reasoning; heylook's parser stack stays
  pass-through via `template_info() = None`), `timings` map to
  prompt/generation tps and cached-token counts. Unload = SIGTERM the
  process group (SIGKILL after 10s), so router LRU/idle-unload just work.
  Pure stdlib -- imports and runs with no MLX present.
- `GGUFModelConfig` (`models.toml`): model_path, mmproj_path,
  draft_model_path + spec_type/spec_draft_n_max (MTP/speculative -- sidecar
  drafters and embedded-MTP GGUFs both expressible as FIELDS on the one
  entry), ctx_size, n_gpu_layers, server_binary
  (default `$HEYLOOK_LLAMA_SERVER` or the coderef build), extra_args,
  default_sampler, supports_thinking/modalities capability description.
- Sampler cascade for gguf mirrors MLX: `GLOBAL_SAMPLER_FLOOR` (now shared
  in `samplers.py`) -> model `default_sampler` -> request `sampler` ->
  explicit request fields; `repetition_penalty` maps to llama.cpp's
  `repeat_penalty`; `max_tokens` is ALWAYS sent (llama-server's default is
  unlimited); `enable_thinking` -> `chat_template_kwargs`.
- `_infer_model_capabilities` gguf branch: chat always; vision from
  mmproj/modalities; thinking from the explicit flag; never
  hidden_states/logprobs (MLX-only). Fixes the silent eval-bank
  under-testing a capability-less provider would cause.
- Live-verified on gemma-4 E4B UD-Q4_K_XL (Metal build of
  coderef/llama.cpp): 12s load, text + vision (mmproj) + pre-split
  thinking (993-char reasoning trace on a multi-step problem;
  `<|think|>` injection confirmed via /apply-template), prefix-cache
  hits reported, clean SIGTERM shutdown.

## [1.40.0]

### Changed

- **Providers now yield a heylook-owned `GenerationChunk`** (plan Phase 7a
  seam hardening). The de-facto provider contract was mlx-lm's
  `GenerationResponse` dataclass, mutated in flight (three runtime-patched
  attrs) and scraped by getattr across the API layer. `run_generation`, the
  VLM first-token path, and the diffusion path now convert to a slotted
  `GenerationChunk` (`providers/base.py`) at the engine boundary --
  attaching undeclared attributes fails loudly; new telemetry means a new
  FIELD absorbed in `ChunkTelemetry`, not an attr-patch at a call site.
  `_VisionTokenResponse` deleted. `ChunkTelemetry.absorb()` latches
  snapshot fields on truthy values (every field now exists on every chunk,
  so absence no longer protects first-chunk-only stats).
- **BaseProvider capability surface**: `provider_name` class attribute
  (fixes the router's DEAD `mx.clear_cache()` teardown gates, which
  matched a `provider.provider` attr nobody set, and replaces api.py's
  class-name sniffing), `template_info()` method (parser selection no
  longer reads the private `_template_info`), neutral `is_vlm` /
  `effective_loader` defaults. `MLXEmbeddingProvider.create_chat_completion`
  and the test mock gained the `abort_event` parameter (signature drift vs
  the ABC).
- **Provider config registry**: `PROVIDER_CONFIG_CLASSES` in config.py is
  the single source for known providers; `validate_config_type` dispatches
  from it instead of an if/elif chain.

### Added

- **Pre-split thinking passthrough**: `GenerationChunk.thinking` carries
  reasoning an engine already separated (llama-server's
  `reasoning_content`); all four consume paths (chat + messages,
  streaming + non-streaming) route it straight to the thinking channel,
  bypassing the text parsers. MLX providers never set it -- no behavior
  change for existing models. Groundwork for the Phase 7 gguf provider.
- Tests: `tests/unit/test_generation_chunk.py` (chunk shape, telemetry
  latch semantics, capability surface, registry) +
  `tests/contract/test_prethought_passthrough.py` (pre-split thinking
  end-to-end through the HTTP surface). OpenAPI schema verified
  byte-identical vs the pre-change worktree.

## [1.39.17]

### Changed

- **`uv sync` now installs everything, no extras.** The seven optional-extras
  (`performance`, `analytics`, `profile`, `cli`, `scripts`, `test`, `all`) are
  gone. The performance stack (`xxhash`, `PyTurboJPEG`, `cachetools`, `uvloop`)
  and the CLI (`questionary`) moved into CORE deps -- they were required for a
  fully-working server anyway, and gating them behind `--extra all` was a
  long-standing footgun (a plain `uv sync` silently produced a degraded install
  that failed on the first image request). Dev tooling (`pytest` + plugins,
  `httpx`, `build`, `twine`, `rich`, `py-spy`) is now the `dev`
  dependency-group, which uv installs by default. Net: `uv sync` = full runtime
  + dev; `uv sync --no-dev` = runtime only; `pip install heylookitsanllm` (from
  PyPI) = runtime only (groups aren't published). `uvloop` carries a
  `sys_platform != 'win32'` marker so a base install still resolves on Windows.
- **De-confused two version constraints.** `torch` is now an exact pin
  (`==2.13.0`, the version already resolved) instead of an arbitrary `>=` floor
  -- it's only a transitive/availability dep, so freezing it is honest and
  reproducible. The `[tool.uv] override-dependencies` for transformers was
  `>=5.3.0` while core said `>=5.13.1`; they now match (`>=5.13.1`) so they
  don't appear to contradict, with a comment explaining the override exists to
  strip mlx-lm/mlx-vlm's upper bounds (still resolves to the latest 5.x, 5.14.1).
- **Dropped dead/duplicate deps.** Removed the bare unmarked
  `mlx`/`mlx-lm`/`mlx-vlm` duplicates (the `sys_platform == 'darwin'` copies
  already cover Apple Silicon; the unmarked ones were what broke non-darwin
  resolution). Removed `datasets` entirely -- it was declared in the
  `analytics` extra but imported nowhere in the repo, and dropping it also shed
  a heavy transitive chain (pandas, pyarrow, dill, multiprocess). Both git
  sources carry an explicit `rev` again (mlx-lm had drifted to a floating
  source in the working tree) so the lock stays reproducible.
- **Fixed stale install instructions in code + docs.** `router.py`/`server.py`
  told users to run `uv sync --extra mlx` -- an extra that never existed (mlx
  is a core, darwin-marked dep); now they say `uv sync`. The OpenAPI
  description's `pip install heylookllm[performance]` block and the README's
  `--extra analytics/performance/all` recipes are replaced with the single
  `uv sync`. `CLAUDE.md`'s `--all-extras` gotcha and `setup.sh` (which offered
  the nonexistent `mlx`/`all` extras and a nonexistent `models.toml.example`)
  were rewritten to the one-step flow.
- **`scripts/update_deps.py`** (new, tracked) supersedes the removed
  `update-packages.sh`. Bumps git-sourced packages to their latest commit and
  writes the resolved SHA back as a pinned `rev` in `[tool.uv.sources]` -- plain
  uv can pin HEAD in `uv.lock` but leaves the source floating in pyproject, and
  this project's policy is an explicit rev in both. Also does `--release` PyPI
  bumps with an optional `--pin` floor-raise, plus `--dry-run`. PEP 723 header
  provisions `tomlkit` without touching the project env. Indexed, with the rest
  of `scripts/`, in the new `scripts/README.md`.

## [1.39.16]

### Added

- **Applied-preset provenance is durable** (`applied_preset_id`, schema v6 on
  conversations AND notebooks). The chip's "(edited)" label used to die with
  the session -- a reload could only re-infer a preset by exact state match,
  so a document you had applied a preset to and then tweaked came back
  unlabelled. The stamp now lives on the document, like every other piece of
  per-document state in v3 (`params`, `system_prompt`), so it survives a
  reload and reads the same on every device. What is stored stays strictly
  EXPLICIT: Apply and Save stamp it, Del clears it for the active document,
  and a match that is merely coincidental is still reported by live
  client-side matching and never written -- storing a derived association
  could bind stale state to the wrong document after a failed load (the
  v1.39.7 lesson). The field holds a preset id or null; never any prompt,
  message, or model output. A stamp naming a deleted preset is self-healing
  (stamps resolve against the live preset list, so a dangling id reads as
  "no stamp"). Deliberately NOT carried inside `params`: that object is the
  sampler bag and everything in it reaches the model, so non-sampler state
  may never live there. E2E: chip provenance asserted across a reload -- a
  check that could not have passed before this field existed. Contract:
  `docs/frontend_v3_spec.md` section 4.
- **Parser invariants are now properties, not examples**
  (`TestParserInvariants`). Both 2026-07-23 parser bugs were failures of an
  unstated invariant rather than of a missing case, so two are stated and
  checked over randomised chunkings of a representative corpus: output is
  invariant to how the stream was chopped (the property finding #1 violated),
  and text carrying no structural tokens survives intact (the property
  finding #2 violated). Seeded, so a failure reproduces.

### Fixed

- A bare trailing `<` is no longer swallowed at end of turn. The
  partial-control-token guess now requires at least two characters: `<|` and
  longer are essentially never natural model output, while a lone `<` is
  ordinary text (comparisons, code, generics). This is what makes the
  no-silent-loss property hold, and it retires the proposed `in_name`
  streaming bail-out -- the invariant covers that case without a heuristic.
- **Selected items no longer lose their tint on hover.** `.btn:hover`,
  `.nav-item:hover` and `.conv-item:hover` outranked the state classes
  declared right after them, so hovering an active nav item, selected
  conversation, or the preset chip made it look unselected. The hover
  pseudo-class is now wrapped in `:where()` (zero specificity), so state wins
  by source order -- the pattern is fixed once rather than per component.
- The preset bar's select and save-as input carry `aria-label`s. `title` is a
  tooltip whose exposure as an accessible name is inconsistent, and neither
  control has a visible label to associate.

## [1.39.15]

### Changed

- **The per-document system-prompt editor is one shared factory**
  (`js/prompt-section.js`), used by chat and notebook. The two pages had
  divergent copies of the same grammar, and the hard-won lessons lived in
  ONE copy each -- chat's per-keystroke commit (a save-on-blur commit lost
  the text whenever the drawer closed under focus, and a preset saved in
  that window captured `system_prompt=null`), its stale-editor guard, and
  its blur flush. A page can no longer drift from that by copy-paste. Both
  editors now live in the same settings drawer, so the incidental
  differences (rows, CSS class, collapsed-when-empty) are unified rather
  than parameterised: one `.sysprompt` / `.sysprompt-input` rule set, always
  expanded. The factory FLUSHES a pending write on teardown rather than
  cancelling it -- navigating away inside the debounce window is the same
  data-loss shape the module exists to prevent.
- **The notebook's system prompt has its own field-scoped PUT** instead of
  riding the whole-document autosave, matching chat. A preset apply used to
  re-upload title + full content + prompt + model; it now writes just
  `system_prompt`, serialized through a per-mount chain (chat's ordering
  guard). `doSave` no longer sends the field at all, so a slow document save
  cannot clobber a newer prompt write. Deleting the last notebook also
  resyncs the widget -- it outlives any single document now.
- The shared prompt input keeps its focus ring. The notebook's old copy
  suppressed it, borrowing the distraction-free-editor exemption that
  belongs to the writing surface; DESIGN.md section 7 scopes that exemption
  to the surface, not to every borderless field.

### Added

- E2E lib consolidation: `serverGet(page, path)` (lib/server-state.mjs) --
  the server's own view of a document, fetched inside the page so it shares
  the app's origin and session; the suites' four state readers compose on it
  instead of re-implementing evaluate + fetch + json. `proveQuiet(watch,
  ...)` (lib/harness.mjs) holds the ONE sanctioned bounded sleep: claims of
  absence ("no second fetch follows") have no condition to wait on, and
  keeping the exception in one place stops it being re-argued per site.
  A third proposed helper (`raceState`) was deliberately NOT added -- the
  two "observed-state branching" sites make genuinely different decisions,
  and one abstraction over them would obscure more than it shares.

## [1.39.14]

### Fixed

- **`finish_reason` no longer reports `"stop"` on truncated responses.**
  Both non-streaming paths (`/v1/chat/completions` and `/v1/messages`)
  hardcoded it, so a response cut off by `max_tokens` was indistinguishable
  from one the model chose to end -- clients could not tell a complete
  answer from a truncated one. mlx-lm reports the real reason on the final
  chunk (the streaming path already forwarded it); it is now scraped in
  `ChunkTelemetry.absorb` -- one place, so the four consume loops cannot
  drift -- and latched, since a trailing empty chunk carries no reason. The
  Messages converter already mapped `length` to `stop_reason`; it was simply
  never fed one. Live-verified on both paths: `max_tokens=12` -> `"length"`,
  natural completion -> `"stop"`.

### Changed

- `tests/eval` judges the budget check on the server's own `finish_reason`
  instead of inferring truncation from `completion_tokens == max_tokens`.
  The inference stays as a fallback for servers that report no reason (it
  was the only signal while the field was hardcoded).

## [1.39.13]

### Fixed

- **A spurious channel-open no longer swallows the whole reply** (the
  "post-abort immediate-EOS" investigation, root-caused live). gemma-4
  sometimes emits a stray `<|channel>` mid-answer and simply keeps
  answering; because the header was never terminated by a newline, both
  channel parsers accumulated every following token into the channel-NAME
  buffer and dropped it as structural at flush. The user saw an EMPTY
  reply while the server reported a normal `stop` -- captured raw tokens
  `['<|channel>', ' to', ' the', ' movies', '!', '<turn|>']` produced ''.
  At end of turn, unrouted model text now goes to content (both gemma and
  harmony). Trade-off recorded in the code: an abort inside a legitimate
  header now surfaces a short fragment instead of vanishing, which is the
  lesser failure.

  Measurement notes, since this had a standing (wrong) explanation: the
  memo's template/history theory was falsified first -- 0/64 empty replies
  across four history shapes (complete / truncated / empty-content /
  omitted assistant turn), and 0/6 vs 0/6 with REAL mid-stream client
  disconnects, so aborts were never the variable. The empty replies came
  from the parser, not the model, and not from v1.39.12: an A/B against a
  v1.39.11 worktree on the exact captured token sequence swallowed it
  identically.

### Changed

- tests/e2e/README.md's "empty-EOS is legal model output" principle carries
  a caveat: tolerating empty replies stays correct, but a rising rate is a
  bug signal -- this class of them was real text loss.

## [1.39.12]

### Fixed

- **Declared special tokens can no longer leak across a chunk boundary**
  (the P2 parser strip/holdback unification). Stripping was implemented
  three ways across four streaming parsers: harmony and gemma sized their
  partial-token holdback to their OWN structural control tokens (<= 10
  chars), so a longer declared special -- `<|reserved_200000|>` and
  friends -- straddling an emit boundary leaked both halves into
  user-visible text, and PassThrough had no holdback at all. There is now
  ONE implementation, `reasoning_parser.StripSpecials`, composed by the
  factory over whichever routing parser it selected (and only when the
  model declares specials); its rolling per-kind holdback is sized by the
  STRIP SET, not by any parser's structural grammar. The holdback also no
  longer assumes specials start with `<`, so Mistral-family `[INST]`-shaped
  tokens are held correctly too.
- **Abort landing mid-control-token no longer flushes garbage on harmony
  models.** The final drain emits the whole buffer, so harmony's trailing
  partial-token strip ran against an already-empty buffer and never fired
  (dead since it was written); an abort inside `<|channel|>` flushed
  literal `<|chan`. Gemma's 2026-07-20 fix -- pre-strip at the TOP of the
  final drain -- is now a shared `_strip_partial_token` used by both, and
  both parsers' consequently-unreachable leftover blocks are gone.

### Changed

- Parsers no longer take `strip_tokens`; `HybridThinkingParser` is a thin
  protocol adapter again, `_safe_prefix_len` is one free function taking
  the caller's own token-length constant, and the design note
  (docs/parser_strip_unification.md) is now an implemented record. Eight
  failing-first tests pin the boundary behavior that regressed silently
  before.

## [1.39.11]

### Fixed

- **Failed document loads are now recoverable** (frontend v3; /code-review
  finding). After a failed conversation fetch, the previous conversation's
  messages/prompt lingered under the new id and the select guard's
  `messages.length` condition made re-clicking the same conversation a
  permanent no-op; the failure path now clears the stale state, renders the
  empty thread, and re-clicking retries. Notebook had the same class via
  its `activeId === id` guard -- a failed load now drops back to "no active
  notebook" so re-clicking retries.
- **System-prompt PUTs are serialized** (chat). The textarea's blur-flush
  and a preset-apply's write were independent fetches with no ordering
  guarantee; all prompt writes now chain through one per-mount promise so a
  stale flush can never land after an apply (pre-existing class, near
  unobservable on localhost, closed outright).

## [1.39.10]

### Fixed

- **Cap-gated sampler keys no longer ride requests to incapable models**
  (frontend v3; the P2 "pinned invisibly" TODO item). `enable_thinking` /
  `vision_tokens` set on a capable model stayed in the settings cache when
  the panel hid their controls on a model lacking the cap, and
  `samplerParams()` kept emitting them on every request until Reset.
  `samplerParams(caps)` now drops capability-gated keys at request-build
  time; chat/notebook/explore pass their page's current caps. The cache
  deliberately keeps the value -- switching back restores the control and
  its setting; only the wire is filtered.
- **E2E `newFreshConversation` resolves the fresh conversation by max
  `created_at`, not list position** -- the conversations list orders by
  `updated_at`, and a prior check's trailing debounced params PUT could
  bump its old conversation past the fresh one, handing back the wrong id
  (the intermittent image-check failure, finally reproduced with its
  message and fixed).

### Added

- E2E: composer paste-path check (synthetic ClipboardEvent with a real
  File); notebook applied-preset chip assertions (shows on save, gains
  "(edited)" on drift, clears on delete); the cap-filter wire check
  (request intercepted + aborted -- the negative model never loads).
  75 checks total (chat 39 + pages 36), live-verified 75/75.

## [1.39.9]

### Added

- **E2E coverage for thinking, images, and capability gating** (chat suite
  31 -> 37 checks, 73 total, live-verified 73/73). New section: thinking
  toggle + `vision_tokens` gate on the selected model's capabilities, with
  the negative model discovered from `/v1/models` and never loaded (gating
  is pure metadata); `vision_tokens` localStorage round-trip; the thinking
  wire contract asserted on captured request bodies (`enable_thinking`
  absent when off -- never false -- `true` when on); the thinking block UI
  on a real thinking generation; attach cap (9 canvas-generated PNGs -> 8
  thumbs + the aria-live message); and an image-message round-trip (2
  image content_blocks persist server-side, render as `.message-image`,
  survive reload). Building these surfaced two more harness truths, now in
  the README: per-document params beat the localStorage seed after any
  reload (hydration replaces the seeded cache), and a reload can kill a
  debounced params PUT and resurrect stale settings -- so checks control
  settings through the drawer panel on a fresh conversation and end by
  waiting for the params PUT to land server-side.

## [1.39.8]

### Changed

- **E2E suites audited and hardened end-to-end** (both suites, 67 checks,
  live-verified 67/67 in one run). Every check now verifies its stated claim
  via persisted outcomes rather than transient UI states: notebook autosave
  checks wait for the debounced PUT to land server-side instead of sleeping
  past the debounce; save-and-regenerate waits on server-confirmed
  truncation instead of the Stop label; stop-mid-generation branches on the
  observed state (a generation legally finishing before Stop is clickable is
  logged, not flaked); post-abort health asserts pipeline health and only
  checks persistence when content was produced (empty-EOS replies legally
  persist nothing); generate-at-cursor asserts head/tail preservation
  unconditionally but no longer fails on an empty completion; a vacuous
  edit/cancel check now verifies prefill + exact restore; a perf-profile
  vacuous pass (Loading placeholder sharing the empty-state class) is
  closed; regenerate prompts reworded away from "single word" framing to
  reduce the empty-EOS rate; hidden order couplings (terse sysprompt
  persisting on the suite conversation, preset-bar prompt dependency,
  jspace session continuity) are now explicit comments. The audit principle
  is recorded in tests/e2e/README.md.

## [1.39.7]

### Fixed

- **Applied-preset chip can no longer bind stale state to the wrong
  document** (/code-review finding on v1.39.6). The chip's exact-match
  seeding wrote an inferred association into the per-document map; a failed
  conversation/notebook load (id flipped, old state lingering) could stamp
  the new document with the old one's preset and persist a false "(edited)"
  claim. Inference is now read-only -- the map holds explicit Apply/Save
  stamps exclusively -- and both pages resync the chip on load-failure paths
  and on deleting the last notebook (chat parity). Notebook builds its
  skeleton before the preset bar (chat parity), dropping a defensive guard
  for an unreachable ordering.
- **E2E: regenerate check de-raced.** It required observing the transient
  Stop label, but a one-word regeneration on the fast MoE starts and
  finishes inside a single poll interval (deterministic timeout on a warm
  machine). The check now waits for the outcome -- the regenerated reply
  persisted under a NEW message id -- instead of the transient UI state.
  Chat suite live-verified 31/31.

## [1.39.6]

### Added

- **Applied-preset chip** (frontend v3, chat + notebook). A chip beside the
  model select names the preset the active conversation/notebook is running,
  gains "(edited)" the moment the prompt or sampler panel drifts from it, and
  opens the settings drawer on click. The shared preset bar tracks the
  association per document (Apply/Save stamp it, delete clears it) and feeds
  the chip through a new adapter callback; provenance is session-local by
  design -- only explicit Apply/Save stamps are stored; a document whose
  state exactly matches a preset is labeled by live inference, never stored
  (the server keeps no preset association, and a stored inference could bind
  stale state to the wrong document). E2E: chip
  visibility/(edited)/clears-on-delete checks in the chat suite (31 checks,
  live-verified 31/31; the notebook chip has no dedicated E2E check yet).

## [1.39.5]

### Fixed

- Preset drift line carries `role="status"` so its Matches/Differs flips are
  announced to screen readers, not just shown (DESIGN.md §7 rule; /code-review
  finding). Stale comments from the preset rework corrected (drawer header
  still described the retired apply-on-select; chat's `setSystemPrompt` still
  claimed to be the only prompt writer).

## [1.39.4]

### Fixed

- **Sampler inputs no longer lose a typed-but-unblurred edit on drawer close**
  (frontend v3). The v1.39.1 fix protected the chat system prompt but the
  data-loss class lived in the drawer: any commit-on-`change` field loses its
  value when `close()` removes it while focused (Escape/hashchange). The
  drawer now blurs the focused field before clearing its body, flushing every
  such field (sampler numbers included) at the one place all close paths
  converge.

### Changed

- Quality pass over the v1.39.1-.3 diff (4-agent /simplify): the preset bar
  owns its sampler-side drift subscription (consumers can no longer forget it
  and show a stale line) and exposes `onDrawerOpen` (both pages' hand-rolled
  refresh wiring folded in); drift writes are change-gated and the detached
  section subtree is released after drawer close; preset save no longer
  refetches the list it just wrote (patches from the server's response);
  notebook's sysprompt widget sync is one helper shared by populateFields and
  the preset adapter; the drift line rides the shared `.settings-note` style.
  E2E lib gained `handleByText` (clickByText rebased on it) + `driftText`,
  and `openDrawer` takes an opener selector -- the suites' hand-rolled
  copies are gone.

## [1.39.3]

### Added

- **Notebook preset bar** (frontend v3). The preset section was extracted
  from chat into a shared `preset-bar.js` module (`createPresetBar` with a
  `getPrompt`/`setPrompt`/`onStatus` adapter) and the notebook page now
  contributes it to the settings drawer ahead of its system-prompt editor --
  identical grammar to chat: inert select, live drift line, explicit Apply
  armed only when it would replace a differing non-empty prompt, save-by-name
  upsert. Applying a preset writes the notebook's system prompt and sampler
  params (a preset is a prompt+sampler bundle everywhere). Chat now consumes
  the same module (no behavior change). E2E: new notebook preset check
  (save, live drift flip, armed apply, delete).

## [1.39.2]

### Changed

- **Preset apply is explicit and legible** (frontend v3 chat). Selecting a
  preset in the drawer no longer applies it as a side effect of the `change`
  event -- selection only records the choice, prefills the save-as name, and
  drives a new live drift line ("Matches current settings." / "Differs from
  current settings -- Apply copies it here, Save overwrites it."), updated in
  place on prompt keystrokes and sampler edits. Applying is a dedicated Apply
  button, armed-confirmed ("Replace prompt?") only when it would overwrite a
  non-empty conversation prompt with something different -- the copy-on-apply
  semantics are unchanged, but the copy now happens on a deliberate click and
  announces what it will do. `armedConfirm` gained an optional `when`
  predicate for sometimes-destructive buttons. E2E: preset round-trip check
  now asserts inert selection, explicit apply, and both drift states.

## [1.39.1]

### Fixed

- **Chat system prompt silently lost on drawer close** (frontend v3). The
  editor committed only on `change` (blur); closing the settings drawer with
  Escape or a route change removed the focused textarea before `change` could
  fire, discarding the typed prompt from both state and the server. A preset
  saved in that window captured `system_prompt: null`, so applying it later
  actively erased the conversation's prompt. The editor now commits state on
  every keystroke (notebook parity) and debounces the PUT (400 ms, flushed on
  blur), so no close path can outrun the save.

### Changed

- Chat settings reachability + system-prompt legibility (frontend v3): a gear
  in the chat top bar opens the shared settings drawer in context (desktop and
  mobile; the sidebar-foot and bottom-nav gears remain); the system-prompt
  section is always expanded instead of collapsed-when-empty; the editor grew
  from 3 rows to a 9rem-minimum auto-growing textarea; the drawer widened from
  22rem to 26rem. E2E: 2 new chat checks (in-context gear; Escape-close
  persistence regression).

## [1.39.0]

### Added

- **Masked-diffusion generation path** (`DiffusionStrategy`). Diffusion LMs
  (`diffusion_gemma`) denoise a fixed-length canvas instead of extending a
  sequence, so mlx-lm's `stream_generate` -- heylook's only text path -- drove
  them as autoregressive: one meaningless token sampled from the last prompt
  position, landing on EOS immediately. Requests completed with **zero emitted
  tokens** and an empty reply, no error. Detection is mlx-vlm's own
  `is_diffusion_model` predicate (keys on `config.canvas_length`, not a
  model_type match, so new diffusion architectures are picked up without a
  code change), resolved at load and routed ahead of the text/vision split --
  the denoising loop takes images inline, so there is no separate vision path.
  Text and vision both verified end to end.

### Notes

- The engine is mlx-vlm's `stream_diffusion_generate`, called **directly**
  rather than through `stream_diffusion_generate_from_kwargs`: the from_kwargs
  wrapper collects the entire generation into a list before yielding any of it,
  which on a streaming server is full-duration dead air followed by one burst.
  Called directly it streams **block-by-block** -- one flush per denoised
  canvas, which is as incremental as masked diffusion gets (a 683-token reply
  arrives as 3 canvas flushes, not 1).
- Generation parameters come from the checkpoint's own `generation_config`
  (canvas length, denoising steps, entropy-bound sampler, confidence/stability
  thresholds, linear temperature schedule). heylook's AR sampler cascade does
  not apply -- `top_p`/`top_k`/`repetition_penalty` have no meaning for a
  canvas update. Only `temperature` and `max_tokens` set **explicitly** on the
  request override the checkpoint; the AR sampler floor deliberately does not.
- The radix/prompt cache is bypassed: the denoising loop owns its own KV cache,
  and radix assumes AR trim-to-a-prefix semantics a canvas rewrite violates.
- `warmup()` now primes the denoising loop for diffusion checkpoints instead of
  mlx-lm's AR decode -- priming a path real requests don't take is how the VLM
  `LanguageModelOutput` bug stayed hidden.

## [1.38.1]

### Fixed

- **Unknown `sampler` name returned a bare 500**: the deep `SamplerNotFound`
  raise happens inside the provider's `_apply_model_defaults`, which runs
  lazily on first generator advance -- past the chat route's guarded stage --
  so it escaped to Starlette's default handler as `Internal Server Error`.
  Both `/v1/chat/completions` and `/v1/messages` now validate the sampler
  name at the route boundary: immediate 400 naming the known samplers, no
  model load spent on a typo. Contract test added.
- **Eval bank: refusals could score as passes**: the large-image "sanity"
  task judged only prose-shape, so "I'm sorry, but I cannot describe this
  image" PASSED (live case: the 26B QAT 4-bit refusing synthetic images).
  New `not_refusal` judge (refusal-shaped openers fail), and the noise
  image is replaced by a four-quadrant color-block PNG (same size, now
  describable) with a >=2-colors-named requirement.

## [1.38.0]

### Added

- **Server-owned load+warm readiness**: `POST /v1/admin/models/{id}/load`
  accepts `?warm=true` -- after loading, runs a 1-token generation through
  the normal generation path (FIFO gate, sampler cascade), paying the
  first-forward-pass Metal-kernel JIT. Returns `warmed`/`warm_ms`, or
  `warmed: false` + `warm_error` with 200 (model loaded, server usable).
  This is the fix for the code-review finding that `scripts/dev_server.sh`
  and `tests/e2e/lib/server.mjs` each hand-rolled poll-the-model-list +
  warm-generation heuristics (and drifted): both harnesses are now thin
  clients of this one endpoint -- spawn, wait for HTTP-up, call
  `load?warm=true`, done. Cross-references added in both harness headers,
  the dev-server skill, CLAUDE.md, and spec section 4. Live-verified on
  Metal (0.8B: warm 164ms; unknown-id start fails fast with no orphan;
  foreign server on another port untouched).

## [1.37.0]

### Changed

- **The sampler registry is now "samplers" -- "preset" belongs exclusively
  to the v3 user-preset system** (`/v1/presets`, DuckDB prompt+sampler
  bundles). The registry had worn three names (profiles on import/admin,
  presets on the request API, colliding with user presets); final
  vocabulary: module `samplers.py` (`SamplerRegistry`, `SamplerNotFound`,
  `get_sampler_registry`), data dir `data/samplers/`, request field
  `ChatRequest.sampler` (a request still sending `preset` gets an explicit
  400 with a migration hint -- ChatRequest ignores unknown keys, so a
  silent drop was the alternative), models.toml `default_sampler`, admin
  routes `/v1/admin/models/samplers` + `/bulk-default-sampler` (body
  `sampler`), `ModelImportRequest.default_sampler`, capabilities block
  `samplers` (`request_field: "sampler"`), model_service
  `stamp_default_sampler`/`get_samplers`/`bulk_set_default_sampler`/
  `available_samplers`. Import CLI: `--sampler` canonical,
  `--preset`/`--profile` kept as aliases. batch-labeler follows
  (`--sampler` flag, task TOML `sampler` key). Explicit per-request
  sampler fields (temperature/top_p/...) continue to override everything,
  named sampler or not -- the cascade is unchanged.

### Fixed

- **Code-review findings** (10/10 confirmed by adversarial verify,
  all fixed):
  - `scripts/dev_server.sh` (6): readiness now waits for HTTP-up then
    exact-id membership (was substring/regex grep against the configured
    list -- prefix ids false-READYed and dots were metachars), fails FAST
    on an id the server doesn't list instead of spinning the full
    deadline; failed starts kill the spawned pid + clear state (was: exit
    1 with a multi-GB orphan and a retry that reported false success);
    warm-request failure is a warning, not a `set -e` abort right after
    READY (timeout raised to 600s for big cold loads); `probe()` cd's to
    the repo root so the foreign-server port guard works from any cwd;
    `status` no longer lists the script's own spawned pids under "NOT
    ours -- never kill these"; port-guard branch probes once, not twice.
  - `ModelImportRequest` gets `extra="forbid"` -- an old client's renamed
    body field now 422s instead of being silently dropped and stamped
    with the default.
  - `/v1/capabilities` `server_version` reads the real package version
    (`heylook_llm.__version__`, bumped to match this release) -- was
    hardcoded "1.0.1".
  - model_service's vestigial adapter tower (dead `apply_sampler_preset`
    + its test mock, single-caller `_presets_view`, import-time snapshot,
    stale "during the transition" docstring, duplicate listing pipeline)
    collapsed onto the registry: `stamp_default_sampler()` helper +
    registry-direct reads.
  - `ChatRequest.sampler` Field uses keyword-form default (repo rule;
    the last positional `Field(None, ...)` in config.py).
  - Stale docs: config.md's `profile_name` reference and roster wording;
    packaging: `[tool.setuptools.package-data]` pointed at a nonexistent
    `data.profiles` and never shipped the sampler TOMLs in a wheel build.

## [1.36.1]

### Fixed

- **gemma-4 stop-set regression under transformers 5.x**:
  `extend_eos_from_generation_config` assigned `tokenizer.eos_token_ids`,
  which transformers 5.x `SpecialTokensMixin.__setattr__` intercepts and
  rejects ("Cannot set a non-string value as the eos_token") -- the
  `<turn|>` terminator silently dropped out of the stop set and gemma-4
  responses ran past end-of-turn to the token cap (channel-block loops).
  The union is now stored under a private attr that `resolve_stop_tokens`
  reads first, out of reach of the special-token machinery.
- **Behavioral eval bank request hygiene**: every task is now explicit
  about `enable_thinking` with budgets sized for what is enabled
  (thinking tasks 1536, thinking-off tasks keep tight budgets honestly);
  the near-greedy `temperature=0.1` stop task now uses server sampling
  defaults; thinking-on runaway is caught by a budget-exhaustion check on
  `thinking_requested_split` instead of under-budgeted stop tasks.

## [1.36.0]

### Added

- **Sampler-preset discovery on `GET /v1/capabilities`**: new
  `sampler_presets` block (`available` name+description list straight from
  the PresetRegistry, plus `request_field`/`model_default_field` pointers)
  so scripted clients (batch-labeler etc.) can enumerate preset names
  without admin access. Previously names only leaked via the admin route
  or the text of an unknown-preset 400. Contract tests assert the block
  mirrors the registry exactly.

### Changed

- **"Profile" terminology retired for sampler presets** (it was the same
  registry as `ChatRequest.preset` wearing a second name in the
  import/admin paths, and collided with `/v1/performance/profile`).
  Renames -- admin API: `GET /v1/admin/models/profiles` ->
  `/sampler-presets` (response field `profiles` -> `presets`),
  `POST /bulk-profile` -> `/bulk-default-preset` (body `profile` ->
  `preset`), import body field `profile` -> `default_preset`; code:
  `ModelProfile` -> `SamplerPreset`, `load_profiles` ->
  `load_sampler_presets`, `get_profiles` -> `get_sampler_presets`,
  `bulk_apply_profile` -> `bulk_set_default_preset`,
  `get_available_profiles` -> `available_sampler_presets`. The import
  CLI keeps `--profile` as an alias for `--preset`. CLAUDE.md now carries
  the two-preset-systems note; full rename map in
  `docs/architecture/config.md`.

## [1.35.1]

### Removed

- **Unused flavor sampler presets** `moderate`, `code`, `creative` (audit
  finding: zero consumers anywhere -- the v3 frontend's user-preset system
  owns interactive sampler preferences, and their only references were
  tests asserting their own existence). Bundled roster is now mechanism +
  wired defaults only: `thinking` (auto-applied for enable_thinking
  models), `vlm-extract`/`vlm-describe` (VLM-safe subsets, used by
  batch-labeler tasks), `balanced` (import-default profile),
  `deterministic` (repro/eval). Registry test now pins the exact roster so
  additions must name a consumer. No API change; a request naming a
  removed preset gets the existing unknown-preset 400.

## [1.35.0]

### Added

- **batch-labeler rebuilt as a task-based CLI (app v0.2.0)**: subcommands
  `run` / `try` / `models` / `tasks` replace the single flag-soup invocation.
  Built-in task templates with curated system prompts (`label` structured
  taxonomy, `caption` training-style captions, `tags` keyword tags, `ocr`
  verbatim extraction), each carrying its own sampler preset
  (`vlm-extract`/`vlm-describe`), max_tokens, and required-key validation;
  custom tasks via `--task-file` TOML (unknown keys rejected). Now exposes
  the server capabilities the old client ignored: `--think/--no-think`
  (enable_thinking; thinking stored in its own record field), `--vision-tokens`
  (visual token budget), `--resize-max`/`--image-quality` (server-side
  resize), `--preset`, `--seed`. Model auto-pick when the server has exactly
  one vision model; transient-failure retries with backoff; `--limit` for
  sampling; `--dry-run` prints fully-resolved settings; records carry
  parse_ok/missing_keys, usage, performance, and a settings echo. Live tok/s
  in the progress bar. 37 new tests (62 total in the app suite).

### Fixed

- **`include_performance` on `/v1/chat/completions` was dead**: declared on
  ChatRequest but never read; `ChatCompletionResponse.performance` was never
  populated. Non-streaming responses now fill it from native mlx-lm telemetry
  (prompt_tps, generation_tps, peak_memory_gb) when requested. Contract test
  added; schema unchanged (field already existed as Optional).

## [1.34.66]

### Fixed

- **Review-pass fixes** (xhigh multi-agent review over the day's 13-commit
  range; findings adversarially verified before fixing):
  - `<think>`-family parser (HybridThinkingParser) now strips declared
    special tokens from routed text like every other parser -- with the
    decode-level hygiene gone, non-think control tokens could render
    literally for Qwen-family models; implemented with a rolling per-kind
    holdback because the inner parser's buffering can split a special
    across deltas.
  - Gemma channel parser: an abort landing mid-`<channel|>` no longer
    flushes literal partial-token garbage (final drain strips trailing
    partials of BOTH gemma control-token shapes; harmony's helper only
    knew `<|`-prefixed ones).
  - `effective_thinking_flag`'s absent-key fallback now mirrors the
    template-side resolution (True) so a raw un-normalized config dict
    can't produce a prefilled-thinking prompt with a content-state parser.
  - config API docstrings/OpenAPI descriptions no longer claim an
    env-override layer that settings.py deliberately abolished.
- **v3 touch/mobile fixes from the impeccable audit** (desktop + iPhone 17
  Pro Safari): attach-strip remove buttons get a 44px hit area (glyph
  stays 24px); composer icon buttons meet the 44px floor on touch
  devices; the composer placeholder no longer wraps and clips on phones
  (Enter hint is desktop-only).

## [1.34.65]

### Added

- **`tests/eval/` -- opt-in LLM behavior-eval harness** (13-task seed bank,
  7 programmatic judges), generalized from the 2026-07-20 live-verification
  scripts. Covers the four bug classes found that day: thinking split/leak,
  stop discipline, vision single/multi-image correctness, vision_tokens
  budgets. Needs a running server (never spawns one); not wired into
  /test-suite -- run it when touching templates, parsers, stop tokens, or
  the vision pipeline.

### Changed

- **Pyright noise triage.** Real fixes: deprecated `datetime.utcnow`;
  untyped `= None` defaults; batch responses could hand pydantic
  `model=None` (runtime 500) -- now coalesced; float re-binding in
  `_format_bytes`; route discovery duck-typed via getattr. Systemic:
  all 96 positional Pydantic `Field(default, ...)` calls converted to
  explicit `default=` (this pyright build only recognizes keyword form --
  the source of every false "arguments missing" constructor error).
  Idioms annotated per repo convention (`# type: ignore[...]`): request
  attr-attach, streaming-logprobs union, psutil optional import, RLM's
  provider-specific batch fast path. hasattr-ternary tokenizer access
  swapped for `getattr(x, "tokenizer", x)` (same runtime, narrows).

## [1.34.64]

### Added

- **Model-agnostic vision token budget** (`vision_tokens` on chat requests +
  per-model default in models.toml; v3 drawer control, cap-gated on
  `vision`): one wire knob mapped by duck-typing the loaded model's image
  processor -- gemma-4 discrete buckets (snapped: 70/140/280/560/1120 soft
  tokens ~ 0.16-2.58MP), qwen2/3-VL continuous pixel budget
  (tokens x (patch x merge)^2 -> max_pixels), unknown families degrade to
  processor defaults. Vision-feature cache key carries the budget (feature
  shapes differ per bucket). Verified live on gemma-4-31b + Qwen3.5-27B
  (pixel patches 630/2520/10080 at 70/280/1120; full think x images x
  budget matrix clean on both).

### Fixed

- **Qwen3.5 thinking never split into the thinking field**: its template
  PRE-FILLS `<think>\n` into the generation prompt when thinking is on, so
  the model output starts inside the block without an opening tag; the
  parser routed everything to content. New `prefills_thinking` template
  detection + `initial_thinking` parser state, armed per request from the
  effective thinking flag.
- **Removed hardcoded Qwen3 vocab ids from the thinking parser**
  (151667/151668): the token-level mode keyed on them silently failed to
  split for any other `<think>`-family vocabulary whenever token ids were
  present. Parser is text-based only now, same as the harmony/gemma
  channel parsers (tag strings are format grammar; token ids are
  per-model vocabulary).

## [1.34.63]

### Fixed

- **Gemma-4 on the mlx-vlm path generated past end-of-turn** (visible as
  answers that "keep going", thought-babble tails, repeated `<turn|>`):
  two stop-set gaps, both fixed data-driven from the model's own files.
  (1) Raw HF tokenizers don't absorb generation_config.json's eos list
  (gemma-4 declares [1, 106, 50] incl. `<turn|>`; the tokenizer said 1) --
  `extend_eos_from_generation_config` now unions it at load. (2) mlx-lm's
  `stream_generate` auto-wraps raw tokenizers with only the single
  `eos_token_id` -- `run_generation` now wraps raw tokenizers itself
  (`ensure_gen_tokenizer`) with the full resolved stop set.
- **Removed the decode-path special-token hygiene patch** (v1.34.5x
  `tokenizer_hygiene.py`): defaulting every `decode()` to
  `skip_special_tokens=True` stripped structural channel markers before
  the reasoning parser could see them (gemma-4 thinking could never split
  on the VLM path). The parsers themselves strip declared specials from
  ROUTED text (`strip_tokens`), which is the correct layer. The vision
  first-token decode also yields raw now (a leading `<|channel>` must
  reach the parser).
- Live-verified on gemma-4-31b (vision + text, think on/off, 1/2/large
  images): thinking splits into the thinking field everywhere, turns
  terminate, multi-image + thinking works. The previously reported
  multi-image + thinking gibberish did not reproduce on the fixed stack.

## [1.34.62]

### Fixed

- **Gemma-4 canonical template was rejected by the stop-token gate**, which
  silently disabled thinking parsing + capability sniffing (template_info
  emptied -> PassThroughParser -> thinking leaked inline as plain text with
  the channel markers stripped -- the "thought" first-line leak). Cause:
  `_read_eos_tokens` resolved eos ids only via tokenizer_config.json's
  `added_tokens_decoder`, which gemma-4 fast tokenizers don't have; the
  `<turn|>` terminator (eos id 106) lives in tokenizer.json's
  `added_tokens`. The id map now unions both files (same dual-source rule
  as `_read_special_tokens`). Generation itself was never affected
  (transformers loads the jinja directly); only the sniffing layer was
  blind. The existing v3 collapsible thinking block now receives the
  split thinking as designed.

## [1.34.61]

### Changed

- v3 chat composer: attach button is now an icon (was "+ Img"), and a
  thinking-toggle icon sits next to it for thinking-capable models (same
  capability gate + true/unset semantics as the drawer checkbox, kept in
  sync via onSettingsChange; pressed state styled off aria-pressed). New
  `.btn--icon` style (40px touch-target floor, currentColor SVGs).

## [1.34.60]

### Added

- **Gemma-4 thinking support, end to end.** The canonical gemma-4 template's
  `<|channel>thought ... <channel|>` reasoning format is now recognized:
  new `GemmaChannelParser` (streaming-safe state machine, `thought` channel
  -> `thinking` field, unknown channels -> content) selected via template
  sniffing (`has_gemma_channel_structure`); previously gemma thinking leaked
  inline into `content` on every path.
- **Thinking capability auto-detected from the template.** A chat template
  referencing `enable_thinking` (Qwen3 <think> blocks, gemma-4 thought
  channels -- transformers forwards extra template kwargs as variables) now
  reports the `thinking` capability on /v1/models without a manual
  models.toml flag -- the v3 thinking checkbox appears for these models.
- v3 chat: multi-image attach hardening -- cap at 8 with an aria-live
  announcement when exceeded, per-image accessible remove labels ("Remove
  image N"). (The strip, picker `multiple`, paste-append, N-block store/wire
  shipped in v1.34.20; this closes the cap + a11y gaps.)

### Fixed

- **VLM template path dropped `enable_thinking` entirely** (both the
  text-only-request leg and the image leg via `prepare_vlm_inputs_parallel`):
  the flag never reached `apply_chat_template`, so thinking was
  uncontrollable on VLM-loaded models (mlx-lm's TokenizerWrapper silently
  injects `true` when absent). Both legs now forward the resolved bool
  (request > model config default).
- **Messages API used a hardcoded `<think>`-only parser** in streaming AND
  non-streaming; both now route through `select_reasoning_parser` like
  chat/completions (harmony/gemma/think formats all split correctly).

## [1.34.59]

### Added

- **`mlx_cache_limit_gb` operational setting** (`/v1/admin/config`, default
  unset): opt-in cap on MLX's buffer cache. The allocator keeps freed buffers
  for reuse and never returns them to the OS, so server RSS pins at the
  prompt-spike high-water mark; the cap bounds idle RSS (useful when other
  memory-hungry jobs, e.g. lens fitting, share the box). Clearing the override
  restores MLX's own default, captured from the first apply.

### Fixed

- `DELETE /v1/admin/config/{key}` now re-applies settings immediately like PUT
  does; previously a reset only took effect after a restart while GET already
  reported the default as effective.

## [1.34.58]

### Changed

- **Per-document sampler settings unified across chat AND notebook** (no branched
  copies of the same wiring):
  - Backend: `notebooks` gain a `params` JSON column (like conversations);
    `_SCHEMA_VERSION` 4 -> 5. Conversations + notebooks share ONE encode/decode
    pair (`_encode_params`/`_decode_params` in db.py). Threaded through notebook
    create/update + `Notebook{Create,Update}` + `PUT /v1/notebooks/{id}`. 3 tests.
  - Frontend: one shared `bindDocumentParams({activeId, updateDoc, onError})` +
    `hydrateDocParams(doc)` in settings.js. Both chat.js and notebook.js call them
    (chat's bespoke `saveConversationParams` copy removed) -- sampler knobs bind to
    the active conversation/notebook's `params`, hydrate silently on select,
    debounce-PUT on change, and carry forward on create.

## [1.34.57]

### Changed

- **Sampler settings are now per-conversation** (v3, settings-storage unification
  frontend): the drawer's sampler knobs bind to the active conversation's `params`
  (server), mirroring the per-conversation system-prompt editor -- no longer
  browser-only global state. On conversation select the panel hydrates from
  `conversation.params` (silent, no re-PUT); a knob change / preset apply
  debounce-PUTs `{params}` to the conversation; new chats + first-send create with
  the current panel (`snapshotSettings()`), so knobs carry forward. `settings.js`
  gains `onSettingsChange` (mirrors `onDisplayChange`) + a `silent` hydrate option
  on `applySettings`. localStorage is now just the new-chat seed. Resolves the
  "some settings in browser, some on server" split (redesign note §3b).

## [1.34.56]

### Added

- **Per-conversation sampler settings** (settings-storage unification, backend):
  conversations gain a `params` JSON column (temperature, top_p, ... -- next to
  `system_prompt`, so per-conversation tuning lives WITH the conversation on the
  server instead of split into browser localStorage). Threaded through
  `create_conversation`/`update_conversation` + `Conversation{Create,Update}` +
  `PUT /v1/conversations/{id}`. `_SCHEMA_VERSION` 3 -> 4 (drops tables per the
  solo-deploy policy -- accepted). Frontend wiring (hydrate the settings drawer
  from `conversation.params`, PUT on change) is the v3 lane. 5 tests.

## [1.34.55]

### Fixed

Chat-template robustness -- a broken/corrupted `chat_template.jinja` no longer
causes silent runaway generation (owner hit this after a re-import force-installed
a gemma jinja that emitted `<|turn>model` with no `<end_of_turn>`, so the model
generated to the max_tokens cap):

- **Load-time self-heal** (`read_template_info`): a resolved template that renders
  none of the model's OWN stop tokens is rejected; resolution walks the remaining
  file sources for a valid one, and if none, installs nothing so the loader's
  built-in template stands (rescues already-broken configs without a re-import).
- **Import-time guard** (`detect_chat_template_source`): a stop-less
  `chat_template.jinja` is no longer recorded as `chat_template_source = jinja`.
- **Non-hardcoded** stop-token detection: `_read_eos_tokens` reads the model's
  declared `eos_token`/`eos_token_id` (resolved via `added_tokens_decoder`, across
  `tokenizer_config.json` + `generation_config.json`) -- we validate against the
  model's real stop set, never a hardcoded marker list. Conservative: an empty
  template, or a model whose stop set can't be determined, is never rejected.
- 6 tests.

## [1.34.54]

### Fixed

- **Aborted / manually-stopped streaming requests are now logged** (owner report):
  on client disconnect, `GeneratorExit`/`CancelledError` is thrown into the
  streaming generator mid-yield and unwinds past the normal finalizer, so a
  stopped request produced NO `request_complete` and left a silent gap. A new
  `except (GeneratorExit, CancelledError)` on the stream loop emits a partial
  `request_complete` (`success=false`, `stop_reason="abort"`, partial
  `completion_tokens` + elapsed) then re-raises. Sync-only in the handler (no
  await/yield during unwind). `provider` type derivation extracted to a shared
  `_provider_type` helper.

## [1.34.53]

### Fixed

/code-review pass on the observability + config work:

- **record_event field collisions** (correctness): the ingestion API took
  `**fields`, so a diag caller or v3 client sending a key named `source`/`tier`/
  `min_level`/`type` would either raise `TypeError` (500 + dropped batch on the
  telemetry endpoint; broke `diag_event`'s best-effort on the request path) or
  silently override the record's own `type`/`source`. `record_event` now takes an
  explicit `fields=<dict>` (reserved record keys spread last, always win); all
  call sites updated.
- **Rotation clobber**: two same-second rolls overwrote the earlier archive
  (`rename` is silent on POSIX). Rotation now picks a non-existing archive name.
- **Config stored raw value**: `PUT /v1/admin/config` persisted the request value,
  not the Pydantic-coerced one (`stored` could be `"30"` vs effective `30`); now
  stores the coerced value.

### Documentation

- CLAUDE.md, README, and `docs/observability_guide.md` updated for the spine
  (`logs/metrics.jsonl` + `logs/events.jsonl`), `observability_level` as the single
  control, the `logs/` location, and the accurate content model (metrics tier
  content-free; events tier may carry bounded error text -- `minimal` is NOT
  "content-free").

## [1.34.52]

### Changed

- **`observability_level` is now the master telemetry kill switch** (unifying the
  control, owner ask): setting it to `off` silences memory.py's legacy streams
  (`request_events`/`model_events`/`memory_baseline`) too, not just the spine --
  one place to turn ALL telemetry off. Additive gate; the legacy per-stream env
  toggles still work for granular control. (Full retirement of those toggles +
  removing the streams the spine now duplicates is deferred until the spine is
  fully live-verified -- deleting proven streams for an unverified replacement is
  the risk the live check just caught with the provider bug.)
- Tests: an autouse fixture resets the observability global level per test
  (deterministic; the level is a mutated module global).

## [1.34.51]

### Fixed

- Telemetry `provider` field was always null (found via live verification): the
  per-request emission + memory.py's `provider_type` read `getattr(provider,
  "provider")` on the provider OBJECT, which has no such attribute. Now derived
  from the provider class (`MLXEmbeddingProvider` -> `mlx_embedding`, else `mlx`).
  Live run confirmed the text/mlx-lm path: `request_complete` with
  `effective_loader=mlx-lm`, `is_vlm=false`, and real token/tps/memory metrics.

## [1.34.50]

### Changed

- **Operational settings are DB-authoritative -- the env-override layer is removed**
  (owner feedback: env silently overriding a value set in the admin UI is a
  footgun). `observability_level`/`retention` now resolve DB > default only; there
  is no `HEYLOOK_OBSERVABILITY_LEVEL`. Env vars are reserved for bootstrap paths
  that have no UI counterpart and thus can't conflict (`HEYLOOK_LOGS_DIR`,
  `HEYLOOK_DB_PATH`). The `/v1/admin/config` response drops `env_overrides`.

## [1.34.49]

### Added

- `POST /v1/telemetry/events` (redesign Phase 3, backend): frontend telemetry
  ingestion. v3's client logger batches events (JS errors, fetch failures, stream
  stalls) and posts them here; each is appended to the observability events stream
  with `source=frontend-v3`, level-gated, batch- and field-size-bounded. Metadata
  only. 5 tests.

## [1.34.48]

### Changed

Error/event consolidation (redesign Phase 2) -- one writer, one schema for
`logs/events.jsonl`:

- `diagnostic_logger.diag_event` now **delegates to `observability.record_event`**
  (events tier) instead of owning its own file writer + rotation. All api.py/
  router.py call sites are unchanged. Consequences: diag fields are **flattened**
  onto the record (queryable top-level keys) instead of nested under `data`;
  the diag `level` (severity) is carried as a field and mapped to the spine's
  verbosity gate (errors/warnings surface at `minimal`, info at `standard`); and
  events are now level-gated + rotated like the rest of the spine. `exception_detail`
  is unchanged. `HEYLOOK_DIAG_LOG` is retired (use `HEYLOOK_LOGS_DIR`).
- Tests: `HEYLOOK_LOGS_DIR` is isolated to a temp dir in the root conftest, so the
  suite no longer writes telemetry into the repo's `logs/` (also fixes a
  pre-existing `events.jsonl` pollution).

## [1.34.47]

### Added

- Per-request metrics emission (redesign Phase 1): completed requests now emit a
  content-free `request_complete` line to `logs/metrics.jsonl` (tokens, tps, ttft,
  timings, peak memory, kv bytes, cache, stop reason, image count), plus the
  frozen §4.3 registry dims (`provider`/`effective_loader`/`is_vlm`) read null-safely
  via `getattr` -- an embedding provider yields null, not a crash. This is the
  highest-value telemetry the spine carries. 3 tests.

## [1.34.46]

### Added

Observability spine wired live (redesign Phase 1, slices 2-5):

- **Startup wiring + disclosure**: the spine is `configure()`d from the settings
  layer at boot (level/retention resolved env > DB > default), and a startup log
  line discloses what's written and that it's local ("nothing transmitted").
  `/v1/admin/config` PUT refreshes the in-process cache so a level change takes
  effect immediately.
- **Env hardening**: `resolve_settings_safe` never raises -- a bad `HEYLOOK_*`
  value falls back to defaults + a warning (and is surfaced in the config API
  response) instead of crashing startup.
- **File rotation**: `logs/*.jsonl` streams roll past a size cap to timestamped
  archives; archives older than the retention window (default 30d) are swept
  hourly on the maintenance tick. Best-effort, never raises.
- **`internal/log/` reconcile**: `memory.py`'s telemetry streams now write under
  `logs/` (runtime data), not `internal/log/` (human session diaries). Overridable
  via `HEYLOOK_LOGS_DIR`.
- **First real emitters**: model load/unload/evict now emit `record_event` (events
  tier) alongside the existing diagnostics.

### Changed

- README monitoring paths updated to `logs/`.

## [1.34.45]

### Added

Observability spine core (redesign Phase 1, slice 1) -- the single JSONL
ingestion path. Internal foundation; wiring + emission from real call sites and
the `internal/log/` reconcile follow in subsequent slices:

- `observability.record_event(type, *, tier, min_level, source, **fields)` --
  appends one JSON line (with `ts` + local-time `iso`) to the right stream
  (`logs/metrics.jsonl` content-free, `logs/events.jsonl` correlated), gated by
  the configured verbosity (`off < minimal < standard < debug`). **Best-effort:
  never raises** (observability must not break inference). Level + log dir are
  cached in-process (`configure()`), so the hot path never touches the DB.
- 8 tests (write shape, tier routing, level gating, never-raises).

## [1.34.44]

### Added

Config foundation (observability + config redesign, Phase 0) -- the runtime-mutable
operational-settings layer that the JSONL observability spine will build on:

- App DB `settings` table (key -> JSON value), added additively alongside presets.
  Schema-stable (a new setting is a new row, never DDL), so it survives the
  drop/recreate schema policy without a drop-list carve-out; treated as config,
  not data. Store CRUD in `db.py` (`get_setting`/`get_all_settings`/`set_setting`/
  `delete_setting`).
- `settings.py`: the `SettingsSchema` Pydantic contract (types, defaults,
  `extra="forbid"`) + `resolve_settings()` with **env > DB > default** precedence
  (`HEYLOOK_<FIELD>` overrides stay the always-wins escape hatch). First fields:
  `observability_level` (off/minimal/standard/debug, default minimal) and
  `observability_retention_days` (default 30) -- consumed by the Phase 1 spine.
- `/v1/admin/config` router (GET effective+stored+env-overrides, PUT validated
  updates, DELETE resets a key) -- the backend for the v3 admin/settings config
  panel. 422 on unknown key / invalid value before anything persists.
- 23 tests (store, resolver precedence, HTTP contract); no new database (reuses
  the App DB), no registry change.

## [1.34.43]

### Added

- Model registry now describes modality and engine routing as two separate
  fields on `MLXModelConfig`, decoupling what a model IS from how it loads
  (Phase 6 refinement; see `docs/project/plan_2026-07.md`):
  - `modalities: list[str]` -- author-declared capability set
    (`text`/`vision`/`audio`/`video`), detected at import from the config's own
    `vision_config`/`audio_config` blocks + `*_token_id`/`image_token_index` keys
    (`model_importer.detect_modalities`), with `mmproj`-style files as a
    fallback. Represents genuinely multi-modal models (e.g. gemma-4 declares
    text+vision+audio) that a single `vision` bool could not. Validated against a
    19-model modelzoo audit (LLaVA/Mistral use `image_token_index`, not `_id`).
  - `loader: "auto" | "mlx-vlm" | "mlx-lm"` -- engine routing within
    `provider="mlx"`. `auto` picks mlx-vlm when the model declares vision AND
    mlx-vlm registers its `model_type`, else mlx-lm; it degrades to mlx-lm only
    on POSITIVE knowledge that mlx-vlm lacks the type (uncertainty keeps the
    historical vision->mlx-vlm default). An explicit value forces the engine
    (e.g. run a dual-capable VLM as text via `mlx-lm`). Resolution lives in
    `providers/common/loader_routing.py`; `is_vlm` + a new
    `MLXProvider.effective_loader` derive from it.
- `/v1/models` entries now carry `modalities` (full description); `capabilities`
  stays gated to what the server actually serves (image input) -- description !=
  served.

### Changed

- `vision: bool` (MLXModelConfig) is demoted to a derived mirror of
  `"vision" in modalities` (kept for back-compat readers of `config["vision"]`);
  `modalities` is authoritative. Absent `modalities` derives from `vision`, so
  existing `models.toml` entries and the provider load path are unchanged. The
  richer modality set (e.g. audio) lands on re-import.

## [1.34.42]

### Fixed

- jspace `/v1/jspace/analyze` crashed (`AttributeError: 'NoneType' object has no
  attribute 'offset'`) on hybrid mlx-vlm models -- specifically Qwen3.5 (the
  KVCache+ArraysCache GDN architecture). Their full-attention block dereferences
  `cache.offset` with no None-guard, so the cache-less inner forward the lens
  used for read-out (`ModelAdapter.logits` / `capture_residuals`) blew up. gemma
  was unaffected (its attention tolerates a missing cache). The adapter now
  sources a fresh, empty per-layer cache from the model's own `make_cache()`
  (length-matched to the block count) and passes it into every inner forward;
  each analyze forward re-prefills the whole sequence, so a throwaway offset-0
  cache reproduces the old no-cache semantics. Models without a matching
  `make_cache` still run cache-less, unchanged.

## [1.34.41]

### Changed

Diagnostic log (`logs/events.jsonl`) now actually explains errors:

- Every event carries a human-readable `iso` field (local time with UTC offset)
  alongside the epoch `ts` -- `ts` stays authoritative for sorting/latency math,
  `iso` makes the file legible without converting epoch seconds by hand.
- `request_error` events now record `error_type` (the exception class), `stage`
  (where in setup it failed: routing / provider_get / capacity_check /
  generator_create / streaming), `model`, and -- for wrapped errors -- a `chain`
  of the underlying causes. Previously the record held only `str(e)`, so the
  actual "why" (exception type, root cause) only reached the console logger.
- Mid-stream generation failures are now logged. The streaming `GenerationFailed`
  path previously wrote nothing to `events.jsonl`, leaving a `generation_start`
  with no matching completion; a new catch-all also logs unexpected mid-stream
  errors and closes the SSE stream cleanly (in-band error payload + `[DONE]`)
  instead of propagating a raw exception into an already-started response.
- Cause chains are captured via `traceback.format_exception_only` (type +
  message only, never frame locals), so prompt/response text cannot leak.

## [1.34.40]

### Fixed

Chat-template hardening batch (the quick-fix findings from the v1.34.38 review):

- Server-side import (`/v1/admin/models/import`) and the CLI wizard now share one
  detection helper (`template_info.detect_chat_template_source`) -- the two inline
  copies had already drifted -- and tilde/relative model paths no longer silently
  skip jinja detection (`expanduser`).
- `chat_template_source = "auto"` no longer takes the force-install branch (it is
  documented as fill-only-when-missing; force could clobber a natively-loaded dict
  of named templates). `"chat_template_json"` is now an accepted explicit value --
  it was a resolved-source label users could see in load logs but not configure.
- The missing-template error is decided from tokenizer state
  (`chat_template`/`has_chat_template`) instead of string-matching transformers'
  error prose (version-fragile), mentions all three supported sources, and now
  covers all three apply sites (chat, batch, hidden-states) instead of one.
  mlx-lm wrapper-level python templates (`chat_template_type`, e.g. DeepSeek-V3.2
  conversions) are recognized, so their render errors are no longer mislabeled.
- The load-time "NO chat template" warning consumes `install_chat_template()`'s
  result and checks the wrapper's `has_chat_template`, fixing both a missed-warning
  case (resolved template whose install failed) and a false-alarm case
  (`chat_template_type` models that render fine with a None tokenizer attr).

## [1.34.39]

### Fixed

- **`heatmap_top_k` is now schema-bounded (0-64)** on `POST /v1/jspace/analyze`: previously any
  int was accepted and clamped only to vocab size, letting one request decode band x positions x
  vocab tokens (a multi-GB response) while holding the process-global generation gate.
- v3 jspace interaction fixes from the review pass: arrow-key pin walking now respects the
  layer-range scope (it could land the pin on a hidden row); `scrollIntoView` on the pinned cell
  now fires only for keyboard navigation (strip clicks no longer yank the viewport to the
  heatmap's far-right onset column); non-answer logit bars no longer show a literal "undefined"
  tooltip; the echo highlight accepts the empty-string token (rendered as the empty-token glyph);
  slider drags dedupe unchanged ranges and route the aggregation repaint through the frame
  throttle.

### Tests

- Contract: out-of-range `heatmap_top_k` rejected (422). E2E pages suite grows to 35 checks:
  arrow-walk-respects-scope assertions and a heatmap-off analyze check (strip-only render,
  onset_strip aggregation fallback, strip-row pin) -- the path the review found uncovered.

## [1.34.38]

### Added

- **Chat-template resolution hardened as a registry concern.** The server-side
  scan+import route (`/v1/admin/models/import`, what the v3 models page uses) now
  applies the same `chat_template.jinja` auto-detection as the CLI import wizard,
  recording `chat_template_source = "jinja"` on the imported entry (request
  overrides still win). Template resolution (`template_info.py`) gains a last
  auto-fallback to processor-side `chat_template.json` -- previously a model
  shipping ONLY that file looked template-less at the tokenizer level. In auto
  mode the provider now installs the resolved template onto the tokenizer when
  the tokenizer loaded none (explicit `chat_template_source` still force-installs);
  the install logic is consolidated into a tested `install_chat_template()` helper.
  The v3 models page shows `template: <source>` in each model's meta line when set.

### Fixed

- A model folder with no chat template anywhere (no `chat_template.jinja`, no
  embedded `tokenizer_config.json` template, no `chat_template.json`) previously
  surfaced transformers' raw `ValueError` as the HTTP 500 detail on the first chat
  request. The provider now warns at load time and raises an actionable error
  naming the model and the fix (add a template file or set `chat_template_source`).

## [1.34.37]

### Added

- **Per-cell top-N analyze extension** (`POST /v1/jspace/analyze`): new `heatmap_top_k`
  (default 0 = unchanged) makes each heatmap cell carry `top_k: [{token, logit}...]`
  (reduced on-device via argpartition -- the full per-layer logits never leave the GPU
  path). The v3 jspace page sends it with the heatmap toggle, so pinning ANY cell now
  shows its full silent-token readout (spec section 4 updated in the same commit).
- **J-space visualizer sequence item 2 -- layer-range slider + aggregation** (v3, pure
  client-side): a slot-per-band-layer slider (click = one layer, drag = a contiguous
  range, hover = live preview, reset) scopes the strip and heatmap rows; the detail
  panel's unpinned state becomes a most-common-silent-tokens aggregation (top-k
  appearance counts over the scoped layers, heatmap-wide when per-cell top-k data
  exists), and clicking an aggregation row echo-highlights where that token wins in
  the grid.
- **Lens provenance surfaced**: `GET /v1/jspace/models` now returns
  `meta: {model: {provisional, fit_date, fit_source, n_prompts}}` from the lens
  sidecar (`provisional` = no own-fit `hf_model_name` stamp); the jspace page shows a
  "provisional lens" badge for such models. Pairs with the fitting-track change that
  stamps own-fit sidecars with provenance.

### Tests

- analyze() pipeline now unit-tested end-to-end on a tiny random-weight gpt2 (per-cell
  top-k shape/ordering + back-compat cell shape); registry provenance unit test;
  contract tests for heatmap_top_k forwarding and models-meta. E2E pages suite grows
  to 34 checks (aggregation panel, slider scoping/reset; the non-onset pin check now
  expects full bars).

## [1.34.36]

### Added

- **v3 design language seeded** (`apps/heylook-frontend-v3/DESIGN.md`, plan Phase 4 item 2):
  formalizes the token roles and the OKLCH data-strength chip formula that lived only in
  `css/app.css` comments, defines the selection/pin grammar, and records the j-space
  visualizer paradigm decision (matrix-first; Neuronpedia-style layer-range slider +
  aggregation sidebar as the growth path).
- **J-space visualizer sequence item 1 -- click-to-pin readout** (v3 `jspace` page, no
  backend change): workspace strip rows and heatmap cells pin a per-(layer, position)
  detail panel. Answer-onset pins (the strip; the heatmap's last column) show the full
  top-k silent tokens as logit bars with the first-answer-token emphasized; other cells
  show top-1 + entropy with a note that the per-cell top-N analyze extension (scoped in
  TODO.md) unlocks them. Esc unpins, arrow keys walk layers/positions, cells sharing the
  pinned cell's top token get an echo highlight, and the heatmap gains a prompt-token
  header row with an answer-onset column marker.

### Tests

- E2E pages suite extends the lens-gated jspace block to 32 checks total (heatmap render +
  onset marker, row pin, Escape unpin, non-onset cell pin/unpin); 32/32 live-green.

## [1.34.35]

### Changed

- `scripts/jspace_convert_lens.py` tolerates `--out-dir` already ending in the model id
  (avoids the double-nested `adapters/jspace/<id>/<id>/` that the registry never sees).

### Docs

- README, CLAUDE.md, and `docs/architecture/api.md` document the j-space feature + endpoints;
  CLAUDE.md gains the MLX off-event-loop / thread-stream, lazy-`mx.load`, and pipeline-`.layers`
  gotchas.

## [1.34.34]

### Fixed

- **`/v1/jspace/analyze` crashed the process from the frontend** (`There is no Stream(cpu, 0) in
  current thread`). Two causes, both fixed: (1) it ran the MLX forwards on starlette's ephemeral
  `run_in_threadpool` worker, which has no thread-local MLX stream (and a dying MLX thread aborts
  the process). Analyze now runs on a **pinned `mlx-stream` executor** (`streaming_utils._executor_pool`,
  same as generation) inside `mx.stream(generation_stream)`. (2) The lens `J` matrices are
  `mx.load`'d lazily (mmap-backed); their first eval landed on the worker thread and dispatched to
  the CPU default stream that thread lacks. `JSpaceLens` now force-evaluates them at load time
  (on the loading thread). Verified end-to-end on the served 26B MoE via a worker thread.

### Changed

- **Convert helper is easier to run** (`scripts/jspace_convert_lens.py`): carries PEP 723 inline
  deps, so `uv run scripts/jspace_convert_lens.py …` provisions torch+jlens itself (a clear error
  points there if run via the torch-less server venv); accepts a lens *directory* (finds the single
  `*_jacobian_lens.pt`, e.g. a neuronpedia model dir); rejects a path-shaped `--model-id`.

## [1.34.33]

### Fixed

J-space code-review fixes (xhigh review of the feature branch):

- **Capture is pipeline-safe (#1).** `ModelAdapter` now mutates the underlying block
  list on the text decoder (`inner.layers`/`.h`), not the top `model.layers` property,
  which for pipeline-parallel models (Qwen3.5, deepseek, glm4_moe) returns a FRESH slice
  each access — so capture was silently recording nothing → KeyError 500. `capture_residuals`
  now also raises a clear error if a forward never hits the recorders (instead of a silent
  empty read-out).
- **Analyze respects the concurrency invariants (#2/#3/#4).** `/v1/jspace/analyze` now pins the
  model (no LRU-evict / idle-unload mid-analyze) and runs the forwards under the process-global
  FIFO generation gate, so it serializes with generation and other analyze calls — no concurrent
  Metal command buffers (the documented crash class) and no racing mutation of the shared block list.
- **Unguarded 500s (#5/#6/#7).** Lens/normalizer/router load inside the error handler; `has()`
  requires the sidecar too (a partial convert no longer 404-passes then 500s); `router()` picks an
  available variant instead of KeyError-ing on a missing 'combined'; `format_prompt` accepts
  OpenAI content-block (list) messages instead of TypeError-ing on the default path.
- **Feature correctness (#8/#9/#10).** Reuse `resolve_stop_tokens` (honors plural `eos_token_ids`);
  `greedy_generate` returns the real first token even when it's a stop token (no redundant fallback
  forward); `workspace_readout` tolerates empty hedge sets and out-of-vocab ids.
- **E2E (#11).** The jspace mount check waits for the `/v1/jspace/models` fetch to resolve before
  asserting (was reading the select before it populated).

### Changed

- J-space efficiency + robustness: heatmap reduces to (top-1, entropy) on-device instead of
  materializing full-vocab float64 host arrays; lens/model `d_model` mismatch raises a clear error;
  `LensRegistry.from_env` falls back to `<cwd>/adapters/jspace` for non-editable installs; the
  convert script serializes the sidecar before the safetensors; the risk badge uses `Number.isFinite`.

## [1.34.32]

### Added

- **Lens convert+register helper + `adapters/` store.** `scripts/jspace_convert_lens.py` (git-tracked;
  torch+jlens, separate env) downloads/loads a jlens `.pt` and writes
  `adapters/jspace/<model_id>/lens.safetensors` + sidecar. `adapters/` is git-tracked via `.gitkeep`
  with gitignored contents (mirrors `modelzoo/`); `LensRegistry.from_env` now defaults there (repo
  root), so a converted lens is served with zero config (`HEYLOOK_JSPACE_DIR` still overrides).
- **E2E coverage for the J-Space page** (`tests/e2e/suites/pages.mjs`, lens-gated): mounts the page
  and, when a lens for the E2E model is installed, drives Analyze and asserts the workspace strip
  renders. Cuts the 26B-reload cost of future iteration.

### Docs

- **`docs/jspace_guide.md`** -- how-it-works + end-to-end tutorial (install a lens, call
  `/v1/jspace/analyze`, use the v3 J-Space page, interpret the output, enable risk). Indexed in
  `docs/README.md` + CLAUDE.md.

## [1.34.31]

### Added

- **v3 `J-Space` page** (`apps/heylook-frontend-v3/js/pages/jspace.js`, new `jspace` nav route).
  Model picker (lens-gated), prompt, `raw`/`chat` + heatmap toggles; renders the layer×top-k
  "silent words" strip (colored by within-layer rank), an optional layer×position heatmap (colored
  by confidence), and a hallucination-risk badge. Reuses the explore-chip OKLCH formula.

### Changed

- **`/v1/jspace/analyze` defaults to raw-completion prompting** (`chat=false`). The chat template's
  final position is the generation-prompt boundary, where the lens top-k is formatting junk; a raw
  completion reads a real content token so the workspace surfaces sensible tokens (verified on the
  served 26B MoE: "...the city of" -> Paris). `chat=true` (chat template) is kept for the risk
  features. Answer decode now strips special tokens.

## [1.34.30]

### Added

- **`/v1/jspace` interpretability API** (`jspace_api.py`, tag `JSpace`). `GET /v1/jspace/models`
  lists served models with a fitted lens; `POST /v1/jspace/analyze` formats the prompt exactly as
  the provider does (chat template + `<bos>`), greedily generates a short answer, captures the
  residual stream, and returns the Jacobian-lens workspace: per-band-layer top-k "silent" tokens
  at the answer-onset, an optional layer x position heatmap, workspace features, and (when a
  per-model normalizer + router are configured) a hallucination-risk score. New
  `jspace/analyze.py` (pipeline) + `jspace/registry.py` (`HEYLOOK_JSPACE_DIR/<model_id>/` lens
  cache; offline-converted safetensors). Lenses are loaded, never converted, at runtime.
  Registry unit tests + endpoint contract tests (routing/guards, no model needed).

## [1.34.29]

### Fixed

- **J-space `_Recorder` proxies attribute access to the wrapped block** (`capture.py`). The
  gemma-4 text forward reads `layer.layer_type` during mask construction, so the temporary
  capture wrapper must delegate attributes -- without this, residual capture on the served
  gemma-4 VLM raised `AttributeError`. Verified end-to-end on `gemma-4-26b-a4b-it-8bit-mlx`:
  the late-band workspace surfaces the correct entity (e.g. "Eiffel Tower ... city of" -> Paris),
  confirming the MoE capture point (through 128-expert routing) and 8-bit-lens transfer. NOTE:
  gemma requires an explicit `<bos>` (id 2) or the residual stream degrades to garbage.

### Changed

- **Bumped mlx-lm / mlx-vlm to latest upstream commits** (mlx-lm `a790972`: quantized-SDPA GQA
  batched-padding crash fix; mlx-vlm `05440cc5`). Same versions (0.31.3 / 0.6.5), newer commits.

## [1.34.28]

### Changed

- **J-space `ModelAdapter` now resolves multimodal-wrapper nesting** (`capture.py`). Walks
  `model` -> `.model` / `.language_model` -> `.model` to find the text decoder, softcap, and
  tied/untied head -- so it handles the served gemma-4 VLM (text stack under
  `model.language_model.model`, `final_logit_softcapping` on `language_model`) as well as the
  flat mlx-lm gpt2/gemma layouts. New unit test covers the nested-VLM resolution.

## [1.34.27]

### Added

- **J-space workspace features + hallucination-risk router** `src/heylook_llm/jspace/features.py`.
  Reproduces solarkyle/jspace's feature math: `workspace_readout` (per-band-layer ignition /
  rank / entropy / hedge-rank from lens logits), `router_feature_vector` (the 10 named features),
  `baseline_features` (output-confidence), and `HallucinationRouter` + `FeatureNormalizer`
  (per-model z-scored logistic regression predicting P(answer wrong)). Verified against the
  shipped e4b TriviaQA trace: AUC 0.795 workspace-only / 0.815 combined, both beating the
  first-token-logprob baseline (0.771) -- the paper's "workspace beats/adds to output confidence"
  result. Download-free unit tests in `tests/unit/test_jspace_features.py`.

## [1.34.26]

### Added

- **J-space (Jacobian lens) backend core module** `src/heylook_llm/jspace/` -- a post-hoc
  interpretability read-out of a model's verbalizable "workspace" (per-layer, which vocabulary
  tokens a residual is disposed toward), from Anthropic's July-2026 global-workspace work.
  `capture.py` (`ModelAdapter` architecture-introspection + `capture_residuals` via a temporary
  block wrapper) and `lens.py` (`JSpaceLens`: load converted safetensors + sidecar, transport,
  apply through the model's real head so gemma soft-cap / tied embeddings stay correct).
  Download-free unit tests in `tests/unit/test_jspace.py`; apply-parity verified cos ~1.0 vs the
  genuine reference `jlens` on gpt2 (no softcap) and gemma-2-2b (RMSNorm + softcap 30). Not yet
  wired to an endpoint -- see [docs/jspace_integration_plan.md](docs/jspace_integration_plan.md).

### Docs

- Promoted the j-space build + verifier plan into git-tracked `docs/jspace_integration_plan.md`
  (indexed in `docs/README.md` + CLAUDE.md Orient).

## [1.34.25]

### Removed

- **Legacy React frontend `apps/heylook-frontend/` deleted.** v3 (served at `/v3`) has
  parity and the app was no longer mounted by the backend. Retired with it: the OpenAPI
  drift guard that served only the legacy app -- `scripts/check_openapi_sync.sh`, the
  pre-commit OpenAPI block, and the `/openapi-regen` skill (v3 hand-writes `js/api.js`;
  nothing consumes the generated TS types). The live schema stays at `/openapi.json` +
  `/docs`. Also deleted the stale `docs/frontend_api_reference.md` (React-era integration
  guide, superseded by the v3 spec §4 + live OpenAPI).

### Docs

- CLAUDE.md, README.md, tests/README.md, docs/frontend_v3_spec.md, .gitignore, and a
  `perf_collector` docstring updated to drop references to the removed app/guard and to
  point at v3 (spec §4, `tests/e2e/`). New `internal/frontend/v3.md` (gitignored) maps
  v3's done/left + the backend<->v3 coupling; CLAUDE.md Orient now names
  `docs/project/plan_2026-07.md` as the roadmap.

## [1.34.24]

### Fixed

- **`/code-review` pass over v1.34.22-.23** (8 finder angles; 10 findings reported, 5 finders
  independently converged on the top two): the background preset-refresh repaint no longer fires when
  the list is unchanged or while the cursor is inside the panel (it destroyed uncommitted text on
  every open); `presets` removed from the schema-recreate drop list (a `_SCHEMA_VERSION` bump would
  have wiped saved presets despite the config-not-data promise — regression-tested); preset `params`
  that orjson can't serialize (>64-bit ints) now 400 instead of 500; `savePreset` decides
  create-vs-overwrite against a freshly fetched list (fixes the stale-cache 404 mirror of the 409 and
  the wrong "saved" toast, and replaces the nested 409 retry); the New button carries a pre-create
  draft prompt (send()'s implicit create already did); a prompt/preset applied while the first-send
  create is in flight is delivered to the new conversation instead of reverted; a stale sysprompt
  blur now PUTs to the conversation the textarea was built for instead of dropping the edit;
  `resetSettings()` = `applySettings({})`; spec §4 points at `PARAM_META` instead of re-enumerating
  knobs; E2E preset-option lookup deduplicated. Skipped as design/pre-existing: cap-gated
  `enable_thinking` pinning via global settings (predates presets; needs caps-aware settings),
  unknown-params round-trip stripping, apply-copies-null-prompt. 882 green; E2E 55/55 live.

## [1.34.23]

### Changed

- **`/simplify` pass over v1.34.22** (4 review angles: reuse, simplification, efficiency, altitude;
  applied 7, skipped 4 as deliberate design or house-pattern): settings.js `mergeKnown()` unifies the
  twice-inlined known-keys merge and `samplerParams()` now derives from `snapshotSettings()`; the
  one-caller `lead` option reverted in favor of `panel.prepend(...)` in chat.js; the open-panel
  freshness guard moved inside `rebuildSettingsPanel()` (call sites unconditional, plus a rebind after
  the implicit first-send create); the sysprompt textarea captures its conversation id at build so a
  stale blur can't write one conversation's prompt onto another; settings panel opens instantly from
  cached presets (refresh repaints in the background); preset save-by-name catches a stale-cache 409
  and retries as overwrite; preset router drops its duplicated field-allowlist (db layer enforces).
  Skipped by design: unifying user presets with the TOML sampler registry (client-side copy semantics
  is intentional), generic `_update_row` extraction, shared sysprompt component. E2E chat 28/28 live.

## [1.34.22]

### Added

- **Per-conversation system prompt editing + saved presets in v3** (LM-Studio-style). Backend: new
  `presets` table in the DuckDB store (additive `CREATE TABLE IF NOT EXISTS` — no schema-version bump,
  existing data untouched) holding named `system_prompt` + sampler-`params` bundles; name uniqueness
  enforced in code on the store's single serialized writer; presets deliberately survive
  `POST /v1/data/clear` (config, not data). New `/v1/presets` router (list/create/update/delete;
  409 on name collision, 400 on bad fields) — spec §4 + `generated-api.ts` updated in this commit.
  v3 chat settings panel: a per-conversation system-prompt editor (PUTs to the conversation on blur;
  a prompt typed before the first send rides along on create) and a preset bar (apply = copy params
  into the panel + prompt onto the conversation; save-by-name creates or overwrites; armed delete).
  These are UI-authored and expanded client-side — distinct from the server's TOML preset registry
  (`ChatRequest.preset`). Tests: +25 unit (store + HTTP), suites 880 green; E2E +3 checks, 55/55 live.

## [1.34.21]

### Fixed

- **Code-review pass over v1.34.20** (8 finder angles, 3 independently verified system claims; all fixes regression-tested, +6 tests, suites 855 green + E2E):
  - Store ops are now transactional (BEGIN/COMMIT with rollback-on-exception): DuckDB autocommits per statement, so a crash mid-operation could previously orphan rows or leave stale `updated_at`; an unhandled error could also have wedged the long-lived connection until ROLLBACK (verified live).
  - Store runs on its own dedicated single worker thread instead of asyncio's shared default executor, where multi-second model loads and full generation-consumption loops could starve trivial conversation reads (verified: `to_thread` = the shared pool; aiosqlite previously had its own thread). The threading.Lock became redundant and was removed.
  - `duckdb.connect` retries the file lock for up to 10s (parity with the old aiosqlite `timeout=10`); previously a restart racing the old process's lock hard-failed startup instantly (verified live).
  - Content blocks are validated at the storage boundary: `{"type":"text","text":null}` no longer poisons a row (flatten would TypeError on every subsequent read, making the conversation permanently unreadable -- repro'd), and a malformed image block (missing/invalid `source`) is a 400 instead of persisting and crashing the whole conversation render client-side. Unknown block types still pass through (forward-compatible). FK dropped from the schema (DuckDB's FK check rejects parent deletes even with children deleted in the same transaction -- documented limitation; integrity enforced in code) with a schema v3 recreate for any same-day v2 file.
  - v3 chat: staged images are cleared on conversation switch/new (previously a photo picked in conversation A silently attached to the next send in B); one `imageBlockUrl()` helper feeds both rendering and wire conversion and handles `url`-type sources (previously `data:undefined;base64,undefined`); Edit-save syncs `content_blocks` from the server response; Copy hidden on image-only messages (copied empty string); multi-file reads parallelized.
  - db.py: `_COLS` derived from `_NAMES` (zip could silently mispair on drift), `_touch_conversation` helper, `update_message` merges locally instead of re-SELECTing multi-MB rows; README + conversation_api docstring no longer say SQLite; `*.duckdb` added to the gitignore safety net.
- Noted, deliberately not fixed here (recorded in the plan): the schema module's flat `ImageBlock` vs the stored nested Anthropic shape (the STORED shape is the spec-correct one; Phase 3b conformance reconciles the schema module), per-turn base64 re-upload of full history (Phase 3b design input: server-side history resolution), keyed message rendering.

## [1.34.20]

### Changed

- **Q5 executed: conversations/notebooks store migrated SQLite/aiosqlite -> DuckDB, messages now persist as CONTENT BLOCKS** (`db.py` rewritten; same public surface). Every operation runs on a worker thread under a store lock with explicit statements -- the aiosqlite shared-implicit-transaction defect class is retired by construction (regression test: concurrent appends serialize with correct positions). DuckDB has no ON DELETE CASCADE, so conversation deletes cascade explicitly. No data migration by owner decision: fresh store at `data/conversations.duckdb` (`HEYLOOK_DB_PATH` still honored). aiosqlite dependency removed.
- **Conversation API accepts Messages-style content blocks** (additive): `content` on message create/update takes a string OR a block list (`[{type:"image",source:{type:"base64",media_type,data}},{type:"text",text}]`); responses carry both `content` (flattened text, back-compatible) and `content_blocks` (full list). Spec §4 updated in this commit; `generated-api.ts` regenerated.

### Added

- **Images in the v3 chat UI** (the point of pulling Q5 forward): attach via file picker (iPhone camera roll included) or paste; thumbnail strip with per-image remove; user messages with images stored as content blocks and rendered as images in history (reload included); generation converts stored blocks to OpenAI `image_url` data-URL parts (works against the VLM path today; the conversion disappears when v3 moves to /v1/messages). Editing is hidden on image messages (the text editor would silently drop the blocks) -- delete/regenerate still work. Verified: unit suite green (849), full E2E green, plus a live round-trip (store blocks -> reload byte-identical -> VLM correctly describes the image over the v3 wire shape).

## [1.34.19]

### Fixed

- **optloop-lib spec-decode baseline guard was incomplete** (found by a `/code-review` pass): the CLI-level `--reset-baseline`+`--spec-decode` refusal only caught the explicit flag, but per-model baselines (v1.34.14) mean a spec-decode run against a *not-yet-benched* model hits the implicit `baseline_data is None` branch and would silently write a **speculative** baseline (inflated gen_tps + mismatched fingerprints for all later comparisons). Moved the guard into `run_benchmark` where baseline presence is known, so it fires for both the explicit and implicit cases (before the prompt loop); removed the now-redundant CLI guard.

### Changed

- **`/simplify` cleanup of the session's E2E + optloop code** (4-angle review): shared `resolve_or_download()` in `bench_common.py` collapses the models.toml→HF-download fallback that was copy-pasted across three resolvers; spec-decode result metadata deduped via one `spec_meta` dict; stale text-model default id fixed. E2E harness: new `lib/dom.mjs` helpers (`waitForLabel` for the toggle-button idiom used ~7×, `findModelRow`/`modelRowState` for the models-row lookup duplicated 4× — the value-returning `modelRowState` avoids a handle-per-poll leak, `settingsInputValue`/`setSettingsInput` for the settings panel); `run.mjs` collapses the two identical suite-run blocks into a loop; magic literals (`STOP_TEST_MAX_TOKENS`, cadence thresholds) named. Behavior-identical; Python 70 tests green (re-run the Metal-gated E2E suite to confirm the JS refactors).

## [1.34.18]

### Changed

- **mlx-vlm bumped 0.6.3 -> 0.6.5 in the root venv** (git pin refreshed to upstream e9c5bd7): brings the gemma-4 video/audio-weight loading fixes and the Qwen3-VL mrope fix to the SERVER (they were previously only in optloop-lib's fork clones). Safety net held: the tests/contract/test_mlxvlm_surface.py pins (added v1.34.5 for exactly this moment) plus the full unit suite are green on 0.6.5, and the v3 E2E suite passed 52/52 against a server running it.

## [1.34.17]

### Fixed

- **Five confirmed v3 frontend defects** (from the triaged external review, plan Phase 4 item 3; E2E 52/52 after):
  - Router crash-guard (`js/app.js`): a page that fails to load/mount renders an in-place error panel instead of bricking navigation.
  - Perf page fetches now abort on page teardown (`{signal: ctx.signal}` passed to systemMetrics/perfProfile).
  - Chat settings panel rebuilds on model switch while open, so capability-gated controls (enable_thinking) track the selected model.
  - Notebook content is readOnly during generation -- the streaming painter overwrites textarea.value every frame, so mid-generation keystrokes were silently destroyed; the surface now locks honestly and unlocks on completion/stop/error.
  - Status lines (chat/notebook/explore) moved --ink-faint -> --ink-muted (~3.5:1 -> ~6:1 on white); --ink-faint documented as placeholders-only at the token definition.
  - NOT included on purpose: the enable_thinking tri-state (contract change, coupled to the Messages extension design -- Phase 3b).

## [1.34.16]

### Added

- **First real-vision VLM baselines: gemma-4 dense + MoE** (`docs/optimization_log.md`). The bench's vision path had never run against a real VLM before (the Mar-16 "VLM baseline" was a text model through the loader's text path). Updating the editable forks to the owner's synced versions — **mlx-vlm 0.6.5 (#1529), mlx-lm 0.31.3 (#1431)** (`uv sync` clean, mlx stayed 0.32.0) — resolved the multimodal-RoPE blocker that the stale Mar-15 fork had; gemma-4 dense/MoE and Qwen3-VL all run the manual pre-filled-cache vision path clean. Baselines (8-bit, 14 prompts incl. 9 real photos, runs=3): **dense gemma-4-31b 15.3 gen_tps / 1592ms vision / 33.3 GB**; **MoE gemma-4-26b-a4b 48.1 gen_tps / 524ms vision / 27.3 GB**. The MoE is ~3× faster decode + vision-encode despite similar total params (only ~4B active) — dense is bandwidth-bound, MoE is dispatch-bound, so library optimizations will score very differently on each (the reason to bench both). Per-model baselines keep them separate. Next: the MTP experiment (MoE + the `-assistant-bf16` drafter via mlx-vlm's `draft_kind="mtp"`).

## [1.34.15]

### Changed

- **E2E harness default model → `gemma-4-26b-a4b-it-8bit-mlx`** (`tests/e2e/run.mjs`, README). Any fast A4B MoE in `models.toml` works; override with `E2E_MODEL`. Unverified against this default (needs a run once it's in `models.toml`) — the streaming-cadence guard needs >30 tok/s, which an 8-bit A4B MoE should clear comfortably.

## [1.34.14]

### Added

- **optloop-lib: per-model baselines** so dense vs MoE (and their distinct fingerprints) don't clobber one another. Baselines now live in `data/<bench>/<model-slug>/baseline.json` via a shared `model_bench_dir()` helper. Also `resolve_model_name()` maps an HF cache snapshot path (`.../models--org--name/snapshots/<hash>`) back to the repo short name instead of the opaque hash (the cosmetic model-name bug), and `slugify_model()` keeps the dir filesystem-safe. Applied to both `bench_text.py` and `bench_vlm.py`; +5 unit tests (70 total). Groundwork for benching gemma-4 dense and MoE side by side.

## [1.34.13]

### Fixed

- **optloop-lib VLM bench: advanced the vision baseline past two blockers** (still one open). The `[bench.vlm]` model id was dead (`mlx-community/Qwen3.5-27B-mxfp8-mlx` -- a text model, not local); pointed it at the local vision model `Qwen3-VL-32B-Instruct-8bit`. This also revealed the bench's VISION path had never run against a real VLM -- the Mar-16 "VLM baseline" was a text model through the loader's text path. Ported the server's two transformers-5.x soft-patches (AutoVideoProcessor -> None, lenient ProcessorMixin video check) verbatim into `bench_vlm.py` so Qwen3-VL loads on a torch-free MLX venv. STILL BLOCKED (documented in `docs/optimization_log.md`, not fixed here): Qwen3-VL's 3D multimodal RoPE -- the bench's simplified pre-filled-cache vision path doesn't supply mrope position_ids, so cos/sin broadcast fails against image-expanded queries. Needs either routing vision through `mlx_vlm.generate` or porting the server's wrap_language_model/position-reset. No false baseline was written.

## [1.34.12]

### Added

- **Speculative-decoding baseline -- optloop-lib's first real run** (`docs/optimization_log.md`). Re-established the text baseline on mlx 0.32.0 (`gemma-3-27b-it-bf16`, 6 prompts incl. the new long_context workload; 11.7 gen_tps, matching the Mar-16 continuity point) and ran the first classic-draft speculative-decoding experiment (draft `gemma-3-1b-it-bf16`). Result: NET-NEGATIVE on this bandwidth-bound bf16 target (composite 0.91 at num_draft=2, 0.96 at num_draft=4). Nuance: `num_draft_tokens` dominates -- at 4, short-context prompts turn positive (short +10%), but the benefit collapses as context grows (long_context -40%), and greedy spec-decode is NOT bit-identical (batched-verify float order flips borderline argmaxes, so fingerprints diverge -- a distributional gate, not the fingerprint guard, is needed to certify a speculative run). Confirms the Direction thesis that the decode win is verification-based decoding (DFlash), not classic draft. Added a `--num-draft-tokens` flag to `bench_text.py` for the sweep; corrected the (wrong) "lossless/bit-identical" docstring. The harness validated itself: it flagged every regression and divergence.

## [1.34.11]

### Changed

- **mlx upgraded 0.31.2 -> 0.32.0 in the root venv (and 0.31.1 -> 0.32.0 in optloop-lib's)** -- v0.32.0 (released 2026-07-07) ships upstream PR #3628 "Fix threaded compile cache cleanup", the real fix for the CompilerCache TLS teardown abort we worked around in v1.31.2. Proven with a discriminating A/B repro (a compiled function returning a TUPLE, executed on a worker thread that then exits): SIGTRAP with the exact production `PyThreadState_Get`/GIL fatal error on 0.31.2, clean on 0.32.0 -- the tuple return was the ingredient the original minimal-repro attempt was missing. `_PinnedExecutorPool` stays regardless (it also bounds stream-registry growth, which the upstream fix does not address). Full suites green on 0.32.0: backend 839, optloop-lib 65, E2E 51/51 live.

### Fixed

- **Root venv extras**: plain `uv sync` had silently stripped the `performance`/`test` extras (pyturbojpeg -- the multipart JPEG decoder -- uvloop, xxhash, cachetools, pytest plugins). Restored with `uv sync --all-extras`; gotcha recorded in CLAUDE.md.

## [1.34.10]

### Added

- **optloop-lib: spec-decode baseline prep** (committing the prior session's tested work-in-progress). `bench_text.py` gains a genuinely long-context workload (~2.5-3k prompt tokens, fixed coherent document) so prefill scaling and a large KV cache are exercised, not just decode; `bench_vlm.py` gains folder-based real-photograph prompts (`data/vlm/photos/`, sorted, empty-safe -- adding/removing photos changes fingerprints, re-baseline after any change); `bench_config.toml` fixes the text-model HF id (the prior `google_gemma-3-27b-it-mlx-bf16` does not exist on HF; now `mlx-community/gemma-3-27b-it-bf16`) and adds `draft_model = gemma-3-1b-it-bf16` for the speculative-decoding comparison run (unset it for the plain baseline). Both additions land BEFORE the first baseline on purpose. optloop-lib suite: 65 green.

## [1.34.9]

### Added

- **Streaming-cadence regression guard in the E2E chat suite** (`tests/e2e/suites/chat.mjs`, now 25 checks): an in-page fetch to `/v1/chat/completions` measures client-observed inter-chunk arrival gaps and asserts median gap < 50ms and > 30 tok/s. The Phase 1 delivery fix (`asyncio.wait` instead of a 0.1s poll) is INVISIBLE to server-side telemetry -- only a client timing the stream can catch a revert to the ~100ms poll ceiling, so this is the sole automated guard for it. Live: 64 chunks, 10.8ms median, 92.2/s on the MoE. Requires a fast `E2E_MODEL` (the default MoE); a natively-slow dense model would false-fail by design.

### Changed

- **Root `.gitignore`: anchored `lib/`/`lib64/` to `/lib/`/`/lib64/`.** The bare setuptools-boilerplate `lib/` matched ANY nested source dir of that name (it had already forced a `!apps/heylook-frontend/src/lib/` negation and silently swallowed `tests/e2e/lib/`). Anchoring to the repo root keeps the build-artifact intent without eating source trees; the frontend negation is now unnecessary and removed.

## [1.34.8]

### Added

- **v3 frontend E2E harness (`tests/e2e/`)**: puppeteer-core + system Chrome driving the real `/v3` frontend against a spawned `heylookllm` with an isolated `HEYLOOK_DB_PATH`, so real conversations/notebooks are never touched (the suites clear all data). Two suites, 51 checks, green against a fast A4B MoE (~90 tok/s): **chat** (24 -- streaming, edit/regenerate/delete position-truncation, stop=partial-saved, post-abort health, settings + the `localStorage` `max_tokens` seed, conversation CRUD, 390px mobile) and **pages** (27 -- notebook autosave + generate-at-cursor tail preservation, explore logprob chips + keyboard nav, perf no-polling proof + range switching, models list/load/unload + HF scan + danger-zone clear). Own `package.json`/`bun.lock` (not repo root); run with `node run.mjs [chat|pages]`. Rebuilds the 52 browser checks lost with the v3 build scratchpad (plan Phase 4 item 1). Two gotchas encoded in the harness: settings are seeded via `localStorage` before boot then a forced reload (settings.js caches localStorage once at import, and a hash-only navigation is same-document, so a plain re-goto never re-reads the seed); `finishStream` flips the Send button to idle before it awaits the partial-save and sets the status, so stop-checks wait on the "Stopped" status, not the button.

## [1.34.7]

### Fixed

- **Teardown waiter-safety moved to the right depth**: `MLXProvider.unload()` now waits for gate WAITERS as well as active generations (the active counter decrements BEFORE `gate.release()` admits the next waiter, so an active-only wait could free weights exactly as a woken queued request started generating). Living in `unload()` means every teardown path -- LRU eviction, `clear_cache`, explicit unload, idle unload -- inherits the guarantee; the router-level check `_unload_idle` gained in v1.34.3 remains as an early skip. The gate is process-global, so with multiple loaded models this conservatively waits out other models' traffic too, bounded by the same 30s force-unload cap as before.
- **Capacity-reservation wait is bounded** (10 min default, `router._reservation_wait_timeout`): a wedged model load no longer blocks admission of every other model indefinitely (each blocked `get_provider` also pinned an asyncio default-executor thread), and the all-pinned `RuntimeError` can no longer be starved by an in-flight reservation -- the loop now raises a clear timeout error naming the in-flight loads.

## [1.34.6]

### Changed

- **Simplify pass over the v1.34.1-.5 diff** (4-angle review: reuse, simplification, efficiency, altitude). Router: the `_LoadingPlaceholder` sentinel inside `self.providers` (13 isinstance checks across 9 methods) is replaced by a `self._loading` side-set mirroring the existing `_pinned` precedent -- `self.providers` again always means "real, loaded providers", every reader keeps its filter-free form, and forgetting to skip a reservation becomes structurally impossible; behavior unchanged (same race tests pass untouched). Telemetry: the per-chunk getattr scrape hand-copied at 4 consume loops collapses into `perf_collector.ChunkTelemetry.absorb()`, and the twice-duplicated TTFT-minus-queue-wait formula into `net_ttft_ms()`. Scans: `scan_paths` computes the configured id/path identity once instead of once per source (was K TOML re-reads + K x N path resolves per scan). Tests: `QueueStatsProvider` subclasses the shared `MockProvider`; the two hand-rolled fake-chunk builders merge into `tests/unit/_fake_chunk.py`.

## [1.34.5]

### Added

- **mlx-lm/mlx-vlm surface contract tests** (`tests/contract/test_mlxvlm_surface.py`, 22 tests): executable pins for every private/undocumented library surface this server consumes -- `prepare_inputs` signature and return shape, `apply_chat_template` kwargs, the `encode_image`/`cached_image_features` pattern, `LanguageModelOutput` fields, the `_position_ids`/`_rope_deltas` attribute convention, `mlx_lm.utils._get_classes`, cache classes' `state`/`empty()` surface, and `GenerationResponse`'s exact field set + non-slotted runtime attachment. Each test names its consumption site, so an aggressive library upgrade fails loudly in tests instead of silently at runtime. This is item 1 of the mlx-vlm bus-factor strategy (plan Direction).

## [1.34.4]

### Fixed

- **Scan/import correctness**: `already_configured` now matches on the resolved weights path as well as the id (a rescan that derives a different id for already-configured weights no longer presents them as unconfigured; symlinked spellings compare equal). Re-import has PUT semantics: importing an id that already exists replaces that entry with the freshly built one (smart defaults + profile + overrides) instead of silently skipping -- refreshing an entry from a rescan no longer requires hand-editing models.toml.

## [1.34.3]

### Fixed

- **Router load-capacity TOCTOU closed.** The capacity check + LRU evict ran under `cache_lock` but the load and publish ran outside it, so two concurrent requests for two DIFFERENT models both passed the check and held two full models in memory at once (OOM-class on a box sized for `max_loaded_models`). `get_provider` now reserves a placeholder slot under `cache_lock` before loading; placeholders count toward capacity, are invisible to every reader API (`get_loaded_models`, `get_current_model_id`, `get_model_status`), are never evicted/unloaded, and are released on load failure. A loader that finds the cache full of other in-flight loads waits for one to publish instead of over-committing.
- **Idle unload no longer tears down a provider with queued requests.** A request waiting at the FIFO generation gate is neither "active" (that starts after gate acquire) nor recently-used (its `last_used` was stamped at cache hit, and gate waits can outlast the idle threshold), so the 60s idle tick could delete model weights out from under a request about to run. `_unload_idle` now checks the provider's `generation_queue_stats()` (active + waiting) under the SAME `cache_lock` hold as the pop, and skips busy providers until a later tick.

## [1.34.2]

### Fixed

- **Reasoning parser is now instantiated per request, not shared on the provider.** The parser (with its streaming buffer state) was built once at model load; every request called `reset()` and streamed through the shared instance, so two interleaved streams on the same model corrupted each other's buffers and request B's `reset()` clobbered request A mid-flight. Each request now gets its own parser via `select_reasoning_parser(provider._template_info)`. The load-time rationale (Mistral's ~1000-token strip-regex compile) is preserved by an `lru_cache` on `_compile_strip_pattern` -- the compiled pattern is stateless and shared; only buffers are per-request.
- **Embedding tokenizer pad-token guard**: decoder-only backbones without a `pad_token` broke the `padding=True` batch-encode call; the embedding provider now falls back to `eos_token` at load (warns if both are missing).

## [1.34.1]

### Fixed

- **Streaming delivery is no longer quantized to ~10 chunks/s.** The disconnect-watch loop in `streaming_utils.async_generator_with_abort` slept a fixed 100ms between `chunk_future.done()` checks, so every SSE chunk waited for the next poll boundary -- capping delivered (and recorded) throughput at ~10 tok/s regardless of model speed, and making e.g. a 60->48 tok/s regression invisible. The loop now blocks on the chunk future with a 100ms timeout (`asyncio.wait`), waking the moment a chunk is ready while keeping the disconnect-detection and keepalive cadence. This was also the measurement prerequisite for all Phase 5 perf work.
- **Headline perf metrics are honest now** (from the 2026-07-06 measurement audit): recorded tok/s comes from mlx-lm's native per-chunk `generation_tps` (measured tightly around the decode loop; previously never read anywhere in src/), with a wall-clock fallback that excludes FIFO queue wait; recorded TTFT excludes queue wait (admission pressure stays visible in its own `queue_wait_ms` field); `/v1/messages` non-streaming `prompt_tps` no longer divides prompt tokens by whole-request elapsed time (it reports native prefill tps); hourly trends and the 60s resource-snapshot rolling average aggregate successful requests only (failed/503 events recorded 0.0 tok/s and dragged averages toward zero). `RequestEvent` gains a `prompt_tps` field (defaulted, back-compat), which flows into `request_events.jsonl`.
- **Close-timed-out streaming executors are quarantined, not dropped.** Dropping the last reference let GC fire `ThreadPoolExecutor`'s weakref callback, which enqueues the shutdown sentinel -- so a wedged worker that eventually finished would EXIT its thread and hit the MLX TLS-teardown process abort the executor pool exists to prevent. The pool now holds quarantined executors for the process lifetime (cost: one leaked idle thread per wedge).

## [1.34.0]

### Removed

- **App-level optloop (`apps/optloop/`) retired.** A measurement audit found its benchmarks import mlx-lm/mlx-vlm directly and never exercise the `src/heylook_llm/` serving path they were chartered to optimize -- a change to the router, radix cache, or generation core scored exactly 1.0 either way -- and no optimization cycle had ever run end-to-end (results.tsv was header-only, `data/cycles/` empty; the only artifacts were `--reset-baseline` writes). The scoring/fingerprint harness itself was sound and lives on in optloop-lib. Serving-path benchmarking will instead be a thin HTTP bench against a running server, planned after the streaming-delivery and headline-metrics fixes (see `docs/project/plan_2026-07.md`, Phase 5 measurement section). Also removed: `docs/optloop_advanced.md` (its headline topics -- the bench activation gap and `.pth` monkey patching -- documented the retired app-level mechanism).

### Changed

- **optloop-lib is now the only optimization bench**, reframed as a manual benchmark tool first (agent-driven loop optional): new `apps/optloop-lib/CLAUDE.md` orientation doc, placeholder `AGENTS.md` deleted (cross-session knowledge consolidates in `docs/optimization_log.md`), `program.md` slimmed ("LOOP FOREVER"/"NEVER STOP" ceremony removed, stale references fixed), and `docs/optloop_guide.md` rewritten lib-only with the still-relevant advanced-guide content merged in. Fingerprinting docs now state the limitation plainly: greedy decode + token-ID fingerprint freezes behavior against the harness's own baseline but certifies nothing about output quality (no ground-truth metric exists).

### Added

- **optloop-lib: models.toml path resolution** (ported from the retired app-level harness before deletion): bench model IDs now resolve CLI `--model-path` > `bench_config.toml` id > the server's root `models.toml` local path (no re-download) > HF download fallback, with an org-prefix fallback match and 5 new unit tests (65 total).

## [1.33.0]

### Changed

- **Generation failures are now typed exceptions, not sentinel chunks.** The provider raises `GenerationFailed` (server-side) or `InvalidGenerationRequest` (client-side, e.g. images sent to a text-only model) instead of yielding an `is_error` chunk that each consumer had to remember to check -- the sentinel approach had already missed two consumers (`batch_processor.py` and `rlm.py` concatenated error text into results; RLM fed "Error: MLX generation failed..." back into its REPL loop as a sub-answer). Raising makes every consumer, present and future, fail loudly by default: batch requests now record `group.error` per-request instead of shipping fake content; RLM surfaces the failure through its own error handling. API translation: HTTP 500 / **400 for client errors (new)** on non-streaming; the same SSE error payload as before when streaming (headers already sent, so client errors also arrive as stream errors there). The wire contract for streaming clients is unchanged; non-streaming clients now get a proper 400 for never-going-to-work requests that previously returned a 200 with error text (pre-1.31.1) or a 500 (1.31.1+).

## [1.32.2]

### Fixed

- **1-in-60 flaky test identified and fixed**: `test_perf_collector.py::test_single_hour_bucket` used live `time.time()` with a +60s second event, so its two events straddled an hour bucket whenever the suite ran in the last minute of any hour (the previously-unexplained single-run failure on 2026-07-06). Now anchored to mid-hour.

### Added

- **Real coverage for two previously-untested production paths** (the deleted tautological tests only asserted their own inline math): `resolve_add_generation_prompt()` extracted from `_apply_template` (prefill convention: trailing assistant message = continue, no new generation prompt) with 4 tests, and `normalize_layer_index()` extracted from `_extract_from_layer` (negative layer indexing + bounds) with 7 tests -- both verified non-tautological by mutation (break helper -> tests fail). The batch path's duplicate inline prefill logic now reuses the helper. Sibling suites confirmed unaffected by the backend changes: legacy frontend 880/880, optloop-lib 60/60, batch-labeler 25/25.

## [1.32.1]

### Fixed

- **`mx.metal.device_info()` deprecation** (warned at every startup): migrated to `mx.device_info()` at all three call sites (`memory.py` startup record, `prompt_cache.py` memory-pressure check -- which would have broken outright when the alias is removed -- and `/v1/capabilities`). Verified warning-free with `-W error::DeprecationWarning`.

### Changed

- **Library-drift audit against installed mlx 0.31.2 / mlx-lm 0.31.3 / mlx-vlm 0.6.3 / transformers 5.5.4**: every load-bearing assumption verified against installed source -- `KVCache.state` laziness, `stream_generate` kwargs, `GenerationResponse` fields, mlx-vlm `apply_chat_template`/`prepare_inputs`/`LanguageModelOutput`, `_get_classes`, ArraysCache trim limitation: all still correct, no broken sites. Removed verified-dead code: two of four transformers compat patches (one now unreachable behind a backend gate, one silently no-oping since `transformers.utils.auto_docstring` became a function) and the `_load_vlm_with_weight_fix` TypeError-fallback + `load_model` monkeypatch (mlx-vlm `load()` has accepted `strict` for a long time).
- **Test-suite consolidation** (-26 tests, 760 passing): deleted `test_mlx_provider_safety.py` (its fake provider invented `unload(max_wait=...)` and `_content_cache`, neither exists on the real provider; real coverage in `test_mlx_provider.py::TestUnload`); removed two tautological classes that asserted their own inline math without touching production code (`TestPrefillConvention`, `TestLayerIndexNormalization` -- both note the real coverage gap for a follow-up); deduplicated tests across `test_speculative`/`test_vlm_inputs`/`test_admin`/`test_config`/`test_rlm`; merged `test_rlm.py`'s three identical-setup timeout tests into one; replaced a `time.sleep(10)` payload the interrupt mechanism couldn't break (leaked a live thread ~7s past the test) with an interruptible busy loop (~1s now). `tests/README.md` scrubbed of the false "pre-existing failures" claims; `CLAUDE.md` test-count reference updated.

## [1.32.0]

### Changed

- **Chat-sane request defaults.** The global sampler floor (what a request gets when neither the request, a preset, nor the model config says anything -- i.e. every freshly imported model) was `temperature 0.1, max_tokens 512`: near-greedy sampling and mid-sentence truncation of long answers. Now `temperature 0.7, max_tokens 4096` (`GLOBAL_SAMPLER_FLOOR`), and the batch fallbacks reference the same constant instead of a third hardcoded 512. Admin/CLI import now stamps `default_preset = "balanced"` (temp 0.7 / top_p 0.9) instead of the deprecated `moderate` alias.
- **Radix prefix cache is bypassed for non-standard KV caches.** Snapshot restore prefix-trims `keys[..., :N, :]`, which is wrong for `QuantizedKVCache` (packed tuple state) and impossible for rotating caches -- the risk was documented in `radix_cache.py` but unenforced (silent wrong output on partial prefix hits). Both lookup and store now gate on `cache_type == "standard"` with no `max_kv_size`.
- **Import size is real bytes, not a name regex.** The CLI import path parsed `size_gb` out of the model NAME (`Qwen-7B` -> "7 GB" -- billions of params masquerading as gigabytes; `-4bit` -> 4 GB) and fed it to the RAM-relative smart defaults. `size_gb` now always comes from the safetensors byte-sum (matching the admin scan path); the name regex only supplies the human label, and only from the directory name (the full-path match let parent-dir fragments win).
- **Config TUI cache defaults aligned** with the new policy (8-bit/group-64 when quantizing, no `max_kv_size` by default, truncation warning on the max-size prompt).

### Added

- **Strict model-config validation** (`MLXModelConfig`): `extra="forbid"` so models.toml typos fail at load instead of silently reverting to defaults; `kv_bits` constrained to 2/4/8 and `kv_group_size` to 32/64/128 (what MLX actually supports); `cache_type="rotating"` without `max_kv_size` now fails validation instead of the first generation; `max_queue_depth` is a real config field (it was read by the generation gate but silently dropped by pydantic, making it permanently 8).

### Removed

- **`quantized_kv_start`** config field: written by smart defaults and stored in every import, but never consumed by `_build_cache_config`/`make_cache` -- pure dead config. Existing models.toml entries carrying it must drop the key (this machine's file was migrated). Also removed: `num_draft_tokens` stamping on import (inert without a `draft_model_path`; the field itself remains) and five `RELOAD_REQUIRED_FIELDS` entries from a long-removed audio provider.

## [1.31.3]

### Changed

- **Import-time KV-cache defaults are now RAM-relative, and `max_kv_size` is never defaulted.** `get_smart_defaults` quantized the KV cache for any model over an absolute weight threshold (>13GB: 8-bit KV; >30GB: 8-bit KV *plus* `max_kv_size = 2048`) -- sized for a small-RAM machine and wrong on a 192GB Studio, where 11 of 14 configured models had auto-quantized KV and 6 carried the 2048 cap. The cap is the worst offender: it creates a RotatingKVCache that **silently drops context** beyond 2048 tokens (and rotating caches have known correctness limits with the radix prefix cache). Now: quantize only when weights exceed ~35% of total unified memory (`psutil`), and never emit `max_kv_size` -- context truncation is an explicit user choice. Existing `models.toml` entries are local data and were migrated by hand on this machine (standard cache below the threshold, quantized-but-uncapped above it).

## [1.31.2]

### Fixed

- **Server aborted (SIGTRAP, `Fatal Python error: PyThreadState_Get`) after a streaming request on models with compiled sampler / quantized-KV paths** (e.g. Qwen3-VL-32B with `cache_type = "quantized"`): `async_generator_with_abort` created a fresh single-worker executor per request and shut it down at stream end, so one MLX-tainted thread died per request. MLX keeps a thread-local `CompilerCache` whose entries hold Python objects when `mx.compile`d *Python* functions ran on that thread; pthread TLS cleanup runs after the Python thread state is destroyed, so the cache destructor deallocated those objects without the GIL -- `Py_FatalError` -> abort (confirmed by two macOS crash reports with identical stacks: `~CompilerCache()` -> `tupledealloc` -> `fatal_error` inside `_pthread_exit`). Fix: `_PinnedExecutorPool` in `streaming_utils.py` leases persistent single-thread executors instead of creating/destroying one per request -- generation stays pinned to one thread (unchanged invariant), but threads are reused, never torn down. A worker whose generator close times out is retired (leaked), not shut down. Repro was deterministic: one streaming request to Qwen3-VL-32B-8bit killed the server; 6/6 clean after the fix. Also removes the per-request thread churn noted as a follow-up in the 1.31.1 review (MLX stream-registry growth). Tests: `tests/unit/test_streaming_executor_pool.py`.

## [1.31.1]

### Fixed

- **Radix cache reuse crashed with "There is no Stream(gpu, N) in current thread"** (recurrence of the 1.30.5 bug class in a different spot): `snapshot_kv` published *lazy* KV slice nodes into the shared radix tree. Generation runs on a fresh single-worker thread per request (`streaming_utils`), and both our `generation_stream` and mlx_lm's are `mx.new_thread_local_stream` -- GPU thread-local streams are destroyed with their thread (verified by direct probe; CPU streams and per-thread *default* streams survive, which is why only this path crashed). When a later request on a different thread hit the cached prefix, mlx_lm's `mx.eval([c.state for c in prompt_cache])` couldn't resolve the dead thread's stream. Snapshots are now materialized (`mx.eval`) at store time on the generating thread, making cached entries thread-agnostic. Reproduced deterministically (4/4 identical-resend radix hits crashed without the fix, 4/4 clean with it) via seed-with-`max_tokens=1` + identical resend, which parks the snapshot on a block the resend fully covers. Regression test: `tests/unit/test_snapshot_thread_affinity.py` (skips off-Metal). Audited `vision_feature_cache` for the same hazard: safe -- its features are scheduled on a (globally registered) default stream, not a thread-local one.
- **Generation failures were streamed to clients as assistant content**: the provider yielded error text as a normal chunk, so frontends rendered and even persisted "Error: MLX generation failed: ..." as a model response -- the crash above shipped as a fake completion. `MLXErrorChunk` (now module-level, `is_error=True`) is surfaced properly: OpenAI streaming emits `data: {"error":{message, type:"server_error", code:"generation_failed"}}` then `[DONE]`; Messages API streaming emits an Anthropic-style `event: error`; non-streaming paths return HTTP 500 with the message in `detail`. Frontend v3's `streaming.js` routes error payloads to `onError`. Contract tests: `tests/contract/test_generation_errors.py`. (Legacy/v2 clients that only read `delta.content` now see an empty response instead of fake content.)

## [1.31.0]

### Added

- **Frontend v3 at `/v3`** (`apps/heylook-frontend-v3/`): from-scratch rewrite per `docs/frontend_v3_spec.md`, served alongside `/v2` until cutover. Vanilla JS ES modules, no framework, no build step. Five pages: chat, notebook, token explorer, models (admin), performance (on-demand only, no polling); batch page dropped per spec. Pretext virtualization is gone -- markdown rendering via the vendored marked + DOMPurify path only. Shared layer replaces v2's per-page boilerplate: `createPage` lifecycle (per-mount state, teardown AbortSignal, auto-cancelled rAF throttles, post-await guards), hash router with nav generated from route registration, route-table-generated `api.js`, `streaming.js` (SSE keepalive-comment handling, `reader.cancel()` on abort, abort-as-normal-completion), data-driven sampler settings panel (null = backend cascade, localStorage key `heylook-v3-settings`). Fresh OKLCH warm-minimal design system (pure-white surface, honey-bronze accent) with desktop + iPhone-Safari layouts. Verified end-to-end against a live backend: 25 chat checks + 27 page checks (streaming, position-truncation edit/regenerate, stop/abort partial save, 503-busy retry path, autosave, generate-at-cursor, logprob chips, admin load/unload/scan, clear-all).
- **`/v3` static mount in `api.py`**: duplicate of the `/v2` block (SPA fallback + path-traversal guard), plus contract tests for both mounts (`tests/contract/test_frontend_mounts.py`).

## [1.30.5]

### Fixed

- **All MLX generation failed with "There is no Stream(gpu, 0) in current thread."**: the dedicated generation stream in `mlx_provider.py` was created at import time with `mx.new_stream(mx.default_device())`. MLX streams are thread-local -- bound to the thread that creates them -- but generation runs on FastAPI's thread pool (`asyncio.to_thread` / `run_in_executor`), not the import thread. When `wired_limit` called `mx.synchronize(generation_stream)` on a pool worker, MLX raised `RuntimeError: There is no Stream(gpu, 0) in current thread.`, so every text and VLM request aborted before producing output (clients saw a fixed-length error string instead of a completion). Switched to `mx.new_thread_local_stream(...)`, which materializes the stream per-thread -- the same API `mlx_lm.generate` uses for its own `generation_stream`. Verified on real Metal across multiple concurrent pool workers.
- **Concurrent requests cannibalized each other**: the generation lock used a *preemption* policy -- a new request aborted the in-flight one to take the lock. Under any concurrency this meant only the newest request ever completed. The Batch applet (fires up to 4 concurrent), the batch-labeler client, and multiple frontends all aborted each other's generations. Replaced with a strict-FIFO admission gate (`providers/common/generation_gate.py`): requests queue in arrival order and each completes. Interactive "cancel on new message" is unaffected -- the frontend already aborts its own in-flight HTTP request, which the disconnect handler turns into a cooperative abort.
- **Generation slot could be held until GC**: the non-streaming, batch, and RLM consume loops never called `generator.close()`, so on a consumer-side exception the provider generator's `finally` (which releases the generation slot) only ran when the garbage collector eventually reclaimed it -- stalling every queued request until then. All consume paths now close the generator (via `contextlib.closing`). (The streaming path already did.)
- **One client's disconnect aborted another client's generation**: the cooperative abort signal was a single `_abort_event` shared per provider. Once FIFO made concurrent requests genuinely live, a disconnecting request set the shared flag and the *active* (unrelated) generation saw it and stopped early -- the connected client got a truncated response. The abort signal is now **per-request**: created by the route, passed to both the generation and the disconnect watcher, so a disconnect cancels only that request.
- **A queued request whose client disconnected still ran a full generation**: `acquire()` didn't watch for cancellation, and the per-generation `_abort_event.clear()` wiped the disconnect signal once the turn arrived. The gate's `acquire(cancel_check=...)` now drops a request from the queue (`GenerationCancelled`) when its client has gone, and the fresh per-request event means a disconnect set during the wait survives. The streaming disconnect wait is also bounded so it can't pin the coroutine.
- **Generation didn't serialize across models**: the gate (and the lock before it) was per-provider, so with `max_loaded_models>1` two models could run concurrent generations on the one GPU. The gate is now a process-global singleton shared by all MLX providers.
- **Metrics double-counted queued requests / `mx.clear_cache()` overlapped the next generation / `max_queue_depth=0` rejected everything**: `_active_generations` is now incremented after acquiring (so a queued request counts as `requests_queued`, not `requests_active`); the MLX cache is cleared before releasing the slot (cleanup completes before the next waiter runs); and `check_capacity()` accounts for the active holder so an idle gate admits the first request even at `max_queue_depth=0`.

### Added

- **`requests_queued` in `/v1/system/metrics`**: per-model count of requests waiting in the FIFO generation queue behind the active one (alongside the existing `requests_active`), for observing backpressure and tuning `max_queue_depth`.
- **Per-request queue-wait timing.** Each request's time blocked in the FIFO queue is measured (around `gen_gate.acquire()`), tagged on the generation chunks, and surfaced three ways: `queue_wait_ms` in the streaming usage chunk's `timing` (when `stream_options.include_usage=true`), a `queue_wait_ms` field on the per-request observability record (`request_events.jsonl`), and an average `queue_wait` in the per-model `bottlenecks` breakdown of the performance profile. Distinct from the existing `queue` metric, which is provider-acquisition / model-load time. Covers both `/v1/chat/completions` and the Messages API, streaming and non-streaming.
- **OpenAPI types drift guard.** `scripts/check_openapi_sync.sh` regenerates `generated-api.ts` from the FastAPI app's schema offline (`app.openapi()`, no running server) using the frontend's pinned `openapi-typescript`, and diffs against the committed file. Wired into the pre-commit hook (gated on staged top-level `src/heylook_llm/*.py`, `schema/`, or the generated file) so the types can't silently drift again; also runnable via `bun run check:api`. Degrades gracefully (skips, never false-blocks) when uv/bun/MLX are unavailable.

### Changed

- **503 backpressure responses now report real queue capacity.** The `model_overloaded` 503 previously hardcoded `X-RateLimit-Limit: 1` ("we can handle 1 concurrent request") and said "processing another request" -- both stale now that requests queue. The body says the generation queue is full, and `X-RateLimit-Limit` reflects actual capacity (`1 + max_queue_depth`) from the provider's live queue snapshot.
- **Generation is serialized FIFO with bounded-depth backpressure.** A single GPU + one loaded model + shared KV cache means one generation at a time; the new `GenerationGate` enforces this in arrival order. HTTP entry points call `provider.check_capacity()` before starting, returning **503 (`model_overloaded`, `Retry-After: 1`)** once `max_queue_depth` requests are already queued -- wiring up the `MODEL_BUSY` 503 path in `api.py`/`messages_api.py` that previously existed but was never triggered. Depth is configurable per model via `max_queue_depth` (default 8). Internal orchestration (batch, RLM) skips the capacity check and simply queues. Batch generation now shares the same gate as chat, so batch and chat can never run on the GPU concurrently.
- **Streaming generation is pinned to one thread.** `async_generator_with_abort` drove each `next()` through the default thread pool, so a single generation's tokens could hop worker threads -- fragile for MLX, whose per-generation stream and `wired_limit` context are entered on the first `next()` and synchronized on the last. Each streaming generation now runs start-to-finish on a dedicated single-thread executor, and the generator is closed on that same worker. Non-streaming paths already ran each generation on one thread.

## [1.30.4]

### Fixed

- **VLM warmup never primed the model**: `MLXProvider.warmup()` passed the full VLM model straight to `generate_text`. A VLM's forward pass returns a `LanguageModelOutput`, but mlx-lm's `generate_step` subscripts logits directly (`logits[:, -1, :]`), so every VLM warmup raised `'LanguageModelOutput' object is not subscriptable`. Warmup now routes the model through the text strategy's `_get_generation_model()` -- the same `LanguageModelLogitsWrapper` real requests use -- so VLMs are actually JIT-primed at load instead of paying compilation cost on the first request. Text-only models are unaffected (wrapper returns the raw model).

### Changed

- **Warmup failures now log at WARNING** (was INFO). A consistently-failing warmup means the model is never primed and the first request pays full JIT cost; logging it at WARNING surfaces the regression instead of burying it. Behavior is otherwise unchanged -- warmup stays best-effort and never blocks model loading.
- **Consolidated VLM language-model wrapping** into `wrap_language_model()` in `providers/common/model_wrappers.py`. The text and vision strategies share one definition of "wrap a VLM's language model for mlx-lm" instead of constructing `LanguageModelLogitsWrapper(model.language_model)` inline in two places. Warmup resolves its generation model through `UnifiedTextStrategy._get_generation_model()` -- the same path real requests use -- so it can't drift back into passing the raw VLM model.
- **Removed a redundant VLM mRoPE position reset** in `UnifiedTextStrategy.generate()`. The `_position_ids`/`_rope_deltas` reset already happens in `run_generation` via `_reset_vlm_positions()` (the wrapper forwards to the same language-model instance), so the inline copy was dead. No behavior change.

## [1.30.3]

### Added

- **Format-aware reasoning parser + template-info driven selection**: new `heylook_llm.reasoning_parser` module replaces the previous hardcoded thinking parser call sites. Two classes + a factory: `HarmonyChannelParser` for multi-channel formats (control tokens `<|channel|>`/`<|message|>`/`<|start|>`/`<|end|>`/`<|return|>`/`<|call|>` stripped; analysis/commentary channels route to `message.thinking`; final channel routes to `message.content`), `PassThroughParser` for formats without reasoning structure. Templates with `<think>...</think>` markers route through the existing `HybridThinkingParser` directly (no wrapper). `select_reasoning_parser(template_info)` picks by template-file signals. Non-streaming + SSE streaming paths both route through the factory.
- **Template-info loader** (`providers/common/template_info.py`): reads `chat_template.jinja` / embedded `tokenizer_config.json` template + unions specials from both `tokenizer.json` `added_tokens` and `tokenizer_config.json` `added_tokens_decoder`. Exposes `ModelTemplateInfo` with `chat_template`, `special_tokens`, `has_harmony_structure` (derived from `<|channel|>` + `<|message|>` literals), `has_thinking_markers` (from `<think>...</think>`), `template_source`. The model's on-disk files are the single source of truth; no tokenizer introspection, no format-name lookup table.
- **Decode-path special-token hygiene**: `apply_special_token_hygiene(tokenizer)` patches `tokenizer.decode` to default `skip_special_tokens=True`, closing the leak where `NaiveStreamingDetokenizer` calls `decode(tokens)` bare and control tokens render as literal strings. Patches both the wrapper and the inner HF tokenizer so either detokenizer reference path is covered. Callers that want raw specials (Token Explorer UI) still pass `skip_special_tokens=False`. Vision first-token and batch text decode sites updated in-place.
- **Chat-template source policy**: `MLXModelConfig.chat_template_source` field (`None`/`"auto"` / `"jinja"` / `"tokenizer_config"` / absolute path). Resolved at load time. Useful when a model ships multiple templates or the user wants to point at a custom `.jinja` for testing. Logged at load. Import wizard auto-detects `chat_template.jinja` in scanned folders and records `"jinja"` when present.
- **CLI `--chat-template` flag** on `heylookllm import`: overrides the auto-detection, recorded in generated `models.toml` so the user can edit post-import.
- **`HarmonyChannelParser` + `PassThroughParser` strip-tokens set**: consumed from the template-info's declared specials and compiled into a single alternation regex (sorted longest-first). Strips non-structural control tokens from output deltas as a defense against fast-detokenizer leaks. Optimized for tokenizers with hundreds-to-thousands of declared specials.

### Changed

- **Parser built once at model load** (`MLXProvider._reasoning_parser`), reset per request instead of rebuilt. Saves the regex compile cost for tokenizers with large reserved-token sets.
- **Harmony control-token scan** collapsed from six `.find()` calls per iteration to one module-level compiled regex (`_HARMONY_CONTROL_PATTERN`). Single `re.search` returns position + matched token.
- **`template_info` uses `orjson` + `read_bytes()`** for JSON parsing, matching project convention.
- **Removed `Qwen3ThinkingParser` adapter class**: `HybridThinkingParser` already conforms to the `ReasoningParser` protocol. Factory imports it directly in the thinking branch.

## [1.30.1]

### Added

- **Idle model unload (C2)**: non-pinned loaded models that go unused for longer than their idle window auto-unload. Global default `idle_unload_seconds = 1800` (30min) in `[defaults]` of `models.toml`; per-model `unload_after_idle_seconds` override on `MLXModelConfig`. `0` at either level disables (per-model override still wins for models that set their own non-zero value). `ModelRouter._last_used_ts` tracks last cache hit or fresh load per model; `unload_idle_models(now_ts)` scans and unloads; pinned models always exempt. Events flow through `model_events.jsonl` as `reason="idle_timeout"`. `MemoryManager.tick()` drives this from the existing 60s resource-snapshot loop -- no new thread.

### Changed

- **`max_loaded_models` schema default flipped from 2 to 1**. Apple Silicon is memory-bandwidth-bound so a second loaded-but-idle model doesn't help throughput; it just holds memory that could serve bigger KV caches or higher-resolution vision batches. Field stays configurable (`Field(1, ge=1)`); existing `models.toml` entries are unaffected (they set their own value). Matches the user's already-explicit `max_loaded_models = 1`.

## [1.30.0]

### Added

- **Runtime preset registry (C1)**: new `heylook_llm.presets` module + `src/heylook_llm/data/presets/*.toml` bundle (8 canonical presets: `balanced`, `creative`, `deterministic`, `code`, `thinking`, `moderate`, `vlm-describe`, `vlm-extract`). `ChatRequest.preset` references them at request time. Five-layer cascade in `MLXProvider._apply_model_defaults`: global floor -> thinking-model defaults -> model sampler fields -> request preset -> request explicit fields. Per-model sampler fields still act as defaults; presets overlay at request time; explicit fields still win. Unknown preset name -> HTTP 400. VLM presets deliberately omit `top_k`/`min_p`/`repetition_penalty` since mlx-vlm's `stream_generate` ignores them (prevents silent no-ops). `request_events.jsonl` now records the resolved preset name.
- **Inference API key (C1.5)**: optional bearer-token gate via `HEYLOOK_API_KEY` env var. When set, inference endpoints (`/v1/chat/completions`, `/v1/batch/chat/completions`, `/v1/messages/*`, `/v1/embeddings`, `/v1/hidden_states*`, `/v1/rlm/*`) require `Authorization: Bearer <value>`. Loopback (`127.0.0.1`, `::1`) is exempt by default -- same-machine dev tools don't need to carry the key; set `HEYLOOK_API_KEY_ENFORCE_LOOPBACK=true` to close the carve-out for paranoid setups. `hmac.compare_digest` comparison, case-insensitive `Bearer` scheme per RFC 6750. Admin token (`HEYLOOK_ADMIN_TOKEN`) remains a separate gate on admin endpoints. Both default unset = open, matching the default single-user localhost UX.

## [1.29.0]

### Added

- **LAN hardening (S1.6)**: optional admin-token gate via `HEYLOOK_ADMIN_TOKEN` env var. When set, `/v1/admin/*`, `/v1/data/clear`, and `/v1/cache/clear` require a matching `X-Heylook-Admin-Token` header or return 401; unset/empty is a backward-compat no-op. Inference endpoints (`/v1/chat/completions`, `/v1/messages/*`, `/v1/embeddings`, `/v1/rlm/*`) are intentionally never gated so clients don't need to learn a shared secret. Startup log now advises on non-loopback binds and reports admin-token status. New `docs/lan_setup.md` walks through Caddy `tls internal` + `caddy trust` + hosts-file flow for HTTPS in front of a loopback-bound inference server; nginx alternative documented.

## [1.28.0]

### Added

- **Per-request peak memory + KV cache bytes telemetry**: `/v1/chat/completions` responses now expose `x-heylook-peak-memory-gb` and `x-heylook-kv-bytes` headers on non-streaming responses; streaming emits the same values in the usage chunk's `timing` object when `stream_options.include_usage=true` (SSE headers can't carry post-generation values). Frontend-v2 chat status bar renders "N tokens · P.PP GB peak · K KV" after each completion. `mx.reset_peak_memory()` is called at the top of `run_generation` to scope the counter per-request.
- **Three-stream observability with content invariant**: new `src/heylook_llm/memory.py` owns three disk-backed JSONL streams under `internal/log/` (gitignored) plus a one-shot startup record. `memory_baseline.jsonl` is the periodic resource snapshot (default hourly); `request_events.jsonl` is one line per completed request with sampler settings, timings, peak memory, cache hit rate, thinking/content token counts, stop reason; `model_events.jsonl` records load/unload with weights bytes, quantization, param count, context length. Configurable via `HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS` / `HEYLOOK_REQUEST_LOG_ENABLED` / `HEYLOOK_MODEL_EVENT_LOG_ENABLED` env vars. Content invariant -- numeric + metadata only, never prompts or responses -- is test-enforced via a recursive forbidden-key walk and a `RequestEvent` primitive-fields-only introspection test. See `docs/observability_guide.md`.
- **Vision feature cache byte cap**: `VisionFeatureCache` now evicts on both `max_entries` (20) and `max_bytes` (default 8 GB). Closes the documented leak vector where a few 8K-resolution images could consume multiple GB despite the entry-count cap. `stats()` exposes `bytes` + `max_bytes`; the hourly baseline aggregates across loaded providers.
- **Provider warmup + prefill_step_size passthrough**: `BaseProvider.warmup()` (no-op default) runs after each `load_model()`. `MLXProvider` override runs a ~30-token throwaway generation to prime Metal shader compilation, killing the 1-3s cold-start latency on first user request. `MLXModelConfig.prefill_step_size` (new optional field) flows into per-request `effective_request` and to `lm_stream_generate` when set; `None` lets mlx-lm use its own 2048 default. `MLX_RUNTIME_DEFAULT_FIELDS` is derived from `MLXModelConfig.model_fields` metadata (fields tagged `json_schema_extra={"is_runtime_default": True}`) so adding a new cache/speculative-decoding field auto-propagates without touching `_apply_model_defaults`.

### Changed

- **Deprecated `mx.metal.*` memory APIs** swapped to the top-level non-deprecated `mx.*` equivalents (`get_active_memory`, `get_peak_memory`, `get_cache_memory`, `reset_peak_memory`) across the observability code and tests. `mx.metal.device_info()` stays -- not aliased, not deprecated.
- **Router tests** migrated from YAML fixtures to TOML (matches `router.py:_load_config` which only dispatches on `.toml`).
- **`docs/FRONTEND_HANDOFF.md` renamed to `docs/frontend_api_reference.md`** (git-detected 98% similarity preserved). Cross-links updated across CLAUDE.md, tests/README.md, apps/heylook-frontend/README.md, internal/backend/api.md.
- **`setup.sh` installation menu** collapsed from 4 options to 2; removed every reference to `mlx_stt` / `parakeet-mlx` / the `stt` install extra (MLXSTTProvider was removed in Phase 2).

### Fixed

- **Double JSON serialization** on non-streaming `/v1/chat/completions`: switched from `JSONResponse(content=result.model_dump())` to `Response(content=result.model_dump_json(), media_type="application/json")` -- Pydantic's single-pass serializer saves ~1-2ms per response.
- **`_normalize_path_for_log`** strips the user's home-directory prefix from `ModelMetadata.path` before emitting to JSONL streams; keeps logs portable.
- **Deleted dead `mlx_batch_vision` test references** in `tests/unit/mlx_perf/` that prevented the suite from being green by default after the v1.23.0 batch-labeler extraction.

### Infrastructure

- **pytest-asyncio** added as a dev dependency so `tests/unit/test_conversation_api.py` and `test_notebook_api.py` run in the default suite (previously required manual install).
- **`sandbox.excludedCommands`** in `.claude/settings.local.json` now exempts `uv run pytest*`, `uv run python*`, `uv sync*`, `uv lock*`, `bun install*`, `bun run build*` so they don't trip the uv cache-access error.
- **`.gitignore` hardened** for runtime data: `*.db`, `*.db-*`, `*.sqlite`, `*.sqlite3`, `*.jsonl`, `/data/*` (with `!/data/.gitkeep`), `apps/*/data/*`.

## [1.27.0]

### Added

- **Conversation storage API**: New `/v1/conversations` CRUD endpoints backed by SQLite (aiosqlite). Stores conversations and messages server-side, enabling simpler frontends that don't need client-side persistence. Endpoints: list, create, get (with messages), update, delete (cascade), append message, edit message, truncate messages after position.

## [1.26.0]

### Added

- **Process-wide wired memory limit**: Server now calls `mx.set_wired_limit(max_recommended_working_set_size)` at startup, matching mlx-lm's server. Model weights stay wired between requests, reducing memory churn and improving time-to-first-token.
- **Vision feature cache**: VLM requests cache vision encoder outputs by image URL. Multi-turn conversations with the same image skip the 200-500ms vision tower forward pass. LRU cache with 20 entries, cleared on model unload. Uses mlx-vlm's `cached_image_features` / `encode_image` API.
- **Byte-level prompt cache budget**: New `--prompt-cache-bytes` CLI flag (e.g. `2G`, `512M`) caps total KV cache memory in the radix tree. Radix cache tracks snapshot sizes and evicts LRU leaves when over budget.
- **Cache stats in API responses**: `usage.prompt_tokens_details.cached_tokens` reports how many prompt tokens were served from the radix cache, matching the OpenAI API format.
- **Segment-aware cache eviction**: Radix tree nodes are tagged with segment type (system/assistant). System prompt KV caches are evicted last, keeping shared system prompts alive longer across conversations.
- **SSE keepalive during long prefill**: Streaming responses emit SSE comments (`: keepalive`) every 5 seconds during prompt processing to prevent connection timeouts on long prompts.

## [1.25.2]

### Fixed

- **OOM on model swap**: Router evicted the old model AFTER loading the new one, holding both in memory simultaneously. With `max_loaded_models: 1`, swapping from a 27B to a 120B model would OOM-kill the process. Eviction now happens BEFORE the new model loads.

## [1.25.1]

### Fixed

- **Single source of truth for connection state**: Removed duplicate `isConnected` state in `App.tsx` -- connection status now lives exclusively in `connectionStore`, eliminating drift between the reconnection banner and the app's connection gate
- **Reconnection banner safe area**: Banner now uses `env(safe-area-inset-top)` so it isn't hidden behind iPhone notch/dynamic island
- **Duplicate listener registration**: `initReconnectionDetection()` is now idempotent -- StrictMode double-fire no longer registers two `visibilitychange` listeners
- **Dead streams on tab restore**: After successful reconnect, stale streaming state is cleaned up so the UI doesn't show a spinner indefinitely
- **Connection error message**: Replaced hardcoded "localhost:8080" with generic message that works for LAN connections
- **E2E sidebar assertion**: Replaced tautological `isOffscreen || isOverlay` check with actual verification that sidebar doesn't push main content off-screen
- **Event listener leak**: `_resetReconnectionState()` now removes the `visibilitychange` listener, preventing stacked listeners in tests and HMR

### Changed

- `chat.spec.ts` refactored to use shared `backendPage`/`modelPage` fixtures instead of inline `beforeEach` blocks
- `connectionStore` now uses `withDiagnostics` middleware for consistency with `modelStore`
- Reconnect module delegates to `connectionStore.checkConnection()` instead of calling `fetchModels` directly -- single code path for refreshing server state
- `modelStore.initialize()` owns the startup sequence (fetchModels + fetchCapabilities), called by both initial connection and reconnection
- Dead-stream cleanup in reconnect is now fire-and-forget (`.then()` instead of `await`) to stay off the critical path
- Safe-area banner uses `.pt-safe` CSS class (matching existing `.pb-safe`/`.mb-safe` pattern) instead of inline style
- Removed no-op `hmr.host: undefined` from `vite.config.ts`
- Consolidated duplicate imports in `persistence.spec.ts` and `conversation.spec.ts`
- Added `data-role` attributes to message wrappers in `MessageList.tsx` (fixes `conversationPage` fixture selector)
- Added idempotent init test to `reconnect.test.ts`

## [1.25.0]

### Fixed

- **iOS Safari meta tag**: Changed `mobile-web-app-capable` to `apple-mobile-web-app-capable` in `index.html` -- Safari now treats the app as a web app and is less aggressive about killing the tab under memory pressure
- **Vite HMR over LAN**: Configured `server.hmr` to auto-detect hostname from request and disabled error overlay to prevent reload loops when iOS Safari freezes and restores the tab's websocket
- **Playwright config**: Changed `npm run dev` to `bun run dev` to match actual package manager
- **Persistence test DB name**: Fixed `heylook-db` to `heylook` in `persistence.spec.ts` (tests were passing by accident since `indexedDB.open` creates any DB name)
- **Persistence test assertions**: Replaced weak `body.toBeVisible()` checks with actual data verification (message text presence, conversation count)

### Added

- **Reconnection detection**: New `reconnect.ts` module detects dead connections after iOS Safari tab restore. Pings `/v1/models`, retries with exponential backoff, shows "Reconnecting..." banner via `connectionStore.ts`
- **Shared E2E fixtures**: New `e2e/fixtures.ts` with `backendPage`, `modelPage`, and `conversationPage` fixtures -- eliminates copy-pasted `setupWithLoadedModel` across 4 test files
- **Navigation E2E tests**: All 7 applet routes tested for rendering, lazy loading, unknown-route redirect, and state preservation across navigation
- **Applet E2E coverage**: New test files for Notebook, Models, Batch, Token Explorer, Model Comparison, and Performance applets (previously zero E2E coverage)
- **Multi-turn E2E test**: Conversation test that sends a message, waits for response, then sends follow-up to verify context is maintained
- **Mobile E2E tests**: Viewport tests (layout at mobile width, sidebar behavior, touch target sizes) and persistence tests (visibilitychange flush, rapid-send-then-background)
- **Playwright browser matrix**: Added WebKit (Desktop Safari), Mobile Safari (iPhone 13), and Mobile Chrome (Pixel 5) projects with 60s timeout for mobile
- **Reconnection unit tests**: 6 tests covering ping success/failure, backoff retry, visibility change listener, and model list refresh
- **Optloop multi-turn prompts**: `bench_text.py` gains `multi_turn_short` (2-turn Q&A follow-up) and `multi_turn_long` (3-turn conversation with system prompt and growing KV cache). `bench_vlm.py` gains `vision_multi_turn` (image analysis followed by text follow-up). Both optloop and optloop-lib benchmark suites updated in sync.

## [1.24.2]

### Added

- **Optloop user guides**: `docs/optloop_guide.md` (user walkthrough, scoring, monitoring) and `docs/optloop_advanced.md` (bench activation gap, monkey patching, performance ceilings, failure modes, FAQ)
- **Optloop data artifact reference**: inventory table documenting what gets created, where it lives, persistence rules, and audience
- **Session-end protocol**: documented in program.md files, README, and user guides (teardown, analysis, optimization log update)
- **Cross-session memory references**: both program.md files now reference `docs/optimization_log.md` in setup and loop steps; optloop-lib also references AGENTS.md

## [1.24.1]

### Added

- **Optloop cross-session memory**: `docs/optimization_log.md` accumulates findings across optloop sessions (baselines, what worked/failed, technical gotchas). Optloop pre-flight, loop iteration, and session-end protocol updated to read and write it.

## [1.24.0]

### Fixed

- **VLM position state bleeding**: Qwen3.5 mRoPE models cache `_position_ids` and `_rope_deltas` on the language model instance. Stale values between fresh generations caused broadcast shape mismatches. Position state is now reset before each fresh generation in `run_generation()`

### Changed

- **DraftTuner**: `ensure_and_get()` replaces separate `_ensure_baseline` + `get_num_draft_tokens` (single lock acquisition per request)
- **LanguageModelLogitsWrapper**: simplified `__call__` hot path -- removed try/except, reduced to single `getattr` check

## [1.23.9]

### Added

- **RLM SHOW_VARS**: `SHOW_VARS()` function in REPL namespace lists user-defined variables with their types
- **RLM root prompt re-injection**: original query is appended to feedback after iteration 0, keeping the model on-task during long runs
- **RLM best partial answer tracking**: fallback paths (max_iterations, error_threshold, timeout) prefer the best text answer seen over raw code block text
- **RLM max_timeout**: wall-clock timeout for the entire RLM loop (`max_timeout` request field, `"timeout"` finish reason). Checked per-iteration
- **RLM llm_query_batched**: `llm_query_batched(prompts)` runs multiple sub-queries with GPU batching when available, sequential fallback otherwise
- **RLM rlm_query_batched**: `rlm_query_batched(prompts)` runs multiple recursive sub-calls sequentially (requires `max_depth >= 2`)
- **RLM custom tools**: `RLMEngine(router, custom_tools=[...])` injects server-registered Python functions into the REPL namespace. Propagates to child RLMs
- **RLM event callbacks**: `on_iteration_start`, `on_iteration_complete`, `on_subcall_start`, `on_subcall_complete` callbacks on `RLMEngine.__init__()` for programmatic monitoring

## [1.23.8]

### Added

- **RLM compaction**: history summarization when context fills up (`compaction`, `compaction_threshold`, `max_context_tokens` request fields). Prevents hitting context window limits on long runs while preserving REPL namespace
- **RLM recursive depth**: `rlm_query()` spawns child RLMs with their own REPL loops for divide-and-conquer over sub-problems (`max_depth` request field, `child_traces` in response metadata)
- **RLM max errors**: stop after N consecutive code execution errors to prevent infinite error loops (`max_errors` request field, `error_threshold` finish reason)

## [1.23.7]

### Added

- **RLM endpoint**: `POST /v1/rlm/completions` -- Recursive Language Model inference with sandboxed Python REPL, iterative code execution, `llm_query()` sub-calls, and SSE streaming support

## [1.23.6]

### Changed

- **Python version**: Bump minimum from 3.11 to 3.12 across all pyproject.toml files (main, optloop-lib, batch-labeler)
- **optloop-lib**: Remove stale dependency floor pins (bare package names, let uv resolve latest)
- **optloop-lib**: Switch build system from hatchling to setuptools (no package to build)

### Added

- **optloop-lib**: Smoke import tests for bench_text, bench_vlm, bench_analysis (60 tests total)

## [1.23.5]

### Added

- **optloop-lib**: Library-level inference optimizer targeting mlx-lm and mlx-vlm internals via editable installs from GitHub fork clones (`apps/optloop-lib/repos/`)

## [1.23.4]

### Fixed

- **Optloop JSON error handling**: `load_baseline()`, `load_cycles()`, and `load_json_runs()` now catch corrupt JSON instead of crashing the entire run (warn to stderr, skip bad files)
- **Optloop lazy MLX import**: `bench_common.py` defers `import mlx.core` to `sync_barrier()` so pure functions can be imported without triggering Metal initialization

### Added

- **Optloop test coverage**: 18 new tests covering `baseline_metrics_from_result`, `get_bench_params`, `get_constraints`, TTFT per-prompt regression, prefill/memory hard constraints, partial fingerprint matches, zero-variance guard, and corrupt JSON handling (52 total)

## [1.23.3]

### Fixed

- **Optloop prefill_tps guard**: bench_vlm.py now guards `prefill_tps` division with `if prefill_time_s > 0 else 0.0`, matching bench_text.py
- **Optloop dead code**: bench_analysis.py `print_rankings` removed redundant `and r.get("status") != "baseline"` condition (always true when first condition is true)

### Added

- **Optloop unit tests**: 34 tests for bench_common pure functions (scoring, variance, constraints, suspicion, fingerprinting, config extraction) in `apps/optloop/tests/`

## [1.23.2]

### Fixed

- **Optloop variance**: bench_common.py used population variance (N divisor) instead of sample variance (N-1), causing CV threshold to pass too easily with runs=3
- **Optloop zero-token guard**: bench_text.py and bench_vlm.py now raise RuntimeError if generation produces 0 tokens instead of silently recording all-zero metrics
- **Optloop atomic writes**: save_baseline, save_run, and save_cycle now write to tmp file and rename to prevent corruption on crash
- **Optloop weight validation**: compute_composite_score warns to stderr if scoring weights don't sum to 1.0
- **Optloop dead config docs**: bench_config.toml `[scoring.decision]` comments now clarify these values are read by the agent from program.md, not by bench scripts

### Changed

- **Optloop README**: full rewrite with detailed end-to-end tutorial, configuration reference, scoring explanation, verification walkthrough, and troubleshooting guide

## [1.23.1]

### Fixed

- **Optloop skill names**: program.md referenced `/mlx` and `/mlx-lm` instead of `/mlx-skills:mlx` and `/mlx-skills:mlx-lm`
- **Variance transposition shadowing**: list comprehension variable `prompt_runs` shadowed outer scope in both bench scripts
- **TimingContext `__exit__`**: renamed unused `*exc` to `*_exc` to suppress linter warning

### Added

- **Local source mode docs**: program.md documents how to use editable mlx-lm/mlx-vlm installs for library-level optimization
- **Commented coderef config**: bench_config.toml has commented-out `allowed_paths` and `banned_diff_patterns` for local source mode

## [1.23.0]

### Added

- **Output fingerprinting**: SHA-256 hash of token ID sequences for greedy decode correctness verification -- mismatch = auto-reject
- **Per-cycle structured logging**: `data/cycles/cycle_NNNN.json` with git info, optimizer hypothesis, verification results, cumulative drift tracking
- **Config-driven bench**: `bench_config.toml` controls scoring weights, constraint thresholds, model paths, and optimizer scope
- **Verification phase**: diff inspection, per-prompt regression checks, suspicion flags, variance checks built into the optimization loop
- **Optloop skill**: `/optloop` skill replaces the old `optloop.md` slash command

### Changed

- **Bench scripts relocated**: moved from `scripts/` to `apps/optloop/scripts/` following the batch-labeler self-contained app pattern
- **Scoring weights configurable**: `compute_composite_score()` accepts weights dict from config instead of module-level constants
- **Constraint thresholds configurable**: `check_hard_constraints()` reads thresholds from config
- **CLI args override config**: bench scripts load `bench_config.toml` as defaults, CLI flags take precedence

### Removed

- `scripts/optloop.md` (replaced by `apps/optloop/program.md` + `/optloop` skill)

## [1.22.0]

### Added

- **Bench harness**: Direct-load benchmark scripts for text (`bench_text.py`) and VLM (`bench_vlm.py`) inference paths -- no HTTP server required
- **Composite scoring**: Weighted metric (40% gen_tps, 25% TTFT, 20% prefill_tps, 15% memory) with hard constraint checks and baseline tracking
- **Bench analysis**: `bench_analysis.py` reads results.tsv and per-run JSON to produce summary tables and progress charts
- **Optimization loop**: `optloop.md` agent instructions for continuous autonomous inference optimization with dual-bench scoring

## [1.21.1]

### Fixed

- **VLM vision**: pass mask as `mask` kwarg (not `attention_mask`) to VLM models, fixing `Model.__call__() missing 1 required positional argument: 'mask'` for mistral3, pixtral, and llava_next architectures
- **VLM chat template**: flatten list content before applying tokenizer chat template, fixing `can only concatenate str (not "list") to str` for mistral3/pixtral models

## [1.21.0]

### Added

- **Packaging**: Package is installable from git via `uv pip install git+https://github.com/fblissjr/heylookitsanllm`. Profiles and service templates ship inside the wheel as package data (`heylook_llm.data.profiles`, `heylook_llm.data.services`).
- **Dynamic version**: Single source of truth in `heylook_llm.__version__`, read by setuptools at build time.
- **Platform guard**: `heylookllm` CLI exits with a clear error on non-macOS platforms.
- **Project URLs**: Homepage, repository, and issues links in package metadata.

### Changed

- **macOS-only deps**: `mlx`, `mlx-lm`, `mlx-vlm`, and `parakeet-mlx` now carry `sys_platform == 'darwin'` markers so pip can resolve the dependency tree on non-macOS (even though the server requires macOS to run).
- **Classifiers**: Removed Linux/Windows OS classifiers; added Python 3.11/3.12/3.13, FastAPI, and AI topic classifiers.
- **Data file paths**: `profiles/` and `services/` moved from repo root into `src/heylook_llm/data/`; path resolution uses `importlib.resources` instead of `__file__`-relative traversal (fixes broken paths when installed from wheel).
- **License metadata**: Switched from `license = { file = "LICENSE" }` to SPDX expression `license = "MIT"` per PEP 639.

## [Unreleased]

### Added

- **Model pinning**: `pin_model()`/`unpin_model()` on ModelRouter prevent LRU eviction of models in active use.
- **Dynamic embedding backbone**: Embedding provider loads any mlx-lm-supported architecture via `load_backbone()`, replacing the hardcoded Gemma3 import.
- **Pooling config**: Embedding models accept `pooling` field (`mean`, `cls`, `none`) for future multi-vector/ColBERT support.
- **Stop-token utility**: Shared `resolve_stop_tokens()` standardizes EOS token resolution across all generation paths.
- **Embedding weight sanitization**: Strips rotary embedding frequencies, vision tower weights, and multimodal projector keys for architecture-agnostic backbone loading.

### Changed

- Pydantic validators migrated from V1 (`@validator`) to V2 (`@field_validator`, `@model_validator`).

### Removed

- **STT provider**: Removed `mlx_stt` provider, `/v1/audio/transcriptions` endpoint, `parakeet-mlx` dependency, and `stt` optional dependency group.
- **Batch vision labeling pipeline**: Decoupled from backend into standalone client app at `apps/batch-labeler/`. Removes `batch_vision_pipeline.py`, 4 API endpoints, provider-specific prefix cache methods, and SQLite/threading infrastructure. The client app calls the existing `/v1/chat/completions` VLM endpoint instead.

### Fixed

- **Radix cache crash on VLM hybrid models (Qwen3.5)**: Editing/deleting messages caused `broadcast_shapes` ValueError when the radix cache restored a partial prefix match. Three root causes: (1) VLM `LanguageModel._position_ids` persisted across requests, causing stale position slicing; (2) KV snapshots contained entries beyond the matched prefix length, corrupting cache offsets; (3) failed prefill stored broken snapshots that cascaded into future requests. Fixed by resetting VLM position state per-request, trimming KVCache to matched prefix on restore, and skipping snapshot storage on generation errors.
- **VLM model loading crash with transformers 5.x**: Four bugs in transformers 5.x prevent VLM processor loading when torchvision is absent (the correct state for MLX-only setups). Patched at import time in `mlx_provider.py`: `VIDEO_PROCESSOR_MAPPING_NAMES` None values, `auto_docstring` IndexError, `AutoVideoProcessor.from_pretrained` hard-fail, and `ProcessorMixin` type-check rejection of optional `None` sub-processors. Qwen3.5 VLM models now load correctly.
- **Batch processor eos_token_ids null safety**: `hasattr` returns `True` when attribute is `None`; switched to `getattr(..., None) or set()` to handle tokenizers that define `eos_token_ids` as `None`.
- **transformers version**: Pinned `>=5.0.0` to match mlx-lm 0.30.8 requirement. Added `override-dependencies` in `[tool.uv]` to force latest regardless of transitive pins.
- **Mobile state persistence**: Chat and notebook stores now flush to IndexedDB on `visibilitychange`/`pagehide`, preventing data loss when mobile Safari kills the tab.
- **iOS delete button**: Sidebar delete button fires from `onTouchEnd` directly, working around iOS Safari not synthesizing click events for small tap targets in scrollable lists.
- **Gitignore silently dropping src/lib/**: Python `lib/` ignore rule was catching `apps/heylook-frontend/src/lib/`. Added exclusion so frontend lib modules are tracked.
- **DB connection retry**: `getDB()` no longer caches a rejected `openDB` promise forever. If IndexedDB open fails (quota, permissions), the next call retries instead of failing permanently.
- **Thread safety in unpin_model**: `unpin_model()` now acquires `cache_lock` to prevent TOCTOU race with `_evict_lru_model()`.

### Removed

- **torchvision dependency**: Removed unused `torchvision` from core dependencies (nothing imports it; it just pulled in PyTorch unnecessarily).

- **MLX Embedding Provider**: New `mlx_embedding` provider for EmbeddingGemma models. Produces contextual 768-dim embeddings via full bidirectional transformer forward pass with padding-aware attention masking, mean pooling, dense projections, and L2 normalization. Supports task-specific prefixes (query, document, code_retrieval, clustering) and quantized model loading (4bit, 8bit via nn.quantize). 30 unit tests.
- **EmbeddingGemmaModel**: Pure MLX encoder reusing mlx-lm Gemma3 internals with bidirectional attention and padding mask. Located in `src/heylook_llm/models/embedding_gemma.py`.
- **Embedding model import**: Model importer now detects embedding models (bidirectional attention config or `*_Dense` projection dirs) and imports them as `provider: "mlx_embedding"` with correct config (no vision/temperature/sampling params). Model service validates and imports `mlx_embedding` provider correctly.

### Fixed

- **Embedding padding attention**: Padding tokens no longer contaminate content token hidden states. EmbeddingGemmaModel now creates a (B, 1, 1, seq_len) additive padding mask instead of passing mask=None to all layers. Identical content with different padding now produces identical embeddings.
- **Diagnostic logging**: Frontend ring buffer (5000 events) with JSONL download from Settings panel. Backend writes structured events to `logs/events.jsonl`. Request IDs (`X-Request-ID` header) correlate frontend and backend events. Console verbosity adjustable via Settings or `window.__setLogLevel()` in devtools.
- **Stream timeout setting**: Configurable stream timeout (default 30s) in Generation Settings panel. Prevents permanently stuck streaming state when the backend hangs.

### Changed

- **Model selection consolidated**: Removed per-applet model dropdowns from Chat, Batch, and Token Explorer. All applets use the globally loaded model from the top-level selector. Model Comparison multi-select unchanged.
- **VLM guards**: Batch shows a warning when a VLM is loaded (batch mode is text-only, submit disabled). Notebook hides image attachment UI when a text-only model is loaded.
- **Tokenizer extraction consolidated**: Provider base class now exposes `get_tokenizer()` method; tokenizer extraction consolidated from 2 duplicated call sites in api.py.
- **Frontend re-render optimization**: Bare `useModelStore()` calls replaced with individual selectors in 9 components (reduces unnecessary re-renders).
- **Logprobs init deduplication**: Logprobs collector initialization extracted into shared `_init_logprobs_collector()` helper (removes ~30 lines of duplication between streaming and non-streaming paths).
- **Frontend package manager**: Migrated from npm to bun.

### Fixed

- **Generation lock deadlock after client disconnect**: `async_generator_with_abort` now calls `sync_gen.close()` in a finally block, deterministically releasing the provider's `_generation_lock` on all exit paths (disconnect, aclose, normal completion). Previously the abandoned generator only released the lock when GC collected it, causing the next request to hang for up to 30s.
- **Bare except handlers**: Replaced `except:` with `except Exception:` in Metal info query and STT cache cleanup (was swallowing SystemExit/KeyboardInterrupt).
- **Logprobs helper exceptions**: `_decode_token` and `_get_token_bytes` exception handlers narrowed to specific types instead of broad `except Exception`.
- **Non-streaming logprobs init error path**: Missing diagnostic event in non-streaming logprobs init `except` block (now shared via extracted helper).
- **Logprobs exception handling**: `add_token()` exception handler narrowed from `except Exception` to specific types (IndexError, ValueError, RuntimeError, TypeError).
- **Non-streaming logprobs diagnostic**: Non-streaming logprobs path now logs `logprobs_missing_data` diagnostic event (was streaming-only).
- **Redundant provider lookup**: Removed redundant `router.get_provider()` call in streaming logprobs initialization.
- **Token Explorer logprobs**: Added full traceback logging and diagnostic events to the logprobs pipeline. Silent exceptions in `logprobs.py` now log `exc_info=True`. Missing tokenizer logged at `WARNING` level instead of silently producing no logprobs.
- **Chat: concurrent stream guard**: Sending a message while a stream was already in-flight silently started a second stream without aborting the first. Both streams wrote to the store simultaneously. A new `ChatStreamManager` singleton ensures the previous stream is aborted before any new one starts.
- **Chat: AbortController leak**: The controller was only nulled on the success path; errors left a stale reference. The controller is now nulled in a `finally` block in all cases.
- **Chat: wrong-conversation targeting**: `finalizeStream` read `activeConversationId` at callback time. Switching conversations mid-stream caused streamed content to be written into the newly active conversation. The conversation ID is now pinned when the stream starts and passed through to `finalizeStream`.
- **Chat: orphaned streams on navigation**: Navigating away from the Chat applet did not stop in-flight streams. The backend kept generating and eventual callbacks wrote to a detached store. `ChatView` now calls `stopGeneration()` on unmount.
- **Chat: no timeout**: A hung backend (stuck MLX kernel, stalled connection) left `isStreaming: true` permanently with no way to recover without a page reload. A 30-second timeout is now applied via `AbortSignal.timeout()`, producing a user-visible error message.
- **Chat: "Failed to fetch" on cancel-and-resend**: Cancelling a streaming generation and immediately sending a new message caused `TypeError: Failed to fetch` because the browser HTTP connection from the aborted SSE stream was never released (relied on GC). The `ReadableStreamDefaultReader` is now explicitly cancelled on abort. `stopGeneration` saves partial content to the message. Stale-callback guards prevent old stream callbacks from corrupting a new stream's state.

## 1.22.0

### Added

- **Performance profiling backend**: `GET /v1/performance/profile/{time_range}` now returns real aggregated data (was a 503 stub). In-memory ring buffer (10K events, ~2MB) records timing breakdown, per-model bottlenecks, hourly trends, and resource timeline. Background task collects system snapshots every 60s. Performance applet in frontend now renders live data.
- **Benchmark script** (`scripts/benchmark.py`): HTTP-based benchmark for measuring TTFT, generation TPS, and memory against a running server. Supports both OpenAI and Messages endpoints, streaming and non-streaming modes, configurable prompt sets, and `--json` output for CI.

## 1.21.0

### Removed

- **llama.cpp provider and all GGUF support**: Deleted `LlamaCppProvider`, `LlamaCppModelConfig`, `LlamaCppEmbeddingExtractor`, `LlamaCppHiddenStatesExtractor`, and all associated config types, router entries, factory functions, test fixtures, and frontend type unions. The `gguf` pyproject extra is removed. Provider type narrowed from `mlx | llama_cpp | gguf | mlx_stt` to `mlx | mlx_stt` throughout backend and frontend. GGUF model entries removed from `models.toml`. A future `llama-server` subprocess provider will replace this.
- **Dead scripts and config**: Removed `setup_analytics.py`, `analytics_config.json`, `.env.example` (analytics system removed in v1.20.0), and broken `tests/run_tests.sh`.

### Changed

- **tests/README.md**: Full rewrite -- documents all 34 backend test files (unit, contract, integration), corrected coverage matrix, updated run commands to `uv run pytest`.

## 1.20.0

### Removed

- **14 dead/broken API endpoints**: Removed analytics endpoints (`/v1/data/summary`, `/v1/data/query`, `/v1/data/request/{id}`), evaluation endpoints (`/v1/eval/create`, `/v1/eval/run`, `/v1/eval/run/{id}`, `/v1/eval/list`), replay endpoint (`/v1/replay/{id}`), async batch processing (`/v1/batch/process`, `/v1/batch/{id}`), and server restart (`/v1/admin/restart`). All were broken at runtime or had no consumers.
- **6 dead files**: `data_endpoint.py`, `api_capabilities.py`, `openapi_enhancements.py`, `analytics_config.py`, `metrics_db.py`, `metrics_db_wrapper.py` -- never imported or only consumed by removed endpoints.
- **Analytics from core path**: Removed metrics database logging from chat completion request/response handlers and server startup initialization.
- **STT dead endpoints**: Removed broken `/v1/audio/translations`, stub `/v1/stt/stream` WebSocket, hardcoded `/v1/stt/models`. Simplified transcription response to `json` and `text` formats only (removed fake `srt`, `vtt`, `verbose_json`).
- **Sync streaming generator**: Removed unused `stream_response_generator()` (async version is what's actually used).

### Changed

- **`/v1/admin/reload` moved to admin_api.py**: New `admin_ops_router` with `/v1/admin` prefix, consistent with other admin endpoints.
- **Image resize logic extracted**: Duplicate ~25-line resize blocks in `create_chat_completion` consolidated into `_apply_image_resize()` helper.
- **Shared streaming utilities**: New `streaming_utils.py` with `async_generator_with_abort()`, `get_provider_or_503()`, and `consume_sync_generator()` -- used by both `api.py` and `messages_api.py`.
- **`/v1/performance/profile/{time_range}`**: Re-added as a stub returning 503, so the frontend Performance applet gets a clean error instead of 404.

## 1.19.0

### Fixed

- **Profile apply bug**: `ModelProfile.apply()` now unconditionally sets profile values instead of only filling gaps. Previously, smart defaults ran first and set `top_k`, `max_tokens`, `cache_type`, etc., so profiles could never override them. Precedence is now: base -> smart_defaults -> profile overrides -> user `--override`.
- **Sub-1B model size regex**: `Qwen3-0.6B` was reported as "(6B)" because the integer pattern `(\d+)b` matched before `(\d+\.\d+)b`. Swapped regex order so decimal patterns match first.
- **Admin `/status` route shadowed by catch-all**: `GET /v1/admin/models/{id}/status` was unreachable because the greedy `GET /{model_id:path}` catch-all was registered first. Reordered route registration so sub-resource routes (`/status`, `/toggle`, `/load`, `/unload`) register before catch-all routes.

### Changed

- **Profiles moved to TOML files**: The 9 hardcoded Python profile definitions are now standalone TOML files in `profiles/`. Each file has `[meta]` (name, description) and `[defaults]` (flat key=value). Lambda-based dynamic values removed; size-based logic stays in `get_smart_defaults()`. Profiles are loaded once at module import via `tomllib`.
- **Profile renames**: `fast` -> `tight_fast`, `balanced` -> `moderate`, `quality` -> `wide_sampling`, `performance` -> `high_throughput`, `max_quality` -> `widest_sampling`, `background` -> `low_resource`, `memory` -> `quantized_kv`, `interactive` -> `conversation`, `encoder` -> `embedding`.
- **Dynamic profile discovery**: `--profile` choices in CLI are now discovered from `profiles/*.toml` filenames instead of a hardcoded list. Adding a new profile is just dropping a `.toml` file.
- **Profile values printed on import**: When `--profile` is used, the parameter table is printed before writing. The "Available profiles" listing now includes a parameter summary for each profile.

### Removed

- **`/v1/performance` stub endpoint**: Deleted the stub that returned "Removed in v1.17.1". No consumers.
- **`/v1/performance/profile/{time_range}` endpoint**: Deleted ~170 lines of DuckDB queries for performance profiling. The analytics SQL endpoint (`/v1/data/query`) provides equivalent ad-hoc access.
- **`test_performance_monitoring()` integration test**: Deleted stale test that tested the removed endpoint.
- **`mlx` optional extra**: Removed from pyproject.toml. `mlx`, `mlx-lm`, `mlx-vlm`, `transformers` are already in core `dependencies`; the extra duplicated them and added unused packages.
- **Unused dependencies from extras**: Removed `torch`, `torchvision`, `opencv-python`, `scipy` (never imported in production code). Moved `datasets` to `analytics` extra (only used by data loader). Moved `rich` to `scripts` extra (only used by `scripts/metrics_dashboard.py`).

### Added

- **`gguf` placeholder extra**: Empty extra with commented `llama-cpp-python` for Phase Next.
- **`scripts` extra**: Contains `rich` for dashboard scripts.
- **`load_profiles()` / `get_available_profiles()` API**: Public functions for programmatic profile access.
- **Contract test suite**: TestClient-based API tests (39 tests) covering `/v1/models`, `/v1/chat/completions`, `/v1/messages`, `/v1/admin/models/`, and OpenAPI conformance. Runs in-process, no server or models needed.
- **Profile unit tests**: 22 tests covering TOML profile loading, profile application with provider filtering, model size regex, and the `load_profiles()` caching API.
- **`httpx` test dependency**: Added to `[test]` extra for Starlette TestClient support.

## 1.18.1

### Added

- **`--interactive` flag for model import**: `heylookllm import --interactive` launches a TUI (via ConfigEditor) to customize sampler and KV cache settings for each discovered model before writing `models.toml`. Compatible with `--profile` (profile applies first, interactive tweaks override).

### Changed

- **Documentation refresh for v1.18.0**: Updated CLAUDE.md, architecture.md, mlx.md, mlx_optimization_plan.md, and TODO.md to reflect vision path unification via pre-filled cache pattern.

## 1.18.0

### Changed

- **Vision path unification**: Replaced `mlx_vlm.generate.stream_generate` with a pre-filled cache pattern (inspired by vllm-mlx). The full VLM model runs a single forward pass to encode vision + text into a KV cache, then the language model generates tokens using `generation_core.run_generation()` -- the same code path as text-only requests. Vision requests now get the full sampler suite (top_k, min_p, presence_penalty, logit_bias, XTC), abort support, and speculative decoding acceptance tracking. Eliminates the hardcoded Qwen `[1, 24, 24]` image grid -- `mlx_vlm.utils.prepare_inputs` handles grid dimensions natively per model.
- **Syntax check auto-discovery**: `scripts/syntax_check.py` now uses `glob.glob("src/heylook_llm/**/*.py")` instead of a hand-curated file list. Adding or removing source files no longer requires updating the script.
- **Cache miss logging**: Radix cache misses now log at INFO level with model ID for observability parity with cache hits.

### Removed

- **`vlm_generation.py`**: Deleted entirely (55 lines). The `stream_generate_vlm_vision()` wrapper around `mlx_vlm.generate.stream_generate` is replaced by the pre-filled cache approach in `VLMVisionStrategy`. The Qwen model-type string sniffing for `image_grid_thw` is no longer needed.
- **`BatchVisionEncoder` import**: Removed unused import from `mlx_provider.py`.

## 1.17.1

### Removed

- **`PerformanceMonitor`**: Deleted `performance_monitor.py` (234 lines) and all `@time_mlx_operation` decorators. No consumers, no tests, generator-wrapping overhead on every token. `/v1/performance` endpoint returns a stub response.
- **`VLMGeneratorWithSampling` class**: Flattened to standalone `stream_generate_vlm_vision()` function (~55 lines, down from 161). Dead text-only branch, `LanguageModelLogitsWrapper` cache, and `lm_stream_generate` import removed -- all unreachable after phase 5 unification.
- **Duplicate `_reconstruct_thinking`**: Deleted copy from `mlx_provider.py`. Canonical version lives in `providers/common/vlm_inputs.py`.

## 1.17.0

### Added

- **`generate_text()` entry point**: New high-level function in `generation_core.py` that builds sampler/processors internally and delegates to `run_generation()`. Strategies call this instead of building samplers externally, keeping sampler construction co-located with the generation loop.
- **Dynamic draft token tuning (`DraftTuner`)**: Module-level singleton in `generation_core.py` that dynamically adjusts `num_draft_tokens` per model based on rolling acceptance rate. Conservative policy: increase by 1 (max 8) when acceptance > 80%, decrease by 1 (min 1) when < 50%, over a 50-sample window. Integrated into `run_generation()` automatically.
- **Standalone VLM input preparation**: Extracted `VLMVisionStrategy._prepare_vlm_inputs_parallel` (92 lines) to `providers/common/vlm_inputs.py` as a standalone function. Testable without instantiating a full strategy.
- **Unified path equivalence tests**: Parameterized tests proving `UnifiedTextStrategy` produces equivalent `generate_text()` calls for `is_vlm=True` and `is_vlm=False`.

### Changed

- **Sampler construction moved out of `create_chat_completion()`**: `build_sampler()` no longer called in the routing layer. `UnifiedTextStrategy` uses `generate_text()` (builds sampler internally); `VLMVisionStrategy` builds its own sampler at the start of `generate()`. Strategy signatures no longer include `sampler`/`processors` parameters.
- **`run_generation()` consults DraftTuner**: When a draft model is active, `run_generation()` queries `DraftTuner` for the current token count before calling `lm_stream_generate`, and feeds acceptance data back in the `finally` block.

## 1.16.0

### Changed

- **Provider strategy unification**: Merged `TextOnlyStrategy` and `VLMTextOnlyStrategy` (~400 lines of duplication) into a single `UnifiedTextStrategy` (~130 lines) that dispatches on `is_vlm` for chat template application and model wrapping. All shared logic (cache config, prompt cache, generation loop, acceptance tracking, KV snapshot storage) extracted to `generation_core.run_generation()`. Strategy keys changed from `text_only`/`vlm_text`/`vlm_vision` to `text`/`vision`.
- **`LanguageModelLogitsWrapper` moved**: Relocated from `mlx_provider.py` to `providers/common/model_wrappers.py` to break circular import with `vlm_generation.py`.
- **Generation core extracted**: New `providers/common/generation_core.py` contains the single generation loop (`run_generation`), cache config construction (`_build_cache_config`), and prompt cache setup (`_setup_prompt_cache`). This is the integration point for future `mx.compile` optimization.
- **Simplified routing**: `_compile_strategies()` now creates 1-2 strategies (text, optionally vision) instead of 2-3. `create_chat_completion()` routing reduced to a simple `has_images` check.

## 1.15.0

### Fixed

- **`num_draft_tokens` passthrough**: Both `TextOnlyStrategy` and `VLMTextOnlyStrategy` now pass `num_draft_tokens` to `lm_stream_generate`. Previously, the configured value (default 6) was never forwarded, causing `speculative_generate_step` to use its hardcoded default of 2. New default changed from 6 to 3 (safe middle ground between mlx-lm's default of 2 and the overly aggressive 6).
- **`VLMTextOnlyStrategy` missing `model_config`**: Added `model_config` parameter so VLM text-only path can access model-level config (thinking mode, cache settings, etc.), matching `TextOnlyStrategy`.
- **`cache_config` ignoring model config**: The `kv_bits` fallback changed from hardcoded `8` to `None`, and `_apply_model_defaults()` now explicitly includes cache config fields (`cache_type`, `kv_bits`, `kv_group_size`, `max_kv_size`, `quantized_kv_start`, `num_draft_tokens`) from model config. Previously, if the request didn't specify cache params, the strategy's fallback defaults would override model config.
- **Analytics DB size limit enforced**: The cleanup thread now prunes the oldest 25% of records and runs VACUUM when the database exceeds `max_db_size_mb`, instead of just logging a warning.

### Added

- **Memory-pressure eviction for radix cache**: `RadixCache` accepts an optional `memory_pressure_fn` callback checked before each node insertion. When GPU memory exceeds 85% of the recommended working set, eviction triggers even if node count is below `max_nodes`. Keeps the radix cache pure (no MLX dependency in the data structure).
- **Speculative decoding acceptance tracking**: Both text-only strategies now count draft token acceptance/rejection during generation and log the acceptance rate in the finally block. Provides visibility into whether speculation is helping without changing behavior.
- **Startup disk usage logging**: Server startup now logs analytics DB size (with limit) and log directory size for disk usage visibility.
- **`num_draft_tokens` in smart defaults**: `get_smart_defaults()` now includes `num_draft_tokens=3` for MLX models, ensuring the field is always present when users configure `draft_model_path`.

### Changed

- **Profile cache threshold aligned**: The `balanced` profile's quantized KV cache threshold changed from >30GB to >13GB, matching `get_smart_defaults()`.

### Removed

- **`models.toml.example`**: Deleted. The `heylookllm import` command and smart defaults generate complete config. README updated to use `heylookllm import --hf-cache` instead of `cp models.toml.example`.

## 1.14.0

### Added

- **Radix-tree prefix cache**: New `RadixCache` data structure (`providers/common/radix_cache.py`) stores multiple cached prefixes per model simultaneously. Editing an earlier message, branching, or regenerating no longer invalidates the entire cache -- only the divergent suffix needs re-prefilling. Configurable block size (32 tokens), LRU leaf eviction, thread-safe.
- **KV snapshot helpers**: `snapshot_kv()` and `restore_kv_from_snapshot()` in `cache_helpers.py` capture and restore KV cache state for radix tree storage. Uses MLX copy-on-write semantics for cheap snapshots.

### Changed

- **Pure-MLX sampler**: `_mlx_unique()` in `samplers.py` reimplemented using `mx.sort` + `mx.cumsum` + scatter, replacing the numpy-based version that forced a full GPU-to-CPU sync on every token when presence penalty was active. Now only a single int32 scalar crosses the device boundary.
- **Prompt cache manager**: `PromptCacheManager` now uses a `RadixCache` per model as the persistent backing store instead of a single linear prefix cache. Public API (`get_or_create_cache`, `process_prompt_with_cache`) unchanged.

### Removed

- **numpy dependency from samplers**: `import numpy as np` removed from `samplers.py`. The presence penalty path now stays entirely on the Metal compute graph.

## 1.13.0

### Removed

- **Dead code**: Deleted `mlx_optimizations.py` (283 lines of hallucinated MLX APIs, zero imports anywhere)
- **Queue manager**: Removed `queue_manager.py` and all queue branches from `api.py`, `messages_api.py`, `router.py`. Queue was always disabled (`queue_config.enabled: false`) and fundamentally incompatible with streaming (`list(generator)` defeated the point). Removes ~400 lines of dead code.

### Added

- **Generation abort mechanism**: New `AbortEvent` (`providers/abort.py`) enables cooperative cancellation of in-flight MLX generation. When a new request arrives while another is generating, the current generation is aborted (per-token check) so the new request starts immediately instead of blocking. Client disconnect during SSE streaming also triggers abort, freeing GPU compute.

## 1.12.4

### Added

- **Models applet sorting**: Sort models by name (A-Z/Z-A), provider, or status (loaded first) via dropdown selector
- **Models applet tag filtering**: Data-driven tag chips extracted from model configs; click to filter by tag (OR logic)
- **Models applet preference persistence**: Sort and filter preferences persist to localStorage across sessions

## 1.12.3

### Changed

- **README.md rewrite**: Updated from v1.1-era to reflect current state -- dual API (OpenAI + Messages), 7 applets, llama-server subprocess (not llama-cpp-python), thinking blocks, logprobs, model management, hidden states. Removed dead links (`guides/SERVICE_SECURITY.md`, `docs/WINDOWS_INSTALL.md`) and stale installation instructions (`--extra llama-cpp`, `CMAKE_ARGS`/`llama-cpp-python` GPU section).
- **pyproject.toml version**: Bumped from 1.2.0 to 1.12.2 to match actual release.
- **performance extra trimmed**: Removed 6 unused packages (imagecodecs, blake3, diskcache, msgpack, aiofiles, aiocache). Kept xxhash, PyTurboJPEG, uvloop, cachetools.

## 1.12.2

### Fixed

- **Mobile delete button**: Tapping the trash icon on iOS Safari triggered conversation selection (and sidebar collapse) instead of deletion. Touch events now stop propagation on the delete button so they don't bubble to the parent's long-press handler.
- **Prompt cache stale KV entries**: After generation, the KV cache contained both prompt AND generated tokens, but token tracking only recorded the prompt. On regeneration or message editing, the trim calculation was wrong, leaving old response tokens in the KV cache and biasing the model toward repeating its previous output. Now tracks generated token IDs so trimming is accurate.

## 1.12.1

### Fixed

- **ModelImporter race condition**: `handleScan` used stale closure-captured `scanResults` after async `scanForModels()`. Now reads fresh state via `useModelsStore.getState().scanResults`.
- **model_service.py update_config mutation bug**: Updates were applied in-place to the models list before validation. Invalid updates corrupted in-memory state during lock hold. Now works on a deep copy and only commits if validation passes.
- **Path validation ordering**: `model_path` validation happened after the value was already written to the config dict. Moved to before any mutations.
- **default_model fallback**: When removing the default model, fallback now prefers enabled models instead of blindly using `models[0]`.
- **ModelDetail save/remove error handling**: `handleSave`, `handleConfigUpdate`, and `handleRemove` now catch errors and display them inline instead of leaving the UI in a broken state.
- **Import modal state persistence**: Local state (customPath, selectedIds, step, profile) now resets when the modal reopens.

### Added

- **Per-action loading states**: `actionLoading` store field tracks which model is being acted on. Load/unload buttons, toggle button, and preset buttons show spinners during operations.
- **Import modal error display**: Errors from scan/import now display inline in the modal instead of behind it.
- **PresetSelector empty state**: Shows "Failed to load profiles" instead of permanent "Loading..." when profile fetch fails. Uses `profilesLoaded` flag to distinguish loading from failure.
- **reload_config error handling**: All 6 admin API endpoints that call `router.reload_config()` now catch exceptions and include a warning in the response instead of crashing.
- **Models applet tests**: 70 tests across 3 files (modelsStore.test.ts, ModelList.test.tsx, ModelImporter.test.tsx). Total test count: 781.

### Changed

- **model_importer.py refactor**: Moved `PROFILES`, `ModelProfile`, `get_smart_defaults`, `get_hf_cache_paths` into `model_service.py` (single source of truth). `model_importer.py` reduced from 874 to 450 lines, now imports shared logic from `model_service`. Extracted `_detect_tags` helper to reduce duplication in MLX/GGUF entry creation.
- **Removed fragile route exclusion list**: `admin_api.py` no longer hard-codes sub-paths that the `{model_id:path}` catch-all must skip. FastAPI's two-router registration order handles this correctly.

## 1.12.0

### Added

- **Model Management System**: Full-stack model management with backend API and frontend applet
  - **ModelService** (`model_service.py`): Service layer for model discovery, validation, and config management with thread-safe atomic TOML writes, path validation, CRUD operations, scan/import, and profile application
  - **Admin API** (`admin_api.py`): 14 endpoints under `/v1/admin/models/` for CRUD, scan, import, validate, profiles, bulk-profile, load/unload, and status
  - **Router enhancements**: `unload_model()`, `get_model_status()`, `reload_single_model()` on ModelRouter
  - **Pydantic models**: `ScannedModelResponse`, `ModelScanRequest`, `ModelImportRequest`, `ModelUpdateRequest`, `AdminModelResponse`, `ModelStatusResponse`, `ProfileInfo`, etc.
- **Models applet** (`/models` route): 7th frontend applet for model management
  - Side-by-side list + detail layout (AppletLayout pattern)
  - Searchable, filterable model list with status pills (Loaded/Available/Disabled)
  - Full config editing with field-level reload indicators
  - Import workflow: scan filesystem/HF cache, select models, apply profile, import
  - Preset selector for quick profile application
  - Provider-specific config forms (MLX 17 fields, GGUF 13 fields, STT 6 fields)
  - Load/unload controls per model
  - Metadata editing (description, tags)
  - Config-only removal with confirmation

### Changed

- **AppNav**: Added Models entry with CubeIcon (7 nav items total)
- **MobileBottomNav**: Inherits Models entry via shared `navItems`
- **App.tsx**: Added lazy-loaded `/models` route

## 1.11.1

### Changed

- **Documentation restructure**: Reorganized flat `internal/` directory (36 files, 9 dead) into `backend/`, `backend/providers/`, `bugs/`, `research/`, `frontend/`, `session/`, `log/`. Deleted 9 obsolete files, consolidated 5 into 3.
- **CLAUDE.md rewrite**: Reduced from 322 lines to 87 lines. Now a nav hub that links out instead of duplicating content.
- **Stale CoreML STT references**: Updated all non-historical references to use MLX STT (docs, tests, scripts). CoreML STT was removed in v1.2.0.

## 1.11.0

### Added

- **Mobile bottom tab navigation**: New `MobileBottomNav` component provides access to all 6 applets on mobile (Chat, Batch, Token Explorer, Model Comparison, Performance, Notebook). Previously only Chat was reachable.
- **Shared `AppletLayout` component**: Reusable responsive wrapper for applets with left panels. Desktop shows inline panel; mobile hides it behind a toggle button with overlay drawer.
- **Model loading from any applet**: `ModelSelector` and right panels (Advanced, Settings) lifted from Chat-only `Layout` to `AppShell`, making model management available on every route.
- **iOS scroll fix**: `100dvh` height with `100vh` fallback, `-webkit-overflow-scrolling: touch` on chat scroll container.
- **Desktop content width constraints**: `max-w-3xl` on chat messages, `max-w-4xl` on batch/notebook/token-explorer, preventing unreadable line lengths on wide screens.

### Changed

- **AppShell absorbs shared chrome**: Header, SystemStatusBar, ModelSelector panel, AdvancedPanel, SettingsPanel, and mobile detection all moved from route-specific `Layout` to `AppShell`. `Layout` reduced to Chat sidebar wrapper only.
- **Header is route-aware**: Sidebar hamburger toggle only renders on `/chat`; other routes show a spacer.
- **ModelSelector no longer uses `onModelLoaded` callback**: Removed prop. ChatView now watches `loadedModel` store state directly to auto-create conversations.
- **AppNav exports `navItems`**: Shared between desktop sidebar and mobile bottom nav.
- **Applet LeftPanels stripped of outer wrappers**: `AppletLayout` now provides width, border, and responsive behavior.

### Frontend Tests

- 707 tests passing across 31 test files (was 686/28)
- New test files: `AppShell.test.tsx` (20), `MobileBottomNav.test.tsx` (6), `AppletLayout.test.tsx` (13)
- Rewrote `Layout.test.tsx` (sidebar-only), `Header.test.tsx` (route-aware), `ModelSelector.test.tsx` (removed callback tests)

## 1.10.1

### Fixed

- **`wired_limit` model mismatch in VLMTextOnlyStrategy**: `wired_limit()` was receiving the full VLM model (vision encoder + language model) but only the language model wrapper was running, causing incorrect Metal memory limit calculations. Now correctly passes `self._cached_wrapper` to match the model actually used for generation.
- **Generator detection in performance_monitor**: `time_operation` used `hasattr(result, '__next__')` which matched any iterator. Changed to `isinstance(result, types.GeneratorType)` for precise generator detection.
- **VLM vision path now forwards logits_processors and repetition_penalty**: `_stream_generate_vision_enhanced()` previously silently dropped `processors` and `repetition_penalty`. These are now forwarded to `vlm_stream_generate` as `logits_processors` and `repetition_penalty` kwargs (both supported by `mlx_vlm.generate.generate_step`).

### Removed

- Dead `_cached_generator` field from `VLMTextOnlyStrategy` (only used by `VLMVisionStrategy`)
- Duplicate `import threading` inside `MLXProvider.__init__` (already imported at module level)

## 1.10.0

### Fixed

- **VLM text-only sampling parity**: `VLMTextOnlyStrategy` now uses `lm_stream_generate` with full sampler/processor pipeline (top_k, min_p, presence_penalty, logit_bias, XTC). Previously bypassed all advanced sampling by calling `vlm_stream_generate` with raw temperature/top_p/repetition_penalty kwargs.
- **VLM text-only prompt caching**: Added prompt cache support to VLM text-only path (same pattern as `TextOnlyStrategy`), reducing token processing on follow-up requests.
- **Performance monitor generator timing**: `time_operation` decorator now correctly times generator functions from first to last yield, instead of measuring generator object creation time (microseconds).
- **`_apply_model_defaults` serialization**: Replaced `request.model_dump()` (serialized entire request including all messages) with direct `getattr()` for the 9 scalar parameter fields.

### Removed

- **`mlx_metal_tuning.py` module**: Deleted entirely. This module was called on every model load and caused active harm: cast all weights to float16 (destroying 4-bit quantized weights), pre-allocated unused KV cache buffers (wasting GPU memory), set wired limits that conflicted with per-generation context managers, and ran `subprocess.run(['sysctl', 'hw.memsize'])` on every load.
- **Broken `_content_cache`**: Removed image detection cache that used `id(messages)` as key (ephemeral per request, never hit, grew without bound).
- **Dead methods**: Removed `VLMTextOnlyStrategy._prepare_vlm_inputs`, `VLMVisionStrategy._prepare_vlm_inputs`, `VLMGeneratorWithSampling._apply_advanced_sampling`, `_get_vocab_size`, `_get_eos_token_id`, `supports_speculative_decoding`, and `vlm_stream_generate_with_sampling` convenience wrapper.
- **Unused imports**: Cleaned up `traceback`, `ABC`, `abstractmethod`, `vlm_generate`, `vlm_stream_generate`, `load_image`, `make_cache`, `BatchVisionStrategy` from mlx_provider.py. Removed `nn` from vlm_generation.py.

### Added

- **`mx.clear_cache()` after generation**: Added to `create_chat_completion` finally block to release MLX internal memory cache between requests, preventing memory accumulation.
- **`LanguageModelLogitsWrapper` caching in `VLMGeneratorWithSampling`**: Wrapper now created once in `__init__` instead of per-request.

## 1.9.0

### Added

- **`/v1/messages` endpoint**: Anthropic Messages-inspired API alongside existing `/v1/chat/completions`. Typed content blocks (text, image, thinking, logprobs), system prompt as top-level parameter, and structured SSE streaming with distinct event types (message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop). Uses `StreamingEventTranslator` state machine for event sequencing.
- **Testing infrastructure**: Root `tests/conftest.py` with shared fixtures (mock_mlx, mock_mlx_provider, mock_vlm_provider, sample requests). Reusable MLX mocking utilities in `tests/helpers/mlx_mock.py` for testing provider code on any platform without MLX installed.
- **MLX provider unit tests**: 26 tests covering initialization, strategy compilation, image detection, model defaults (including thinking mode defaults), metrics, cache clearing, unload safety, and error paths.
- **Glass Box backend tests**: 16 tests validating `_reconstruct_thinking()` round-trip behavior and assistant prefill convention. Covers thinking tag formatting, non-assistant message handling, None/empty thinking, and dict mutation semantics.
- **Config unit tests**: Rewrote `test_config.py` from 6-line script to 25 proper pytest tests covering ChatMessage, ChatRequest, ModelConfig, and AppConfig.
- **Messages API unit tests**: 21 tests for request/response converters and StreamingEventTranslator event sequencing.

### Removed

- Deleted `config_migration.py` (dead code -- YAML-to-TOML migration completed, file imported nowhere).

### Fixed

- Fixed `mlx_provider.py` header comment (was referencing old `mlx_provider_optimized.py` filename).

## 1.8.0

### Added

- **Glass Box: Universal Editability and Transparency**: Every token the model generates is now visible, editable, and round-trips correctly through the API.
  - **Backend thinking round-trip**: `ChatMessage` accepts `thinking` field; MLX provider reconstructs `<think>` tags before template application across all generation paths (text, VLM, batch). Assistant prefill support: when last message is `role: assistant`, sets `add_generation_prompt=False` for mid-response continuation.
  - **Shared `lib/messages.ts`**: Extracted from chatStore, now includes thinking on assistant messages in API payloads. Used by all applets.
  - **Shared `lib/stale.ts`**: Timestamp-based stale detection -- marks downstream messages when upstream edits occur. No stored flags.
  - **Editable ThinkingBlock**: Thinking blocks are now default-open and editable (save/cancel inline).
  - **Shared MessageActions**: Copy, edit, delete, regenerate, continue, next-turn actions in one component. Compact mode for tight layouts.
  - **StaleBadge**: Amber indicator on messages generated before upstream edits.
  - **Chat: Continue from message**: Prefill/append to partial assistant responses.
  - **Chat: Generate next turn**: Fresh assistant response from full history without a user message.
  - **Notebook: Thinking display**: ThinkingBlock appears above editor during and after generation with thinking models.
  - **Model Comparison: Editable results**: ThinkingBlock defaults open and editable on completed results. Compact message actions. New `editResult` store action.
  - **Token Explorer: Thinking token visibility**: Tracks thinking token boundary. Visual separator between thinking and response tokens in the stream.
- **PlayIcon, ForwardIcon**: Added to icon library (22 icons total).

## 1.7.1

### Fixed

- **Notebook persistence hardening**: Migrated document storage from localStorage (5MB limit) to IndexedDB, matching the pattern used by chatStore. Added `loaded` state flag to eliminate the 100ms setTimeout race condition on auto-document creation. One-shot migration reads existing localStorage data into IDB and removes the legacy key. Delete operations go directly to IDB (no debounce). Individual document saves via debounced put instead of full-array serialization.

## 1.7.0

### Added

- **Notebook Mode applet** (`applets/notebook/`): Sixth applet in the platform. Base-model text continuation simulator with a single monospace text buffer and cursor-based generation. The model continues from wherever the cursor is positioned, treating the text as a completion context. System prompt is visible and editable (not hidden). Optional image attachments provide vision model context. Documents persist to IndexedDB with debounced saves. Keyboard shortcuts: Cmd+Enter (generate), Escape (stop), Cmd+N (new document), Cmd+S (force save). Lazy-loaded at `/notebook`.
- **DocumentTextIcon**: Added to icon library (20 icons total).

## 1.6.0

### Added

- **Performance Dashboard applet** (`applets/performance/`): Fifth applet in the platform. Real-time system metrics (RAM, CPU, context usage) with color-coded thresholds. When analytics is enabled (`HEYLOOK_ANALYTICS_ENABLED=true`), displays timing breakdowns by operation type (queue, model load, image processing, token generation), throughput sparklines with TPS/request/error trends, and per-model performance table with response time and TTFT. Graceful degradation shows system metrics with a friendly message when analytics is disabled. Auto-polls system metrics at 5s and profile data at 30s. Lazy-loaded at `/perf`.
- **ChartBarIcon**: Added to icon library (19 icons total).
- **Sparkline component**: Reusable SVG sparkline with gradient fill for inline data visualization.
- **MiniBarChart component**: Reusable horizontal bar chart for comparing values with labeled bars.

### Changed

- **Type adapter layer**: `EnhancedUsage`, `GenerationTiming`, `GenerationConfig` in `types/api.ts` now re-export from `generated-api.ts` with optional-field normalization, reducing manual type drift surface.

## 1.5.1

### Fixed

- **MLX Provider safe unload**: Added reference counting (`_active_generations` counter) to prevent LRU cache eviction from unloading a model during active generation. `unload()` now waits up to 30 seconds for active generations to complete, with force-unload as a safety valve. Fixes potential Metal command buffer crashes when >2 models are requested concurrently (e.g., model comparison with 3+ models).

### Changed

- **Shared utility library**: Extracted 9 duplicated functions across 4 applets into `src/lib/` -- `generateId()`, `tokenFromLogprob()`, `displayToken()`, `probabilityToColor()`, `probabilityToBarColor()`. `ExplorerToken` and `ComparisonToken` are now type aliases for shared `LogprobToken`.
- **Map to Record migration**: `ComparisonRun.results` changed from `Map<string, ModelResult[]>` to `Record<string, ModelResult[]>` for JSON serialization and devtools compatibility. Extracted `updateRunResult()` helper to simplify store mutations.
- **Shared UI primitives**: Extracted `StatusBadge` (10 status variants), `StreamingCursor` (inline/block), `AlternativeBar` (default/compact), and `RunHistoryList` (generic collapsible) from applet-specific duplications.
- **OpenAPI streaming schema**: Backend now exposes `StreamChunk`, `StreamChoice`, `StreamDelta`, `TokenLogprobInfo`, `EnhancedUsage`, `GenerationTiming`, `GenerationConfig` Pydantic models in the OpenAPI spec. Frontend `types/api.ts` restructured as adapter layer with compile-time drift detection against `generated-api.ts`. `npm run generate:api` chains `openapi-typescript` with `sed` to convert `| null` to `| undefined` for TypeScript idiom compatibility.

## 1.5.0

### Added

- **Model Comparison applet** (`applets/model-comparison/`): Fourth applet in the platform. Run the same prompt against 2-6 models simultaneously, streaming responses side-by-side with per-model performance metrics (TTFT, tokens/sec, duration, token count). Supports batch mode (multiple prompts separated by `---`, navigated via tabs). Optional token probability visualization with colored chips and alternative token bars. Models execute sequentially via the backend LRU cache but appear concurrent in the UI. Includes run history, per-model stop controls, and keyboard shortcut (Escape to stop all). Lazy-loaded at `/compare`.
- **ScaleIcon**: Added to icon library (18 icons total).
- **DuckDB persistence stub**: `ComparisonPersistence` interface defined with no-op `sessionPersistence` adapter, ready for real implementation when analytics API is built.

## 1.4.0

### Added

- **Token Explorer applet** (`applets/token-explorer/`): Third applet in the platform. Stream tokens with real-time probability visualization using a continuous red-yellow-green HSL color scale. Click any token to see its rank and top-K alternative tokens as horizontal probability bars. Includes run history, keyboard navigation (arrow keys to move between tokens, Escape to deselect), and auto-scroll during streaming. Lazy-loaded at 13.8 kB.
- **Streaming logprobs callback**: Added `onLogprobs` to `StreamCallbacks` in `api/streaming.ts`. Extracts `logprobs.content` from SSE chunks and forwards to callback. Backward-compatible -- existing callers unaffected.
- **SparklesIcon**: Added to icon library (17 icons total).

## 1.3.0

### Added

- **Batch Processing applet** (`applets/batch/`): Second applet in the platform. Create batch jobs with multiple prompts, process via `/v1/batch/chat/completions` endpoint, view results in dashboard with per-prompt expandable cards and JSON export. Lazy-loaded for zero impact on chat page performance.
- **Shared SamplerControls component** (`components/composed/SamplerControls.tsx`): Reusable sampler parameter sliders (temperature, top_p, top_k, min_p, max_tokens, penalties, seed) extracted from SettingsPanel. Used by both chat settings and batch create form.
- **Shared ModelSelector component** (`components/composed/ModelSelector.tsx`): Moved from `features/models/` to shared layer. Now uses `onModelLoaded` callback prop instead of direct chatStore dependency, enabling reuse across applets.
- **Batch-related icons**: LayersIcon, BoltIcon, DownloadIcon added to icon library (16 icons total).
- **API schema module** (`src/heylook_llm/schema/`): New Anthropic Messages-inspired API schema with typed content blocks (TextBlock, ImageBlock, ThinkingBlock, LogprobsBlock, HiddenStatesBlock), structured streaming events, and bidirectional converters to/from the existing OpenAI-compatible format. Purely additive -- existing endpoints unchanged.
- **Pre-commit safety hook**: Rejects staged files in `internal/`, `coderef/`, `.claude/`, `.archive/`, `modelzoo/*`, `models.toml`, `.env`, and files containing personal filesystem paths.
- **Frontend platform documentation** (`internal/frontend/`): Applet catalog, platform architecture, API schema design, design system, and migration plan.
- **Frontend applet platform architecture**: AppShell with AppNav sidebar rail, react-router-dom routing with `/chat` and `/batch` routes, shared primitives (Slider, Toggle, EmptyState, Modal) and composed components (ThinkingBlock, SamplerControls, ModelSelector), icon component library. Chat restructured as first applet module under `applets/chat/` with own store.
- **Type generation pipeline**: openapi-typescript devDependency with `generate:api` script for auto-generating TypeScript types from the FastAPI OpenAPI spec.

### Changed

- **Dependency update**: Removed `huggingface-hub<1.0` upper bound pin to allow `transformers>=5.0` (required by latest mlx-lm from git)
- **Frontend architecture**: Chat moved from `features/chat/` to `applets/chat/`, chatStore moved to applet-owned `applets/chat/stores/`. Sidebar, ConfirmDeleteModal, AdvancedPanel now live in `applets/chat/components/`. Layout changed from `h-screen` to `h-full` (AppShell owns viewport). SettingsPanel refactored to use extracted SamplerControls.

### Removed

- Empty placeholder directories: `features/advanced/`, `features/batch/`, `features/settings/`, `components/common/`, `components/ui/`, `features/models/`

## 1.2.0

### Changed

- **Dependency modernization**: Added `[tool.uv.sources]` for mlx-lm, mlx-vlm, and parakeet-mlx to pull latest from git instead of PyPI
- **STT migration**: Replaced CoreML STT provider (coremltools + manual RNNT decoder) with MLX STT provider (parakeet-mlx high-level API)
- **Setup scripts**: Rewrote `setup.sh` and `update-heylook.sh` for uv-only workflow, removed all llama-cpp-python references (deprecated, replaced by llama-server subprocess)

### Removed

- `coreml_stt_provider.py` (465 lines) -- replaced by `mlx_stt_provider.py` (265 lines, parakeet-mlx)
- `CoreMLSTTModelConfig` and `coreml_stt` provider type from config
- `llama-cpp` from `[all]` extra (llama-server subprocess is the supported path)
- pip fallback in setup/update scripts (uv required for `[tool.uv.sources]` git resolution)

### Migration

- Users with `provider = "coreml_stt"` in models.toml must change to `provider = "mlx_stt"` and update model path to a HuggingFace repo ID (e.g., `mlx-community/parakeet-tdt-0.6b-v3`)
- `setup.sh` and `update-heylook.sh` now require uv (no pip fallback)

## 1.1.1

### Added

- **MLX Performance Optimizations**: Following mlx-lm reference patterns for better Metal utilization
  - Compiled presence penalty processor with `@mx.compile` decorator for Metal kernel optimization
  - Compiled vision preprocessing (normalize + transpose) for faster image encoding
  - Pre-computed ImageNet constants as module-level `mx.array` objects
  - Use `mx.broadcast_to` instead of list multiplication for memory-efficient grid creation
  - Replaced blocking `mx.eval` with `mx.async_eval` for better pipeline pipelining
  - Added buffer cache cleanup method with configurable retention
  - Release temporaries before sync points to reduce memory pressure

- **MLX Performance Test Suite**: Comprehensive tests for MLX optimizations
  - `tests/unit/mlx_perf/`: Compilation correctness, type consistency, sync boundary tests
  - `tests/integration/mlx_perf/`: Throughput benchmarks, TTFT tests, memory profiling
  - New pytest markers: `mlx_perf`, `slow` for selective test execution
  - Configurable test model path via `HEYLOOK_TEST_MODEL` env var

- **Structured Hidden States Endpoint**: New `/v1/hidden_states/structured` for server-side chat template
  - Accepts chat components separately (user_prompt, system_prompt, thinking_content, assistant_content)
  - Server applies Qwen3 chat template internally with `enable_thinking` support
  - Returns token boundaries for each section (system, user, think, assistant)
  - Returns token counts per section and total
  - Optional `formatted_prompt` field for debugging
  - Enables ablation studies and token attribution research for Z-Image

- **Model Capabilities Discovery**: Expose model capabilities in `/v1/models` response
  - New `capabilities` field in model config (e.g., `["hidden_states", "chat", "thinking", "vision"]`)
  - New `supports_thinking` and `thinking_token_ids` fields in MLXModelConfig
  - `/v1/models` now includes `provider` and `capabilities` when configured
  - Enables programmatic capability discovery for multi-model clients

- **Auto Model Selection**: Fallback to loaded model when no model specified in request
  - Uses most recently used model from LRU cache
  - Falls back to `default_model` from config if no models loaded
  - Provides clear error message with available models if no default

- **Token-Level Thinking Parser**: More efficient parsing of Qwen3 thinking blocks
  - New `TokenLevelThinkingParser` uses token IDs (151667/151668) for precise detection
  - New `HybridThinkingParser` auto-selects between token-level and text-based parsing
  - Eliminates regex buffering overhead when token IDs are available
  - Instant detection of `<think>`/`</think>` boundaries via special token IDs
  - Backwards compatible: falls back to text parsing for models without token IDs
  - Integrated into streaming response generator for automatic use

- **Hidden States Config Defaults**: Model-level defaults for hidden states extraction
  - New `default_hidden_layer` config option in MLXModelConfig (default: -2)
  - New `default_max_length` config option in MLXModelConfig (default: 512)
  - Hidden states endpoint now applies model config defaults when request uses defaults

- **Enhanced Streaming Metadata**: Detailed generation stats in final streaming chunk
  - New `stream_options: {include_usage: true}` parameter in ChatRequest
  - `usage` object with `prompt_tokens`, `completion_tokens`, `thinking_tokens`, `content_tokens`, `total_tokens`
  - `timing` object with `thinking_duration_ms`, `content_duration_ms`, `total_duration_ms`
  - `generation_config` object with sampler settings used (temperature, top_p, top_k, min_p, max_tokens, enable_thinking)
  - `stop_reason` field indicating why generation stopped: `eos_token`, `max_tokens`, `stop_sequence`, or `length`
  - Properly maps MLX `finish_reason` values to OpenAI-compatible stop reasons
  - Enables frontend display of thinking tokens, timing breakdown, and stop reason

- **OpenAI-Compatible Logprobs Support**: Return token log probabilities in chat completions
  - New `logprobs: bool` parameter to enable log probability output
  - New `top_logprobs: int` parameter to specify number of top alternatives (0-20)
  - Non-streaming responses include `choice.logprobs.content` array with per-token data
  - Streaming responses include `choice.logprobs` delta in each chunk
  - Each token entry includes: `token`, `token_id`, `logprob`, `bytes`, `top_logprobs[]`
  - Leverages mlx-lm's full vocabulary log-softmax for accurate probabilities
  - New `logprobs.py` module with `LogprobsCollector` and `StreamingLogprobsCollector`

### Fixed

- **TextOnlyStrategy model_config**: Fixed AttributeError when `enable_thinking` not in request
  - Added `model_config` parameter to `TextOnlyStrategy.__init__`
  - Changed `getattr()` to `.get()` for proper dict access
  - Fixes Qwen3 thinking mode fallback to model config defaults

- **Qwen3-VL-MOE Vision Support**: Fixed "Image features and image tokens do not match" error
  - Use `mlx_vlm.prompt_utils.apply_chat_template` for proper image token insertion
  - Removed manual image placeholder insertion that conflicted with library handling
  - Now properly passes `num_images` to prompt formatting

### Previously Added

- **Qwen3 Thinking Token Support**: Parse `<think>...</think>` blocks from Qwen3 model outputs
  - Non-streaming responses include `message.thinking` field with parsed reasoning content
  - Streaming responses emit `delta.thinking` during thinking, `delta.content` for response
  - Multiple thinking blocks are concatenated with `---` separators
  - New `thinking_parser.py` module for robust parsing
- **Model Configuration for Thinking Mode**: Add `enable_thinking` parameter to MLXModelConfig
  - When enabled, automatically applies Qwen3 optimal sampler defaults:
    - temperature=0.6 (greedy decoding causes repetition)
    - top_p=0.95
    - top_k=20
    - presence_penalty=1.5
  - Pass `enable_thinking` to chat template for Qwen3 tokenizers
- **Presence Penalty Support**: Add `presence_penalty` parameter to ChatRequest and MLXModelConfig
  - Discourages reuse of tokens that have already appeared
  - Recommended value 1.5 for Qwen3 thinking mode
  - Custom logits processor implementation for mlx-lm compatibility

### Changed

- Updated `models.toml.example` with thinking model configuration documentation
- Extended sampler builder to support presence_penalty processor
