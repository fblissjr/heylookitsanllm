# Sharp edges: the why behind the agent rules

This file holds the rationale and incident history behind the rules in the root
[AGENTS.md](../../AGENTS.md) and the per-area rule files it indexes in
`.claude/rules/`. Those state each rule in a sentence or two and link here;
this file says why the rule exists, which release introduced it, and what broke before it did. It is dated history, not a living status
document: facts here were true on the date or version they name, and the code
is the authority when they disagree. Current status and plans live in
[docs/project/](../project/); how the system works end to end is the
[wiki](../wiki/README.md). When you add a rule to one of them that has a story
behind it, put the story here under the matching subsystem heading and link
to it, so the headings stay stable as anchors.

## Contents

- gguf and llama-server: [template ladder](#gguf-chat-template-ladder),
  [context](#gguf-context-allocation), [micro-batch and memory](#gguf-micro-batch-and-memory),
  [non-causal images](#gguf-non-causal-images),
  [Metal residency](#gguf-metal-residency-keep-alive),
  [one engine contract](#one-engine-contract), [speculative decoding](#gguf-speculative-decoding), [binary and build](#llama-server-binary-and-build)
- Thinking and sampling: [on the wire](#thinking-on-the-wire),
  [default and sampler report](#thinking-default-and-the-sampler-report),
  [depth](#thinking-depth), [vendor layer](#vendor-sampling-layer),
  [sampler registry removal](#named-sampler-registry-removal)
- Model registry and config: [discovery](#model-discovery),
  [explicit entries](#explicit-entries-get-no-derived-fields),
  [effect classes](#config-effect-classes), [toml comments](#toml-comments)
- API and store: [routers](#app-assembly-and-routers), [one wire](#one-inference-wire),
  [Messages conformance](#messages-wire-conformance), [cancellation](#cancellation),
  [DuckDB store](#duckdb-store), [observability](#observability)
- Frontend: [static serving](#static-serving-no-fallback-no-catch-all),
  [generation lifecycle](#generation-lifecycle-and-stream-identity),
  [mirror and resume](#store-mirror-and-resume), [attachments](#attachments-and-paste),
  [URL schemes](#markdown-url-schemes), [streaming render](#incremental-streaming-render),
  [presets](#presets-and-the-system-prompt),
  [model switching](#model-switching-and-the-config-editor)
- MLX engine: [media placement](#media-placement), [vision prefill](#vision-prefill),
  [penalty scope](#penalty-scope), [preview and media](#prompt-preview-and-media),
  [loader routing](#loader-routing), [prompt cache](#prompt-cache),
  [logits shape](#logits-processor-shape), [tokenizers](#tokenizers-and-stop-tokens),
  [reasoning parsers](#reasoning-parsers),
  [template ladders](#template-ladders), [latency checks](#streaming-latency-checks)
- Repo conventions: [hand-copied lists](#hand-copied-constant-lists),
  [enforced rules](#enforced-rules-and-the-claude-allowlist), [docs layout](#docs-layout),
  [engine pins](#engine-pins-and-dependencies)
- Tests: [no fail-first rule](#no-make-it-fail-first-rule),
  [live smoke](#live-smoke-and-the-ram-pre-flight), [browser E2E](#browser-e2e),
  [assertions on strings](#assertions-aimed-at-strings-the-code-never-emits),
  [MLX mocks](#mlx-mocks-and-teardown-crashes),
  [git archive](#git-archive-is-not-isolation)

## Providers

### Provider contract

`mlx_embedding` and the `/v1/embeddings` + `/v1/hidden_states` routes were
removed in v2.0.40-41 (owner call: nothing used them). The gguf provider is
pure stdlib with no MLX import: one llama-server subprocess per loaded model,
so "loaded" means "running process" and LRU/idle-unload is spawn/SIGTERM.
Provider output is the owned, slotted `GenerationChunk` (`providers/base.py`);
new telemetry is a field there absorbed in `perf_collector.ChunkTelemetry`
rather than an attribute patch, because attribute patches were how per-chunk
scraping used to drift from what the collector read.

## gguf and llama-server

### gguf chat template ladder

llama-server runs `--jinja` with reasoning pre-split by default, so the
provider's `template_info()` returns None and routes heylook's parsers to
pass-through: another engine's split output is never re-parsed.

The template is resolved as a four-rung ladder at spawn (v1.79.43, plus the
operator override in v2.0.22), and every spawn logs which rung won: an
explicit `chat_template_path` (v1.68.0, `--chat-template-file`) beats the
operator override, which beats a `chat_template.jinja` discovered beside the
.gguf, which beats the one embedded in the GGUF. Sidecar-beats-embedded is the
default because the embedded template is whatever the quantizer baked in,
while a sidecar is the file you can read, diff and edit.
`use_sidecar_chat_template = false` keeps the embedded one without deleting a
file out of a downloaded snapshot dir.

A consequence worth holding: a template can change with no models.toml change
at all. Dropping a file next to the weights is enough, which is why the spawn
log exists, and why any measurement that varies by prompt format must
establish which template each arm ran against before concluding anything
about the model.

`chat_template_source` (MLX-only, `template_info.py`) does not reach this
provider and is deliberately a different name for a different mechanism. All
of the gguf template settings are `requires_reload` because llama-server takes
the template at spawn, which is also why no per-request or per-preset form
exists. The per-request levers are the wire fields `thinking` and
`reasoning_effort`, which the provider forwards to llama-server as
`chat_template_kwargs`. That name itself was never a `/v1/messages` field:
pydantic dropped it silently, a heylook harness sent it for weeks without
ever turning thinking off, and since v2.0.86 the route refuses it (422).

Publishers differ on the same weights, and it is not cosmetic (measured live
on Qwen3.8-27B, both templates, 2026-08-17): ggml-org embeds Qwen's official
template byte-identically (8952 bytes), unsloth a patched one (9993) adding a
`developer` role and merging up to two leading system messages. Two leading
system messages render under unsloth and are a 500 under official, because a
raised jinja exception is a 500 from llama-server. Both still reject a system
message appearing mid-conversation, so "unsloth's is permissive" is false in
general: try the shape, do not assume it.

### gguf context allocation

(v1.79.61) `ctx_size` absent means llama-server's `-c 0`, i.e. the model's
training context, then `--fit` shrinking unset args to device memory.
DeepSeek-V4-Flash got a 1,048,576-token slot that way. The engine contract
carries `engine.context.length` (GGUF header, the ceiling) and
`engine.context.running` (`/props` at ready, what the process got), and `POST /v1/admin/models/{id}/reload?ctx_size=N`
persists through the one config writer and then loads (0 = Auto = drop the
key; unchanged + resident = plain load, no restart). This is gguf-only: MLX
has no fixed context allocation.

### gguf micro-batch and memory

Always send `max_tokens`: llama-server's default is unlimited. `-np 1` is our
choice.

Micro-batch is automatic (v2.0.13, `n_ubatch` None): the provider sizes the
model the way the fit panel does (`ram_fit`, weights + sidecars vs the live
Metal working set) and spawns `-ub 2048` only when the headroom clears
`ram_fit.THIN_HEADROOM_GB`, else inherits llama-server's 512, logging which.
2048 is a measured prefill win on dense and MoE at no generation cost, but it
is several GiB more compute buffer on DeepSeek V4, and the Vision Q4 at thin
headroom loaded at 2048 (with `--fit` quietly trimming its context) and then
died in its first decode with a Metal OOM that llama's pre-flight (which skips
the drafter) never saw. A stored value wins both ways. Raising
`iogpu.wired_limit_mb` (`scripts/gpu_wired_limit.sh`, root-only, the one lever
that enlarges the working set) flips the big models to 2048 by itself.

A llama-server error mid-stream is a `data: {"error":...}` frame after a 200.
The adapter raises on it (v2.0.13); before that it was skipped for lacking
`choices` and the run ended as a clean zero-token `end_turn`. "Compute error."
on Metal is almost always the working set running out, so the raised message
carries the model's headroom and the sysctl.

The KV cache is f16 by default and stays so; `cache_type_k/v` exist for
headroom emergencies, not as a default.

### gguf non-causal images

Some projectors' image tokens are decoded with non-causal attention
(`mtmd_decode_use_non_causal` in llama.cpp's `tools/mtmd/mtmd.cpp`), and
`llama_context::decode` asserts that a non-causal batch fits one micro-batch.
The image is split by `n_batch`, not `n_ubatch`, so an image with more tokens
than the micro-batch aborts the process (found in the 2026-09-23 audit, §7).
The provider reads `clip.projector_type` from the mmproj at spawn and, for a
listed projector whose default per-image maximum exceeds the effective
micro-batch, passes `--image-max-tokens` equal to it (v2.0.70). It only ever
lowers: llama.cpp takes a custom maximum above its default as given, so the
flag is not passed when the default already fits.

The set and the maxima are a hand-copied table
(`LlamaServerProvider.NON_CAUSAL_IMAGE_PROJECTORS`), against the repo's derive
rule, because they live in C++ with no API. Owner call 2026-09-23: a table plus
a test, not a build-time parser in `build_llama.py`, because a parser that
fails loudly on an upstream reshape could block updating llama-server for a
guard no served model needs, and a non-fatal one ends up where the table does
with more code. `test_non_causal_table_matches_the_build` reads the source of
the build the provider spawns, located through its manifest (never
`coderef/llama.cpp`, which can sit at a different commit), and skips with the
reason when there is no build or the tree has moved off the built commit.

Capping every vision model instead was rejected: qwen3vl allows several times
the micro-batch, so it would cut Qwen image resolution for a crash Qwen cannot
have. gemma4v is listed whole although its E2B/E4B text widths decode
causally; that caps those two only at a small micro-batch. gemma3 is a fixed
token count the flag cannot lower, so it gets a spawn warning, not a cap.

### gguf Metal residency keep-alive

ggml-metal keeps the weights' residency sets requested for
`GGML_METAL_RESIDENCY_KEEP_ALIVE_S` after the last GPU work (default 180 s),
then lets the pages go cold. This is llama.cpp's heartbeat, not heylook's
idle unload. On the 145 GB DeepSeek-V4-Vision the first request after a gap
past it paid a first-token delay many times the warm one; raised, the delay
was gone (audit 2026-09-23, §8). Since v2.0.70 the provider sets it at spawn
(`METAL_RESIDENCY_KEEP_ALIVE_S`), and an inherited value wins with a warning.

The plan asked for a value derived from heylook's idle-unload threshold. That
goes stale: the env is fixed at spawn, while `idle_unload_seconds` applies
live and pinning changes the answer. heylook's unload ends the process, so
"resident for the life of the process" is exactly "as long as heylook keeps
the model loaded", and cannot go stale. ggml-metal has no forever value: it
counts the keep-alive down in 5 ms ticks in an `atomic_int`, so a value past
the int range wraps negative and turns residency off, and 0 or below means
its default. The constant (`METAL_RESIDENCY_KEEP_ALIVE_S`, reasoned) sits far
inside the wrap.

The cost, measured on DeepSeek before shipping (owner condition): the thread
wakes on every tick (`time_per_loop_ms` in ggml-metal-device.m) in both cases,
and only the `requestResidency` calls continue past llama.cpp's default
keep-alive. The long keep-alive raised idle CPU, and the rise was judged
negligible. Data and conditions: `internal/claude/perf/gguf_runtime_2026-09-23.json`,
`idle_cpu`. If it ever matters, a bounded value of a few hours keeps most of
the benefit.

### One engine contract

(v2.0.73, plan W13.) Engine facts used to be answered with per-engine
branches in `capabilities.py`, the admin routes and the frontend, and that is
where drift bugs came from: the gguf vendor sampling layer landed in the
provider while the capability report still said MLX-only. Now every engine
answers the same questions through `providers/contract.py`: a static half per
engine (`mlx_describe`, `gguf_describe`; functions over the config, so it
answers for unloaded models) and an observed half (`describe_observed()` on
the provider, read back from fields `load_model` recorded, never a call into
the process). `/v1/models` and the admin row carry one `engine` object, and
the frontend reads it through `js/engine.js`.

Every decision the static half reports is made by the SAME function the spawn
or the request path uses: `resolve_chat_template`, `binary_choice`,
`image_token_cap_decision`, `keep_alive_choice`, and the prompt-cache gates
(`cache_defaults.static_reuse_gate`, then `prompt_cache.reuse_verdict`). A
second derivation for the report is the drift this exists to end.

"Configured" is not "the entry has this key". Admin edits materialize an
entry with the whole derived config, so most models.toml entries are frozen
copies. A stored value counts as configured only when it differs from what
discovery derives for the same file (the router records both at merge time:
`written_ids`, `derived_configs`), else the schema default. An equal stored
value reads as derived, "stored ... same as derived", which is exactly what
the registry-sidecars prune removes. Checked on the live models.toml when
built: only the real overrides lit up.

The static half is cached by a stat-only stamp over every file that can
change the answer (template ladder files including the override and sidecar,
the mmproj, the drafter, the build manifest), the relevant environment, the
installed mlx-lm/mlx-vlm commits (from their install records, not uv.lock)
and a digest of the resolved config. The app lifespan warms it (and the
template probes) in a background thread after each config load
(`ModelRouter.warm_model_facts`), so the first listing after a start or
reload does not pay the cold parse. Only the lifespan turns this on: unit
tests build routers over a mocked MLX tree, where a background thread is the
teardown-crash class.

### gguf speculative decoding

**2026-09-24, owner decision: spec decode is on by default whenever a model
ships a drafter** (a drafter file or an MTP head built into the GGUF). This
replaces the per-model opt-in policy below, which is kept as history. The
case was the owner's examples: DeepSeek V4 Flash Vision Q8 and Qwen3.8-27B
both ship spec decode and neither ran it. Reading their files showed why, and
each reason is a gap in discovery, not a choice. Qwen3.8-27B's head is built
in (`blk.64.nextn.*` in both local GGUFs) and discovery only looks for
drafter files. The DeepSeek Vision Q8 folder has no drafter of its own, while
a drafter for the same model (`dspark-…-MXFP4.gguf`, arch `dflash`, the same
`general.name`) sits in the ggml-org quant's neighbouring folder, where
discovery never looks. And every such model had an explicit models.toml
entry, which receives no derived fields. The evidence against defaulting on
was thin all along: the paragraph on 2026-08-10 below found a wash, not a
loss. Header-based detection is planned with the registry sidecars. Neither
the built-in-head launch nor the cross-quant drafter had been run when this
was written.

Before 2026-09-24: spec decode (`spec_type = "draft-mtp"`) was per-model
opt-in and was to stay off unless you had checked it was a win on your model
at your context.

`spec_type` is not the switch, and an unset `spec_type` does not mean spec
decode is off (found 2026-09-19; the importer's comment asserted the opposite
for a year). `draft_model_path` is the switch: the provider emits `-md` on
that field alone, and llama.cpp infers the type from the drafter's own header
whenever `--spec-type` is absent (`common_speculative_types_from_gguf`: arch
`dflash` + a `markov_w1.weight` tensor = draft-dspark, a trailing
`blk.N.nextn.eh_proj.weight` = draft-mtp). `spec_type` only pins the type, and
is strictly required only for a sharded drafter, where the header read sees
the first split alone. Sharding of the target is irrelevant, which is why
DeepSeek-V4-Flash (5-shard target, single-file dspark sidecar) infers fine.

The trap is discovery: it pairs `draft_model_path` automatically and leaves
`spec_type` unset on purpose, so a model with an `mtp-`/`dspark-`/`dflash-`/
`eagle3-` sidecar beside its weights runs spec decode at llama.cpp's own
`spec_draft_*` defaults with no models.toml entry anywhere and nothing on this
side announcing it. Which models those are is derived, not listed: walk
`merge_discovered(data, discover(data))` for a config carrying
`draft_model_path`. An embedded MTP head (Qwen3.6) pairs no sidecar, so it
gets no `-md` and no spec decode at all; the packaging difference decides it.
"Default off" describes what we write, never what a paired drafter does; to
actually keep it off, the drafter must not be paired.

**No performance numbers in tracked docs.** 2026-08-10 produced a string of
figures that were each confidently wrong in turn, and the ones that survived
were spot observations, not performance testing: nothing controlled for quant
version, which llama.cpp produced the quant, build flags, thermal state,
memory pressure, or a second machine. What was left after a day of chasing
it: on the one case examined most carefully (a gemma-4 MTP model, vendor
sampling, realistic context, matched warm cache, long generation), spec on/off
is a wash, indistinguishable from noise. Every larger effect seen that day
dissolved when one more variable was controlled: a big tuning win was a
greedy artifact, a clear cost was a short-generation artifact, a dramatic
context effect was a cache-ordering mistake, and a "broken drafter" was
refuted by the drafter's own output. Default off for a new model was because it
was unproven here, not because it was known harmful (superseded 2026-09-24,
above).

Carve-out (moot since 2026-09-24, when on became the default for every model
that ships a drafter): the owner had decided the DeepSeek V4 Flash entry
keeps spec decode on, on community evidence and their own judgement rather
than anything measured here. "Default off" means do not enable it elsewhere
without checking; it does not mean disable what is already running. Nothing
above transfers to that entry anyway: it is a different spec type on a
bandwidth-bound MoE, which is the regime most likely to win and the opposite
of the cheap dense target these observations came from. Local detail,
conditions and the history: `internal/research/`.

Checking it yourself (a day of wrong answers produced these, and they are the
durable part):

- Never at temp 0. Greedy acceptance is exact argmax matching while temp > 0
  is rejection sampling, a different regime, not a quieter one. Temp 0 is fine
  for reproducibility, never for a throughput claim; the probe warns.
- Match prompt length, generation length, seed, sampling, which binary, and
  prompt-cache state across arms. An unmatched cache produced a large phantom
  that survived repeats and looked exactly like a finding.
- Tune `spec_draft_n_max` and `spec_draft_p_min` together; they interact and a
  1D sweep finds a different, wrong optimum.
- Short prompts and short generations mislead about ranking, not just
  magnitude.
- Matching every control you thought of does not make a result sound; it only
  rules out the confounds you imagined. The runs that produced the
  since-dissolved cost matched prompt, cache, seed and sampling, and were
  still measuring the wrong thing because nobody had varied generation length.

Drafter packaging (read the GGUF, do not trust vendor docs; unsloth.ai
currently states the opposite): gemma-4 12B ships a sidecar
`mtp-gemma-4-12B-it.gguf` with zero nextn tensors in the main file;
Qwen3.6-27B has `blk.64.nextn.*` embedded and no sidecar. gemma drafters are
sidecar `mtp-*.gguf`, auto-paired by the importer into `draft_model_path`;
llama's own `-hf` sibling discovery does not work for local files.

A LoRA further erodes whatever win exists: `common_set_adapter_lora` has one
call site in tools/server and applies to `ctx_tgt` only, so the drafter
proposes the base distribution while the target generates the adapted one.
That is structural, no adapter escapes it, magnitude unknown and n=1.
`--spec-type` is a spawn flag while `lora` is per-request, so one process
cannot suit both kinds of traffic, but that tradeoff only matters if spec
decode is a win at all, which is not in evidence.

### llama-server binary and build

The llama-server binary is the canonical local build (a fixed dir under the
user's home, written only by `scripts/build_llama.py`): one build, one source,
owner rule 2026-08-13; update = re-run the build script. `server_binary` /
`$HEYLOOK_LLAMA_SERVER` remain as escape hatches for experiments but warn at
every spawn, naming the canonical build they shadow. The silent-rot mode (an
exported var pointing at a stale working-tree binary while a fresh canonical
build sat unused) shadowed exactly that on 2026-08-13 and is retired: the
owner's shell export is gone, and any re-introduction announces itself. Only
if no binary exists anywhere does load fail loudly.

`heylookllm import` (with its merge-preserving writer and `--fresh`, and the
admin `/scan` + `/import` routes behind the models page's one-off scan) was
retired in v2.0.72 (owner, 2026-09-23). Every model lives in a `[scan].folders`
watch folder and is served with no entry, so writing entries had no remaining
use, and each written entry froze the derived config. The scanner it shared
stays: discovery calls it. Its writer-schema guard was not re-homed, because
the one writer left, `update_config`, validates the whole entry through
`ModelConfig` before it writes.

`scripts/build_llama.py` is the only thing that clones or builds llama.cpp;
`uv sync` cannot, since it is C++, not a uv package. It builds the newest
`b<N>` release tag by default (llama.cpp's releases are those tags; `--rev`
for anything else) and never touches pyproject/uv.lock. It builds three
targets, the other two being instruments the server never calls, from the
same commit as the binary they explain: `llama-bench` and, from v2.0.19,
`llama-fit-params`. The latter is llama.cpp's own memory projector, which
prints each device's model/context/compute split from metadata, loading no
weights. It is not a llama-server flag (upstream gates `--fit-print` to that
tool's own example), which is why it looks absent from `--help`; output is one
machine-readable line per device, and it refuses `--mmproj`, so a projector is
costed separately.

Why it exists here: `ram_fit` sizes a gguf model by file bytes and reads no
placement field at all (not `n_cpu_moe`/`cpu_moe`/`override_tensor`, not even
`n_gpu_layers`), so its Metal-working-set line assumes every byte lands on the
GPU. That holds for most models and is false for an architecture that keeps
large tables host-side (measured on qwen4exp/Qwen3.8-Flash-Next: about a third
of the file stays on the host as per-layer embedding tables, and fit-params
reproduced a full load's numbers exactly). The blast radius is bounded and
worth knowing before chasing it: the reclaimable-RAM line is unaffected (host
bytes are still RAM), and over-working-set is a `warn` for gguf, never a
`fail`, so nothing is wrongly refused. What you get is an overstated GPU need,
a false thin-headroom warning, and `_auto_ubatch` picking the narrow
micro-batch when the wide one would have been safe.

Owner decision 2026-09-08: the tool ships, and `ram_fit` is not rewired to
it. Priced and declined: one model of twelve is affected, the workaround is
one `n_ubatch` line, and adopting it would restate `THIN_HEADROOM_GB` in units
its two calibrating spawns were never measured in. Revisit if expert offload
starts being used (then the panel is wrong by construction, not by
architecture) or the false warning grates; conditions in `internal/research/`.

llama.cpp is not vendored and not a submodule: the clone and build tree live
outside the repo (a fixed dir under the user's home; `dir` /
`$HEYLOOK_LLAMA_CPP_DIR` relocate), so upstream source can never be committed
or packaged and there is nothing to `submodule init`. Build flags and their
rationale (why no LTO, no OpenMP, and why `GGML_METAL_NDEBUG` stays off) are
in `scripts/README.md`.

## Thinking and sampling

### Thinking on the wire

(v1.79.62) `ChatMessage.thinking` reaches llama-server as `reasoning_content`
(`_wire_message`). The unrenamed key was silently ignored, so every replayed
assistant turn rendered an empty think block.

llama.cpp's continuation is its own: a trailing assistant message with
`reasoning_content` and empty content resumes inside the open block
(`COMMON_CHAT_CONTINUATION_REASONING`); with content it closes the block and
continues the content. The prefill echo is on both channels, and reasoning
comes back minus its leading whitespace (measured on b10814), so
`_continuation_echo_chars` returns a pair and the reasoning strip is sized
lstripped (it can only under-strip).

MLX resumes a thought by rendering a fresh generation prompt with thinking on
and appending the trace after the family's opener (`_append_thinking_resume`:
`<think>\n`, `<|channel>thought\n`, `<|channel|>analysis<|message|>`); all
three routing parsers take `resumes_thinking` / `initial_thinking` (v1.79.63).

History thinking on MLX goes to the template the way that template takes it
(`vlm_inputs.thinking_for_template`, keyed on
`ModelTemplateInfo.reads_reasoning_content`): `reasoning_content` where the
template reads it (gemma-4's does, and renders it only for tool-call turns
while stripping channel markers from content; Qwen3.8's keeps every turn's by
default via `preserve_thinking`), reconstructed `<think>` tags for a marker
template that does not, and nothing for a family with neither. Baking
`<think>` text into a gemma prompt was the v1.79.62 regression.

Preview is `provider.render_prompt()` (gguf `/apply-template`, MLX the same
`build_prompt` generation uses) behind `POST /v1/conversations/{id}/prompt`,
which renders resident models only: a preview must never load one.

A content continuation on MLX used `continue_final_message` alone, which
re-renders the partial reply as a HISTORY turn. gemma-4 with thinking off
opens and closes an empty thought channel in its generation prompt that its
history render omits, so Continue asked the model to extend a turn shaped
unlike the one it wrote, and the continuation degraded into repetition. A
fixed-seed A/B in mlx-vlm alone settled the cause (record in
`internal/claude/w2/`). Since v2.0.85 `vlm_inputs.continue_from_generation_prompt`
continues from the generation prompt plus the reply's text whenever that
prompt extends what the continuation render put before the text, on the text
and vision paths alike; the Qwen families' history render already matches
and is unchanged. The same question on gguf (llama-server's own prefill
render) is open, filed in `TODO.md`.

### Thinking default and the sampler report

(v1.79.62) The cascade resolves request > models.toml `enable_thinking` > the
thinking capability, passed in as `thinking_capable=` by every caller
(providers pass `self.thinking_capable`, admin passes `"thinking" in caps`).
From v1.50.0 unset meant off everywhere, a choice made when the UI could only
send true-or-absent. `MLXModelConfig.enable_thinking` is Optional (None =
follow capability).

`samplers.thinking_default()` is the admin row's `thinking_default` and must
stay the cascade's own answer, never a re-derivation.
`samplers.sampler_defaults()` (v2.0.21) is its sibling under the same rule and
reports every key on the admin row and `/v1/models`, which v3 prints as the
placeholder in a blank sampler field so "auto" stops hiding the number in
force.

It has been one flat bag since v2.0.33, and its `enable_thinking` equals
`thinking_default` by construction: one cascade call feeds both, so a second
code path cannot drift from it. It used to be `{"off": ..., "on": ...}`, and
that nesting was earned while the anti-loop overlay moved `presence_penalty`
off the thinking switch: the panel's thinking control is live, so reporting
one state while the user had selected the other put a wrong number on screen.
v2.0.32 removed the overlay, the halves became identical in every key but
`enable_thinking`, and `settings.js`'s `thinkingOn()` resolver (cap gate and
all) was picking between two identical objects. Both went in v2.0.33, which
also retired one of the two copies of the thinking resolver (`chat.js
effectiveThinking` is the survivor).

The report takes each engine's vendor layer exactly where that engine's
provider takes it. Since v2.0.74 each engine's describer says where
(`mlx_describe.vendor_sampling`: generation_config.json via
`load_vendor_sampling`; `gguf_describe.vendor_sampling`: the header's
`general.sampling.*`), and `capabilities._vendor_sampling_pairs` stays the one
cached entry point that dispatches to them, per row. temperature/top_p/top_k are the vendor
keys, so an engine present in its provider and absent there reports the
global floor while generation uses the vendor values; gemma and Qwen3.6 must
each report their own header top-k, not the floor's. That drifted within one
commit (gguf gained its layer in v2.0.22, the gate still said mlx), which is
why the pairing now has a test rather than a comment
(`test_vendor_layer_reaches_the_report_on_every_engine`). Header floats are
rounded at the reader: float32 widening gives a publisher's short decimal a
long expansion tail, harmless to the sampler and not harmless as the
placeholder of a `step=0.01` field.

The panel marks an overridden key (accent label + border) and shows a
per-field reset, where "overridden" is `key in samplerParams(caps)` and not
`cache[key] != null`. Those disagree, because `samplerParams` drops a
`top_k`/`presence_penalty` of 0 and every capability-gated key, so the naive
test made the panel claim a value the model never receives. The reset hides by
visibility, keeping its box so a row cannot jog sideways as you edit it.

The `hidden` attribute is the trap that class of control used to fall into:
an author `display` beats the UA's `[hidden]{display:none}`, which un-hid
every reset button and kept `.chat__ctx` on screen for models its own code
believed it hid. A DOM check reading `el.hidden` sees neither; a screenshot
sees both. `app.css` now carries one global `[hidden]{display:none!important}`
so the next `display` cannot re-open it.

### Thinking depth

`reasoning_effort` (v1.71.0) is a chat-template variable, not a sampler knob:
it rides `chat_template_kwargs` beside `enable_thinking` (gguf) /
apply_chat_template kwargs (MLX). It is sent whenever set, never gated on
`enable_thinking`: gpt-oss/harmony reads it unconditionally and has no
`enable_thinking` at all, so a gate made it unreachable for the one family the
docs name as taking low|medium|high. Its capability is separate from
`thinking` for the same reason (Qwen3.5 reads one, gpt-oss the other): MLX
probes the template file precisely, gguf rides `supports_thinking` because the
template is inside GGUF metadata.

The accepted set is per model (Qwen3.8: xhigh|medium|low, and it raises
otherwise; harmony: low|medium|high), so the schema Literal is their union and
a wrong-for-this-model value reaches the template, where llama-server turns a
raised jinja exception into a 500. Absent = send nothing, leaving the template
default (xhigh on Qwen3.8, which is why the field exists).

In MLX it must not ride `base_kwargs`: the TypeError retry re-passes those
verbatim and strips only the explicitly-named kwargs, so a narrow
TokenizerWrapper has to be able to lose it the same way it loses
`enable_thinking`.

### Vendor sampling layer

The vendor layer is the per-model answer and should normally win (v2.0.23,
promoted in v2.0.30 when the named-sampler layers were deleted). The floor
beneath it is deliberately small: two fallback values (`FALLBACK_TEMPERATURE`
1.0, `FALLBACK_TOP_P` 0.95) that apply only where the model's own metadata is
silent, one safety stop (`DEFAULT_MAX_TOKENS`, because llama-server's
`n_predict` default is unlimited), and four `KNOBS_OFF` identity values. Those
last are load-bearing for a reason easy to miss: the engines' defaults are not
neutral (llama.cpp ships `top_k = 40`), so dropping them would hand each engine
its own taste back and let the two diverge on identical input. The vendor
layer sits directly above `GLOBAL_SAMPLER_FLOOR`, so models.toml fields and
request fields all still win.

Each engine reads the same values from where its models keep them: MLX from
the model dir's `generation_config.json` (`samplers.load_vendor_sampling`),
gguf from the GGUF header's `general.sampling.*`
(`gguf_metadata.vendor_sampling`), which converters write from that same
generation_config.json. gguf went without it until v2.0.23 on the stated
reasoning that a gguf dir ships no generation_config.json. That was true, and
the wrong conclusion: the values had moved into the header, and because
heylook sends every sampler key explicitly on every request, llama.cpp's own
read of that block (`common_init_sampler_from_model`, which fills any key not
set on the CLI) was overridden every time. The server was sending `top_k 0` at
models whose own files ask for 20 (Qwen3.6) and 64 (gemma-4).

Only the three keys the layer takes are read. The spec also defines min_p,
xtc, penalties, mirostat and a sampler `sequence`, but real files carry none
of them because generation_config.json has no such fields. So a publisher's
documented min_p or repeat penalty reaches nothing automatically, and neither
does a thinking-vs-instruct split, since a GGUF holds one set of values and
publishers document two.

### Named-sampler registry removal

The bundled sampler registry is gone (v2.0.30): `data/samplers/*.toml`,
`SamplerRegistry`, `ChatRequest.sampler`, models.toml `default_sampler`,
`/v1/admin/models/samplers`, `/v1/capabilities.samplers`, the
`bulk-default-sampler` route, `request_guards.py` and the `--sampler` /
`--preset` / `--profile` CLI arguments. It shipped generic guesses applied to
every model, which is the opposite of what the vendor layer does. Most of its
entries had no consumer at all, `thinking` was provably a no-op because the
cascade hardcoded the same constant as a fallback, and `balanced` (the import
default) carried `temperature = 0.7`, the value the owner had overturned when
raising the floor to 1.0. The frontend never touched any of it: no JS ever
sent `sampler`, no JS read either roster endpoint, and the e2e suite asserts
the generate wire stays sampler-free. A request still sending `sampler` or
`preset` gets a 422 naming the removal (the guard is on
`MessageCreateRequest`; on `ChatRequest` it would be dead, since nothing binds
that as a request body), pinned through the route in `test_messages.py`.

Presets are client-expanded, so a preset reaches the wire as explicit sampler
fields and needs no server-side layer. Which preset a document is running is
`applied_preset_id` on conversations/notebooks (schema v6), written on explicit
Apply/Update/Save-as-new only. A document whose state merely matches a preset
is labelled by live client-side matching and never stamped, because storing a
derived association can bind stale state to the wrong document.

## Model registry and config

### Model discovery

(v1.69.0, `model_registry.py`) models.toml is override-only: anything under
`[scan].folders` is served with derived defaults, so a new download needs no
import, no symlink, no edit. The merge is load-time (`ModelRouter._load_config`,
so startup and reload both get it) and never writes models.toml; a
`[[models]]` entry is served exactly as written and always wins, and discovery
can only add.

Matching is the resolved `model_path` (`.resolve()` follows symlinks), never
the id. An id is derived from the directory name, so a hand-renamed entry
stops matching itself, and vendor symlinks in a model folder make one file
reachable by two spellings sharing no prefix. That pair silently duplicated a
Muse-Glimmer entry (with a wrong `supports_thinking`) on 2026-08-17, which is
also why the importer now dedups on resolved path. Discovery is best-effort: a
failing scan is logged and dropped, never fatal.

Admin edits materialize an entry on write (`update_config` /
`toggle_enabled`) because editing is the override; reads never do, or browsing
the models page would grow the file. `remove_config` deliberately does not
materialize: the next scan would serve it back, and a "removed" model that
reappears is worse than a clear no.

The router keeps `max_loaded_models=1` by default (LRU evict + pin +
idle-unload via `idle_unload_seconds` / `unload_after_idle_seconds`).

### Explicit entries get no derived fields

An explicit entry receives none of discovery's derived fields, and that is the
sharp edge of "served exactly as written": `merge_discovered` skips a
discovered model whose resolved path an entry already names, so nothing
re-derives for it ever again. Adding one field means hand-writing every other
field that model needs.

It bit twice on 2026-09-06 in two different mechanisms. Materialization wrote
a thin entry (identity + `model_path`) on the stated reasoning that the rest
was "re-derived at load", so a `reload?ctx_size=` cost a vision model its
`mmproj_path`, and the next spawn had no `--mmproj` with the projector sitting
unreferenced beside the weights (fixed v2.0.8: materialization now writes the
whole derived config, and the comment claiming otherwise went with it). And
enabling `spec_type` on a text model required writing `draft_model_path` out
longhand, because the drafter the importer would have auto-paired is not
contributed to an entry that exists.

### Config effect classes

Every provider-config field declares when a change takes effect
(`json_schema_extra={"effect": ...}`, the classes in `config.EFFECT_CLASSES`).
The reload set and `/v1/admin/model-options` derive from it (the import
allowlist did too, until import was retired in v2.0.72), because hand-written
second copies drifted; a new field must be classified or `config.py` refuses
to import.

Invariants (v1.55-56, design record [config.md](./config.md)):
`reload_config()` pushes per_request defaults into loaded providers, which are
construction-time snapshots otherwise ("applies immediately" was a lie before
this). Admin responses serialize config with `exclude_unset`, since absent is
the default's spelling; a validator that assigns derived fields must restore
`__pydantic_fields_set__` or they leak back as "stored". `stale_reload_fields`
on admin responses is the server-derived "saved but the process runs the old
value" truth, never rebuilt client-side.

### toml comments

models.toml comments survive admin writes (v1.58.0, `toml_comments.py`), but
only while their anchor is unchanged, so a note can never outlive what it
describes. A comment on a top-level key needs that key's value unchanged;
every comment inside a `[[models]]` entry needs that whole model
byte-identical (normalized through `tomli_w`, so old hand-formatting does not
pin anything); a block sitting above a `[[models]]` header additionally needs
that following model unchanged and still next. Consequence: a comment on the
value you are patching is deliberately dropped, so provenance for a value you
are changing belongs in the agent rules, this file, or `internal/`, not next to the
value.

Mechanism invariants: `tomli_w` stays authoritative for values, layout and
order; tomlkit is used strictly read-only for comment extraction, and comments
are injected as lines into the fresh render, gated on the merged text parsing
to exactly the fresh render's values. Any doubt degrades to a comment-less
write, never a refusal. Never graft comments into a tomlkit-parsed document
instead: mutating any item of a parsed array-of-tables (even comment trivia)
makes `tomlkit.dumps` re-render the AoT as an inline array, which is malformed
for nested tables. That is the failure mode that sank the first attempt
against `test_import_reimport.py`.

## API and store

### App assembly and routers

`api.py` has been app assembly only since v1.79.67 (lifespan, the MODEL_BUSY
handler, CORS, router mounting). Every route is a `*_api.py` router except
one (`rlm.py` carries its own), the OpenAPI narrative is `openapi_doc.py`, and
the static frontend is `frontend_static.py` (extracted v1.79.77). Root is gone
(v1.79.76: the frontend serves `/`). The shared inference-route guards lived
in `request_guards.py` until v2.0.30 removed it with the named-sampler
system; what remains of that concern is a wire-model validator on
`MessageCreateRequest`.

The OpenAPI drift guard (`generated-api.ts`, `scripts/check_openapi_sync.sh`,
the pre-commit block, `/openapi-regen`) was retired 2026-07-09 with the legacy
React app that consumed the generated TS types; v3 hand-writes `api.js`.

### One inference wire

(v1.79.66) The inference wire is `/v1/messages` (Anthropic Messages-conformant
plus the documented heylook extensions) and the conversation generate route
that shares its grammar. The OpenAI-compatible `/v1/chat/completions` +
`/v1/batch/chat/completions` routes, the route-level batch processor (and in
v2.0.57 the batch internals behind it: `mlx_batch_text.py`, `schema/batch.py`,
`create_batch_chat_completion`; `batch_vision.py` is parallel image loading
and stays), the server-side image resize and the `: keepalive` SSE comment
were removed. Owner call: v3 and the owner's other project speak Messages, and
nothing else that matters spoke OpenAI.

`ChatRequest` stays: it is the internal request every provider takes and still
speaks OpenAI's vocabulary (content parts, `finish_reason`); the rename to
Anthropic's happens once, at the Messages boundary (`converters`).

A consequence worth holding: nothing binds `ChatRequest` as a request body any
more, so a validator on it cannot see a client. A guard refusing a removed or
renamed field belongs on `MessageCreateRequest`, the wire model. Pydantic's
default `extra` policy is ignore, so a field the wire model does not declare
is dropped in silence and the request succeeds. v1.79.74 put the `logprobs`
refusal on `ChatRequest` and shipped green, because its test constructed a
`ChatRequest` directly (the one caller shape no wire produces) while
`POST /v1/messages {"logprobs":true}` answered a normal 200 (fixed v1.79.79;
the `preset` rename guard had been dead the same way since v1.79.66). A
model-level test passes whether or not any route binds the model it tests.

`/v1/models` keeps the OpenAI list shape because v3 and external clients read
`data`; that is a shape, not a wire. Still targeting the removed route and
pending port (owner: small potatoes, port later): `apps/batch-labeler`,
`tests/eval`, `scripts/benchmark.py`'s OpenAI arms, and one measurement script
in the owner's other project.

### Messages wire conformance

(v1.79.39-40) `/v1/messages` was Anthropic Messages-shaped but not
Messages-conformant for three payloads, each of which failed silently for a
client written against Anthropic's spec.

- Media blocks now accept both the nested `source` object and the original
  flat `source_type` form (`content_blocks._flatten_source`, gated on
  `source_type` being absent so a flat block carrying an unrelated `source` key
  is left alone). `source` is a declared `MediaSource` field, not
  validator-only, because a `mode="before"` validator contributes nothing to
  the generated JSON Schema: `/openapi.json` advertised only the flat form
  while the docs recommended the nested one.
- Thinking blocks and `thinking_delta` carry the text under both `thinking`
  (Anthropic) and `text` (v3's `streaming.js` reads `text` in two places);
  dropping either breaks one of the two readers.
- `stop_reason` is Anthropic's vocabulary via one table,
  `converters.STOP_REASON_FROM_FINISH_REASON`. Providers speak OpenAI's
  `finish_reason` because the internal ChatRequest does, and the rename happens
  at that boundary.

There are two routes on this grammar: `/v1/messages` and
`/v1/conversations/{id}/generate` share `StreamingEventTranslator`, so block
payloads agree by construction, but each assigned `stop_reason` itself, and
fixing one left the other emitting `"length"` for a commit. The whole per-path
suite stayed green throughout: per-path behavioural tests are structurally
blind to cross-path divergence, which is why `TestStopReasonHasOneMapper`
asserts the shared mapper is the only writer rather than asserting either
path's output.

An aborted generate run reports `max_tokens`, not `end_turn`: Anthropic has no
cancellation value and `end_turn` positively asserts the model finished.
`error` is not a stop reason: it was added on an untraced claim that api.py
set it, and a non-streaming failure raises HTTPException, so no
MessageResponse exists at all. Deliberate remaining differences are enumerated
in `docs/api_integration.md`; that list is hand-maintained and has been wrong.
Asymmetry: the `/v1/conversations` store accepts only the nested `source`, so
nested is the spelling that works on every surface.

### Cancellation

(v1.79.44, `request_registry.py`) A streaming request is cancellable by
hanging up: the server is writing, so it notices the peer is gone. A
non-streaming one writes nothing until it finishes and never notices, so an
abandoned run continued to completion and blocked everything behind it.
`DELETE /v1/requests/{request_id}` sets that run's existing `AbortEvent` (the
plumbing was already there; what was missing was a way to name a running
request from outside it).

It makes a run stoppable, not self-stopping: a client that hangs up without
calling DELETE still leaves it running, and disconnect polling was
deliberately not built (owner call: an explicit endpoint cannot mistake a
proxy hiccup for a departed client and kill a live generation).

The id is client-supplied, so `/v1/messages` had to stop generating its own
and honour `X-Request-ID`, via the one shared `resolve_request_id`, which
bounds and charset-restricts it: these reach logs and JSONL, and a newline
would forge a log line. The registry maps an id to a set, because two live
requests can share an id and a single-slot map would let the second orphan the
first. A streaming body outlives its route function, so it is registered by
wrapping the generator (`tracked_stream`), never by a `with` around the
return.

### DuckDB store

`db.py` holds conversations, notebooks, presets and `settings`, with a single
serialized writer thread and transactional ops; `HEYLOOK_DB_PATH` overrides the
location. Dynamic field names are gated by allowlists: the
`_UPDATABLE_*_FIELDS` frozensets plus `UPDATABLE_CONVERSATION_FIELDS`, which is
public because the update route pre-filters with the same set and carried a
hand-written second copy of it until v2.0.6. A `_SCHEMA_VERSION` bump drops
all tables; `settings` and `presets` are additive and drop-safe (key->value /
config) and are not in the drop list.

DB/config policy (solo deploy, no data to preserve): never write migration
code. Dropping, recreating or truncating any DuckDB store or config on a
schema change is fine and preferred.

### Observability

The spine (`observability.py`) has one ingestion path,
`record_event(type, *, tier, min_level, source, fields=<dict>)`, writing
level-gated JSONL under `logs/`: `metrics.jsonl` is content-free and
aggregatable; `events.jsonl` is correlated and may carry bounded error text
(type + message + cause chain), still never prompts, responses or token IDs.
`fields` is an explicit dict, not `**kwargs`, so caller/client keys cannot
collide with the reserved kwargs. It is best-effort and never raises
(inference must not break). `diag_event` (`diagnostic_logger.py`) delegates
here; `memory.py`'s legacy streams also write under `logs/` and are gated by
the master off switch.

Control is a single knob, `observability_level`
(off|minimal|standard|debug), an operational setting resolved DB > default,
and the default is `off`. File logging is opt-in (owner rule 2026-08-13: no
files under `logs/` unless the level is raised; `logs/` resolves CWD-relative,
so an on-by-default level sprinkled log dirs wherever the server was started).
There is no env override, because env silently beating the admin UI is a
footgun; env is bootstrap-only (`HEYLOOK_LOGS_DIR`, `HEYLOOK_DB_PATH`). `off`
is the master kill switch: it silences the spine, memory.py's streams, and the
llama-server subprocess `.log`. The gguf provider checks the level at spawn,
so capturing llama-server output needs level > off at load time, then a
reload.

Settings live in the App-DB `settings` table (`db.get_setting` /
`set_setting`), the contract in `settings.py` (`SettingsSchema` +
`resolve_settings`), CRUD via `/v1/admin/config`; the level and retention are
cached in-process (`observability.configure`) and refreshed at startup and on
PUT. Rotation is file-based (size + age, hourly on the tick). The content
invariant is level-independent: `minimal` is not "content-free" (its events
carry error text); only the metrics tier is guaranteed content-free.
Redesign status and the memory.py-stream consolidation follow-ups:
`docs/project/TODO.md` and `internal/research/observability_and_config_redesign.md`.

## Frontend

### Static serving: no fallback, no catch-all

There is no SPA fallback and no catch-all. These are two separate decisions,
and both are load-bearing.

The app routes on the hash, so the server only ever sees `/` and real asset
paths. A fallback would have destroyed 404 for the whole API behind it (a
typo'd `/v1/mesages` answering 200 with a web page). But the first fix was a
`/{rest:path}` route that 404s, and that broke routing a second way: matching
every path means starlette always finds a partial match, so
`redirect_slashes` never fires and a method mismatch reports 405 instead of
404. Measured in v1.79.79: `POST /v1/messages/` went from 307 -> 200 to
405 Method Not Allowed, which breaks any client that builds URLs by
concatenation, and every unknown non-GET path answered 405.

`mount_frontend` therefore registers the tree's real shape (`/`,
`/index.html`, `/js/*`, `/css/*`) and nothing else: unknown paths 404 on every
method, `/v3` and `/v2` get their gone-answer for free, and
`frontend/DESIGN.md` stops being served at the web root.

Two more things became true only when the handlers went sync (threadpool):
the gzip cache must not be iterated while mutated (it raised
`dictionary changed size` -> 500 on a static asset under concurrent cold
load), and `resolve()` raises `ValueError` on a NUL byte before `is_file()` can
swallow it, so `_serve` catches it (`GET /%00` was a 500). Revisit the
fallback only if the app ever moves to the History API.

Retired frontends: `apps/heylook-frontend-v2/` and its `/v2` mount were
deleted at cutover 2026-08-18 (v1.77.0); the older legacy React app on
2026-07-09. Both live in git history.

### Generation lifecycle and stream identity

Chat generates over `POST /v1/conversations/{id}/generate` (v1.65-66): the
server builds the request from the store and owns persistence, including abort
and disconnect; it speaks the Messages SSE grammar plus a final
`heylook_saved` event with the authoritative rows, and the client's
post-stream state is adoption, never position arithmetic. Notebook speaks
`/v1/messages` since v1.74.0.

A terminal path that awaits must re-check stream identity, not just
conversation identity (v2.0.5). `finishGenerate` calls `releaseStream` first,
which nulls `s.stream`, so for the whole of the resync GET that follows a run
ending without `heylook_saved`, the composer reads "Send" and `startStream`'s
`if (s.stream)` bar is down: a second run can be live when the first resumes.
That branch is reached by any ending leaving no usable saved rows, and an
abort is not required; a transport death that never delivered `heylook_saved`
lands there from an otherwise normal run. Conversation identity cannot close
the window either, because several such endings leave `activeId` untouched: a
mid-stream model switch, and `deleteConversation`, which aborts before it
clears `activeId`. The superseded run then paints its ending over the live
one. Rows were never at risk (`resyncMessages` re-checks `s.stream` after its
own await); the status line was, and the line that lands is the recovery
notice, telling the reader the generation on screen is a dead stream being
recovered. `handleStreamError` has the same shape and is safe only because
nothing in it awaits between `releaseStream` and its writes.

The composer being unbarred in that same window is correct, not a second bug.
It was re-raised as one on 2026-09-06 after a read of `finishGenerate` alone.
`refuseWhileStreaming` bars a send on `s.stream` or `remoteGenerating`, and
both are false across the resync GET; the server is the arbiter instead.
`conversation_api._refuse_while_generating` 409s every message write while an
`_ACTIVE` claim is held (the claim is taken before the row snapshot precisely
so a later write 409s rather than being destroyed by the positional commit),
and chat's send catch restores the typed text and the staged attachments on
that 409 (2026-08-13 review finding). So the two outcomes are: run still
live -> 409, composer restored, live run untouched; run genuinely finished ->
the send is legitimate and the `s.stream` re-check above stops the old run
painting over it. A client-side bar would have to guess which, and guessing
wrong blocks a send that should have worked.

### Store mirror and resume

Chat takes image (and gguf audio) input and renders image content blocks out
of the DuckDB store. The page is a mirror of the store with exactly two
invalidation points: document select, and resume (`ctx.onResume` ->
`refreshAfterResume`, v1.79.2). Nothing polls, and re-clicking the active
conversation deliberately does not refetch. Resume exists because iOS Safari
brings a backgrounded tab back with the heap it had, and every write the page
makes is whole-value from that mirror (prompt keystroke PUT, params PUT,
preset Save snapshot), so a stale mirror re-plays old state over newer edits.

The lifecycle edges are `createPage`'s `ctx.onHide` / `ctx.onResume`, each of
which binds both event spellings. Every debounced writer owns its hide flush
the way it owns its teardown flush (prompt-section, `bindDocumentParams` via
its `onHide` arg, notebook's `scheduleSave`); a consumer that has to remember
it is how one shipped without. Hide flushes send with `keepalive` and are
dispatched ahead of the PUT chain, because a request queued behind an
in-flight PUT is never sent if the page unloads. A prompt section's hide hook
lives as long as the section; detached is not dead (notebook keeps one through
every drawer close), and chat, which builds one per drawer render,
`release()`s the one it replaces.

Resume fetches the conversation body only when the list's `updated_at` moved,
and commits the new stamp only after everything it covers is adopted (else one
failed fetch is a permanent "unchanged", since nothing else ever refetches the
active conversation). It adopts via the same `adoptConversationMeta` select
uses, and never touches a live stream's rows, the prompt while its box is
being typed in (only that field; every other drawer field commits on change),
or the sidebar during a rename.

### Attachments and paste

Three attach inputs (picker, paste, drop; v1.72.0) funnel through one
`addFiles` -> `addPendingFiles` routine, which is where the cap gate, the count
cap and the aria-live announcement live. Paste was image-only for exactly as
long as it had its own copy. Only the picker has an `accept` list, so that
routine is also the backstop for everything that has no accept list to
respect.

Paste listens on `document`, not on the page root, and that is not a
preference: clicking a message leaves focus on `document.body`, which is an
ancestor of the chat root, so a root-scoped listener never sees the event. It
only ever fired when a field or an in-thread selection held focus, i.e. the
case that already worked. Any "paste anywhere" feature has this shape; verify
the target rather than the listener. A synthetic event dispatched at a
convenient node proves nothing, which is how v1.72.0 shipped this broken with
a passing check. Document scope is also why the other-editable guard is
load-bearing: the drawer's system-prompt box is a body child outside `#app`.
And `preventDefault` waits until something will really stage: a clipboard
payload carries text and an image, so cancelling on a refusal eats the text
too.

Capability-gated chrome (attach button, picker accept list, thinking toggle,
drop-overlay label) is refreshed after `modelSelect.value` moves, never
before: it all reads `currentCaps()` off that select, and `selectConversation`
had the order backwards, so every one of them described the conversation
being left. Drag/drop is desktop-only on purpose and is not a DESIGN.md §7
violation: it duplicates paths that exist on the phone rather than being the
sole route to anything.

### Markdown URL schemes

(v1.79.73) URL schemes are checked at the renderer. marked does not filter
them: verified on 18.0.11, it emits `<a href="javascript:...">` for four
markdown spellings (inline link, image, autolink, reference link), so
DOMPurify was the sole guard despite markdown.js's comment calling it a
backstop. `markdown.js` now allowlists schemes in the `link` / `image`
renderer overrides, and DOMPurify is genuinely the second layer.

It decodes HTML entities first, and that is the whole correctness argument:
the browser resolves the decoded attribute, so a check on the raw text checks
a different string. The first version tested for entities only before a
literal colon and so missed `javascript&colon;alert(1)`: no literal colon at
all, it took the "relative, therefore safe" early return, was emitted
verbatim, and executed in real Chrome (v1.79.79). Decoding once matches the
parser, so `&amp;#58;` correctly stays literal text. A regex for
`javascript:` passes on `href="javascript&colon;..."`, which is how the first
check was vacuous for exactly the vectors it was added for.

A renderer returning `false` falls back to marked's own implementation;
returning `''` does not (it drops the content silently), so the accept path is
the one a wrong answer breaks quietly.

The vendored libs are pinned by `js/vendor/vendor.json` +
`scripts/vendor_frontend.py` (offline integrity in pre-commit, staleness
reported at release). They have no lockfile entry, and sat a major version
behind for five months before v1.79.72.

### Incremental streaming render

A streaming message is rendered incrementally (`markdown-stream.js`, v1.79.9),
and this is load-bearing, not a micro-optimization. The painter used to
re-parse the whole accumulated response through marked + DOMPurify into
`innerHTML` every animation frame, and marked's parse is superlinear in
length, so per-frame cost grew with the response and a long generation
saturated the main thread. On a phone that is heat and battery, and no check
that renders a finished document can see it.

MarkdownStream cuts at a boundary no markdown construct can span (not inside a
fence; at column 0 after a blank line; never a list marker or `>`, which would
merge with a block above), renders each segment once into a committed prefix
whose nodes are never touched again, and re-renders only the tail. A
link-reference/footnote definition disables splitting for that message
because it reaches forward arbitrarily far. The boundary rule is a property,
not a set of examples: `tests/e2e/render.mjs` grows generated documents one
chunk at a time and diffs against a whole-document render, the same technique
and the same reason as the backend's `TestParserInvariants`.

Painters whose cost scales with the document use `ctx.throttleTime` (about
15/s), never `ctx.throttle` (per-frame, correct only for cheap work like a
token strip).

Scroll-follow is measured at the top of the painter, before it mutates, and
both halves of that are load-bearing. Before the write, the reads are cache
hits (layout is still clean from the last paint) rather than a forced
re-layout. That matters most on iOS, where nothing skips off-screen rows, so a
forced layout walks every row. (content-visibility was removed in v1.79.18: it
moved `scrollTop` behind the app's back on every engine, not just WebKit,
stranding the tail-follow and opening conversations thousands of px above
their end; the layout it saved was one-time and desktop-only.) And only
before the write is the measurement honest: measured after, one paint
appending more than the slack (a code block, a table) reads as "the reader
scrolled away" and strands the view for the rest of the generation. A cached
flag fed by scroll events was tried and is wrong: pinning coalesces scroll
events to a handful across a whole generation, so the flag goes stale exactly
when the viewport changes under it, which on a phone is every keyboard open
(`tests/e2e/render.mjs` resizes mid-stream and was shown red against it).

### Presets and the system prompt

Chat has a per-document system-prompt editor and a saved-preset bar: two
shared drawer sections, `prompt-section.js` + `preset-bar.js`, used by chat
and notebook.

The system prompt is an override box (owner rule, v1.62.3). A preset owns a
prompt and carries it, but a preset with an empty one makes no claim and
leaves the document's prompt alone. Empty never means "set it to empty",
which is what turned one blank Save into two presets losing their prompts.
Only a carrying preset can arm "Replace prompt?" or count as drift.

Both directions are armed (v1.79.20). Apply overwrites the document
(recoverable: re-apply the preset); Save overwrites the stored preset with an
update that keeps no history, so Save is the one "only loss gates" actually
names. It was the bare one, and the select pre-fills the save-as name box, so
picking a preset to look at it armed that preset as Save's target: one click
wrote the document's prompt over a 35k-char stored one on 2026-08-28.

Save's guard is an ordered set of questions in `wouldOverwritePresetPrompt`.
Read it there: it carries the order, the reasons and its own known boundary
(restating the branch list here would be exactly the hand-copied second copy
the repo warns about). The shape: only a save onto a preset the document is
not running arms, so the apply/edit/save-back iterate loop stays one click.
v1.79.20 armed all of them and thereby charged the loop for the accident, the
same click-through failure the rule exists to prevent, reintroduced by the fix
for it. Blanking always arms, because a null write leaves an override-box
preset present but inert ("my preset disappeared").

Enter in the name box goes straight to Save as new, which is correct now and
was not before: that rule existed because a second entry point past an arm is
the same hole with a keyboard on it, and Save as new has no arm to get past,
since it cannot overwrite anything. Update, which can, is reachable only by
its own button.

An arm is a promise about one action, and that is enforced in the primitive:
`armedConfirm` takes a `target()` describing destination + payload, captures
it at arm time and re-reads it on the confirming click, re-arming instead of
firing if it moved. It cannot live in consumer wiring: Save's payload is the
document prompt, edited in a different drawer section the bar gets no events
from, so "arm, clear the prompt box, confirm" blanked a preset straight past
the blanking guard, and no `disarm()` call in the bar could have seen it.
`disarm()` stays for visible honesty (a button still reading "Overwrite
prompt?" while aimed elsewhere is a lie even once clicking it is safe), and
each control disarms only what it re-aims: the select is the only control
that re-aims, so it is the only one that disarms, and the name box disarms
nothing because it feeds Save as new.

The reason that click happened at all is structural: the drawer renders the
preset section directly above the per-document prompt box, which shows the
document's prompt whatever the select says, so every preset looked like it
held the same text. The section now carries a read-only preview of the
selected preset's own prompt, and the document's box names its owner (`label`
on createPromptSection's adapter). The select also follows the document's
`applied_preset_id` until an explicit pick, and a pick is remembered against
the document it was made on.

A prompt typed before any conversation exists is parked in localStorage until
a conversation adopts it (it was page-state-only, so a reload ate it while the
sampler params beside it survived), and `.chat__sysprompt-chip` states what is
in force, including the "No system prompt" case, which is rendered, not
hidden.

### Model switching and the config editor

Since v1.54-1.57 the models page edits per-model config (`js/model-config.js`,
schema-driven off `/v1/admin/model-options`, so a new backend config field
appears in the UI with no frontend change; `ui:"hidden"` on a field is what
keeps it out), and chat switches models mid-conversation honestly. History
media the current model cannot take is dropped at the wire with a per-message
disclosure, while staged attachments still block (a deliberate asymmetry,
commented at both sites). Since v1.72.0 the cap is checked at staging time
too, so that send-side block's only remaining case is media staged on a
capable model and then switched away. A drop or paste onto a model without the
cap now refuses immediately and stages nothing, because a staged blob the user
must later hunt down and clear is worse than a straight no.

Chat also consumes `/v1/admin/models` (residency) and `load?warm=true` (its
Load button). Only loss gates a switch: load cost is disclosed, never
confirmed (owner call, v1.62.3). Residency dots, the Load button and a live
pre-first-token status say what is happening. Choosing a model is choosing to
pay for it, so a confirm there only trains click-through; the one removed
fired hardest with nothing resident, where its "may evict the resident model"
was false.

## MLX engine

### Media placement

(v2.0.18) Media placement is per message, and MLX puts it on user turns only.
mlx-vlm attributes media by counting explicit `{"type":"image"}` markers in
block-form content (`_content_media_count`) and dumps whatever it cannot
attribute onto the last user turn. So passing flattened strings plus a bare
`num_images=` total, which is what `vlm_inputs` did until v2.0.18, attributed
nothing and moved every image to the final user message. Text-only messages
still travel as a plain string, so that is the blast radius. The marker is
bare because mlx-vlm re-derives each message's content from text + count in
the model's own order (llava appends, qwen prepends). Verified at the rendered
prompt on gemma4 + qwen3_vl: identical token multiset before and after, only
the marker's turn moves.

The role gate is upstream and triple-layered (`_content_media_count` skips
non-user, the surplus reallocates, and `_format_list_with_image` re-tests
`role == "user"`), so an assistant-turn image does not error there; it moves,
silently. `_non_user_image_roles` in mlx_provider refuses it instead, naming
gguf, whose server rewrites an image part into a positional media marker at
any role. Owner decision 2026-09-07: not forking mlx-vlm for this.
Assistant-turn media is gguf-only, and that is the answer, not a backlog item.

The images themselves had to arrive in that same order, and until 2026-09-24
they did not: `BatchVisionProcessor.load_images_parallel` sorted its results by
`enumerate(as_completed(...))`, which is completion order, so a request whose
first image decoded slowest handed the model its images shuffled against the
markers (found by the improvement loop's request-path review; [measured] live,
a large red image followed by a small blue one read as two reds at
temperature 0: internal/claude/improve/harness/probe_image_order.py).
Results are now collected in submission order. In the same path every
unreadable image used to become a small red image and the request succeeded;
`utils.load_image` now raises and `prepare_vlm_inputs_parallel` turns it into
a 400 naming the problem. The unit tests had mocked the loader, which is why
neither was ever seen; `test_images_reach_the_model_in_the_order_they_were_sent`
runs the real one with staggered load times.

### The template draws the image markup

(v2.0.100) Since March 2026 heylook flattened mlx-vlm's structured message
content to a string, substituting a bare image token for each image marker,
because mistral3/pixtral templates cannot render list content. Every other
template lost its own image markup with it: Qwen3.5 and Qwen3-VL prompts
carried a bare `<|image_pad|>` with no `<|vision_start|>`/`<|vision_end|>`
(and their mRoPE counts images by `vision_start`, so image tokens took text
positions), and gemma-4 prompts carried a stray space after `<|image|>`.
mlx-vlm's own rendering had neither. Answers stayed mostly right, which is
why nothing noticed; `vlm_parity_probe` could not see it because it replays
heylook's own captured inputs into mlx-vlm. The template now renders the
structured content and only a template that refuses it is flattened, and the
probe compares heylook's prompt with mlx-vlm's own rendering of the same
request.

A message with no image is NOT rendered in list form. mlx-vlm turns every
message into a list, and a list of one text item picked up template artifacts
the plain string never had: gemma-4 renders a space after each item, so a
system prompt became `Be brief. <turn|>`. That alone emptied gemma-4 replies
in three E2E chat checks (v2.0.103). Text-only messages go back to their plain
string; only messages carrying media stay structured.

### Vision prefill

(v2.0.55) The vision path prefills all but the last prompt token and hands
mlx-lm the rest, mirroring mlx-vlm's own loop (`generate/ar.py`):
`get_input_embeddings` once, then `model.language_model` over embedding
chunks, then `run_generation(prompt_tokens=[last_token], pre_filled_cache=)`.
So the first generated token is sampled by the same code as every other.

Until then `VLMVisionStrategy` ran the full VLM forward and sampled token one
itself, which cost it the stop check (continuing a finished answer ran past
end-of-turn), the logits processors, an exact `max_tokens`, and the
detokenizer. The last is why every BPE (Qwen) image reply lost the space after
its first word ("Astylized") until v2.0.51 papered over it.

Five things that are each a bug if undone:

1. No `mask` goes to the language model. A caller's mask replaces the family's
   own causal/sliding/bidirectional masks where it is honoured, and the int32
   ones mask heylook used to pass is "no mask", i.e. non-causal: inert on qwen,
   live on gemma-4/gemma3/pixtral/llava_next.
2. `_reset_vlm_positions` runs before the prefill and stays skipped inside
   `run_generation`. Direct language-model calls never clear mRoPE state, and a
   reset after the prefill nulls the rope delta the decode steps read: fluent,
   wrong output.
3. Unchunked, the `_PER_TOKEN_PREFILL_KWARGS` allowlist is cut to N-1
   (gemma-4's bidirectional overlay silently no-ops on a length mismatch),
   while chunked they go whole, as upstream passes them.
4. The split is refused when the prompt ends on a media placeholder.
5. `run_generation` takes `prefill_progress_offset` so mlx-lm's own `(0,1)`
   cannot paint progress going backwards after the strategy's `(k,N)` frames.

It leans on two upstream-private pieces (`_chunked_prefill_enabled`, the
`n_to_process` kwarg), pinned by `TestChunkedPrefillSurface`.

The instrument is `scripts/vlm_parity_probe.py`: it replays the exact tensors
heylook built through mlx-vlm's `generate_step` and compares token ids at
greedy. Read a near-tie there as drift: the old all-N prefill diverged from
upstream only at one-quantum bf16 margins, and once both sides prefill
the same way the match is exact, multi-chunk included. Continuation with image
history was never an mlx-vlm limit: `vlm_apply_chat_template` always took
`continue_final_message`, and the strategy did not pass it.

### Penalty scope

(v2.0.60, owner decision) A penalty counts generated tokens only on MLX,
enforced in one place: `run_generation` wraps every logits processor in
`generation_core.generated_only`. mlx-lm hands a processor whatever prompt
tokens it prefilled plus the reply, so the scope used to be an accident of the
path: the whole prompt (system prompt, earlier turns, their end-of-turn
tokens) on a cold text request, only the uncached suffix on a prompt-cache
hit, nothing but the reply on an image request. The same request could sample
differently depending on what ran before it.

The wrapper assumes no token count: at the first processor call nothing has
been generated, so the history's length there is the prompt part. It is
recorded once and sliced off every call, which is why it holds for the normal
loop, the speculative loop, a cache hit and the vision path without knowing
which is running. A processor added later gets this for free; one called bare
(a test, a script) penalises whatever it is handed.

gguf is not aligned and cannot be by request: llama.cpp's server feeds every
prompt token to its sampler and penalises over a recent-token window. So
`presence_penalty` is not the same knob on the two engines, and a value tuned
on one does not transfer.

### Prompt preview and media

A preview that cannot show media must say so. `render_prompt` on MLX goes
through the text strategy (images stripped), so the preview is the text
template alone. `PromptPreviewResponse.unrendered_media` (sent, not shown) is
a different field from `dropped_media` (not sent), and both are painted: a
panel headed "what the model will see" that quietly omits the picture reads as
the image having been lost, which is the opposite of the truth. Providers
answer `render_prompt_represents_media` themselves (gguf True) rather than the
route switching on a provider name.

### Loader routing

Until plan W10 this picked a library: `MLXProvider.effective_loader`
(`providers/common/loader_routing.py`) resolved `"mlx-vlm"` or `"mlx-lm"`
from the config's `modalities` + `loader` fields. Every MLX model now loads
with mlx-vlm, the `loader` field is retired, and since v2.0.89 the answer is
one bool, `resolve_serves_vision` (`is_vlm`): vision iff the model declares
it and mlx-vlm registers the `model_type` (it degrades only on positive
non-support). The raw `vision` bool is a derived mirror of `"vision" in
modalities`. Modality description
(`model_importer.detect_modalities`: config `*_config` blocks +
`image_token_id` / `image_token_index` / `audio_token_id`...) is deliberately
separate from this library-aware routing.

The reported `vision` capability derives from it too (v1.79.43,
capabilities.py): the provider's image guard reads `is_vlm`, so reading the
checkpoint's declaration instead let `/v1/models` advertise images that a 400
then refused. One resolver for both surfaces is what makes them agree by
construction. `modalities` still carries the declaration; description and
served capability are different fields on purpose.

The library is on the wire as `engine.runtime` (v2.0.73; `effective_loader`
on the admin row from v1.79.31), and served vision as the `vision`
capability, derived via `serves_vision_for_config` so both answer for
unloaded models, which is what a live harness picking its arms needs. For
gguf the runtime reads `llama.cpp`, so one field names the engine on every
row.
Because it reads each model dir's `config.json`, the two admin read routes
that build a model response are plain `def` (threadpool), not `async def`.

### Prompt cache

(v1.75.0, Q7; the radix tree is deleted) The prompt cache is a per-model
single slot of immutable (state, meta_state) snapshots: extension continues,
divergence goes through mlx-lm's `trim_prompt_cache`, and non-trimmable layers
(hybrid ArraysCache, rotated windows) re-prefill rather than slice, so hybrids
are now correct, not "limited". Never store or hand out live cache objects:
arrays are immutable, objects are not, and a quarantined zombie generator
keeps mutating its own (that was live-verified process-poisoning). See
[mlx_provider.md](./mlx_provider.md) §4.2 and the
[postmortems](./postmortems/).

The two engines' cache classes do not share a contract. mlx-lm folded
`meta_state` into `state` (#1778), and for a while the slot kept `state` alone
on that basis. mlx-vlm vendors its own cache module, whose `RotatingKVCache`
keeps offset and write index only in `meta_state`, so every gemma-4
sliding-window layer restored at offset 0: each follow-up's trim was refused
and reported as `trim_refused` (found through W5's cause field, 2026-09-23;
fixed v2.0.84, `_Slot.metas`). An extension on that restore would have
continued from the wrong position. Check both engines' cache classes before
trusting a round-trip, and prove it with the greedy chain probe
(`scripts/chain_probe.py`), not a unit test on one engine's class.
The wrong-position extension was not reachable in practice: on the pre-fix
code every chat flow (append, follow-up, continue, regenerate, thinking on and
off) diverged inside the stored tail and took the trim path, which refused, so
the cost was lost reuse, never a reply decoded from a wrong position.

The vision feature cache is keyed by the request's whole image-URL list joined
in order, so adding one image to a conversation re-encodes every image in it;
the pixel-hash fallback in the module is never reached from that caller. A
per-image key is part of W10 of
[plan_runtime_visibility.md](../project/plan_runtime_visibility.md).

Since v2.0.86 the live MLX path is mlx-vlm's engine and its prefix cache
(APC), not the slot (plan W10, A2; the slot, the mlx-lm decode loop and the
vision prefill handoff were deleted in v2.0.87). Three traps the switch
found, each silent:
- **The APC salt.** Left to `BatchGenerator`, it folds in a hash of the whole
  prompt's `inputs_embeds`, so no two different prompts ever share a block or
  checkpoint and every follow-up misses with nothing in the logs. mlx-vlm's
  own server precomputes the salt from the request's media only;
  `vlm_engine.semantic_hash` does the same.
- **Checkpoint settings.** At mlx-vlm's defaults a checkpoint model keeps only
  its prompt end and one far boundary, so a follow-up (which diverges at the
  previous reply) restores nothing. `APC_CHECKPOINT_INTERVAL_TOKENS` and
  `APC_CHECKPOINT_CAPTURES` are the spike's measured settings.
- **One number was both the checkpoints per request and the store size.**
  mlx-vlm's `checkpoint_lengths` spends the store's entry cap on one
  request's captures, so at the spike's setting every request evicted every other
  conversation, and a request snapshotted only its last stretch: a new chat
  with the same system prompt and a first message longer than the snapshot
  spacing reused nothing, and going back to a chat after another reused
  nothing, while gguf reused both ([measured] 2026-09-24 by the improvement
  loop, internal/claude/improve scoreboard pairs final2-vision and final2-27b;
  its smoke probe used a short question and passed by coincidence).
  heylook now installs its own capture rule per generator
  (`install_capture_policy`: mlx-vlm's rule with its own count, plus the end
  of the system prompt), and `APC_CHECKPOINT_ENTRIES` is the store size alone,
  bounded first by the byte budget. `test_our_capture_rule_is_upstreams_with_its_own_count`
  pins the copy to mlx-vlm's rule. The system-prompt snapshot is stored first
  and later turns restore from newer ones, so without `refresh_snapshots` it
  aged out of the LRU after a few turns of one chat (found by the run's
  review; [measured] live the same day, internal/claude/improve scoreboard pair
  aging-check). Retention also broke an instrument's
  premise: `chain_probe.py` made its "fresh" run fresh by sending one
  unrelated request, which no longer evicts anything, so it clears the cache
  instead and fails if a fresh run reports reuse.
- **The detokenizer.** mlx-vlm's BPE streaming detokenizer flushes only on a
  token that starts with a space, so a count, code or CJK text arrived in one
  lump at the end; the engine streams through mlx-lm's (vendored in stage 3
  as `providers/common/lm_detokenizer.py`, when mlx-lm left the dependency
  set). The smoke walk-away
  check caught it, as a false "truncated" verdict: the whole answer had been
  delivered in the final flush.
The one-hash image salt is also why a turn that adds an image re-prefills;
any single salt heylook could choose either misses or risks restoring a
different image behind identical placeholder tokens (owner decision: accepted,
`internal/claude/w10/apc_new_image_turns.md`).

### Logits processor shape

A logits processor receives `(tokens, logits)` with logits shaped
`(1, vocab)`: mlx-lm passed `logits[:, -1, :]`, batch axis kept, and
mlx-vlm's engine passes the same shape.
`zeros_like(logits).at[tokens]` scatters along the size-1 batch axis, and MLX
does not bounds-check a Metal scatter, so it is silent memory corruption
followed by a mid-generation Metal fault and a poisoned process (v1.79.63,
presence penalty, gemma-4-26B). Unit tests with 1-D logits stayed green
against it.

### Tokenizers and stop tokens

Stop-token/eos resolution has the same dual-source trap as chat templates: a
model's full eos set can be split across tokenizer_config.json's
`added_tokens_decoder` and tokenizer.json's `added_tokens` (gemma-4's
`<turn|>` terminator lives only in the latter). Raw HF tokenizers also do not
absorb `generation_config.json`'s eos list, and mlx-lm's `stream_generate`
auto-wraps a raw tokenizer with only the single `eos_token_id`, which is why
`run_generation` wraps it itself (`ensure_gen_tokenizer`) with the full
resolved stop set; otherwise a model generates past its own end-of-turn.

(v1.79.66) MLX has two tokenizer shapes at generation time. mlx-lm's `load`
hands the text path a `TokenizerWrapper` whose streaming detokenizer (SPM/BPE)
was chosen from tokenizer.json; mlx-vlm's processor hands the vision path a
raw HF tokenizer (its own detokenizer sits unused on
`processor.detokenizer`). `run_generation` wraps the raw one via
`ensure_gen_tokenizer`, and a wrapper built without `model_path` takes
mlx-lm's default, `NaiveStreamingDetokenizer`: quadratic per line (it
re-decodes the current line on every token), and `text` is a read-only
property. `MLXProvider.load_model` therefore primes the wrapper with
`model_path` (mlx-lm's own loader picks the class), and
`continuation_detokenizer` seeds only a class with a settable `text`
(`_seedable`). v1.79.64 seeded unconditionally and raised inside the first
`next()` of every continuation on every mlx-vlm-routed model, behind a 48/48
browser run whose resume check had a legal early exit that reported green.

(v2.0.88) Since plan W10 there is one tokenizer shape: every model loads with
mlx-vlm, and the tokenizer is its raw HF one. The wrapper is gone with
mlx-lm. `generation_core.detokenizer_source` builds the vendored class
`lm_detokenizer.detokenizer_class_for` picks from tokenizer.json on that
tokenizer, primed at load with `model_path`, and the stop set is the
provider's own, resolved once at load and checked in `vlm_engine`'s loop. The
Naive-default and `_seedable` lessons above still hold for that source.

(v2.0.88) `BaseProvider.get_tokenizer` took `processor._tokenizer` first, a
branch for mlx-lm's wrapper (whose `_tokenizer` was the HF tokenizer). Under
mlx-vlm a text model's "processor" IS the HF tokenizer, and its `_tokenizer`
is the Rust backend, which resolves no eos ids: from v2.0.86 Qwen3-0.6B ran
every reply to its budget. Nothing noticed because every smoke check
accepted `max_tokens` as an ending, and the population run checked "loads
and answers". Smoke now asserts a one-word reply ends on `end_turn`.

mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects
`enable_thinking=True` when the kwarg is absent. The kwarg is the cross-model
thinking control: transformers forwards extra apply_chat_template kwargs as
template variables (Qwen3 renders `<think>`, gemma-4 renders thought channels;
others ignore it), and "template references enable_thinking" is the
thinking-capability signal.

### Reasoning parsers

`reasoning_parser.py` has four routing parsers (harmony/gemma channels,
`<think>` markers, pass-through) that never strip anything themselves;
declared-specials stripping is one wrapper, `StripSpecials`, composed over the
selected parser by `select_reasoning_parser`, and only when the model declares
specials, so a bare parser is the no-strip case. Its rolling holdback is sized
by the strip set, not by any parser's own control tokens, and is prefix-set
based because declared specials are not all `<`-shaped (Mistral's `[INST]`
family). Behaviour is pinned by properties, not just examples
(`TestParserInvariants`): output is invariant to how the stream was chunked,
and text carrying no structural tokens survives intact. Both 2026-07-23 parser
bugs were violations of those two properties. Design record:
[parser_strip_unification.md](../parser_strip_unification.md).

### Template ladders

Adding a rung to a ladder invalidates every hand-written subset of it. The MLX
stop-less fallback (`read_template_info`: a template rendering none of the
model's stop tokens is refused and the other sources are walked) listed
`(TOKENIZER_CONFIG, CHAT_TEMPLATE_JSON)`, which was correct only while `JINJA`
was the top auto rung, since omitting the winner was the point. v2.0.22 put
the operator override above it and silently made that omission a bug: a
stop-less override sent the model straight past the perfectly good vendor
jinja in the same directory and installed nothing, i.e. a rejected override
cost the model its only working template. It is now `_AUTO_LADDER` minus the
source that failed, which cannot rot when a rung is added. Same class as the
reload set and the import allowlist; the tell is a tuple that enumerates a
subset of an ordered list defined elsewhere in the same file.

The operator's template override (v2.0.22, `chat_template_files.py` +
`GET/PUT/DELETE /v1/admin/models/{id}/chat-template`) is one file,
`chat_template.heylook.jinja`, in the model's own folder, discovered at load by
both ladders and beaten only by an explicit `chat_template_path` /
`chat_template_source`. It writes no config (nothing calls `update_config`),
so editing a template cannot materialize a discovered entry, and revert is
deleting one file.

The filename differs from the vendor's on purpose: on MLX `chat_template.jinja`
is usually the only copy of the template (checked 2026-09-08: of the model
dirs here none had an embedded `tokenizer_config` template and all but one had
the sidecar as the sole source), so writing through to it destroys the
original with nothing to revert to. A distinct name also survives a
re-download, since `huggingface_hub` prunes nothing while `chat_template.jinja`
is in the manifest and gets refreshed.

Three rules that are each a bug if forgotten:

- `use_sidecar_chat_template=false` must not suppress it. That flag chooses
  between the publisher's sidecar and the embedded template; applying it here
  makes the editor write a file nothing reads.
- The gguf origin phrase is the constant `HEYLOOK_OVERRIDE`, not a prettier
  spelling, because the admin view decides "is this actually in force" by
  comparing against it, and a second spelling is a comparison that never
  matches (it shipped that way for an hour and a test caught it).
- The routes must be declared above admin_api's bare `/{model_id:path}`, whose
  greedy converter otherwise eats `<id>/chat-template` and 404s a model that
  exists.

The MLX half needs `install_chat_template` to target the processor as well as
the tokenizer under force: mlx-vlm's `get_chat_template` reads
`processor.chat_template` first (transformers fills it from a
`chat_template.json`), so a tokenizer-only install is read, installed,
reported successful and never used on the vision path, with every text-model
check green.

Validation runs before the write (a jinja exception is a 500 from
llama-server, so a bad file breaks the model at its next load) in an
environment mirroring the engines' (`raise_exception` / `strftime_now` /
`tojson`), because a bare jinja2 environment rejects most real templates and
would refuse valid work. Both engines bind at load, so a write is not live
until a reload; `stale` on the response is the file-backed equivalent of
`stale_reload_fields` (which cannot see it, since no config field moved), and
it is null for an unloaded model, which is not the same as false.

### Streaming latency checks

When live-verifying streaming or latency changes, the 31B dense gemma natively
decodes slowly enough to look identical to the old delivery cap; the MoE
`gemma-4-26B-A4B` decodes several times faster and is the discriminating model.

Perf numbers have been honest since v1.34.1: recorded tok/s is native mlx-lm
`generation_tps`, TTFT and tok/s exclude queue-wait (its own `queue_wait_ms`
field), and trends are success-only.

## Repo conventions

### Hand-copied constant lists

A hand-copied constant list is a defect with a delay, not a style issue. This
repo already derives rather than copies: the reload set, the import allowlist
and `/v1/admin/model-options` all come off `effect` metadata precisely because
a hand-written second copy drifted. Three more copies drifted in one session
(2026-08-17): `conversation_generate_api._SAMPLER_KEYS` was a copy of
`samplers.REQUEST_SAMPLER_FIELDS` and silently dropped `reasoning_effort` on
the only surface that generates server-side; the resolved-path identity rule
existed in three places; and the reasoning-effort `Literal` union in three
more. The failure is silent in every case.

### Enforced rules and the .claude allowlist

`.claude/` is local by default with a per-file tracked allowlist (2026-07-26,
pruned 2026-09-23). A bare directory negation in `.gitignore` would silently
publish future files, which is why tracking a new file needs both the
negation and the pre-commit hook's `ALLOWED_PATHS` entry. Files inside a
tracked skill dir track by default; that is the publish-intent boundary, and
deliberate. As of 2026-09-23 the tracked files were `.claude/settings.json`
and the dev-server and eval-ab skills.

Repo rules are enforced, not reminded (2026-09-23). hookify is retired: its
plugin had been disabled and all four of its rules were silently dead, which
let eight positional `Field` defaults back in. A reminder nobody can see
failing is not a control.

### Docs layout

CLAUDE.md carries mechanisms; status lines there rot into being actively
wrong. The perf-distrust note did exactly that within a day. The release
bookkeeping rule (bump `__version__` with the CHANGELOG entry) exists because
the version was hardcoded-stale for weeks before 2026-07-20.

### Engine pins and dependencies

Root venv: plain `uv sync` has been the whole story since v1.39.17. uvloop and
cachetools are core deps; questionary was retired 2026-07-28 with config_tui;
pyturbojpeg and xxhash were dropped 2026-08-18 (both only flipped a status
flag, never called); build/twine were dropped 2026-08-18 (`uv build` /
`uv publish` cover both).

The MLX engines are committed git pins (owner decision 2026-09-05, v1.79.69):
`[tool.uv.sources]` pins mlx-lm and mlx-vlm to exact upstream SHAs. `rev =`,
never `branch =`, because a branch pin moves under a plain
`uv lock --upgrade` and the manifest stops saying what it depends on. The
reason is that mlx-lm is release-starved (PyPI's newest was five months old,
with the behaviour the server is written against sitting unreleased on main),
and the ecosystem posture has said "SHA-pin rather than wait" all along.
`scripts/guard_stable_channel.sh` still blocks a git pin by default so that a
pin commit stays a named act rather than a stray relock, which is also why a
trying-things `branch = "main"` pin in the working tree is safe: it cannot
land by accident. A cloner's `uv sync` needs git and GitHub reachable; that is
the price, priced.

(v2.0.88) mlx-lm is no longer a dependency: plan W10 moved every MLX model
onto mlx-vlm's engine. Its streaming detokenizer is vendored, because
mlx-vlm's BPE one holds space-free text. Samplers come from mlx-vlm's
`sample_utils`, which gave identical results to mlx-lm's on seeded runs of
every knob heylook sends. mlx-vlm is the one pinned engine. protobuf left the
lock with mlx-lm: every MLX model here ships a `tokenizer.json`, and a
sentencepiece-only checkpoint would now fail at load with transformers naming
protobuf.

The optloop-lib bench exists because the app-level optloop (retired
2026-07-06) bypassed the server code it claimed to measure.

## Tests

### No make-it-fail-first rule

It was asked for twice (2026-08-17, 2026-08-28); the second time the ask was
explicitly "entirely", carve-outs included, because the habit had been
codified into CLAUDE.md, `tests/e2e/README.md`, `render.mjs` and
`plan_chat_orchestration.md` after being rejected, and then justified itself
on every read. The `/test-suite` skill that wrapped the backend run was
removed 2026-08-17; the legacy React app that carried a frontend unit suite
was deleted 2026-07-09.

The unit suites cannot certify templates, parsers, stop tokens or vision: the
2026-07-20 turn-overrun and thinking-leak bugs passed 1000+ of them, which is
why the behavioural eval bank exists.

### Live smoke and the RAM pre-flight

The browser suite drives the real frontend against a stubbed `/v1`, which
left the store's own rules and the generation lifecycle unverified; live
smoke talks to a real server and no stub at all. "Covered mlx" is a claim
about a config value, since `"mlx"` routes to two separate upstream repos
(mlx-lm text / mlx-vlm vision, separate release trains). "Served but not run"
prints differently from "no model of this engine exists", because the first
is the quiet one.

(v2.0.88) With mlx-lm gone, every MLX model's runtime reads mlx-vlm, and
classifying by runtime alone left an mlx-lm arm that was empty on every run:
a permanent UNCOVERED that people learn to skim past. The arms are now
mlx-text / mlx-vision / gguf. The MLX split is the served `vision`
capability, which is exact now that the retired `loader` field can no longer
make it disagree with `is_vlm`. The older-server fallbacks and their
"unconfirmable" bookkeeping went with it; a row with no runtime is
unclassified.

`scripts/dev_server.sh`'s RAM pre-flight sizes through `scripts/ram_report.py`,
which resolves a model through the same `discover()` / `merge_discovered()` the
router uses, so a discovered model sizes like an explicit one. It did not
until v1.79.44: it read models.toml alone, which is override-only, so
`--model <discovered-id>` refused to start with "not in models.toml?" on a
model the server serves. The related trap is worth more than the fix:
`ram_fit` returns 0.0 GiB for a path it cannot read, and 0 GiB clears every
ceiling, so a broken `model_path` printed `RAM pre-flight OK: ~0 GiB` and
exited 0, the gate waving through the exact case it exists to refuse. An
unsizeable model is now exit 2 with a reason, distinct from a memory refusal.
The same hole in the `/v1/admin/{id}/fit` route was closed in v1.79.56 (it now
422s through the shared `unsizeable_reason` predicate).

The release coverage standard is a Phase 4 standard, not a CI gate: there is
no CI here, and a gate nobody can run is worse than a rule somebody follows.
The standing Phase 3 gap, thinking depth on both MLX arms, exists because (as
of this writing) the only served MLX model advertising `reasoning_effort` is
gpt-oss-120b.

eval reports how many tasks ran on no model because its
`required_capabilities <= model_caps` filter is how a text-only `--models`
list ran zero vision tasks under a full green. `tests/helpers/engines.py`
replaced eval's own `fetch_models`.

### Browser E2E

The E2E harness refuses to start when anything already listens on `E2E_PORT`
(v2.0.2), and that guard is load-bearing rather than tidy. The spawned child
loses the bind and exits, but that takes seconds, while readiness polls
`/v1/models` immediately: a stranger on the port answers, lists the model, and
every suite runs against it. Teardown then makes it self-perpetuating
(`stop()` returns early because our own child really did exit), so the
squatter captures the next run too. An orphan four releases stale was found
holding that port on 2026-09-06, and nothing in the output said so. A bind
test races the child for the port, which is why the probe connects.

It must run unsandboxed because bun's non-interactive script shell resolves
the real node binary, while bare `node run.mjs` from an interactive-derived
shell hits the nvm lazy-load function, which `export PATH` cannot beat. The
harness itself still executes under node by design, via the package.json
scripts.

The streaming-cadence guard is the only automated check for the Phase 1
delivery fix; server telemetry cannot see it.

`bun run e2e:render` guards that the chat message list is reconciled, not
rebuilt. The sharpest reason used to be `content-visibility: auto` on
`.message` (a row's laid-out height lived on the node, so a rebuild collapsed
`scrollHeight` mid-tick and every pixel-based scroll aimed at a list about to
grow underneath). That is gone as of v1.79.18, and the CSS says so at its own
site. The check stands on its own footing: a rebuild still drops open editors
and the unsaved drafts in them. `E2E_V3_ROOT` is how each check was shown to
fail against a deliberately broken copy of the frontend.

`skip()` (harness.mjs, v1.79.66) exists because the chat suite's resume check
hid the mlx-vlm continuation crash behind a legal early exit that reported
green.

`bun run e2e:ios` (`ios-sim.mjs`) drives real Mobile Safari in the iOS
Simulator through Apple's `safaridriver` because Chrome cannot see iOS
keyboard behaviour: with the keyboard up, iOS shrinks only the visual viewport
and scrolls, while Chrome shrinks the layout viewport, and the fixed bottom
nav, the `100dvh` shell and the composer all follow the layout viewport. An
emulated pass proves nothing about the phone.

### Assertions aimed at strings the code never emits

An assertion aimed at a string the code never emits is indistinguishable from
a fixed bug, and it is the most-repeated own-goal in this repo. v1.79.79's URL
check matched `javascript:` against rendered HTML that says
`javascript&colon;`; v2.0.5's superseded-stream check matched the completion
line when the recovery line is what lands; v2.0.6 found the same check still
missing `MODEL_SWITCH_PREFIX` entirely and matching `still generating`
lowercase against a capital-S constant. The author of the code is the
worst-placed person to spot this in their own check, which is the argument
for an independent review pass rather than more self-checking.

Until the 2026-09-23 rewrite, CLAUDE.md's E2E paragraph also said "run it red
first -- a green here means nothing until you have seen it fail". That
sentence contradicted the standing owner rule that there is no make-it-fail-first
rule, carve-outs included (see [No make-it-fail-first rule](#no-make-it-fail-first-rule)),
so the rewrite states the lesson as "read the constant it is supposed to
match". The owner confirmed the same day that a self-administered red "never
works" here, and the optional-tool carve-out was removed as well. The remedy
for a check that asserts on the wrong string is an independent review pass.

### MLX mocks and teardown crashes

A module-level `.start()` of an MLX `sys.modules` mock leaks mocks across the
whole session and fakes about 50 "Metal context" failures; that bug produced
the old pre-existing-failure allowlist.

The `gilstate_tss_set` interpreter-teardown crash needs the MagicMock MLX
tree, and reproduces model-free and pytest-free in a bare interpreter:
`import heylook_llm.api` under `patch.dict(sys.modules,
create_mlx_module_mocks())` aborts at finalization, while the same import with
real MLX exits 0, and the mock tree without that import exits 0. No stray
Python thread survives the import, so the foreign thread doing it was not
identified (timeboxed). Contract runs on Apple hardware no longer hit it at
all since `mlx_mocks` stopped patching there (v1.77.1); it remains a residual
on the mocked path.

Real-MLX failure poisoning: one `RuntimeError: [read] Unable to read from
file` inside the (since-deleted) embedding-provider test also took down an
unrelated test in another file, which passed the moment the first was fixed.
Its cause is worth knowing generally: `mx.load` is lazy/mmap-backed, so
`mx.save_safetensors` back over the same path without `mx.eval`-ing the loaded
arrays first corrupts them. It was latent in that test until mlx 0.32.1
surfaced it.

Invocation order used to be load-bearing (fixed v1.77.1, verified on Apple
hardware in both directions and each directory alone):
`tests/contract/conftest.py`'s session-scoped `sys.modules` MLX mock tore down
only at the end of the whole run, so contract-first left every later unit test
looking at MagicMock arrays (about 57 failures + 8 collection errors that all
passed in isolation). The fix is that `mlx_mocks` skips the patch when real
MLX imports (`helpers.mlx_mock.real_mlx_available`), because contract tests
drive FakeProvider and only ever needed the mock so imports would succeed
where MLX is absent. Narrowing the fixture's scope would not have sufficed: a
heylook module first-imported under the patch binds MagicMocks into its own
namespace permanently, since the module object outlives the patch. Where MLX
is genuinely absent, the session mock still applies and that residual
order-sensitivity stands, untestable from here.

Mocking optional probe paths breaks absent-dependency tests: mocking
`mlx_vlm.generate.diffusion` turned `_detect_diffusion`'s
returns-False-when-unavailable test from green to red.

### git archive is not isolation

The package is installed editable (`__editable__.*.pth` in site-packages), so
`import heylook_llm` resolves to this repo's own `src/` from any cwd.
Exporting a commit to a temp dir and running pytest there with the repo venv
therefore gives you that commit's tests against the working tree's source:
neither commit nor tree, and worse than either because it looks like the
strictest option available. It produced a confident "HEAD is green" and a
confident "HEAD is broken" ten minutes apart on an unchanged HEAD
(2026-09-08), where the failures actually belonged to a third session's
uncommitted work, and nearly stopped a push over a break that did not exist.
It is a live hazard here because parallel sessions are normal and someone
else's half-finished work is what gets imported.

The print of `heylook_llm.__file__` is not optional, and the second-order
failure is why: the setup step can fail silently. A sandboxed `mktemp -d` is
blocked, the `tar`/`cp` after it fails, and the probe still prints a confident
answer about the wrong tree. Verified in both directions.
