---
paths:
  - "src/heylook_llm/providers/**"
  - "src/heylook_llm/{router,samplers,capabilities,thinking_controls,thinking_parser,reasoning_parser,chat_template_files,image_plan,model_ops_api,admin_api}.py"
  - "frontend/js/{settings,engine}.js"
  - "frontend/js/pages/chat.js"
---

# Provider and engine contract, thinking, sampling

## Providers

- Two providers, `Literal["mlx", "gguf"]`. The single source of truth is `config.PROVIDER_CONFIG_CLASSES`; the router's `provider_map` must stay key-synced with it.
- MLXProvider does text and vision. LlamaServerProvider (gguf) runs one llama-server subprocess per loaded model: "loaded" means "running process", so LRU and idle-unload are spawn and SIGTERM. It is pure stdlib with no MLX import.
- Provider output is the owned, slotted `GenerationChunk` (`providers/base.py`). New telemetry is a field there, absorbed in `perf_collector.ChunkTelemetry`, never an attribute patch. `thinking` carries engine-pre-split reasoning (e.g. llama-server's `reasoning_content`). Errors raise `GenerationFailed` / `InvalidGenerationRequest`; they are never chunks.
- One engine contract (`providers/contract.py`): `/v1/models` and the admin row carry one `engine` object (runtime, context, template, every setting with `{value, configured, auto, reason, provenance}`), built from a static half per engine (`mlx_describe`/`gguf_describe`, config-only, answers unloaded) plus `provider.describe_observed()` (recorded at load, never a call into the process). The report calls the SAME decision functions the spawn and request paths use; never re-derive for the report. "Configured" means stored AND different from what discovery derives (`router.written_ids`/`derived_configs`); a materialized copy reads as derived. No absolute paths in `engine`. A slot a later workstream fills is an explicit null until reported. `tests/contract/test_engine_contract.py` checks it through both routes.
- Audio input (`input_audio` parts) is gguf-only and must fail loudly on MLX, where audio towers are stripped at load. The 400 guard lives in `MLXProvider.create_chat_completion`.

## Thinking

- On the wire to llama-server, `ChatMessage.thinking` must be sent as `reasoning_content` (`_wire_message`). A trailing assistant message with reasoning and empty content resumes inside the open block; the prefill echo comes back on both channels, so `_continuation_echo_chars` returns a pair and the reasoning strip is sized lstripped. When content was prefilled too (a closed thought), the echo also carries the template's newline before `</think>`; `_stream_chunks` drops whitespace-only reasoning right after the echo in that case only (an open thought may resume with a newline).
- MLX resumes a thought by rendering a fresh generation prompt with thinking on and appending the trace after the family's opener (`_append_thinking_resume`); every routing parser takes `resumes_thinking` / `initial_thinking`. A content continuation resumes on the generation prompt plus the reply's text whenever that prompt extends what `continue_final_message` rendered before the text (`vlm_inputs.continue_from_generation_prompt`): gemma-4's history render drops the empty thought channel its generation prompt opens.
- History thinking on MLX goes to the template the way that template takes it (`vlm_inputs.thinking_for_template`, keyed on `ModelTemplateInfo.reads_reasoning_content`): `reasoning_content` where the template reads it, reconstructed `<think>` tags for a marker template, nothing for a family with neither. Never bake `<think>` text into a gemma prompt. On the VLM path mlx-vlm's `apply_chat_template(return_messages=True)` rebuilds messages as role + content only; `vlm_inputs.carry_message_extras` puts every other key back, or `reasoning_content` silently never arrives.
- Prompt preview is `provider.render_prompt()` behind `POST /v1/conversations/{id}/prompt`. It renders resident models only; a preview must never load a model.
- Thinking default cascade: request > heylook.toml `enable_thinking` > the thinking capability, passed as `thinking_capable=` by every caller. `MLXModelConfig.enable_thinking` is Optional (None = follow capability).
- `samplers.thinking_default()` and `samplers.sampler_defaults()` report the cascade's own answer and must never be re-derived. `sampler_defaults()` is one flat bag whose `enable_thinking` equals `thinking_default` by construction. `chat.js effectiveThinking` is the one frontend thinking resolver.
- Always pass an explicit `enable_thinking` bool to a template: an absent key means the template's own default. "Template references enable_thinking" is the thinking-capability signal.
- Thinking controls come from RENDERING the in-force template (`thinking_controls.detect`, on `engine.thinking`): the switch, the depth variable, and its values in the template's own spellings. No list of levels exists anywhere else (owner rule): the wire field `reasoning_effort` is a bounded word, sent under the template's OWN variable (`depth_variable`), whenever set and never gated on `enable_thinking` (gpt-oss has none). A value the model does not offer is a 400 before any stream (the route and the cascade both check; a stored default is dropped with a warning); the generate route and `samplerParams` drop it instead. Depth capability = a detected depth; gguf `thinking` = a detected switch. In MLX the depth kwarg must not ride `base_kwargs` (the TypeError retry must be able to strip it).

## Sampling

- The vendor sampling layer is the per-model answer and should normally win. MLX reads it from `generation_config.json` (`samplers.load_vendor_sampling`), gguf from the header's `general.sampling.*` (`gguf_metadata.vendor_sampling`); only temperature/top_p/top_k are read, so a publisher's documented min_p or repeat penalty reaches nothing automatically. The vendor layer sits directly above `GLOBAL_SAMPLER_FLOOR`, so heylook.toml and request fields still win. The floor stays small: `FALLBACK_TEMPERATURE`, `FALLBACK_TOP_P`, `DEFAULT_MAX_TOKENS`, and the `KNOBS_OFF` identity values. Keep `KNOBS_OFF`: engine defaults are not neutral.
- `capabilities._vendor_sampling_pairs` is the one cached entry point for the vendor layer; where each engine keeps it is that engine's describer's `vendor_sampling()` (called through the source module, never a from-import, so a patch on the reader reaches it); `test_vendor_layer_reaches_the_report_on_every_engine` pins it at the source readers. Header floats are rounded at the reader.
- Presets (`/v1/presets`) are the one named-bundle system, and they are the user's. They are client-expanded, so no server-side sampler layer exists. The old bundled sampler registry is gone; a request sending `sampler` or `preset` gets a 422 from the guard on `MessageCreateRequest`, pinned through the route in `test_messages.py`.
- The sampler panel's "overridden" test is `key in samplerParams(caps)`, never `cache[key] != null`.

## Image cost (plan W4)

- `image_plan.plan` behind `POST /v1/models/{id}/image-plan` reads an image's cost off the resident engine itself: MLX runs the loaded processor through `vlm_prepare_inputs` on a synthetic image per size (on a pinned executor, in `generation_stream`); gguf asks the running llama-server's `/v1/chat/completions/input_tokens`, which runs no vision encode. Never a hand copy of a family's resize arithmetic: the replica-and-rule-name design was dropped because the engine's own answer cannot drift from the engine.
- Resident only, 409 otherwise, like the prompt preview: planning never loads a model. It takes no generation gate (pricing must not queue behind a long run); an eviction mid-plan costs one failed badge, which the page shows as "cost unavailable".
- `target` is null on gguf (llama.cpp does not say what size it resized to), so the page offers no Fit there. The chat page sends two sizes per staged image in one call; `MAX_SIZES` and chat.js `MAX_ATTACH_IMAGES` are pinned together by `TestImagePlan` in `tests/contract/test_model_load.py`.
- The cost is disclosed, never a gate. Fit resizes to the `target` the engine reports for the image at its source size.

## Template overrides and ladders

- The operator template override (`chat_template_files.py`, `GET/PUT/DELETE /v1/admin/models/{id}/chat-template`) is one file, `chat_template.heylook.jinja`, in the model's folder, found by both engines' ladders and beaten only by an explicit `chat_template_path`/`chat_template_source`. It writes no config; revert is deleting the file. Never write through to the vendor's `chat_template.jinja`. `use_sidecar_chat_template=false` must not suppress it. The gguf origin phrase is the constant `HEYLOOK_OVERRIDE`. The routes must stay declared above admin_api's bare `/{model_id:path}`. Under force, `install_chat_template` targets the processor as well as the tokenizer. Validate before writing, in an environment mirroring the engines' (`raise_exception`/`strftime_now`/`tojson`). A write is live only after a reload; `stale` is null for an unloaded model, which is not false.
- Adding a rung to a ladder invalidates every hand-written subset of it. The MLX stop-less fallback walks `_AUTO_LADDER` minus the source that failed; never enumerate a subset of an ordered list defined elsewhere.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "Providers", "Thinking and sampling" and [#template-ladders](../../docs/architecture/sharp_edges.md#template-ladders).
