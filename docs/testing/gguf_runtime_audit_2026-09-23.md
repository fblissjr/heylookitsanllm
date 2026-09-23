# gguf runtime audit: build, spawn, vision, thinking, prompt cache

**A dated record, not a living document.** Investigated on 2026-09-23, against
heylook `2edeb2a` (v2.0.64) and llama.cpp `4e416ee73` (build 11138, the
canonical build from `scripts/build_llama.py --rev master`), on an M2 Ultra
with 192 GB running macOS 27.0 and Xcode 27.0. The llama.cpp source read was
`42916d83f`; `git log 4e416ee73..42916d83f -- ggml/ tools/` touches nothing
this record depends on.

**No performance figures here, by design** (CLAUDE.md, and the wiki's
principle 6). This record carries mechanisms, method and findings stated as
relationships. The measured data is a local JSON file with its conditions
attached (commit, argv, model file identity, template hash, sampling, per-request
image dimensions): `internal/claude/perf/gguf_runtime_2026-09-23.json`, with the
harness that produced it beside it. Re-run the harness before trusting a
relationship on a different build.

Claims are marked **[measured]** (live llama-server, harness below),
**[source]** (read in the llama.cpp or heylook source, with the site) or
**[reported]** (a research pass with external citations, not re-derived here).

## Method

The harness spawns llama-server through heylook's own `LlamaServerProvider`,
so the argv, the template ladder and the automatic micro-batch are the
production ones, and builds every request with the provider's own
`_build_payload`, so the sampler cascade and wire shape are the production
ones too. Only the HTTP read is the harness's own, so it keeps llama-server's
raw `timings` object (prompt tokens, cached tokens, prompt and decode rates).
llama-server runs at `-lv 4` because the vision-encode boundary and every
checkpoint event are trace-level: vision encode time is the gap between the
`encoding mtmd batch` line and the next `decoding image batch` line.

- "Cold" requests carry a nonce at the head of the system prompt, which
  defeats every cache and makes each measurement a full prefill.
- Sampling is the model's own vendor defaults, never temperature 0, with a
  fixed seed per request.
- The request never goes through heylook's HTTP layer, so these results say
  nothing about that layer's overhead.

## 1. Build flags and toolchain

**Nothing to change.** [source]

- **Metal shaders.** `GGML_METAL_EMBED_LIBRARY=ON` embeds the Metal source in
  the binary and the OS compiles it at startup (`newLibraryWithSource`). The
  shader compiler therefore tracks the installed macOS, not Xcode; updating
  macOS is what changes shader code generation.
  - The missing Xcode "Metal Toolchain" component is irrelevant. It is needed
    only for the non-embedded path (`EMBED=OFF`, `GGML_METAL_SHADER_DEBUG`),
    which is not faster.
  - Every other `GGML_METAL_*` build option applies only to that non-embedded
    path.
- **bf16.** Detected at runtime and on for apple8; `GGML_METAL_USE_BF16` was
  removed upstream (#15995) and must not be re-added.
- **Metal tensor API / "neural accelerators".** Gated to M5 and A19 or newer
  by GPU name. Upstream's comment records the M2 Ultra as slower with it
  forced on (`GGML_METAL_TENSOR_ENABLE=1`).
- **CPU flags.** `GGML_NATIVE` reaches only the `ggml-cpu` target. The
  compiled-in options for CPU-resident work (`GGML_LLAMAFILE`,
  `GGML_CPU_REPACK`, KleidiAI) are inert when every layer is on Metal.
  - The M2 has no SME, so KleidiAI's SME kernels do not apply.
  - OpenMP and LTO only affect CPU-side glue.
  - Which options were CUDA/x86-only: none of the ones discussed here. They are
    all Apple-relevant, and all are either already right or inert at `-ngl 999`.
- **BLAS (Accelerate).** Gets no work at `-ngl 999`: it accepts host buffers
  only, and offloaded weights live in Metal buffers.
  - One edge: if DENSE weights are ever placed on the CPU (partial offload, or
    an `override_tensor` of dense tensors), BLAS claims their prefill matmuls
    ahead of Metal's offload path. MoE expert offload is unaffected, because
    `MUL_MAT_ID` is not a BLAS op.
- **Flash-attention tuning.** The Metal flash-attention decode tuning table
  was re-keyed from SKU to GPU family on the day of this record (#29075). The
  M2 Ultra no longer has rows of its own, and uses rows pooled from the other
  M2 chips. Unmeasured.
- **Toolchain state.** No OS or Xcode updates pending; cmake, ninja and ccache
  current. After an Xcode or macOS update, `scripts/build_llama.py --rebuild`
  rebuilds the same commit with the new toolchain.

## 2. Neural Engine and CPU+GPU: what could use the rest of the chip [reported]

On this machine the GPU is the only engine that matters for these models.
Memory bandwidth is the limit, not idle silicon.

**Could help:**

- Keeping weights resident between requests. See §8: small effect measured
  on a mid-size model.
- Raising `iogpu.wired_limit_mb`, which gives headroom, not speed. It is
  already `scripts/gpu_wired_limit.sh`.
- Continuous batching (`-np` > 1), under genuinely concurrent traffic only.

**Does not apply here:**

- **ANE for LLMs.** Published ANE LLM work tops out around 8B and loses to the
  GPU at every size tested. The one study measured on an M2 Ultra found the GPU
  several times faster on dense kernels.
- **MLX.** Has no ANE backend, and the maintainers declined it (closed API).
- **ANE for the vision encoder.** An mtmd Core ML backend exists only as a
  draft upstream PR covering two older projectors.
- **ANE-hosted drafters.** Published only on multi-device research rigs.
- **MoE expert offload and CPU-side drafters.** On unified memory these add no
  capacity and compete for the same bandwidth. Expert offload is only a
  working-set lever.
- **M5-only accelerators.** The Metal tensor API and "neural accelerators" do
  not exist on M2.

## 3. Spawn flags: defaults against what heylook passes [source]

These are the current llama-server defaults and heylook's choice for each.
Everything not listed is a default heylook inherits unchanged and should
keep.

| Flag | llama-server default | heylook | Verdict |
|---|---|---|---|
| `-fa` | `auto`: probes whether the fused op lands on the layer's device, and on Metal resolves ON for these head dims. The vision tower inherits it. | nothing | Keep auto. §9: forcing it off is slower. |
| `-c` | 0 = training context, which `--fit` then shrinks to fit memory | `ctx_size` when set | Keep. |
| `-b` / `-ub` | 2048 / 512 | `-ub 2048` when working-set headroom allows | Keep. §7 has the one hard constraint. |
| `-np` | auto (4 slots, unified KV) | 1 | Keep. A single serialized user gains nothing from more slots. |
| `--cache-prompt` | on | nothing | Keep, and never send it. |
| `--cache-reuse` | 0 | nothing | Dead for every vision model regardless: forced to 0 whenever an mmproj loads (`server-context.cpp`, "cache_reuse is not supported by multimodal"). |
| `-cram` | 8192 MiB | `cache_ram_mb` when set | See §5. |
| `-ctxcp` / `-cms` | 32 checkpoints, 8192 min spacing | nothing | Keep. |
| `--swa-full` | off | nothing | Keep off. It sizes the SWA cache to the whole context. |
| `--context-shift` | off (forced off with an mmproj) | nothing | n/a |
| `-lm` | auto = mmap | `load_mode` when set | Keep. |
| `--mmproj-offload` | on (vision tower on Metal) | nothing | Keep. |
| `--image-min/max-tokens` | per projector (§4) | nothing | The one vision lever on the server side. §4. |
| `--reasoning-preserve` | on (since b10763) | nothing | Keep. It keeps past reasoning in history, which keeps the prompt prefix-stable. |
| `--jinja`, `--reasoning-format` | on, `deepseek` | nothing | Keep. |

## 4. Vision cost and image geometry

**Cost model [measured].** On a dense model with a separate vision tower
(Qwen3.8-27B, `qwen3vl_merger`), time to first token for an image request is:

    image decode (CPU) + vision-tower encode + LLM prefill of the image tokens + text prefill

- Vision encode grows with the image's token count, slightly faster than
  linearly at the top of the range, because the tower's attention is
  quadratic in patches.
- Everything after the encode is ordinary prefill.
- Across 0 to 10 images of one size, TTFT is linear in the total image-token
  count.
- Several images in one request are encoded one after another. For this
  projector there is no batching across images, and each image costs the same
  as it would alone.
- **Decode speed does not depend on how many images are in context**, beyond
  the ordinary slow-down from a deeper KV cache.

**Images above the cap are resized down before encoding [measured].** Every
projector caps the tokens per image.

- On Qwen via llama.cpp the cap is 4096 tokens. A 5504x3072 image and a
  2752x1536 image produce the same token count and the same encode time. The
  only extra cost of the larger file is the CPU decode of its bytes.
- The client-side resize is therefore pure upload and decode savings above
  the cap. Below the cap it is the direct lever on TTFT.

**Image geometry is deterministic and knowable in advance [source; token counts
cross-checked live for Qwen].** Each engine maps WxH plus a few per-model values
to a resized size and a token count. Unit px is the side length of one output
token (patch size x merge).

| Model family / engine | Unit px | Tokens per image | Resize | Aspect handling |
|---|---|---|---|---|
| Qwen3.8 (`qwen3vl_merger`), llama.cpp | 32 | 8-4096 | smart_resize to multiples of 32 | pads (black bars); rounds .5 up |
| Qwen3.5 / Qwen3-VL, MLX | 32 | 64-16384 (processor config) | same rule | stretches; rounds .5 to even; refuses beyond 200:1 |
| DeepSeek-V4-Flash-Vision (`deepseek4v`) | 42 | at most 384, including row separators | largest size in budget | pads grey; only W/H is capped, at 8:1 |
| MiniMax-M3 | 28 | 8-576 | smart_resize to 28 | stretches |
| Muse-Glimmer | 28 | 1-4096 | aspect-closest grid, never upscales | stretches (Lanczos) |
| gemma-4 26B/31B (`gemma4v`), llama.cpp, reference | 48 | 70-1120 | smart_resize to 48 | pads |
| gemma-4, MLX | 48 | always about 280 | fills a fixed budget, upscales small images | stretches |

- **The two engines disagree on the same family**, so geometry has to be
  reported per engine and per model, never per family.
  - Qwen caps at 4096 tokens on llama.cpp and 16384 on MLX.
  - Qwen rounds differently at half a unit. 720 px is exactly such a side:
    1280x720 comes out 1280x736 padded on llama.cpp and 1280x704 stretched on
    MLX.
  - gemma-4 on MLX always spends about 280 tokens; on llama.cpp it follows the
    image's size.
- **The limits live in llama.cpp code, not in the mmproj header.** The unit
  (patch, merge) is in the header; min/max tokens, pad-vs-stretch, and the
  DeepSeek budget and ratio cap are hard-coded per projector. A table like the
  one above, copied into heylook, would drift. It is recorded here as a dated
  fact, not as a source to code against.
- **llama-server can count for us without encoding.**
  `POST /v1/chat/completions/input_tokens` tokenizes with the engine's real
  preprocessing and uses a placeholder bitmap, so no vision encode runs. It
  returns the total prompt tokens, so subtract a no-image baseline to get the
  image's share.
  - A solid-colour PNG of any WxH is a few hundred bytes, which makes probing
    a hypothetical size cheap.
  - It answers the count only, not the resized dimensions.
  - The DeepSeek count includes a 0-3 token lead pad that depends on the text
    before the image.
- **The ideal client-side size is exactly the size the engine would pick**:
  multiples of the unit, token count between the min and the max. The server
  then copies the image as-is instead of resampling it a second time, and on
  llama.cpp there are no pad bars.
  - The frontend's fixed long-edge cap sends several times more pixels than
    DeepSeek, MiniMax and MLX gemma keep.
  - It sends fewer than llama.cpp Qwen allows.

## 5. Prompt cache across requests [measured + source]

**How llama-server reuses a prompt (`-np 1`).**
- The single slot keeps the last request's tokens (prompt plus reply). A new
  request reuses the longest common prefix.
- An image chunk matches only if its SHA-256 and token count match. Images
  before the match point are never re-encoded; every history image is still
  decoded and hashed on the CPU each request.
- A request that matches nothing in the slot can restore a better-matching
  earlier state from the host-RAM prompt cache (`-cram`), which is how
  alternating conversations or apps keep their prefixes.
- **Full-attention models** truncate at the divergence point and prefill the
  suffix.
- **SWA and hybrid/recurrent models** (every served vision model here: Qwen3.8
  is hybrid, DeepSeek-V4 and Muse-Glimmer are SWA) cannot roll back to an
  arbitrary position. They restore the latest **context checkpoint** at or
  before the divergence. Checkpoints are taken only during prompt processing:
  - at the start of the last user message;
  - at user-message starts spaced by the minimum spacing;
  - 4 + n_ubatch tokens and 4 tokens before the prompt's end;
  - never in a batch that has just decoded an image.
- `cache_reuse` (KV shifting) is disabled whenever an mmproj is loaded, and
  context shift is too.

**Consumer-app shape: a fixed system prompt, varying images and questions.**
[measured on all three models]
- The system prompt is reused on every request, and only the new image and
  question are processed.
- **It survives interleaving.** After a request from a different app (a
  different system prompt), returning to the first app reused its whole
  system prompt, restored from the RAM prompt cache.
- So "pre-warming" a fixed system prompt happens by itself after its first
  use. It does not survive a process restart or unload, because the RAM
  cache lives in the process.

**Multi-turn with images and thinking.** [measured]
- **Muse-Glimmer and DeepSeek-V4-Vision: correct.** Each turn processes only
  what is new: the new image and question, plus the previous visible answer
  where the template re-renders it differently from how it was generated.
  - DeepSeek's template drops past reasoning from history.
  - Muse-Glimmer keeps it.
  - A text-only follow-up processes a handful of tokens.
- **Qwen3.8-27B (unsloth), as found: broken by its template.** Every turn:
  - restored the checkpoint at the start of the *previous* user message;
  - re-encoded that turn's image;
  - re-processed the entire previous reply, thinking included.

  A text-only follow-up paid a full image encode and more than a thousand
  tokens.
- **Diagnosis** (slot-debug token diff, then `/apply-template`):
  - The hand-placed template's `{# ... #}` comment lines carry no whitespace
    control. They leak blank lines before the system turn, after it, and
    before the generation prompt: `<|im_end|>\n\n<|im_start|>assistant`.
  - Replayed as history, the same turn renders `<|im_end|>\n<|im_start|>`.
  - So each new prompt diverges from the cache 5 tokens before the previous
    prompt's end, one position before the "4 before end" checkpoint. The
    server can only fall back to the user-message-start checkpoint, which
    precedes the image.
- **Fix, verified live.** Whitespace-stripping comments render the canonical
  Qwen format and make the history prefix-stable (§10). Re-running the same
  multi-turn conversations with the fixed template:
  - every turn processed only the new image and question, with no re-encode of
    earlier images, so a new-image turn's TTFT dropped to roughly a
    single-image request's;
  - a text-only follow-up processed a few dozen tokens;
  - a second conversation opening with the same system prompt and first image
    was restored from the RAM cache on its first turn.
  This held with thinking on and off.

**What forces a full re-process on these models.**
- Any change near the top of the prompt: system prompt, preset, or an early
  edited message.
- On Qwen3.8, **changing the thinking depth**, because the depth instruction
  is written into the system message.
- Different image bytes for the same image.
- A divergence earlier than every usable checkpoint (for edits, roughly at
  checkpoint-spacing granularity).
- A state evicted from, or never admitted to, the RAM cache.
- A reload, sleep, or heylook LRU eviction.

**The RAM prompt-cache budget (`-cram`, default 8192 MiB).**
- An entry is the slot's whole state: KV, plus recurrent/SWA state, plus every
  checkpoint. An entry larger than the budget is **skipped, not stored**, which
  is logged only as a warning.
- On Qwen3.8 the per-entry cost is a fixed recurrent-state amount per
  checkpoint plus a small KV amount per token. The data file has the sizes
  read off the trace log.
- One conversation near the configured context fits in the default. Several
  such conversations, or a larger KV geometry, will not.
- The budget is host RAM, not Metal working set, so on this machine a
  mid-size model can afford a much larger one. A model near the working-set
  ceiling cannot.

**What the product shows of any of this: nothing.** heylook captures the
cached-token count per request and drops it before any client, and
llama-server reports only a final cached count. Whether the prefix came from
the slot, the RAM cache or a checkpoint, and whether a state was skipped for
size, is visible only in trace-level logs, which heylook discards by default.
See the follow-up plan.

## 6. Thinking depth: what each template actually accepts

**Controls [source: every in-force template rendered with a broad candidate
set, grouped by output].**

| Template (in force) | Thinking switch | Depth variable | Values that change the prompt | Default when absent | Unknown value |
|---|---|---|---|---|---|
| Qwen3.8 official (embedded, e.g. the uncensored 27B) | `enable_thinking` | `reasoning_effort` | low, medium, xhigh | xhigh | raises (llama-server returns 500); `high` raises too |
| Qwen3.8 unsloth-patched (embedded; Flash-Next) | `enable_thinking` | `reasoning_effort` | low, medium, high = xhigh | xhigh | raises |
| Qwen3.8 hand-placed sidecar (unsloth 27B, **in force there**) | `enable_thinking`, or effort off/none | `reasoning_effort` | low, medium, high, xhigh, plus aliases | medium (no instruction line) | raises |
| DeepSeek-V4-Flash, unsloth (text and vision) | `enable_thinking`, default off | `reasoning_effort`, applied only with thinking on | high, max | none | silently ignored; low, medium and xhigh do nothing |
| DeepSeek-V4-Flash, ggml-org | same | same | max only | none | silently ignored |
| Muse-Glimmer | not read | `reasoning_strength` | any string, pasted in as "Reasoning strength: X." | high | accepted verbatim |
| MiniMax-M3 | not read | `thinking_mode` | enabled, disabled, adaptive | adaptive | accepted |
| gpt-oss (MLX) | not read | `reasoning_effort` | any string | medium | accepted verbatim |

**Consequences for the UI today.** The depth dropdown is a fixed list (low,
medium, high, xhigh), and the wire `Literal` is the same four values.

- `high` 500s on the official Qwen3.8 template.
- DeepSeek's strongest setting, `max`, cannot be sent at all, and three of the
  four listed values do nothing on it.
- Muse-Glimmer's and MiniMax-M3's depth controls are unreachable.
- Muse-Glimmer shows a thinking toggle that its template never reads.
- The "auto" label cannot name the default, because the default is not
  reported anywhere.

**Detection works mechanically.** Render the in-force template once per
candidate value (the template's own string literals plus a small fixed set)
and group the values by output. This recovers the accepted values, the
aliases, the default (whichever group matches the absent render) and the
strictness (does a garbage value raise, get ignored, or get pasted in). It
matched a by-eye reading of every template.

Pitfalls:
- Depth is often applied only when thinking is on, so render with thinking on.
- Some templates raise without a user message.
- `strftime_now` makes output date-dependent: cache the groupings, never the
  hashes.
- Whether Python jinja matches llama.cpp's engine on every gguf template is
  unverified. `/apply-template` on a loaded model is the cross-check.

**What depth costs [measured].**
- Time to first token does not depend on the thinking level. The level changes
  a line in the system prompt and nothing else before generation.
- Decode speed does not depend on it either.
- The whole difference is how many thinking tokens the model chooses to
  produce, and **that varies more between two runs of the same prompt than
  between adjacent levels.** Two requests with byte-identical prompts (absent
  vs medium on the sidecar, which render the same) differed several-fold in
  thinking length. A single run per level says nothing. See the repeats in
  the data file.

## 7. Latent crash: non-causal image decode with a small micro-batch [source]

- Some projectors decode an image's tokens with non-causal attention:
  `gemma4v` except E2B/E4B, `gemma4uv`, `gemma3` and `deepseek4v`
  (`mtmd_decode_use_non_causal`).
- `llama_context::decode` asserts `n_ubatch >= n_tokens` for non-causal
  batches, but the image is split by `n_batch`, not `n_ubatch`, and nothing
  sizes `n_ubatch` to the image. An image with more tokens than the
  micro-batch therefore aborts the process with `GGML_ASSERT`.
- heylook's auto micro-batch spawns 2048 when headroom allows and otherwise
  inherits llama-server's 512.
  - DeepSeek-V4-Vision caps an image at 384 tokens, so it is safe at 512.
  - gemma3 is a fixed 256 tokens, so it is safe.
  - gemma-4 26B/31B on llama.cpp allows up to 1120, so it would abort at 512
    on any image above roughly 1.2 megapixels.
  - No gemma-4 gguf is served today.
- The durable fix is a spawn-time rule. When the projector decodes
  non-causally, spawn with `--image-max-tokens` no larger than the
  micro-batch, or raise the micro-batch to the projector's maximum. Either
  way it has to be derived from the projector, not listed.

## 8. Residency after idle [measured]

- Metal keeps the model's buffers resident with a heartbeat that stops after
  180 s without GPU work (`GGML_METAL_RESIDENCY_KEEP_ALIVE_S`). This is
  llama.cpp's Metal residency keep-alive, not a model unload; heylook's idle
  unload is a separate setting.
- On the two ~30 GB models, a fully cached request after 240 s idle had a
  slightly longer first token than an immediate repeat, and an unchanged
  decode rate.
- On the 145 GB DeepSeek-V4-Vision, **the first request after the idle gap
  paid a first-token delay many times the warm one**, with an unchanged decode
  rate. That is the cost a sporadic consumer app pays on every request that
  follows a quiet spell.
- **Raising the keep-alive removes it.** The same model, spawned with
  `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600` and left idle for the same gap
  twice over, answered its first request after each gap as fast as a warm
  repeat. The cost is weights that stay wired between requests, which is
  already the point of a resident model. This is the evidence behind the
  plan's W9.

## 9. Flash attention on vs off [measured]

Forcing `-fa off` against the default (auto, which resolves on):
- The **vision encode is markedly slower**. The tower inherits the flag.
- TTFT for image requests is modestly slower.
- Decode is slightly slower at a 10-14k-token context.
- Text-only prefill is unchanged.

Nothing measured favours off. It is worth being able to set it per model to
test a new architecture, but not as a default.

## 10. Template copies: why there are several, and what happened here

**The unsloth Qwen3.8-27B folder holds two templates, and the hand-placed one
wins.**

- The GGUF embeds unsloth's patched Qwen template.
- A `chat_template.jinja` beside it, written by hand on 2026-08-30 with no
  download record and no counterpart in the publisher's repo, is the
  sidecar rung of the gguf ladder, so it is what loads.
- It is Qwen's official template plus a custom effort block. It changes the
  effort default from xhigh to "no instruction", adds a `high` level, and
  drops unsloth's fixes: it raises on a `developer` role and on two leading
  system messages, which the embedded one renders.
- It also carries the whitespace defect in §5, which the embedded template
  does not have.

**Resolved the same day.** The whitespace-fixed body (every `{#`/`#}` made
whitespace-stripping, content otherwise unchanged) was written through
`chat_template_files.write_override` as that model's
`chat_template.heylook.jinja`. The hand-placed file was renamed to an inert
`.bak`, because leaving it would let the sidecar rung silently restore the
defect if the override were ever removed. A lint of every template source on
disk found no other template with the defect: every GGUF-embedded template,
every MLX sidecar, and the one existing override.

**Why copies multiply.**
- Each publisher embeds its own template in the GGUF.
- A hand-placed file beside the weights looks exactly like a publisher's
  sidecar.
- MLX has three possible homes: `tokenizer_config.json`,
  `chat_template.jinja` and `chat_template.json`.
- The operator override is one more file.
- Nothing shows provenance: the spawn log names the winning rung, but not
  whether that file was downloaded or hand-written, or that other copies
  exist and differ.
- Five MLX gemma `chat_template.jinja` files were also edited in place after
  download, before the override file existed; a re-download reverts them
  silently.

**Prevention**, smallest first:
1. **Provenance on the template view.** Show every copy present with its
   origin: downloaded (repo@commit, from the download metadata), modified
   since download, hand-placed, or embedded. Put the in-force body's hash in
   the spawn log line.
2. **A prefix-stability lint** (§5) wherever a template is accepted or
   discovered. It is a two-render string comparison and would have caught the
   defect that cost this model its multi-turn cache.
3. **Detected thinking controls** (§6) shown beside the in-force template.
4. Hand edits belong in `chat_template.heylook.jinja`, the override file,
   which survives a re-download and is visible as an override.

## 11. Cross-model summary [measured]

| | Qwen3.8-27B (dense hybrid) | Muse-Glimmer-30B (SWA) | DeepSeek-V4-Flash-Vision (SWA MoE) |
|---|---|---|---|
| Image tokens vs size | grows with pixels up to 4096 | grows with pixels up to 4096 | capped near 384 regardless of size |
| TTFT vs image size | grows with pixels up to the cap | same, slower encoder per token | near-flat |
| Several images in one request | linear | linear | linear |
| System-prompt reuse across requests and apps | yes | yes | yes |
| Multi-turn reuse | broken by template; correct after the fix | yes | yes |
| Decode vs images in context | flat | flat | flat |
| Depth levels separate thinking length | only low vs the rest | only low vs the rest | no reliable ordering |
| Idle residency cost (first request after a gap past the keep-alive) | small | small | **large enough to matter** |

The "levels" row is the most important caveat for any UI. Beyond "off" versus
"on" and a shortened "low", no model here showed a depth setting that reliably
changes how long it thinks. Run-to-run variation at a fixed level is larger
than the difference between adjacent levels.

## Follow-up

Everything this record found that should change in the product is planned in
[project/plan_runtime_visibility.md](../project/plan_runtime_visibility.md).
