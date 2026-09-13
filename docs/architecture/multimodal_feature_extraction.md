# Multimodal feature extraction for Qwen3-VL and MiniMax M3

Status: proposed design, 2026-09-13. No extraction implementation or Mac runtime
validation is claimed by this document.

## Purpose and chosen backends

Expose a common feature-extraction API for two models already served through
heylook's provider architecture:

| Model | Runtime | Initial feature target |
|---|---|---|
| Qwen3-VL-32B | `MLXProvider`, using `mlx-vlm` | Per-token residual states after 50 decoder blocks, before the final model normalization |
| MiniMax M3, Unsloth UD-IQ3_XXS GGUF | `LlamaServerProvider`, using llama.cpp | Per-token final model embeddings, after the model's final RMSNorm, without additional vector normalization |

The intended consumer is a small experimental adapter mapping M3 features into
the conditioning space used by the open MiniMax H3 video generator. The server
extracts and describes features; adapter training and ComfyUI integration live
outside heylook. Feature extraction does not itself make M3 interchangeable with
H3's Qwen encoder or establish better generation quality.

Preserve the existing providers and public `/v1/hidden_states` route. Start with
one input sequence per request. Support text and images first. Native video
input, arbitrary M3 layer capture, pruning, and expert-offload implementations
are later work with separate acceptance criteria.

## Evidence and unresolved assumptions

Local source inspected for this design:

- [`hidden_states.py`](../../src/heylook_llm/hidden_states.py): the current
  factory is MLX-only. Extraction embeds token IDs and manually iterates decoder
  layers rather than entering the complete multimodal model path.
- [`hidden_states_api.py`](../../src/heylook_llm/hidden_states_api.py): the raw
  and structured routes already exist; the structured route assumes Qwen-style
  chat formatting and is not a general M3 input formatter.
- [`mlx_provider.py`](../../src/heylook_llm/providers/mlx_provider.py): the
  vision-generation path already calls `mlx_vlm.utils.prepare_inputs` and passes
  processor extras such as image grids to the model.
- [`llama_server_provider.py`](../../src/heylook_llm/providers/llama_server_provider.py):
  model lifecycle, `mmproj_path`, context/batch settings, native HTTP requests,
  and `extra_args` are already supported. Feature extraction is not implemented.

Upstream source inspected on the design date:

- [Qwen3-VL's MLX model](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl/qwen3_vl.py)
  exposes `hidden_state_at_layer(input_ids, layer, pixel_values, mask, **kwargs)`.
  It uses the model's visual input preparation and executes the decoder with
  `stop_after_layer=layer` and `apply_final_norm=False`.
- [Qwen3-VL's decoder](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl/language.py)
  handles multimodal positions and DeepStack visual injection. The native
  stopping argument counts completed blocks: `50` means after block index 49.
- [llama-server's API documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
  documents multimodal native embedding requests and unpooled token outputs
  with pooling `none`.
- [llama.cpp's M3 graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/minimax-m3.cpp)
  assigns the final RMSNorm output to `res->t_embd`. Its intermediate block
  output is named through the `l_out` graph callback.
- [llama.cpp's M3 vision implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/models/minimax-m3.cpp)
  exists upstream. This establishes a code path, not a successful run on the
  owner's Mac with the chosen Unsloth artifact/projector pair.

These upstream URLs move. At implementation time, record the exact commits
tested and check the actual Mac installation. The repository pins `mlx-vlm`
in [`pyproject.toml`](../../pyproject.toml); the presence of a method on current
upstream does not establish its presence in that pinned revision.

The following are implementation gates, not settled claims:

1. The chosen Unsloth checkpoint includes compatible MSA tensors and pairs
   correctly with the selected vision projector on the installed llama.cpp build.
2. Native M3 embedding requests produce every required sequence row, including
   visual rows, across microbatches while retaining causal MSA semantics.
3. Embedding-enabled serving and ordinary chat can coexist correctly on that
   build. If they cannot, use an explicit reload into extraction mode; do not
   create a second resident M3 process.
4. The complete M3 image-prefill request fits the Mac's available working set.
   File size alone is insufficient. Measure peak usage, cache state, context,
   microbatch size, and image size on the actual machine.

## API and feature semantics

Introduce typed request/result objects shared by the providers. An optional
provider operation, provisionally `extract_features(request)`, returns this
owned result type. Unsupported providers raise a typed unsupported-operation
error. The route must not inspect `provider.model` or require an MLX processor.

Extend `/v1/hidden_states` with a versioned request form. Retain existing
single-text callers through a compatibility adapter. The proposed shape below
is a design sketch, not a currently callable API:

```json
{
  "schema_version": 2,
  "model": "qwen3-vl-32b",
  "input": {
    "presentation": "raw",
    "content": [{"type": "text", "text": "A person opens a red umbrella."}]
  },
  "feature": {
    "kind": "decoder_state",
    "layers_completed": 50,
    "model_norm": "before_final"
  },
  "max_sequence_length": 4096,
  "encoding_format": "base64"
}
```

The example length is a reasoned starting limit, not an H3 or M3 requirement.
Resolve limits against the running backend and report overflow explicitly.

- `raw` accepts ordered content and bypasses chat wrapping. Images reuse the
  existing image-source schema, with the runtime producing native image tokens.
  It does not accept caller-manufactured visual token IDs as a substitute for
  actual image processing.
- `chat` accepts existing message/content types and uses the selected model's
  own template exactly once. Make raw content and messages mutually exclusive.
- For M3's first implementation, the supported feature is `final_embedding`.
  Reject requests for a particular intermediate layer or pre-final-norm output.
- Do not inherit the existing ambiguous default `layer=-2` into the new form.
  Legacy indices remain zero-based block indices; map index 49 explicitly to
  the native MLX `layers_completed=50` convention.
- Distinguish absent parameters from explicitly supplied values. The current
  equality checks against `-2` and `512` cannot distinguish those cases.
- Reject unsupported media before inference. Do not silently drop an image,
  sample video into unrelated images, or truncate a research capture.

Results must describe:

| Field | Meaning |
|---|---|
| Shape | Full sequence length and feature width, with batch handling explicit |
| Encoding | Actual serialized dtype, byte order, contiguous layout, and base64/JSON encoding |
| Computation dtype | Separately recorded when known; never substituted for wire dtype |
| Feature location | Completed decoder blocks or final embedding, and whether final model normalization was applied |
| Vector normalization | Whether additional L2 or other output normalization was applied |
| Sequence metadata | Native token IDs and row-to-text/media mapping when available, with unavailable fields explicit |
| Provenance | Checkpoint, quantization, projector, runtime/build identity, presentation, and effective preprocessing |
| Execution metadata | Effective length, cache policy, timings, and available memory telemetry |

Metadata must describe the exact forward that produced the tensor. A separate
text tokenizer pass cannot establish the expanded visual sequence. If llama's
HTTP API cannot supply reliable visual spans, advertise that limitation; obtain
them from a native runtime extension before training with visual-row alignment.
Never fabricate token IDs for visual embeddings.

Initially serialize explicit little-endian float32 in base64 for both engines,
matching the existing encoder's wire precision. Reuse the serializer and correct
its metadata. A future compact tensor format is independent of extraction.

The existing API accepts a list of texts but returns only the first result.
For this implementation, reject multiple inputs clearly rather than silently
discarding outputs. True batched extraction can be designed separately.

## Qwen implementation through MLX-VLM

Add the provider operation in `mlx_provider.py`, with extraction logic in a
focused helper if needed. Share media resolution and processor preparation with
the existing VLM path, without forcing chat formatting onto raw requests.

1. Resolve inputs through the model's processor, retaining attention masks,
   image/video grids, and every model-specific input required by the native path.
2. Call `model.hidden_state_at_layer(..., layer=50, ...)` for the H3 target.
   Do not manually iterate decoder layers or reconstruct DeepStack/MRoPE logic.
3. Execute and serialize the tensor before releasing request-owned resources.
4. Use the process-global generation gate, `router.pin_model()`, the existing
   pinned executor pool, MLX generation stream, and wired-limit management.
   The current synchronous extraction call inside an async route is not the
   execution model to retain.
5. Do not mutate the loaded chat model's layer list or remove its output head.
   Native early stopping avoids running later blocks for extraction while
   preserving the model's ordinary generation behavior.

If the pinned MLX revision lacks the method, prefer a verified upstream pin
update through the existing dependency workflow. A missing method must fail
clearly, not silently fall back to the generic text-only extractor.

### H3-specific input ownership

H3 consumes a particular presentation, not a normal Qwen chat transcript. Its
caller owns raw prompt/reference ordering, release-tokenizer configuration,
reference sizing, and H3 modality tags. Keep those rules in the H3 integration
instead of copying a second prompting manual into heylook.

Before claiming an H3-compatible tensor, establish parity of tokenizer IDs,
image preprocessing, visual-row ordering, layer selection, and final-norm
behavior with the existing H3 encoder. A stock Qwen chat processor with an
otherwise correct layer capture is not sufficient evidence of that parity.

The server may return generic text/visual spans; the H3 caller derives its own
DiT tags from verified spans. Audio conditioning is outside this initial API.

## M3 implementation through llama-server

Add the provider operation in `llama_server_provider.py`. Use its existing
subprocess and HTTP lifecycle; no Python llama binding or new provider type is
required for the initial final-embedding path.

During the backend spike, use existing `extra_args` to verify embedding mode
and pooling `none`. After validation, expose supported settings as typed
`GGUFModelConfig` fields with reload metadata and include them in argv tests.
Keep flash attention/MSA enabled and report a dense fallback as incompatible
with the selected extraction profile. Validate causal behavior explicitly.

Use llama-server's native multimodal embedding payload and endpoint. The chat
payload cannot be forwarded unchanged: raw content/media markers and the
embedding request's media representation must follow the installed API. In
chat presentation, apply the native template once and preserve image order.

Collect all returned vectors in input-sequence order. Verify output behavior
for multiple microbatches and repeated requests; a final microbatch or cached
suffix is not a complete capture. Start with request-isolated extraction state
and no reuse of chat-prefix cache. Discard extraction state after completion
or cancellation using the backend's supported lifecycle.

M3's feature profile is deliberately different from Qwen's target:

| Property | Qwen H3 target | M3 initial source |
|---|---|---|
| Feature width | 5120 | 6144, verified against loaded metadata |
| Layer location | After 50 decoder blocks | Final decoder output after model RMSNorm |
| Additional vector normalization | None | None |
| Sequence alignment | Qwen tokens and visual layout | M3 tokens and visual layout |

The adapter can learn between these representations. Equal sequence length or
matching token index does not imply semantic alignment across tokenizers.

### Conditional native extension

If final embeddings work, no llama.cpp graph patch is required for the first
experiment. If arbitrary M3 layers, pre-final-norm states, or missing sequence
metadata become necessary, implement a bounded native extraction extension.
Candidate seam: the existing `l_out` graph callback after a decoder block.

That extension must preserve selected tensors until copied, track sequence and
microbatch row ordering, request all required output rows, avoid retaining
every layer, and expose a versioned capability. Use the same loaded model.
A backend evaluation callback or dedicated extraction output is a design choice
to resolve against the then-current llama.cpp API, not a working implementation
assumed here. Keep experimental builds explicit and follow heylook's canonical
build policy; do not silently patch a second llama-server binary.

## File-level implementation map

| Existing file | Work |
|---|---|
| `src/heylook_llm/hidden_states.py` | Typed extraction contracts, legacy request adapter, provider dispatch, serialization, explicit multi-input rejection |
| `src/heylook_llm/hidden_states_api.py` | Validate the versioned request, expose backend capabilities/errors, await managed execution |
| `src/heylook_llm/providers/base.py` | Optional extraction operation and supported-feature description |
| `src/heylook_llm/providers/mlx_provider.py` | Native Qwen multimodal extraction using the existing worker/stream lifecycle |
| `src/heylook_llm/providers/common/vlm_inputs.py` | Reuse/factor ordered media preparation without coupling raw extraction to chat formatting |
| `src/heylook_llm/providers/llama_server_provider.py` | Native embedding request construction, subprocess configuration, result conversion and capability checks |
| `src/heylook_llm/config.py` | Verified GGUF embedding/pooling settings with reload semantics |
| `src/heylook_llm/capabilities.py` | Advertise feature kinds/media support from the same provider decisions used at execution |
| `pyproject.toml`, `uv.lock` | Update MLX pin only if the required upstream API is missing; record tested revision |
| `docs/api_integration.md` | Document the implemented wire contract and limitations after implementation |

Keep the legacy structured hidden-state route compatible for existing callers.
Do not route M3 through its Qwen-specific formatter. No frontend or model
registry redesign is required to begin this work.

## Validation and implementation order

### 0. Prove the backend surfaces on the Mac

Before building a generalized capture system, verify the installed Qwen native
method and run M3 native embeddings with one text and one image request. Record
the actual builds, checkpoint/projector identities, feature widths, sequence
lengths, and peak memory. Check embedding/chat coexistence and all-token output.
If M3's stock endpoint cannot provide correct outputs, resolve that backend gap
before presenting it as an available feature.

### 1. Implement the shared API and Qwen extraction

Add focused tests for legacy compatibility, index translation, explicit-default
handling, unsupported-feature errors, multi-input rejection, and serialization
round trips. Validate request execution on the real pinned worker thread.

Compare Qwen captures against the current H3 encoder on text, one image, and
multiple images. Compare prepared inputs before comparing numerical outputs.
Match precision where possible; otherwise establish quantization/runtime error
separately and do not demand byte equality or invent a universal tolerance.

### 2. Implement M3 final-token extraction

Test media ordering and native payload construction, startup options, response
conversion, and failure propagation. Live-check complete rows across microbatch
boundaries, repeated requests, changed image inputs, cancellation, and a
subsequent ordinary chat request. Use explicit extraction-cache state.

Do not equate successful JSON or the correct tensor shape with valid features.
Where possible compare against the same runtime's ordinary causal forward.
For non-final layers, compare against a native reference capture before exposing
the feature publicly.

### 3. Hand off a small capture pilot

Capture a handful of reproducible paired examples first. A reasoned initial
pilot is 8–16 examples with separate untouched examples; it is not a sample-size
claim or a requirement to collect hundreds of prompts. The pilot establishes
plumbing and limited fit behavior, not generalization or superiority over Qwen.

Run the models sequentially, persist selected features between runs, and train
only the adapter outside the server. Store research captures in an explicitly
selected artifact location, not ordinary request logs. The existing observability
policy continues to exclude prompts, token IDs, and tensors from routine logs.

### 4. Expand only against an observed need

Native video needs an explicit ordered-frame/timestamp contract and validation
against each runtime's video processor. Arbitrary M3 layer capture needs the
native extension above. Larger datasets and compression experiments require
their own evidence and are not prerequisites for the first extraction test.

## Completion criteria

The first release is complete when one public API can return verified Qwen H3
target features and M3 final-token features for text and images, with truthful
metadata, bounded memory, correct lifecycle handling, and explicit unsupported
cases. Runtime and numerical evidence must identify the tested Mac, builds,
artifacts, and input conditions. No claim about adapter quality is part of this
server feature's completion criteria.
