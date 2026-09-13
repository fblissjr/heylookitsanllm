# H3 conditioning through the resident Qwen3-VL model

Status: revised proposal, 2026-09-13, following the owner's supplied review from
Claude working on the Mac installation. No implementation or live validation is
claimed here. This replaces the earlier shared Qwen/M3 feature-API proposal.

## Decision and scope

Expose a narrow H3 conditioning operation through heylook's resident Qwen3-VL-32B
`MLXProvider`, using the H3 presentation builders and `MiniMaxH3Conditioner`
already shipped in the pinned mlx-vlm dependency. Return conditioning states
**and `token_tags`**. Preserve normal chat behavior on the same loaded model.

The value of heylook is ownership and scheduling of one resident encoder, usable
by ComfyUI or an offline capture client. A ComfyUI process on another machine
can request conditioning from the Mac. If ComfyUI runs on the Mac and can import
mlx-vlm directly, direct conditioning is another integration option, but an
import alone does not share a model instance held in heylook's process.

The current editing environment is the Linux ComfyUI checkout. Whether the
owner also runs ComfyUI with mlx-vlm on the Mac is unverified and does not block
designing the shared-resident-model route.

Defer M3 visual feature extraction as a separate project. Do not introduce a
generic schema-version-2 feature API or change the GGUF provider for this first
release. Preserve the legacy hidden-state routes and address their confirmed
execution/correctness defects without a broad API redesign.

## Evidence and attribution

The owner supplied a review reporting inspection of the actual pinned mlx-vlm,
canonical llama.cpp build, and Mac model directories:

- The pin includes `qwen3_vl.hidden_state_at_layer`, plus
  `models/minimax_h3/` with `MiniMaxH3Conditioner`, fl2va/ref2va presentation
  builders, H3-specific image processing, and the layer-50 constant.
- H3 presentation is built at the token-ID level, with special-token insertion
  disabled during tokenization and image pads derived from H3 image processing.
  Heylook's generic chat image path is not a substitute.
- The target is the 5120-wide state after 50 of Qwen's 64 decoder blocks,
  before final normalization. The local Qwen tokenizer's added tokens agree
  with the release; the local checkpoint uses 8-bit affine quantization whereas
  the reference weights are bf16.
- The installed M3 GGUF has no paired projector and is currently text-only for
  heylook. The reviewed Unsloth repository did not supply one. The canonical
  converter has a vision implementation, so obtaining one is separate download
  and conversion work rather than proof of an impossible model architecture.
- On the reviewed llama.cpp build, visual chunks use separate decode batches
  marked no-output. The embedding response omits those rows even with pooling
  `none`. A projector alone cannot provide a complete visual-token capture.

These are attributed Mac review findings, not independent Mac measurements made
while editing this document. Record exact dependency/build/artifact identities
with implementation evidence. Local [`pyproject.toml`](../../pyproject.toml)
pins mlx-vlm at `d68a25e71e842e8924a54bb3d84d3a3b4d4a2ee1`, consistent with
the review's premise; no dependency update is currently justified.

Local code independently confirms that both legacy routes explicitly re-raise
`ModelBusyError` in [`hidden_states_api.py`](../../src/heylook_llm/hidden_states_api.py),
and [`api.py`](../../src/heylook_llm/api.py) maps it through the shared 503
handler. The relevant [`TODO`](../project/TODO.md) is marked closed. Preserve
and verify this behavior; do not schedule a duplicate busy-to-500 fix on the
strength of that TODO's historical examples.

## Narrow request and result

Provisional route: `POST /v1/h3/conditioning`, authenticated like the existing
inference endpoints. Final naming and concrete image fields should follow the
existing API conventions and the actual pinned builder signatures.

Request fields:

- Resident model identifier.
- H3 mode, limited to modes explicitly supported by the pinned builders.
- Prompt and ordered image inputs, with mode-specific count/role validation.
- A bounded sequence/image policy that rejects overflow rather than silently
  truncating the conditioning presentation.
- An explicit tensor encoding if more than one is supported.

The endpoint is H3-shaped. Do not expose arbitrary layer selection, chat
messages, generic raw-content formatting, or caller-fabricated visual tokens.
Use the dependency's H3 builder as the sole owner of presentation and image
processing. Resolve supported modes from its actual API; do not infer a mode's
semantics or accepted reference layout merely from its name.

The result must contain:

| Result | Contract |
|---|---|
| Conditioning tensor | Complete native H3 conditioning, explicit shape and feature width |
| `token_tags` | The conditioner's aligned tags, explicit shape and integer encoding; not reconstructed from a separate tokenizer pass |
| Tensor serialization | Actual wire dtype, byte order, layout, and encoding for each tensor |
| Capture identity | Completed block count and final-norm behavior from the native path |
| Provenance | Model/quantization, dependency revision, mode, effective image preparation |
| Sequence length | Consistent with the conditioning and tag arrays |

Return additional native fields only where the consumer needs them. Do not
invent a universal feature metadata system for one consumer. If using the
existing float32/base64 serializer, describe the actual float32 wire bytes
separately from the forward's compute dtype. Reuse exact native H3 tag semantics.
The ComfyUI client owns mapping response fields into its conditioning container.

## Provider execution

1. Validate the supported Qwen/MLX-VLM model and obtain the existing resident
   instance through heylook's router lifecycle. Do not load another encoder.
2. Enter the existing generation gate and model-pin lifecycle, following the
   established acquisition order. Keep the pin until all request GPU work ends.
3. Build the H3 presentation and run the dependency's conditioner against that
   model. Verify its construction/injection API before implementing the wrapper;
   do not assume a constructor signature or invoke an implicit weight loader.
4. Run MLX operations on the existing pinned executor and generation stream,
   with the established wired-limit management. Evaluate and serialize outputs
   before releasing request-owned GPU resources.
5. Preserve cancellation, resource release, busy responses, and later chat.
   Do not mutate the model's layer list or output head to implement early stop.

The dependency already owns MRoPE, DeepStack, H3 image preparation, token-level
presentation, and token tags. Heylook owns input transport, residency, scheduling,
authentication, error handling, and tensor transport.

## Legacy route corrections

Retain existing request semantics while replacing unsafe execution paths with
the established executor/gate/pin lifecycle. For Qwen multimodal support, use a
native model entry point rather than manually walking decoder blocks. Do not
reinterpret all legacy requests as H3 requests.

Resolve the existing wire-dtype mismatch and accepted-multiple-inputs/first-only
response behavior explicitly. A clear multi-input rejection is preferable to
silent data loss if true batching is outside scope. Preserve absent-versus-
explicit parameter semantics. Keep these changes bounded and documented.

The existing busy-to-503 path is a regression check, not an open implementation
task in this checkout. Recheck the actual implementation branch before editing.

## Implementation map and validation

| Area | Work |
|---|---|
| New narrow H3 route module, mounted in `api.py` | Typed H3 request/result, auth, input validation, error propagation |
| `providers/mlx_provider.py` and a focused helper if useful | Adapter around the dependency conditioner and resident execution lifecycle |
| `hidden_states.py` / `hidden_states_api.py` | Bounded legacy correctness and execution fixes; preserve public compatibility |
| `docs/api_integration.md` | Document the actual narrow endpoint once implemented |
| Focused tests and a Mac probe | Presentation/tag parity, quantization comparison, lifecycle and wire behavior |

First validate the wrapper against direct use of the pinned conditioner on the
same local quantized model: identical presentation IDs, prepared image inputs,
tag values and ordering, and conditioning shape. This isolates wrapper errors.
Then compare the local 8-bit result with reference bf16 H3 conditioning on a
small set of text/reference inputs. Quantization is a numerical question; do not
claim exact tensor equality across those checkpoints or invent a universal
tolerance. Verify actual reference preprocessing, not only feature width.

Cover supported modes, image order/count failures, over-limit inputs, encoding
round trips, concurrent busy behavior, cancellation, pin release, and a normal
chat request afterward. No model downloads, server restarts, or tests were run
as part of this documentation revision.

Completion means a client can obtain verified H3 conditioning and aligned tags
from the already-loaded Qwen model with correct lifecycle and explicit limits.
M3 availability and bridge quality are not completion criteria for this route.

## Deferred M3 visual feature work

The earlier proposal treated complete native M3 embedding output as a gate to
test before deciding whether an extension was needed. The supplied build review
answers that gate negatively for **visual rows**. A native extension or another
proven runtime path is required for the intended complete visual sequence.

Reopening this work needs three concrete deliverables:

1. A compatible projector, obtained and verified through a separate conversion
   task. A converter class does not supply an already-usable artifact.
2. Native capture of visual and text rows with preserved causal/MSA semantics,
   row-to-input metadata, and complete microbatch ordering. Investigate the
   actual mtmd decode and output-collection paths, not only a model graph tap.
3. A measured fit for image prefill and extraction on the Mac with explicit
   context and microbatch limits. The reviewed GGUF's large header context must
   not become an accidental runtime allocation target. File size is insufficient.

The review reports that embeddings/chat appear to coexist on the installed
build; retain a live verification because embedding mode changes output limits.
M3 uses MiniMax Sparse Attention and its own cache type, which capture work must
preserve rather than treating as generic dense attention.

Missing visual output rows does not logically mean every returned text state is
uninfluenced by an input image. With a working projector and causal processing,
later text states can incorporate image context. That could support a different,
text-state bridge experiment; it does not yield the full visual sequence required
by the original proposal. Today the missing local projector also prevents that
image-conditioned test. Text-only embeddings remain a separate, smaller scope.

Do not download, convert, patch llama.cpp, or build a generic API as a prerequisite
for shipping the Qwen/H3 route. Preserve the original bridge discussion in the
local internal record as exploration, with this revised scope taking precedence.
