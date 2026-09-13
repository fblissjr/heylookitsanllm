# H3 conditioning through the resident Qwen3-VL model

Status: revised proposal, 2026-09-13, following the owner's supplied review from
Claude working on the Mac installation. No implementation or live validation is
claimed here. This replaces the earlier shared Qwen/M3 feature-API proposal.

## Decision and scope

Expose a narrow H3 conditioning operation through heylook's resident Qwen3-VL-32B
`MLXProvider`, using the H3 presentation builders and `MiniMaxH3Conditioner`
already shipped in the pinned mlx-vlm dependency. Return the native conditioning
bundle: hidden states, `token_tags`, `input_ids`, and the image grid. Preserve
normal chat behavior on the same loaded model.

The owner confirmed that ComfyUI/H3 runs on a separate Linux machine and reaches
heylook on the Mac over HTTP only. Heylook owns the complete encoder operation.
There is no shared Python process, filesystem, MLX array, or consumer-side
mlx-vlm dependency to rely on. The consumer's hostname and address are private
and must not be recorded in repository files, internal notes, logs, or examples.

Initial scope is T2VA, FL2VA, and Ref2VA with still-image pixels and an ordered,
typed reference list that also preserves audio-only entries. Such entries affect
presentation numbering even though they supply no image pixels. Video reference
kinds are explicitly rejected in the first release, not filtered out. Supporting
video later requires a frame sampling, ordering, timing, and upload contract.

Defer M3 visual feature extraction as a separate project. Do not introduce a
generic schema-version-2 feature API or change the GGUF provider for this first
release. The earlier `/v1/hidden_states` routes and the `/v1/embeddings` route
were removed in v2.0.40 (owner call: nothing used them), so this route is built
on a clean base rather than beside a legacy one, and the shared MLX worker
lifecycle helper it needs has exactly one consumer.

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
- The follow-up review verifies that the conditioner accepts the resident model
  and an HF tokenizer directly, without a weight loader. On heylook's mlx-vlm
  path, that tokenizer is available as `processor.tokenizer`. The native builder
  tokenizes with special tokens off and uses `convert_tokens_to_ids`.
- The resident model's text dtype and conditioner output are bf16. Preserve that
  representation directly on the wire; choosing a lossy float16 conversion is
  unnecessary.
- In the pinned H3 pipeline, FL2VA keyframes are prepared at the generation
  canvas before conditioning, including stretching the first keyframe. Ref2VA
  reference images are prepared at H3's reference short edge before conditioning.
  Qwen smart-resize and patchification happen afterward inside the conditioner.
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

The app-level `ModelBusyError` handler in [`api.py`](../../src/heylook_llm/api.py)
answers 503 for any route that lets the error out. The new route must let it
out (no broad handler around the provider call) and pin that through the route
in `tests/contract/`, the way the removed routes were pinned.

## Narrow request and result

Provisional route: `POST /v1/h3/conditioning`, authenticated like the existing
inference endpoints. Final naming and concrete image fields should follow the
existing API conventions and the actual pinned builder signatures. Image inputs
must carry encoded image content or use an explicitly supported upload handle;
a path on the consumer machine is not an image source the server can open.

Request fields:

- Resident model identifier.
- H3 mode, limited to modes explicitly supported by the pinned builders.
- Prompt and mode-specific prepared inputs: ordered keyframes for FL2VA;
  the complete ordered reference list with a kind per entry for Ref2VA.
- Image-bearing entries carry client-prepared pixels. Audio-only entries carry
  their native presentation metadata and no pixels. Preserve their positions
  when assigning labels to later image entries. Reject video and unknown kinds
  before inference; never turn a mixed reference list into an image-only list.
- A bounded sequence/image policy that rejects overflow rather than silently
  truncating the conditioning presentation.
- An explicit tensor encoding if more than one is supported.

The endpoint is H3-shaped. Do not expose arbitrary layer selection, chat
messages, generic raw-content formatting, or caller-fabricated visual tokens.
Use the dependency's H3 builder as the sole owner of encoder presentation and
encoder-side image processing. Resolve supported modes from its actual API; do not infer a mode's
semantics or accepted reference layout merely from its name.

The result must contain:

| Result | Contract |
|---|---|
| Conditioning tensor | Complete native bf16 H3 conditioning, explicit shape and feature width, lossless raw-bit serialization |
| `token_tags` | The conditioner's aligned tags, explicit shape and integer encoding; not reconstructed from a separate tokenizer pass |
| `input_ids` | Exact integer presentation IDs returned by the native conditioning path; preserve shape and order |
| Image grid | Native grid metadata and its ordering relative to supplied references; define the no-image representation explicitly |
| Tensor serialization | Actual wire dtype, byte order, layout, and encoding for each tensor |
| Capture identity | Completed block count and final-norm behavior from the native path |
| Provenance | Model/quantization, dependency revision, mode, effective image preparation |
| Sequence length | Consistent with the conditioning and tag arrays |

Carry all four native outputs across the API, even where the current DiT call
uses only states and tags directly. IDs and grids retain the exact presentation
and image-layout context for the remote integration and capture workflow. Map
the native field names and shapes from the pinned conditioner; do not invent or
reconstruct them from a separate tokenizer/image-processing pass. Check the
actual relationships rather than assuming one grid entry per sequence row.

## Image preparation ownership

The client chooses the generation canvas and owns the preparation shared with
its VAE path. The server owns only the subsequent encoder processing:

1. For FL2VA, the client prepares each keyframe using the pinned pipeline's
   `prepare_keyframe_image` semantics at the generation canvas, including the
   first-keyframe stretch. It sends those prepared pixels, not the original upload.
2. For Ref2VA, the client prepares reference images at H3's reference short edge
   using the same preparation as its VAE path, then sends those pixels in their
   original reference-list positions.
3. The server decodes the prepared pixels and feeds them directly into the native
   conditioner. It must not repeat canvas fitting, keyframe stretching, or H3
   reference-short-edge resizing. The conditioner still performs its own Qwen
   smart-resize and patchification.

The decoded upload must be byte-identical to the prepared pixel array that feeds
the client's VAE-side tensor conversion. Specify pixel format, dimensions, and
channel order and use a lossless representation; lossy image re-encoding or an
extra orientation/color conversion can break parity even at the same dimensions.
This comparison is before the conditioner-specific resize and the VAE-specific
normalization, whose final tensors are intentionally different. Verify the
prepared pixels directly in the integration probe; do not rely only on filenames
or dimensions. Keep probe pixel checks out of routine request logs.

The ComfyUI client owns the request, tensor decoding and device transfer, and
mapping the returned bundle into its conditioning container. It also owns scene
prompt authoring and separate VAE/DiT processing. Heylook owns H3 token-level
presentation, reference labels, encoder preprocessing, and encoder execution.
There is no shared-file or consumer-side mlx-vlm assumption.

For Ref2VA, label numbering follows the full typed reference sequence. An audio
entry emits its native label and shifts later picture numbering despite having
no pixels. Pass that full sequence to the native builder; map the image payloads
and grids back to the image-bearing entries without renumbering them. This
preserves audio-reference presentation; it does not add waveform encoding to
the Qwen route. Reject a video kind explicitly, including when mixed with valid
image/audio entries, rather than generating a different presentation by omission.

## Wire format decision

The conditioning tensor uses **raw bfloat16 bits**, with `dtype=bfloat16`, an
explicit shape, contiguous row-major layout, and little-endian byte order. The
resident output is bf16 per the supplied pinned-runtime review, so this preserves
the native result losslessly at the same raw size as float16. Do not route it
through the legacy float32 serializer or numerically cast it to uint16/float16.

Standard NumPy does not provide a native bf16 dtype for this serializer. Evaluate
the MLX array within the established worker/stream lifecycle, reinterpret its
storage as uint16 words, and export those words without changing their bits.
Normalize byte order as necessary for the declared wire order. The torch client
reconstructs bf16 from those bytes with the stated shape; a uint16 staging view
is a bit reinterpretation, not a numeric conversion. Give the decoded storage an
appropriate owned lifetime before device transfer. Verify output dtype and fail
clearly if the runtime violates the declared bf16 contract rather than mislabel
another dtype's bytes.

Tags, IDs, and grids retain their integer values and explicit shapes/dtypes.
The payload container remains a small implementation choice: binary transport
or a base64 wrapper can carry the same raw bf16 bytes. Base64 adds roughly one
third to payload size. This is an envelope/overhead decision, not an open
float32-versus-float16-versus-bf16 comparison. Avoid JSON float lists and
unnecessary simultaneous copies of the full tensor.

The response's dominant cost is sequence length times feature width times bytes
per element. Check a representative remote request for upload, serialization
copies, response bytes, transfer, and client decode/device transfer. Put measured
sizes and timings in a local evidence artifact, not this tracked design. Bound
upload size and output sequence length. Require bitwise equality of the bf16
payload after round-trip; compare encoder quantization against reference bf16
weights separately. Lossless transport does not remove weight-quantization error.

## Provider execution

1. Validate the supported Qwen/MLX-VLM model and obtain the existing resident
   instance through heylook's router lifecycle. Do not load another encoder.
2. Enter the existing generation gate and model-pin lifecycle, following the
   established acquisition order. Keep the pin until all request GPU work ends.
3. Construct the dependency's conditioner with the resident model and the HF
   tokenizer at `processor.tokenizer`; this interface is verified by the supplied
   pinned-source review. Build the native presentation and run conditioning on
   client-prepared images. No weight loader or construction-API research gate is
   needed. Leave special-token handling and token-ID lookup to the native builder.
4. Run MLX operations on the existing pinned executor and generation stream,
   with the established wired-limit management. Evaluate and serialize outputs
   before releasing request-owned GPU resources.
5. Preserve cancellation, resource release, busy responses, and later chat.
   Do not mutate the model's layer list or output head to implement early stop.

The dependency already owns MRoPE, DeepStack, encoder-side image preparation,
token-level presentation, and token tags. The client owns pre-encoder canvas
and reference preparation shared with its VAE path. Heylook owns input transport, residency, scheduling,
authentication, error handling, and tensor transport.

## Implementation map and validation

| Area | Work |
|---|---|
| New narrow H3 route module, mounted in `api.py` | Typed H3 request/result, auth, input validation, error propagation |
| `providers/mlx_provider.py` and a focused helper if useful | Adapter around the dependency conditioner and resident execution lifecycle |
| `streaming_utils.py` or `generation_core.py` | One helper that runs an MLX callable under the gate, pin, pinned executor, generation stream and wired limit; this route is its first consumer |
| `docs/api_integration.md` | Document the actual narrow endpoint once implemented |
| Focused tests and a Mac probe | Presentation/tag parity, quantization comparison, lifecycle and wire behavior |

First validate the wrapper against direct use of the pinned conditioner on the
same local quantized model: identical presentation IDs, prepared image inputs,
tag values and ordering, native image grids, and conditioning shape. This isolates
wrapper errors. Verify the complete bundle survives an HTTP round trip to the
remote consumer without dtype, shape, ordering, or integer-value changes. Check
bf16 bit patterns before and after transport, not just approximate float values.
Compare decoded prepared pixels with the client's pre-VAE pixels before running
encoder processing. Cover a non-canvas-sized first keyframe and an audio reference
before an image reference, so resize ownership and label numbering can fail
observably. A mixed list containing video must fail explicitly.
Then compare the local 8-bit result with reference bf16 H3 conditioning on a
small set of text/reference inputs. Quantization is a numerical question; do not
claim exact tensor equality across those checkpoints or invent a universal
tolerance. Verify actual reference preprocessing, not only feature width.

Cover supported modes, image order/count failures, over-limit inputs, encoding
round trips, concurrent busy behavior, cancellation, pin release, and a normal
chat request afterward. No model downloads, server restarts, or tests were run
as part of this documentation revision.

Completion means the remote client can obtain the complete verified H3 bundle
from the already-loaded Qwen model over HTTP, decode it for its conditioning
consumer, and preserve states, tags, IDs, and grids with explicit dtype and limits.
M3 availability and bridge quality are not completion criteria for this route.

## Deferred appendix: M3 visual feature work

The remote deployment does not change the M3 blockers. The missing projector
and omitted visual rows are findings verified by the supplied Mac review; actual
multimodal memory fit remains unmeasured. Do not label that fit as verified.

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
