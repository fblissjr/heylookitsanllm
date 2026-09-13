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

Initial scope is T2VA, FL2VA, and Ref2VA with still-image references, subject to
the pinned builders' supported input contracts. Video references are deferred:
transporting sampled frames requires an explicit sampling, ordering, and timing
contract and a separate upload-cost check. Do not silently interpret video as
unrelated image references.

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
existing API conventions and the actual pinned builder signatures. Image inputs
must carry encoded image content or use an explicitly supported upload handle;
a path on the consumer machine is not an image source the server can open.

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

The ComfyUI client owns HTTP requests, tensor decoding and device transfer, and
mapping the bundle into its conditioning container. It can still construct scene
prompts and perform its separate VAE/DiT work. Server ownership refers to H3's
encoder presentation and encoder image processing, not all H3 preprocessing.
Maintain image order and correspondence across the encoder and VAE paths without
assuming shared files or copying the encoder processor into the client.

## Wire format decision

The response's dominant cost is the conditioning tensor: sequence length times
feature width times bytes per element. Base64 adds roughly one third to binary
payload size, and JSON numeric arrays also add parsing and allocation costs.
Account for image upload, serialization copies, response bytes, transfer, and
client decode/device transfer, not just encoder execution. Put measured sizes
and timings in a local evidence artifact, not this tracked design.

Use an explicit tensor descriptor with shape, actual wire dtype, byte order,
layout, and encoding. Integer IDs, tags, and grids stay integer-valued. Start
validation with the existing float32/base64 path if convenient; that does not
commit the new endpoint's default to it. Compare one representative real
still-reference request using that baseline and a compact tensor payload before
finalizing transport. Prefer a small supported container/encoding contract over
a general streaming-tensor protocol.

A 16-bit payload halves raw bytes relative to float32. Float16 and bfloat16 are
**different formats**, however: float16 has a smaller exponent range. A bf16
DiT does not make conversion through float16 lossless or guarantee that it avoids
overflow. Where the conditioner produces bf16, an explicitly typed bf16 payload
can preserve those bits if both serializer and client support it. Otherwise
compare float16 round-trip numerical error and finite values against the native
output before selecting it. Record any dtype conversion separately from the
model's 8-bit weight quantization. Select the default from the actual output
and supported client decoder rather than assuming float16 is interchangeable.

Bound upload size and output sequence length. Avoid conversion into giant Python
float lists and unnecessary simultaneous copies of the full tensor. Transport
validation should be part of the first remote-client probe, not a separate
research program or a reason to delay defining the H3 operation.

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
tag values and ordering, native image grids, and conditioning shape. This isolates
wrapper errors. Verify the complete bundle survives an HTTP round trip to the
remote consumer without dtype, shape, ordering, or integer-value changes.
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
