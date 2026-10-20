# Multimodal embeddings through both engines

Status: proposed design, 2026-10-20. No embedding implementation has landed.

## Objective and decisions

Serve EmbeddingGemma 2 through one multimodal embedding API on both existing
engines. Text, image, audio, video, and composed inputs are release requirements
on both engines. Later embedding models should reuse the operation and lifecycle.

Keep `provider = mlx | gguf`. Use mlx-vlm as a library and llama-server as the
existing managed subprocess. Do not add mlx-embeddings, SentenceTransformers,
another server, a second scheduler, or model code in heylook.

The public operation is `POST /v1/embeddings`. Its content blocks follow heylook's
existing media vocabulary. This is a documented heylook endpoint, not a promise
of OpenAI protocol compatibility. Messages generation remains unchanged.

## Evidence and boundaries

- The working-tree mlx-vlm pin and local upstream checkout both identify
  `9527f2da3b42ffb4a9613b2a5c38935c27b81039`. Its EmbeddingGemma 2 implementation
  includes the native model, multimodal processor, pooled normalized output,
  prompt guidance, video decoding and reference checks. See upstream's
  `mlx_vlm/models/embedding_gemma2/README.md` and accompanying notebook.
- The local llama-server build manifest identifies
  `4a45de4f2c68f3d144f6c2431ad7c1da7e53c99e`. Its server README explicitly
  documents `input[].content` with text, image_url, input_audio and input_video
  for `/v1/embeddings`; server-common.cpp implements those media conversions.
- Those are source-level observations, not a completed live heylook test.
  The dependency files already contain another session's edits; this proposal
  neither changes them nor treats them as committed.
- The previous embedding provider/routes were removed for lack of use. This
  requested feature supplies the use case; do not restore the old implementation.

## One request model

Canonical input is an ordered list of items, each with an ordered `content`
array. One item produces one vector, regardless of its number of media blocks.
A string is shorthand for one text item; a list of strings is a batch of text
items. Normalize these conveniences once at the HTTP boundary. Do not accept
token-ID inputs or a second vocabulary of media blocks.

```json
{
  "model": "embeddinggemma-2",
  "input": [
    {
      "content": [
        {"type": "text", "text": "A walk beside the river"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "..."}},
        {"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "..."}}
      ]
    }
  ],
  "dimensions": 768
}
```

The identifier above is illustrative; requests always name an actual registry
model. Omitted dimensions mean the model's native output dimension.

Reuse TextBlock, ImageBlock and MediaSource validation where compatible. Define
embedding audio/video blocks without accidentally widening Messages' accepted
modalities. Nested `source` is the documented form. Reject empty batches,
empty content arrays, unknown fields, malformed media and unsupported dimensions.
Never silently discard a block, truncate text, shorten audio, or switch engines.

An optional per-item `prompt_name` selects an exact key from the model's prompt
catalog. Omission means no task prefix; no guessed query/document default.
Clients select the appropriate published query/document prompts. Expose the
catalog in model capabilities and apply the selected prompt once. Preserve
document formatting required by the model; prompt selection alone must not
silently invent titles or claim to implement all SentenceTransformers behavior.
Do not introduce arbitrary prompt templates or a new preset system.

## Media semantics

Multimodal is the default load: keep every available encoder, pair the GGUF with
its multimodal projector, and advertise what the running model actually supports.
Do not route through the MLX chat loader's audio-stripping behavior.

Media resolution happens once per request, before engine-specific processing:
base64 or HTTP(S) URL becomes bounded request-owned bytes or a temporary file.
Reuse existing safe fetch behavior where it exists. Apply byte limits, timeouts
and cleanup consistently. Do not expose arbitrary local file paths or enable
llama-server's filesystem access to make this work. Files remain alive until
the worker finishes, including after cancellation of the awaiting HTTP task.

Upstream owns decoding, model-specific resize, features, media markers and
tokenization. Heylook does not implement image patches, mel features, projection
or a home-grown video processor. Avoid needless decode/re-encode cycles.

For the first contract, `video` means its visual timeline, sampled by the
upstream processor. Its soundtrack is not included implicitly. To embed sound
with video, include an audio block in the same item; both contribute to its
single vector. This makes mixed audio/video possible without a separate
soundtrack extraction API. Verify that both adapters enforce this behavior.

Report effective video sampling settings and any upstream fallback. Sampling
frames is an explicit part of video processing, unlike silently dropping the
end of an overlong clip. Start with upstream-supported model defaults; avoid
public per-frame/FPS tuning until a common mapping is verified. Image geometry,
frame sampling and token counts may differ across engines and must not be
advertised as identical. Same API and semantic item boundaries do not imply
byte-identical embeddings or interchangeable indexes across quantizations.

Before forwards, enforce engine context and media budgets using the available
processor/token-count path. When preflight cannot determine exact cost, the
backend must still reject overflow explicitly. Do not guess that text character
counts cover media costs. Limits come from a single settings/schema definition;
derive validation and reporting from it.

## Provider integration: small operation, existing lifecycle

Add `BaseProvider.embed(request, abort_event) -> EmbeddingResult`, with an
explicit unsupported-operation default. Keep the existing provider classes and
router factory. Generation rejects an embedding model before generation-specific
work, and embedding rejects a chat model before loading it where classification
is already known. No provider subclass hierarchy or adapter registration framework.

Introduce one canonical model-kind classification for chat versus embedding;
continue excluding drafter-only artifacts. Derive operation availability from
that classification and the adapter's verified support. Do not maintain separate
editable `kind`, `is_embedding`, and `supports_embeddings` switches.

Discovery uses upstream model metadata and architecture support, rather than
filename matching. EmbeddingGemma 2's model_type and GGUF architecture are
recognition inputs; a generic sentence-transformers directory alone is not proof
that an adapter supports a checkpoint. Unknown embedding architectures must not
fall through as chat. Explicit config entries also need classification, since
they bypass discovery. One resolver feeds loading, API guards and reporting.

Put model execution in small provider helpers, e.g. `mlx_embeddings.py` and
`gguf_embeddings.py`, called by the existing providers. These are modules, not
additional public providers. Put the typed operation in `schema/embeddings.py`
and its HTTP route in `embeddings_api.py`. Avoid a generic inference framework.

MLX loading branches early for embedding models: call upstream's appropriate
load path, retain all towers, materialize weights, and skip chat template
overrides, samplers, drafters, detokenizers and generation cache initialization.
The embedding helper calls the upstream processor then the model and reads
`text_embeds`. Do not use BatchGenerator or pool `last_hidden_state` ourselves.
Use the pinned executor pool and generation_stream for every MLX forward;
materialize results into host values on that worker before returning.

GGUF retains the existing process lifecycle, health checks, transport and errors.
Resolve a single embedding launch profile, shared by spawn and engine reporting:
embedding mode, correct pooling, paired mmproj, one slot and explicit context/
batch/micro-batch constraints for non-causal encoding. Do not inherit the chat
micro-batch heuristic blindly or pass chat/drafter/KV settings into this profile.
Preserve applicable owner settings such as flash-attention selection. Validate
incompatible overrides rather than silently ignore them. Verify the full expanded
input fits the physical batch. Translate ordered content to the documented
`input[].content` structure and call llama-server's `/v1/embeddings`.

## Scheduling, residency and cancellation

Reuse the process-global gate, busy responses, request IDs, explicit cancellation
registry, model count limits, idle unload and router reservations. Embeddings are
active inference for every eviction/unload check, despite existing identifiers
being named `active_generations`. Audit those consumers; do not introduce an
embedding-only counter invisible to the router.

Initial scheduling is one bounded request holding one gate acquisition. Adapters
may execute bounded internal batches/items under that lease, preserve input order,
and check cancellation between units. This avoids dynamic batching, partial
results, repeated gate admission and an extra residency-lease framework. A bulk
indexer sends several bounded requests, allowing chat requests to interleave.
Use real padded-token/media cost, not only item count, to bound MLX batching.
Do not promise immediate preemption of an executing Metal kernel.

Registration must cover the complete non-streaming operation. Cancellation
must not release the gate, model references or media files while execution is
still running. In GGUF, closing an HTTP connection is not proof the slot has
stopped: verify cancellation behavior and retain ownership until completion or
confirmed stop. Bound backend waits and follow existing process-recovery policy.
Honor the existing explicit-cancel behavior; no new disconnect-polling policy.

The router currently permits only one engine family resident at a time. Keep
that safety rule. Chat plus embeddings on the same engine can remain resident
only when model-count settings and memory allow; defaults may evict between
models. Switching between MLX and GGUF can unload/reload. Expose this cost; do
not promise an always-resident embedding sidecar or change memory policy here.

## Results, capabilities and observability

Return indexed float vectors in input order, the selected model, effective
dimensions, per-item prompt selection, usage and request timing. Unknown token
usage is null, not zero or an estimate disguised as observation. No generation
token count, thinking fields, TTFT or tokens-per-second fabrication.

Use upstream pooled output. A common host-side finalizer validates vector count,
dimension, finite values and nonzero norm; applies supported Matryoshka truncation;
and L2-normalizes the resulting vectors. Do not silently truncate models without
that capability or substitute zero vectors on failure. Requests are atomic:
an item failure returns an error naming its index, not a partial successful batch.

Extend the existing engine description with an optional embedding section:
supported/observed modalities, native and supported dimensions, prompt catalog,
normalization, input limits, preprocessing settings and provenance. Existing
generation-only fields are not applicable. Static reports may say unknown;
loaded observations establish available towers/codecs and actual limits. Do not
claim runtime video availability solely because the checkpoint has vision weights.

Expose the available model artifact/revision identity, engine revision and
preprocessing configuration through the existing engine report. Return the model
ID, dimension and per-item prompt selection with results. Clients building an
index record that provenance; do not build a new weight-hashing or index-version
system here. Matching dimensions do not certify cross-engine interchangeability.

Use existing observability ingestion, with operation=embedding and content-free
counts, queue/load/preprocess/inference/total durations where measurable. Do not
log media or vectors. No DuckDB schema changes or storage of embedding results.

Models/admin views show embedding capability and embedding-relevant load settings;
chat/notebook selectors filter by generation operation. A separate embedding UI
is not required to deliver the API.

## Implementation sequence and acceptance

1. Run direct upstream probes on the exact pins/build: text, image, audio, video
   and composed inputs on both engines. Verify prompt application, vector shape,
   normalization, long-input refusal and actual encoder use. Record codec and
   video behavior, required ffmpeg/ffprobe availability, and gaps in internal/.
   No new Python dependency is planned; installed codec availability is a real
   runtime precondition. Confirm the reference before designing around a gap.
2. Implement schema, classification and capabilities with one resolver; wire
   early load branches and embed operations in both providers. Add the endpoint
   to app assembly/OpenAPI. Keep generation behavior intact.
3. Connect pinned execution, gate/cancellation and resource ownership; enforce
   budgets and normalize results. Update Models-page presentation/filtering.
4. Verify through the HTTP API against direct upstream results, then against
   the publisher's reference on a fixed multimodal corpus. Reference tooling may
   live in a separate probe environment; it is not a server dependency.

Completion requires both engines to pass all four modalities and composed-input
cases. A source inspection, a successful response or a vector of the right size
is insufficient. Fixtures must detect dropped media: changed image content,
different audio content, motion/order-sensitive video, and mixed-item changes.
Check joint composition versus separate items, batch versus singleton, padding,
ordering, prompts applied once, supported dimensions, and finite normalized output.
Use representative retrieval positives/negatives and justified numerical
tolerances; never demand exact equality across different quants/backends.

Exercise cancellation while queued/active, busy responses, unload/reload, idle
eviction, failure cleanup, memory return and chat/embedding alternation, including
cross-engine switches. MLX worker-thread tests must run on real Metal. A broken
GGUF-only environment must not import MLX merely to serve embeddings.

Run the complete unit/contract suites and relevant frontend suites. Provider
release smoke covers the existing generation arms as well as the new embedding
matrix. Run vendor_frontend --check and report the normal release preconditions.
Use existing performance tooling only for measured speed/memory claims; adding
this API does not itself justify performance claims. Update spec section 4,
API integration docs, matching wiki pages, affected area rules and session log
in implementation commits. Scope generation-only wording in rules to generation.
Follow the repository's normal changelog/version policy when landing code.

## Extension rule

Another upstream-supported dense embedding model should require metadata
recognition, any genuinely model-specific prompt/preprocessing adapter, and
reference fixtures. It should not require another HTTP route, scheduler, model
registry or media transport. Add abstraction only when a second implementation
needs it. Rerankers, token-level/multi-vector outputs, vector databases, chunking,
embedding caches and background indexing are separate features.
