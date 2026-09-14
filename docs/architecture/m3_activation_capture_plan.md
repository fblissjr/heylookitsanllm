# MiniMax-M3 activation capture: the spike plan

Status: proposed, 2026-09-13. Nothing in this document is implemented. It
records what was established on 2026-09-13 so the work can be picked up
another day without re-deriving it. Owner decision the same day: this is the
experiment; the Qwen/H3 conditioning route in
[multimodal_feature_extraction.md](./multimodal_feature_extraction.md) is
supporting infrastructure and is built only as far as the experiment needs it.

## The experiment

1. Extract useful MiniMax-M3 representations for a prompt plus references.
2. Capture the corresponding Qwen3-VL conditioning that MiniMax-H3 expects.
3. Train a small bridge between them, outside this server.
4. Check on held-out inputs whether the bridge improves H3, not merely whether
   it reconstructs Qwen's features.

The central research risk is step 4. A bridge that matches Qwen features has
shown compatibility, not improvement. Keep the pilot small enough that a null
result costs a day, and hold out inputs from the first run.

## What is established (verified, not assumed)

Each of these was checked against code, headers or a live run on 2026-09-13.
Where the evidence was a model card it says so.

- **The canonical llama.cpp build requires MSA indexer tensors to load M3.**
  Its M3 loader reads the `minimax-m3.attention.indexer.*` header keys and
  creates the per-layer `indexer.*` tensors as required, not optional.
  Unsloth's M3 GGUFs carry neither (converted from the pre-merge PR) and do
  not load. AesSedai's quants carry them, verified by parsing shard headers
  over HTTP range requests. The owner replaced the Unsloth file with
  AesSedai IQ3_S plus its mmproj; it sits in the models.toml scan folder and
  discovery derives the entry (`AesSedai_MiniMax-M3-GGUF_IQ3_S`, text plus
  vision, projector paired) with no config edit.
- **M3 loads and serves chat with images.** Materialized entry with
  `ctx_size = 32768` and `n_ubatch = 512`, both set before the first load.
  The micro-batch must be explicit: with the owner's raised wired limit the
  auto path picks the wide micro-batch, which is the shape that killed a
  vision model on its first decode once before. Text, image and a follow-up
  text request all end their turn. Memory is the constraint: one model at a
  time, nothing co-loaded, and a capture run needs the daily server's copy
  unloaded first.
- **The stock embeddings endpoint drops the visual rows.** In the server,
  image chunks are decoded through their own embedding batch whose rows are
  all flagged no-output (`decode_embd_batch` in `tools/mtmd/`), and that
  batch never reaches `send_embedding`, which iterates the text batch and
  skips any row without the output flag. So `--pooling none` returns text
  rows only, with the image span silently absent. This is a code-level
  finding, not a measurement; the measurement is step 0 below.
- **Embedding mode and chat appear to coexist on this build.** `--embedding`
  gates only the embeddings routes, and the server toggles embeddings per
  batch. It does change the server's output limits at spawn, so coexistence
  still wants one live check. The native `/embeddings` route is the one that
  accepts `--pooling none` and returns unnormalized per-token rows; the
  OpenAI-shaped `/v1/embeddings` refuses that pooling.
- **The M3 feature the plan chose is the final-norm output**, width 6144,
  after 60 blocks of which the first three are dense. The graph assigns the
  final RMSNorm output to the embeddings tensor; an intermediate layer would
  need the graph callback and is not part of the first spike.
- **M3 on MLX is not an option for the full model.** The pinned mlx-vlm has
  the M3-VL class including the sparse indexer, but every MLX conversion
  that fits this machine is either text-only or REAP-pruned with
  coder-calibrated 2-bit experts. Pruning it here is not possible either:
  the tooling needs the unpruned model running under torch. A pruned MLX
  checkpoint is acceptable only as plumbing validation, never as evidence
  about M3.
- **The H3 side needs no server.** The pinned mlx-vlm ships the whole H3
  conditioning path: the layer-50 conditioner, the presentation builders,
  H3's own image processing and the token tags. For the spike it is called
  directly from a Python script on the resident Qwen3-VL-32B. The HTTP route
  matters only when ComfyUI consumes a bridge that has proven itself.

## The spike, in cost order

### Step 0: text rows from the stock server (nothing to build)

Spawn M3 through the existing provider with `extra_args` of `--embedding`
and `--pooling none`, post the native embeddings request with one prompt and
one image, and count rows against prompt tokens. One run answers:

- whether embedding mode coexists with MSA and the projector on this build;
- whether a chat request still works on the same process afterwards;
- exactly how many rows the image span loses.

That last count is the acceptance test for step 1. Record the build tag,
the artifact, the flags and the counts in `internal/research/`, not here.

### Step 1: the visual-row patch (bounded C++)

Two changes in the server, from reading the code rather than from a
prototype: flag the image-chunk rows for output when the slot wants
embeddings, and collect those rows in position order into the same result
the text rows go into. Final-norm output only. Acceptance: rows returned
equals text tokens plus image tokens, in prompt order, and repeated requests
and a changed image give consistent counts. Do not add a layer tap until the
bridge asks for one.

Where it lives: a branch on a llama.cpp fork, not the canonical build. The
build script clones only upstream and verifies the remote, so it needs a
small change to build from a fork, or the experimental binary runs through
the `server_binary` escape hatch, which warns at every spawn. That warning
is the point: an experiment binary must announce itself. Which fork is an
open decision below.

### Step 1, alternative: a standalone capture tool on the evaluation callback

Codex wrote this on 2026-09-13 as an independent route. Later the same day
it was built on the Mac against the canonical clone at the server binary's
commit and RUN against M3: text-only, one image, a fresh-process repeat, a
different micro-batch, a swapped image, and two images in one prompt. Every
run captured every row, including image chunks split across micro-batches;
the repeat and the micro-batch variant were bit-identical; the swapped
image changed the image rows and the text rows after the image while the
text before it stayed identical. Observations with the numbers are in
`internal/research/`. THIS IS THE ROUTE. Step 0 and step 1 above are no
longer needed for the spike and stay only as the HTTP alternative for a
later consumer. The tool lives in this checkout's gitignored `internal/`
tree (copied from Codex's box; it never travels through git).

The idea: `llama_context_params.cb_eval` sees every intermediate tensor the
scheduler computes, for the text batches and for the image-embedding batch
alike, so the row filter that drops visual rows from the final output never
applies. Verified in the M3 graph: only at the last layer does the graph
gather the output rows (`inp_out_ids`), so the penultimate layer's residual
is the last tensor that still carries every row. The tool captures one
selected layer with the native chunk metadata and refuses a capture whose
row count disagrees with the chunk's token count. That refusal is the
acceptance test, the same one step 1 has.

What it changes and what it does not:

- The M3 feature becomes a pre-norm residual at a chosen layer, not the
  final-norm output the profile table in the sibling doc names. Either is a
  legitimate bridge source; record which one every capture used. Final-norm
  output for all rows is still reachable from a standalone tool by building
  the image batch itself with every row flagged for output, instead of
  going through the helper that flags none.
- Micro-batching splits a layer's tensor across ubatches. Rows must be
  concatenated in position order across them, and the chunk metadata is
  what proves nothing was lost at a boundary.
- It is an instrument beside the server, in the spirit of `llama-bench`
  and `llama-fit-params`: it must be built against the same llama.cpp
  commit as the canonical binary or the graph and tensor names can differ.
  No fork and no second llama-server are needed. Its CMake pulls llama.cpp
  in with `add_subdirectory` and links the native targets, which carry the
  Metal, embedded-shader, Apple-framework and BLAS wiring themselves, so
  nothing is reproduced by hand and nothing goes into the canonical tree.
  Two things make the result match the server binary: point
  `add_subdirectory` at the canonical CLONE under the home directory (it is
  already checked out at the binary's commit; a fresh clone would need
  pinning to that same commit), and configure with the same backend flags
  the build script passes (its `cmake` argument list in
  `scripts/build_llama.py`, rationale in `scripts/README.md`: static libs,
  Metal with the embedded library, native, Accelerate BLAS, no LTO, no
  OpenMP, and `GGML_METAL_NDEBUG` left unset). The cost is a second build
  of the libraries in the tool's own build directory, minutes with ccache.
  The canonical build script does not build it; that stays a deliberate,
  named act.
- The micro-batch repeat is an ORDERING check with a tolerance, not a
  bit-equality check: on Metal a different micro-batch changes how matmuls
  are batched, so rows agree closely, not exactly. Row order and count must
  match exactly; values must match within a small tolerance.
- Running it has the same precondition as everything else here: the daily
  server's M3 must be unloaded first, and the tool loads the same GGUF and
  mmproj that the server does.

Prefer this route over the server patch if the first live run captures
every row: it leaves the serving path untouched. The server patch remains
the route if a bridge consumer later needs capture over HTTP.

### Step 2: a capture script, not an endpoint

Status 2026-09-14: the H3 side of this exists and ran. A script in the
gitignored `internal/claude/h3-target-capture/` calls the pinned mlx-vlm
conditioner directly on the local Qwen3-VL-32B, writes the layer-50 states
as bf16 bits with the tags, ids and grid, refuses any mismatch between them,
and verified bitwise reloads and in-process repeats on the same synthetic
inputs the M3 tool used. Results in `internal/research/`. Those six pairs
are a plumbing check, not a dataset: the M3 side used M3's raw diagnostic
prompt and the H3 side used H3's presentation, and the H3 mode and
reference contract must be fixed before captures count as aligned pairs.

A script that takes a handful of paired prompt-plus-image examples and
writes both sides to files:

- M3 rows from the patched server, with the token ids and the image span
  boundaries the server reports, so a row can be attributed;
- H3 conditioning from mlx-vlm's conditioner called in-process on
  Qwen3-VL-32B: hidden states, token tags, input ids, image grid.

Sequential, one model at a time. The two sides differ in width, tokenizer
and layout by design; equal sequence length never implies alignment. The
client that prepares images for H3's VAE path must prepare the same pixels
the conditioner sees. Store captures under an explicitly chosen artifact
location; the observability policy keeps prompts and tensors out of logs.

Status 2026-09-14, later: the first REAL pair exists, from a prompt in the
H3 repo's own prompt bank (a text-only T2VA example, so no image
dependence yet), captured M3 then Qwen with one encoder resident at a time
and bound to its example by a pair manifest with payload hashes and
revisions, under the gitignored `internal/claude/m3-h3-pilot/pairs/`. Row
counts differ between the two tokenizers, as they must. Capture order for
a bank: every M3 example first, then every Qwen one, because alternating
the two evicts the larger model from the page cache each time.

Status 2026-09-14, evening: the first bank is captured. The H3 side froze
a text-only bank of eleven scenes from its own prompt bank with fixed
train and held-out groups and a capture profile pinned from the first pair;
this side captured the remaining ten in that order and every pair passes
the H3 side's own validator against the frozen bank. Receipts sit beside
the pairs under `internal/claude/m3-h3-pilot/`. From here the work is the
adapter and its controls, on files, off this server.

### Step 3: the bridge, elsewhere

Training happens on the captured files, outside this server, on whatever
machine Codex chooses. Nothing here depends on it.

## Not being built

- A polished or generic extraction API. The removed `/v1/hidden_states`
  routes are not coming back in another shape.
- The Qwen HTTP conditioning route, until a bridge exists to consume it.
- A large dataset. The first pilot is a handful of pairs with held-out
  examples.
- Thinking control for M3 through heylook's toggle. Finding recorded for
  later: M3's template reads only its own `thinking_mode` variable (enabled,
  disabled, adaptive; undefined means adaptive), heylook forwards only
  `enable_thinking` and `reasoning_effort`, and llama.cpp's M3 parser maps
  nothing between them, so the toggle does nothing on this model. The agreed
  wiring, when someone wants it: explicit on and off map to enabled and
  disabled, unset sends nothing so adaptive survives, and the admin row's
  `thinking_default` must be able to say "model decides" instead of a bool.
  It is chat behaviour, not a prerequisite for capture.

## Open decisions

- Whether the HTTP capture path (server patch) is ever needed; the
  standalone tool covers the spike, so this waits for a consumer that needs
  capture over HTTP.
- Capture windows still need the daily server's M3 unloaded each time.
- Which M3 feature the bridge trains on: the final-norm output (server
  path, or a standalone tool that flags every row for output) or a chosen
  pre-norm residual (the callback tool's default). The first live capture
  decides what is cheap; the bridge decides what is useful.

Postmortem of the day this was written and first run (local, gitignored):
`internal/postmortems/2026-09-13_session_m3-capture-and-removals.md`.

## Rules that apply

- One model resident at a time; check reclaimable memory before a spawn.
- No performance numbers in tracked docs; observations go to
  `internal/research/` with the build, artifact and flags named.
- llama-server's own log is captured only when the observability level is
  above off at spawn time.
- Match prompt, image, flags and build across arms before comparing any two
  captures.
