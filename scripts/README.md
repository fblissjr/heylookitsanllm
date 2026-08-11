# scripts/

Last updated: 2026-08-10

Standalone developer/ops scripts. All are run with `uv run` (PEP 723 headers
provision their own deps where noted, so they don't add anything to the project
environment). Flat by design -- a handful of files don't need subdirectories, and several
are referenced by path from `CLAUDE.md`, `docs/`, and `tests/`, so moving them
would just create churn.

| Script | What it does | Run |
| --- | --- | --- |
| `update_deps.py` | **Upstream updater -- the only thing that moves an upstream.** Three are in scope: mlx-lm, mlx-vlm (Python, via uv) and llama.cpp (C++ -- cloned and **built** here into `llama-server`). Each runs on a `stable` or `latest` channel declared in `[tool.heylook.deps]`; the script resolves the channel, writes the resolved pin back to `pyproject.toml`, and relocks. **No default set:** a bare run only reports the current pins; you name what moves, it prints a plan with a compare link, and it asks before writing or building. | `uv run scripts/update_deps.py [pkgs...\|--all] [--channel stable\|latest] [-y] [--pin] [--branch B] [--dry-run]` |
| `dev_server.sh` | Spawns an **isolated** heylookllm server for live verification (temp DB, RAM pre-flight, server-owned load+warm readiness). Same warm contract as `tests/e2e`. The `dev-server` skill wraps this. | `scripts/dev_server.sh start\|stop\|status [--port N] [--model ID]` |
| `gguf_probe.py` | Direct llama-server diagnostics for one GGUF model, BELOW `dev_server.sh` (no FastAPI/DB): /props modalities, thinking-template on/off diff, one-shot gen w/ tps + draft acceptance, LoRA attach + off/on A/B, auto-teardown. Sidecar pairing via the importer's pickers. The `gguf-probe` skill wraps this. | `uv run python scripts/gguf_probe.py <model-dir> [--spec-type draft-mtp] [--lora A.gguf --lora-ab]` |
| `ram_report.py` | **Memory pre-flight.** What is holding RAM (rolled up by app) and the ceilings that actually refuse a load: available RAM, the Metal working-set limit (~161 GB of 192 on an M2 Ultra -- well below total), and the per-allocation buffer cap. With `--model`/`--path` it sizes the model the way it is really loaded (whole shard SET, plus mmproj/drafter sidecars) and checks it against each. `dev_server.sh` calls it in `--quiet` form. | `uv run python scripts/ram_report.py [--model ID \| --path DIR] [--headroom N] [--quiet]` |
| `benchmark.py` | HTTP benchmark against a **running** server: TTFT, generation TPS, memory; OpenAI + Messages endpoints, streaming and not. (Uses `rich`, in the dev group.) | `uv run scripts/benchmark.py [--url ...] [--model ...]` |
| `export_openapi.py` | Exports the OpenAPI spec from the FastAPI app **without** running the server. | `uv run python scripts/export_openapi.py [--format yaml] [-o PATH] [--stats]` |
| `jspace_convert_lens.py` | Converts a Jacobian-lens `.pt` into an mx-safetensors lens for the j-space feature. Self-contained deps via its PEP 723 header (torch, safetensors, jlens). | `uv run scripts/jspace_convert_lens.py ...` |
| `syntax_check.py` | Fast `ast.parse` sweep over the source tree -- a cheap pre-flight for syntax errors. | `uv run python scripts/syntax_check.py` |

## Updating upstreams (common workflows)

```bash
# Report the current pins. Changes nothing, hits no network.
uv run scripts/update_deps.py

# Preview only -- resolves revs, writes nothing, clones nothing, builds nothing
uv run scripts/update_deps.py --all --dry-run

# Move one package (prints the plan, asks before it lands)
uv run scripts/update_deps.py mlx-lm

# Build llama-server at the newest release tag (or at HEAD if its channel is
# `latest`), pin what it built, print the export line
uv run scripts/update_deps.py llama.cpp

# Put llama.cpp on tip-of-master from here on, and build it now
uv run scripts/update_deps.py llama.cpp --channel latest

# Rebuild the ALREADY-pinned rev (new Xcode, changed flags, --clean)
uv run scripts/update_deps.py llama.cpp --rebuild

# Put the whole project on published releases, floors raised to match
uv run scripts/update_deps.py --all --channel stable --pin
```

### Updating a dependency is a deliberate act

Pulling a new rev runs code nobody here has read -- on the `latest` channel,
literally whatever was pushed to a branch since the last pin. So nothing moves
that you did not name. A bare run reports and exits. Naming packages (or
`--all`) resolves the targets, prints a plan with a GitHub compare link per
package and a note saying when a bump pins unreviewed code, and then asks.
`-y/--yes` skips the prompt and is **required** when stdin is not a terminal,
so an automated caller cannot drift the pins by accident. Resolution happens
for every named package before anything is applied, so a multi-package run
cannot half-land before you have seen the whole plan.

This script is the only thing in the repo that changes a dependency version:
there is no CI, no dependabot, no renovate, and `setup.sh` only runs `uv sync`
(which installs the lock, it does not move it).

### Channels

`[tool.heylook.deps]` in `pyproject.toml` says where each upstream comes from:
`stable` (PyPI for the Python packages, newest `b<N>` release tag for
llama.cpp) or `latest` (tip of the tracked branch, pinned to the exact commit).
`channel` is the project default, `overrides` is per package, and `--channel`
changes and **persists** the decision -- a file claiming `stable` while a git
SHA is pinned would be a lie. mlx-lm/mlx-vlm are held on `latest` on purpose
(see `docs/architecture/ecosystem_strategy.md`: mlx-lm is release-starved, so
fixes arrive as commits, not releases).

Why this exists instead of a one-liner: uv can pin a git commit in `uv.lock`
(`uv lock --upgrade-package <pkg>`) but leaves the source **floating** in
`pyproject.toml`. This project's policy is to always keep an explicit `rev`
pinned in both, and `update_deps.py` is the only thing that does that. It
replaced the old `update-packages.sh` (removed 2026-07-24), which only
refreshed the lock.

### llama.cpp is built, not installed

`uv sync` installs the Python packages but **cannot build a C++ binary**, so a
llama.cpp bump is always an explicit run of this script -- never a side effect
of sync. The script clones (blobless, so it stays small while any rev is still
checkoutable), refuses to check out over a dirty tree, builds, verifies the
binary, and writes a `heylook-build.json` manifest next to it recording the
rev, the resolved cmake config and a `sha256` of the binary.

**We do not vendor llama.cpp, and there is no submodule.** No upstream source
or build output is stored in this repo, tracked or untracked; nothing needs
initialising, updating, or remembering. The checkout goes to a fixed directory
under your home directory (`.heylook/llama.cpp`) -- outside the repo, so a
multi-GB build tree can never be committed, packaged into an sdist or wheel, or
shipped to anyone. It is the same path on every machine, so docs and error
messages can name it. `dir` in `[tool.heylook.llama-cpp]` moves it (relative
paths resolve against the repo root, if you *want* it in-tree);
`$HEYLOOK_LLAMA_CPP_DIR` overrides both.

The script does **not** touch `$HEYLOOK_LLAMA_SERVER` -- it prints the export
line, and warns if that variable currently points somewhere else (in which case
the server keeps using the old binary, not the new build).

Build settings, and why. The governing fact is that on Metal the arithmetic
runs in GPU shaders, so anything that only speeds up the CPU-side glue
(sampling, tokenizer, graph build, HTTP) buys noise and costs build time.

| Setting | Why |
| --- | --- |
| `GGML_METAL_EMBED_LIBRARY=ON` | metallib inside the binary -- it moves without dragging a shader file along. Upstream's own macOS release build sets exactly this. |
| `BUILD_SHARED_LIBS=OFF` | one self-contained static binary, which is what a subprocess-spawning provider wants. |
| `GGML_NATIVE=ON` | `-mcpu=native`. We build for this machine, not for distribution -- hence also no `CMAKE_OSX_DEPLOYMENT_TARGET` pin, which upstream needs only for portable release artifacts. |
| `GGML_METAL` / `GGML_ACCELERATE` / `GGML_BLAS=Apple` | all three are already the upstream defaults on Apple; set explicitly so the build does not drift with a default change. |
| `GGML_CCACHE=ON` | rebuilds after a rev bump are near-free. |
| `LLAMA_BUILD_UI=OFF` | the WebUI re-provisions from a network bucket on every build. heylook drives llama-server over HTTP and never serves it. `--ui` opts back in. |
| `GGML_LTO=OFF` | whole-program opt at link only reaches that CPU glue, costs a multi-minute link, and defeats ccache (which caches compiles, not the LTO link). Upstream sets it on **no** platform. `--lto` opts in. |
| `GGML_OPENMP=OFF` | in ggml, OpenMP only threads the CPU graph-compute path, the BLAS backend and the quantizer. Without it `ggml_graph_compute_kickoff` drives ggml's own persistent threadpool -- same decomposition, same affinity and priority handling -- so this is a choice of thread runtime, **not** threading vs none. macOS has no libomp by default, so this is already what every upstream macOS binary and every mac benchmark runs on; setting it explicitly just stops the build changing meaning if libomp ever appears. `--openmp` opts in. |

**`--openmp` is a trap upstream, and the script defuses it.** ggml only
*warns* when OpenMP is missing (`ggml/src/CMakeLists.txt:225-232`) and links the
threadpool build anyway, so `-DGGML_OPENMP=ON` can produce a binary that is
byte-identical to a non-OpenMP one while every log and manifest says `ON`.
Verified on this machine: libomp installed via Homebrew, `-DGGML_OPENMP=ON`
alone (and even with `-DOpenMP_ROOT=`) still resolves
`GGML_OPENMP_ENABLED:INTERNAL=OFF`, because AppleClang needs the flags spelled
out. An A/B against such a build would "prove" OpenMP makes no difference while
measuring nothing. So `--openmp` passes the working AppleClang incantation, then
reads `GGML_OPENMP_ENABLED` back out of `CMakeCache.txt` after configure and
**refuses to build** if it came back OFF. The manifest records the resolved
cache values (`effective`), not just the requested args, for the same reason.

Deliberately **not** set: `GGML_METAL_NDEBUG`. It compiles out load-time
logging only -- including the "allocated size is greater than the recommended
max working set size" warning. That is the ceiling that actually refuses loads
on this box (see `ram_report.py`), and a 138 GiB model against a ~161 GiB
working set is close enough that the warning earns its keep. No throughput to
gain, real diagnostics to lose.
