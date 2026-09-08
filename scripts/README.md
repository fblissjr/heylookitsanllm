# scripts/

Last updated: 2026-09-06

Standalone developer/ops scripts. All are run with `uv run` (PEP 723 headers
provision their own deps where noted, so they don't add anything to the project
environment). Flat by design -- a handful of files don't need subdirectories, and several
are referenced by path from `CLAUDE.md`, `docs/`, and `tests/`, so moving them
would just create churn.

| Script | What it does | Run |
| --- | --- | --- |
| `build_llama.py` | **Clones and builds llama-server** (llama.cpp) -- the one dependency `uv sync` cannot install, because it is a C++ binary. Builds two instruments beside it, from the same commit: `llama-bench`, for any spawn-flag question (`build/bin/llama-bench -m X.gguf -p 512,4096 -n 128 -ub 512,2048 -fa 1 -r 3`), and `llama-fit-params`, llama.cpp's own memory projector, which prints the per-device model/context/compute split without loading the weights (`build/bin/llama-fit-params -m X.gguf -ngl 999 -np 1 -ub 2048 -fitp on`). The projector flag is not accepted there, so an mmproj is costed separately. Newest `b<N>` release tag by default (`--rev` for anything else, `--status` to report, `--rebuild` after an Xcode bump). Never touches pyproject/uv.lock. Build settings and their rationale: below. | `uv run scripts/build_llama.py [--rev REV] [--status] [--rebuild] [--clean] [--openmp] [--lto] [--ui]` |
| `guard_stable_channel.sh` | **Pre-commit guard** (wired into the git hook): blocks committing `pyproject.toml`/`uv.lock` while they carry a git pin (a `[tool.uv.sources]` rev, a git-sourced lock entry). See "Dependencies" below. | runs from the hook; manual: `scripts/guard_stable_channel.sh` |
| `dev_server.sh` | Spawns an **isolated** heylookllm server for live verification (temp DB, RAM pre-flight, server-owned load+warm readiness). Same warm contract as `tests/e2e`. The `dev-server` skill wraps this. | `scripts/dev_server.sh start\|stop\|status [--port N] [--model ID]` |
| `gguf_probe.py` | Direct llama-server diagnostics for one GGUF model, BELOW `dev_server.sh` (no FastAPI/DB): /props modalities, thinking-template on/off diff, one-shot gen w/ tps + draft acceptance, LoRA attach + off/on A/B, auto-teardown. Sidecar pairing via the importer's pickers. The `gguf-probe` skill wraps this. | `uv run python scripts/gguf_probe.py <model-dir> [--spec-type draft-mtp] [--lora A.gguf --lora-ab]` |
| `gpu_wired_limit.sh` | **GPU wired limit** (`iogpu.wired_limit_mb`), the one lever that enlarges the Metal working set both engines size against (~84% of RAM at the OS default). `status` reads it and reports whether the boot job is actually LOADED and last exited 0 (not merely whether the plist exists); `set`/`install` raise it now / at every boot (root); `uninstall` removes the daemon. The admin fit panel and a Metal compute-error message print the value to use. | `scripts/gpu_wired_limit.sh status` · `sudo scripts/gpu_wired_limit.sh install [MB]` |
| `ram_report.py` | **Memory pre-flight.** What is holding RAM (rolled up by app) and the ceilings that actually refuse a load: available RAM, the Metal working-set limit (~161 GB of 192 on an M2 Ultra -- well below total), and the per-allocation buffer cap. With `--model`/`--path` it sizes the model the way it is really loaded (whole shard SET, plus mmproj/drafter sidecars) and checks it against each. `dev_server.sh` calls it in `--quiet` form. | `uv run python scripts/ram_report.py [--model ID \| --path DIR] [--headroom N] [--quiet]` |
| `benchmark.py` | HTTP benchmark against a **running** server: TTFT, generation TPS, memory; OpenAI + Messages endpoints, streaming and not. (Uses `rich`, in the dev group.) | `uv run scripts/benchmark.py [--url ...] [--model ...]` |
| `export_openapi.py` | Exports the OpenAPI spec from the FastAPI app **without** running the server. | `uv run python scripts/export_openapi.py [--format yaml] [-o PATH] [--stats]` |
| `vendor_frontend.py` | **Vendored frontend dependency manager** (marked, DOMPurify). `--verify` is offline and wired into the git hook; `--check` also reports what npm has published; `--update` installs the latest and rewrites the manifest. Stdlib only and run directly, NOT under `uv run` -- a pre-commit hook must not depend on the venv being synced. See "Vendored frontend libraries" below. | `scripts/vendor_frontend.py [--verify [--staged] \| --check \| --update [--package NAME]]` |
| `syntax_check.py` | Fast `ast.parse` sweep over the source tree -- a cheap pre-flight for syntax errors. | `uv run python scripts/syntax_check.py` |

## Dependencies

Dependency updates are plain uv -- there is no updater script, no CI, no
dependabot:

```bash
uv lock --upgrade && uv sync              # everything, newest releases
uv lock --upgrade-package mlx-lm && uv sync   # one package
```

The committed `pyproject.toml`/`uv.lock` always point at published releases,
so a clone gets releases, never someone's experiment. Pinning a dependency to
a git commit (a `[tool.uv.sources]` entry) is a machine-local, working-tree
choice: `guard_stable_channel.sh` blocks committing it, because uv honors no
gitignored file for source pins and every pin propagates into `uv.lock`.
`HEYLOOK_ALLOW_CHANNEL_COMMIT=1` is the deliberate-exception escape hatch
(e.g. a reviewed, intentional git dependency).

## Vendored frontend libraries

The frontend has no build step, so its only two external dependencies -- `marked`
and `DOMPurify` -- are committed files under `js/vendor/`, not lockfile entries.
Nothing about a committed file says where it came from, so before v1.79.72 the
only way to answer "are we current?" was to read a version banner out of a
minified bundle and hand-check npm. They had been a major version behind for
five months.

`js/vendor/vendor.json` is now the pinning record, and `vendor_frontend.py`
keeps it and the files honest:

```bash
scripts/vendor_frontend.py --verify    # offline: files vs manifest
scripts/vendor_frontend.py --check     # ... plus what npm has published
scripts/vendor_frontend.py --update    # install latest, rewrite manifest
```

The two halves are split on purpose. **Integrity is offline and fatal** -- two
file reads and a regex, so the pre-commit hook can run it without a network and
a mismatch blocks the commit, because the files and the record must never
disagree. **Staleness needs the network and is only reported** -- an npm outage
is not a broken repo, so `--check` prints what is behind and still exits 0.

In the hook it runs as `--verify --staged`, reading staged blobs rather than the
working tree: like `guard_stable_channel.sh`, it judges what is being committed,
so a dirty checkout cannot block an unrelated commit. The hook itself lives in
the untracked `.git/hooks/pre-commit.local`; the logic lives here, in the repo.

**Release step:** run `--check` before a release and name the answer, the same
way an uncovered `tests/smoke` arm gets named. That is the only thing that will
tell you a vendored library has moved.

Bumping is not automatic and should not be. `marked` in particular is a markdown
parser feeding an incremental renderer, and its majors have carried real
behavioural changes; after any update run `bun run e2e:render`, which diffs an
incrementally-grown document against a whole-document render and carries the
scheme-guard and raw-HTML checks.

## llama.cpp is built, not installed

`uv sync` installs the Python packages but **cannot build a C++ binary**, so
llama-server comes from an explicit `uv run scripts/build_llama.py` (or any
binary you point `$HEYLOOK_LLAMA_SERVER` / models.toml `server_binary` at --
Homebrew's `llama.cpp` and upstream's prebuilt release binaries both work).
llama.cpp has no semver: upstream tags every merge to master as `b<N>`, and
that tag IS the release; the script builds the newest one by default.

The script clones (blobless, so it stays small while any rev is still
checkoutable), refuses to check out over a dirty tree, builds, verifies the
binary, and writes a `heylook-build.json` manifest next to it recording the
rev, the resolved cmake config and a `sha256` of the binary.

**We do not vendor llama.cpp, and there is no submodule.** No upstream source
or build output is stored in this repo, tracked or untracked; nothing needs
initialising, updating, or remembering. The checkout goes to a fixed directory
under your home directory (`.heylook/llama.cpp`) -- outside the repo, so a
multi-GB build tree can never be committed, packaged into an sdist or wheel, or
shipped to anyone. It is the same path on every machine, so docs and error
messages can name it, and the server's binary fallback finds a build there
with zero config. `--dir` or `$HEYLOOK_LLAMA_CPP_DIR` move it.

The script does **not** touch `$HEYLOOK_LLAMA_SERVER` -- it prints a note if
that variable currently points somewhere else (in which case the server keeps
using the old binary, not the new build).

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

**A configure failure is retried once with `CMakeCache.txt` dropped.** cmake
caches every `find_package` result as a `FILEPATH` and never re-checks that the
file is still there, so a package manager upgrading a library out from under an
existing build tree leaves dangling paths that still read as *found*. Seen
2026-09-06: Homebrew replaced `openssl@3` 3.6.3 with 3.6.4, `FindOpenSSL`'s
`REQUIRED_VARS` (non-empty strings, not files) still passed, its imported
targets (guarded on `EXISTS`) were never created, and llama.cpp's vendored
cpp-httplib failed with `OpenSSL::SSL ... target was not found` -- a message
that names neither OpenSSL's absence nor the cache. The source was fine; the
tree was stale. The retry drops the cache file and nothing else, so objects and
the last good binary survive a heal that goes on to fail, and it fires only for
a **reused** tree: a fresh tree that fails to configure has a real problem.
`--clean` remains the bigger hammer.

Deliberately **not** set: `GGML_METAL_NDEBUG`. It compiles out load-time
logging only -- including the "allocated size is greater than the recommended
max working set size" warning. That is the ceiling that actually refuses loads
on this box (see `ram_report.py`), and a 138 GiB model against a ~161 GiB
working set is close enough that the warning earns its keep. No throughput to
gain, real diagnostics to lose.

## convert_gguf.py

Wrapper around the canonical checkout's `convert_hf_to_gguf.py`: local HF
safetensors -> GGUF text model (default `q8_0`, near-lossless, no
llama-quantize step) + `--mmproj` perception encoder at f16. torch rides a
`uv run --with` overlay -- nothing touches the repo venv. A brand-new
architecture may need the checkout at master
(`uv run python scripts/build_llama.py --rev master`) -- converter classes
land in llama.cpp's Python side separately from the C++ inference side.
Verify output with `scripts/gguf_probe.py` before registering it.
