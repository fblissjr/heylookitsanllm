# heylookitsanllm LLM Wiki

Welcome to the internal engineering wiki for **heylookitsanllm** (`heylook_llm`), a high-performance personal inference engine and user interface optimized for Apple Silicon (Metal) and macOS.

This wiki provides comprehensive, deep-dive technical documentation covering the core backend, frontend, provider layer, performance optimizations, and an exhaustive examination of how the system builds, configures, and spawns `llama-server` for GGUF models.

---

## Wiki Contents & Roadmap

| Document | Focus Area | Description |
| :--- | :--- | :--- |
| [**Architecture Overview**](./architecture_overview.md) | System & Architecture | High-level system topology, core invariants, data flow, thread model, and process boundaries. |
| [**Core Backend**](./backend_core.md) | Backend Subsystems | FastAPI app assembly and the real router/prefix map, the Messages wire, the single-threaded DuckDB store, the effect-class config engine, override-only model discovery, and request cancellation. |
| [**Core Frontend**](./frontend_core.md) | Vanilla JS Web UI | Zero-build frontend, hash routing, DuckDB store mirroring, document lifecycle, preset system, settings drawer, and page architecture. |
| [**Providers & Generation**](./providers_architecture.md) | Multi-Engine Providers | Unified provider abstraction (`BaseProvider`), `GenerationChunk` contract, MLXProvider (`mlx-lm` vs `mlx-vlm`), MLXEmbeddingProvider, reasoning parsers, and FIFO generation gate. |
| [**Llama-Server & GGUF Deep Dive**](./llama_server_build_and_spawn.md) | GGUF Subprocess & Build | **Core Deep Dive**: How `llama-server` is built from C++ source, how the backend spawns and manages it, exact CLI parameters sent and why, the chat template ladder, and how settings are configured by model. |
| [**Performance & Optimizations**](./performance_optimizations.md) | Cross-Stack Optimization | Complete guide to performance: Metal GPU shaders vs CPU glue, MLX prompt caching & KV snapshots, vision feature cache, speculative decoding, and incremental streaming UI rendering. |

---

## Architectural Principles at a Glance

1. **One Engine Wire, Dual Purpose**:
   - The server speaks an **Anthropic Messages-conformant wire** (`/v1/messages` and `/v1/conversations/{id}/generate`).
   - Legacy OpenAI completion endpoints have been completely retired. Internal code communicates via `ChatRequest`, and conversion to the Messages wire format occurs at the boundary.
2. **Apple Silicon First**:
   - Inference runs primarily on Apple Silicon Unified Memory and the Metal framework.
   - Compute-heavy arithmetic runs in GPU shaders via Metal, while the CPU handles scheduling, tokenization, and request routing.
3. **Subprocess Isolation for GGUF**:
   - While MLX models run in-process via Metal streams, GGUF models run as dedicated, isolated `llama-server` subprocesses (one process per loaded model).
   - Process lifecycle is tied directly to model residency: loading spawns a process group; unloading cleanly SIGTERMs the process group.
4. **Zero-Build, High-Fidelity Frontend**:
   - The web interface (`frontend/`) is 100% vanilla JavaScript, HTML, and CSS—no npm build step, no bundler, no SPA catch-all router.
   - The frontend acts as an exact mirror of the DuckDB store with strict invalidation rules and incremental, linear-time markdown streaming.
5. **Override-Only Model Configuration**:
   - `models.toml` is an override manifest, not an exhaustive inventory.
   - Local weights placed in scanned folders are automatically discovered, inspected via GGUF header/config analysis, and served with derived defaults without modifying `models.toml`.
   - The trade: an **explicit entry receives none of discovery's derived fields**. Adding one field means hand-writing every other field that model needs.
6. **No Numbers in This Wiki -- Link to Where They Live**:
   - Measures and tunable values do not belong in prose. A number copied out of the code is a second copy that drifts, which this repo already names as a defect with a delay; a number copied out of a measurement is worse, because it arrives without the conditions that make it mean anything.
   - So: **link to the constant, do not transcribe it.** Thresholds, window sizes, timeouts, batch sizes and defaults are named here by *what they do*, with a link to the file that declares them.
   - A measurement is only worth recording **with its conditions attached** -- the commit of every moving codebase (llama.cpp above all), the model and quant, the sampling, the prompt and generation lengths, the cache state. The ecosystem underneath this project changes constantly, and one measurement in isolation is usually not a finding. Where a single measure genuinely is meaningful, it gets recorded deliberately, in `internal/research/` or at the code site it explains -- not repeated here.
   - The same rule governs identifiers: a private helper named in passing becomes a **link**, not a spelled-out name. What stays in prose is the stable, load-bearing vocabulary -- the architectural type names, the wire routes, the llama-server CLI flags, HTTP status codes -- things that either do not move, or whose movement would mean rewriting this prose anyway.

---

## Key Repository Paths

- **Backend Package**: [`src/heylook_llm/`](../../src/heylook_llm)
- **Frontend App**: [`frontend/`](../../frontend)
- **Llama.cpp Build Automation**: [`scripts/build_llama.py`](../../scripts/build_llama.py)
- **GGUF Provider Implementation**: [`src/heylook_llm/providers/llama_server_provider.py`](../../src/heylook_llm/providers/llama_server_provider.py)
- **Configuration & Schemas**: [`src/heylook_llm/config.py`](../../src/heylook_llm/config.py)
- **Model Registry & Scanner**: [`src/heylook_llm/model_registry.py`](../../src/heylook_llm/model_registry.py), [`src/heylook_llm/model_importer.py`](../../src/heylook_llm/model_importer.py)
