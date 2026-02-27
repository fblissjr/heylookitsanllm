# Test Suite

last updated: 2026-02-27

## Overview

Backend tests for the heylookitsanllm API server. For frontend tests (874 tests across 38 files), see `apps/heylook-frontend/`.

## Organization

```
tests/
  unit/              # Fast isolated tests, no server required
  unit/mlx_perf/     # MLX performance correctness tests
  contract/          # API contract tests with TestClient (no server required)
  integration/       # Tests requiring running server
  integration/mlx_perf/  # Performance benchmarks (requires server + MLX)
  helpers/           # Shared mocking utilities (mlx_mock.py)
  fixtures/          # Shared test data
  input/             # Test input files (images, audio)
  conftest.py        # Root fixtures (chat requests, temp config)
  README.md          # This file
```

## Running Tests

```bash
# Unit + contract (no server needed)
uv run pytest tests/unit/ tests/contract/ -v

# Unit only
uv run pytest tests/unit/ -v

# Contract only
uv run pytest tests/contract/ -v

# Integration (requires running server on port 8080)
uv run pytest tests/integration/ -v

# All tests
uv run pytest tests/ -v
```

**Pre-existing failures:** 5 router tests (YAML config vs TOML parser), 3 mlx_perf tests. Do not investigate.

## Unit Tests (`tests/unit/`)

17 test files + 3 mlx_perf sub-files:

- **test_mlx_provider.py** -- MLX provider: loading, generation, vision, streaming
- **test_mlx_provider_safety.py** -- MLX provider safety guards
- **test_config.py** -- Config loading, validation, provider types
- **test_router.py** -- ModelRouter LRU cache and provider loading (5 tests, all fail -- pre-existing)
- **test_model_service.py** -- Model profiles, smart defaults, size regex
- **test_generation_core.py** -- Core generation logic
- **test_messages_api.py** -- Anthropic Messages-style API
- **test_hidden_states.py** -- Hidden states extraction, base64 encoding, structured requests
- **test_thinking_parser.py** -- Thinking block parsing
- **test_thinking_roundtrip.py** -- Thinking block roundtrip fidelity
- **test_radix_cache.py** -- Radix/prefix cache
- **test_speculative.py** -- Speculative decoding
- **test_draft_tuner.py** -- Draft model tuning for speculative decoding
- **test_vlm_inputs.py** -- VLM input handling
- **test_abort.py** -- Request abort/cancellation
- **test_samplers.py** -- Sampling strategies
- **test_unified_equivalence.py** -- OpenAI/Messages API equivalence

**mlx_perf/** (3 files):
- **test_type_consistency.py** -- MLX type consistency checks
- **test_sync_boundaries.py** -- Synchronization boundary tests
- **test_compilation_correctness.py** -- Compilation correctness (3 tests fail -- pre-existing)

## Contract Tests (`tests/contract/`)

API contract tests using FastAPI TestClient with mocked router/service. No real models or server needed.

- **test_chat_completions.py** -- OpenAI chat completions: streaming, non-streaming, error cases
- **test_messages.py** -- Anthropic Messages: content blocks, streaming events, typed input
- **test_openapi_conformance.py** -- OpenAPI schema structure, route coverage, core endpoints
- **test_admin.py** -- Admin API: config listing, profiles, scan, status
- **test_models_endpoint.py** -- Model list: structure, required fields, provider info

## Integration Tests (`tests/integration/`)

Tests requiring a running heylookllm server (`heylookllm --port 8080`):

- **test_hidden_states_api.py** -- Hidden states endpoints (raw + structured)
- **test_logprobs.py** -- Log probability extraction
- **test_api_integration.py** -- API endpoint validation, MLX provider
- **test_stt_integration.py** -- MLX STT transcription
- **test_batch_integration.py** -- Batch text processing
- **test_embeddings_integration.py** -- Embeddings API
- **test_keepalive.py** -- Keepalive and prompt caching

**mlx_perf/** (2 files):
- **test_baseline_benchmarks.py** -- Performance baseline measurements
- **test_memory_profiling.py** -- Memory usage profiling

## Coverage Matrix

| Feature Area | Coverage | Test Location | Notes |
|---|---|---|---|
| MLX Provider | Good | unit/test_mlx_provider*.py | 45 tests |
| Config/Validation | Good | unit/test_config.py | 23 tests |
| Model Service | Good | unit/test_model_service.py | Profiles, defaults, size regex |
| Model Routing | Broken | unit/test_router.py | 5 tests, all fail (YAML vs TOML) |
| Generation Core | Good | unit/test_generation_core.py | |
| Messages API | Good | unit/test_messages_api.py, contract/test_messages.py | Unit + contract |
| Chat Completions | Good | contract/test_chat_completions.py | Streaming + non-streaming |
| Hidden States | Good | unit/test_hidden_states.py, integration/test_hidden_states_api.py | Unit + integration |
| Thinking Blocks | Good | unit/test_thinking_parser.py, test_thinking_roundtrip.py | 47 tests |
| Radix Cache | Good | unit/test_radix_cache.py | 26 tests |
| Speculative Decoding | Good | unit/test_speculative.py, test_draft_tuner.py | 27 tests |
| Logprobs | Good | integration/test_logprobs.py | Requires server |
| VLM Inputs | Good | unit/test_vlm_inputs.py | |
| Abort/Cancel | Good | unit/test_abort.py | |
| Samplers | Basic | unit/test_samplers.py | 5 tests |
| OpenAPI Conformance | Good | contract/test_openapi_conformance.py | Schema validation |
| Admin API | Good | contract/test_admin.py | Config, profiles, scan |
| STT | Basic | integration/test_stt_integration.py | Requires server |
| Batch Processing | Basic | integration/test_batch_integration.py | Requires server |
| Embeddings | Basic | integration/test_embeddings_integration.py | Requires server |
| **Error Handling** | **Poor** | Scattered | **No dedicated error tests** |

## Testing Guidelines

- Use pytest for new tests
- Create fixtures for shared test data in `conftest.py`
- Test error cases, not just happy path
- Mock external dependencies in unit tests (see `helpers/mlx_mock.py`)
- Descriptive names: `test_router_evicts_lru_model_when_cache_full`
- Contract tests use `TestClient` -- no server needed, fast iteration

## Related

- `apps/heylook-frontend/` -- Frontend tests (874 tests, 38 files, Vitest)
- `docs/FRONTEND_HANDOFF.md` -- API reference for integration tests
