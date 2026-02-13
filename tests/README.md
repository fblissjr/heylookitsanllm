# Test Suite

last updated: 2026-02-12

## Overview

Backend tests for the heylookitsanllm API server. For frontend tests (711 tests across 31 files), see `apps/heylook-frontend/`.

## Organization

```
tests/
  unit/              # Fast isolated tests, no server required
  integration/       # Tests requiring running server
  fixtures/          # Shared test data and helpers
  input/             # Test input files (images, audio)
  archive/           # Old/superseded tests (62 files)
  README.md          # This file
  run_tests.sh       # Test runner script
```

### Unit Tests (tests/unit/)

Fast, isolated tests that don't require a running server:

- **test_config.py** - Config loading validation
- **test_llama_cpp_provider.py** - LlamaCppProvider unit tests with mocking (4 tests)
- **test_router.py** - ModelRouter LRU cache and provider loading tests (5 tests)

### Integration Tests (tests/integration/)

Tests requiring a running heylookllm server:

- **test_api_integration.py** - API endpoint testing, MLX provider validation (4 tests)
- **test_batch_integration.py** - Batch text processing endpoint (1 test)
- **test_embeddings_integration.py** - Embeddings API endpoint (1 test)
- **test_keepalive.py** - Keepalive and prompt caching (1 test)
- **test_queue_integration.py** - Request queuing system (5 tests)
- **test_stt_integration.py** - MLX STT endpoint (3 tests)

## Running Tests

### Prerequisites

For integration tests, start the server:
```bash
heylookllm --port 8080
```

### Commands

```bash
cd tests && pytest                # All tests
cd tests && pytest unit/          # Unit only
cd tests && pytest integration/   # Integration (requires server)
cd tests && pytest -m unit        # By marker
cd tests && pytest -v             # Verbose
```

## Coverage Matrix

| Feature Area | Coverage | Test Files | Gaps |
|---|---|---|---|
| Model Loading/Routing | Good | test_router.py | No concurrent access tests |
| API Endpoints | Good | test_api_integration.py | -- |
| Batch Processing | Basic | test_batch_integration.py | No error handling tests |
| Embeddings | Basic | test_embeddings_integration.py | No multi-model tests |
| STT (MLX) | Good | test_stt_integration.py | No format validation |
| Queueing | Good | test_queue_integration.py | No stress tests |
| Keepalive/Caching | Basic | test_keepalive.py | No cache invalidation tests |
| Llama.cpp Provider | Good | test_llama_cpp_provider.py | No vision support tests |
| **MLX Provider** | **None** | -- | **No unit tests** |
| **Error Handling** | **Poor** | Scattered | **No dedicated error tests** |

## Testing Guidelines

- Use pytest for new tests
- Create fixtures for shared test data in `conftest.py`
- Test error cases, not just happy path
- Mock external dependencies in unit tests
- Descriptive names: `test_router_evicts_lru_model_when_cache_full`

## Related

- `apps/heylook-frontend/` - Frontend tests (711 tests, 31 files, Vitest)
- `docs/FRONTEND_HANDOFF.md` - API reference for integration tests
