# Test Suite

Reorganized: 2025-11-17

## Overview

This test suite has been aggressively reorganized from 69 files to 10 focused tests. The goal is a maintainable, clear test structure that covers core functionality without duplication or one-off debug scripts.

## Organization

```
tests/
├── unit/              # Fast isolated tests, no server required (3 tests)
├── integration/       # Tests requiring running server (7 tests)
├── fixtures/          # Shared test data and helpers
├── input/             # Test input files (images, audio)
├── archive/           # Old/superseded tests (62 files)
├── README.md          # This file
└── run_tests.sh       # Test runner script
```

### Unit Tests (tests/unit/)

Fast, isolated tests that don't require a running server:

- **test_config.py** (6 lines, 0 test functions)
  - Basic config loading validation
  - Issue: Not using proper test framework

- **test_llama_cpp_provider.py** (99 lines, 4 test functions)
  - LlamaCppProvider unit tests with mocking
  - Uses unittest framework

- **test_router.py** (165 lines, 5 test functions)
  - ModelRouter LRU cache tests
  - Provider loading/unloading
  - Uses unittest framework

### Integration Tests (tests/integration/)

Tests requiring a running heylookllm server:

- **test_api_integration.py** (410 lines, 4 test functions)
  - Comprehensive API endpoint testing
  - MLX provider optimization validation
  - Uses pytest

- **test_batch_integration.py** (199 lines, 1 test function)
  - Batch text processing endpoint
  - Recent (Nov 2025)
  - Uses requests directly

- **test_embeddings_integration.py** (105 lines, 1 test function)
  - Embeddings API endpoint testing
  - Starts/stops server
  - Uses requests directly

- **test_keepalive.py** (62 lines, 1 test function)
  - Keepalive and prompt caching
  - Uses requests directly

- **test_queue_integration.py** (198 lines, 5 test functions)
  - Request queuing system
  - Uses pytest

- **test_stt_integration.py** (186 lines, 3 test functions)
  - Speech-to-text API endpoint
  - Recent (Sept 2025)
  - Uses requests directly

### Fixtures (tests/fixtures/)

Shared test data and helpers:
- `__init__.py` - Module initialization

### Input Data (tests/input/)

Test input files (images, audio, etc.)

## Running Tests

### Prerequisites

For integration tests, start the server:
```bash
heylookllm --api openai --port 8080
```

### Run All Tests

```bash
# Using pytest
cd tests && pytest

# Using shell script
./tests/run_tests.sh
```

### Run Specific Categories

```bash
# Unit tests only (no server needed)
cd tests && pytest unit/

# Integration tests (requires server)
cd tests && pytest integration/

# Specific test file
cd tests && pytest integration/test_api_integration.py

# Verbose output
cd tests && pytest -v

# Show print statements
cd tests && pytest -s
```

### Run Individual Tests

```bash
# Unit tests
python tests/unit/test_router.py
python -m unittest tests.unit.test_llama_cpp_provider

# Integration tests (server must be running)
python tests/integration/test_api_integration.py
python tests/integration/test_batch_integration.py
```

## Test Coverage Matrix

| Feature Area | Current Coverage | Test Files | Quality | Gaps | Priority |
|--------------|------------------|------------|---------|------|----------|
| **Model Loading** | Good | test_router.py | Good (unittest) | No edge case tests for LRU eviction | Medium |
| **Model Routing** | Good | test_router.py | Good (unittest) | No concurrent access tests | Medium |
| **API Endpoints** | Excellent | test_api_integration.py | Excellent (pytest) | None identified | Low |
| **Batch Processing** | Good | test_batch_integration.py | Fair (no framework) | No error handling tests | High |
| **Embeddings** | Basic | test_embeddings_integration.py | Fair (no framework) | No multi-model tests | Medium |
| **STT (Speech-to-Text)** | Good | test_stt_integration.py | Fair (no framework) | No format validation tests | Medium |
| **Queueing** | Good | test_queue_integration.py | Good (pytest) | No stress tests | Medium |
| **Keepalive/Caching** | Basic | test_keepalive.py | Fair (no framework) | No cache invalidation tests | High |
| **Llama.cpp Provider** | Good | test_llama_cpp_provider.py | Good (unittest) | No vision support tests | Low |
| **MLX Provider** | None | None | N/A | No unit tests for MLX provider | High |
| **CoreML STT Provider** | None | None | N/A | No unit tests for STT provider | High |
| **Config/YAML** | Minimal | test_config.py | Poor (no framework) | No validation tests | Medium |
| **Error Handling** | Poor | Scattered | N/A | No dedicated error tests | High |
| **Performance** | None | None | N/A | No performance regression tests | Medium |
| **Vision Models** | Basic | test_api_integration.py | Fair | No multimodal edge cases | Medium |

## Quality Assessment

### Test Quality Issues

| File | Issues | Recommendations |
|------|--------|-----------------|
| test_config.py | No test framework, just a script | Convert to pytest, add validation tests |
| test_batch_integration.py | No framework, no assertions | Add pytest, proper assertions, error cases |
| test_embeddings_integration.py | Server management in test, no framework | Use pytest fixtures for server |
| test_keepalive.py | No framework, minimal coverage | Add pytest, test cache invalidation |
| test_stt_integration.py | No framework, generates audio in test | Move audio generation to fixtures |

### Good Practices Found

| File | Good Practices |
|------|----------------|
| test_router.py | Proper unittest, mocking, good coverage |
| test_llama_cpp_provider.py | Proper unittest, comprehensive mocking |
| test_api_integration.py | Good pytest usage, comprehensive |
| test_queue_integration.py | Good pytest usage, proper fixtures |

### Testing Anti-Patterns Identified

1. **No Framework**: Several tests are raw scripts without pytest/unittest
2. **No Fixtures**: Shared setup code duplicated across tests
3. **Server Management**: Tests manually start/stop server (should use fixtures)
4. **No Assertions**: Some tests print results but don't assert correctness
5. **Inline Data**: Test data generated inline instead of in fixtures
6. **No Error Tests**: Very few tests validate error handling

## High-Priority Gaps to Fill

### Critical (Highest Priority)

1. **MLX Provider Unit Tests** - Core provider has no unit tests
2. **CoreML STT Provider Unit Tests** - STT provider has no unit tests
3. **Error Handling Tests** - No dedicated error scenario testing
4. **Keepalive Cache Tests** - Cache invalidation not tested

### High Priority

5. **Batch Error Handling** - Batch endpoint needs error tests
6. **Provider Edge Cases** - Vision model edge cases (multi-image, large images)
7. **Config Validation** - models.toml validation not tested
8. **LRU Cache Edge Cases** - Concurrent eviction scenarios

### Medium Priority

9. **Performance Regression Tests** - Track performance over time
10. **Multi-Model Embeddings** - Test different embedding models
11. **STT Format Validation** - Test various audio formats
12. **Router Concurrency** - Stress test model loading under load

## Testing Guidelines

When writing new tests:

### DO

- Use pytest for new tests (preferred) or unittest
- Create fixtures for shared test data
- Use proper assertions (assert, self.assertEqual, etc.)
- Test error cases, not just happy path
- Mock external dependencies in unit tests
- Keep unit tests fast (< 100ms each)
- Document what each test validates
- Use descriptive test names: `test_router_evicts_lru_model_when_cache_full`

### DON'T

- Create standalone scripts without test framework
- Print results instead of asserting
- Duplicate test setup across files
- Start/stop server in test code (use fixtures)
- Test multiple unrelated things in one test
- Leave debug print statements
- Create one-off test files for investigations (use archive/)

### Test Structure Template

```python
#!/usr/bin/env python3
"""
Brief description of what this test file validates.

Requires: server running / no server / etc.
"""

import pytest
# other imports

@pytest.fixture
def setup_test_data():
    """Fixture for shared test data."""
    # setup
    yield data
    # teardown

def test_specific_behavior(setup_test_data):
    """Test that X behaves correctly when Y."""
    # Arrange
    input_data = setup_test_data

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value
    assert result.status == "success"
```

## Pytest Configuration

Create `pytest.ini` for test discovery:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests (no server required)
    integration: Integration tests (server required)
    slow: Slow tests (> 1s)
```

Mark tests with:
```python
@pytest.mark.unit
def test_router_loads_model():
    pass

@pytest.mark.integration
def test_api_endpoint():
    pass
```

Run specific markers:
```bash
pytest -m unit        # Only unit tests
pytest -m integration # Only integration tests
```

## Coverage Reporting

Install pytest-cov:
```bash
uv pip install pytest-cov
```

Run with coverage:
```bash
cd tests && pytest --cov=heylook_llm --cov-report=html
open htmlcov/index.html
```

## Next Steps

1. **Add pytest.ini** - Configure test discovery and markers
2. **Convert Raw Scripts** - Convert test_config.py, test_batch_integration.py to pytest
3. **Add MLX Provider Tests** - Critical gap in coverage
4. **Add Error Tests** - Dedicated error scenario testing
5. **Add Fixtures** - Create conftest.py with shared fixtures
6. **Performance Tests** - Add performance regression tracking
7. **CI/CD Integration** - Add GitHub Actions for automated testing

## Troubleshooting

### Server Not Running

```bash
# Check if server is running
curl http://localhost:8080/health

# Start server
heylookllm --api openai --port 8080
```

### Import Errors

```bash
# Ensure package is installed (recommended: uv sync)
uv sync

# For MLX support (macOS only)
uv sync --extra mlx

# For llama.cpp support
uv sync --extra llama-cpp
```

### Test Collection Issues

```bash
# Verify pytest can find tests
pytest --collect-only

# Run with verbose output
pytest -v

# Show why tests are skipped
pytest -v -rs
```

### Port Conflicts

If port 8080 is in use:
```bash
# Find process using port
lsof -i :8080

# Kill process
kill -9 <PID>

# Or use different port
heylookllm --api openai --port 8081
```

## Related Documentation

- `/docs/TESTING_GUIDE.md` - MLX provider optimization testing guide (older, pre-reorganization)
- `/tests/archive/README.md` - Details on archived tests
- `/internal/BATCH_TEXT_IMPLEMENTATION_PLAN.md` - Batch processing implementation
- `/docs/CLIENT_INTEGRATION_GUIDE.md` - API integration guide
