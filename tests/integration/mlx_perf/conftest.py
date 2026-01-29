# tests/integration/mlx_perf/conftest.py
"""
Pytest configuration and fixtures for MLX performance integration tests.

These tests require:
- MLX installed (macOS only)
- A test model available
- Server not required for most tests (direct model tests)
"""

import os
import sys

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "mlx_perf: MLX performance optimization tests (requires MLX and model)",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def mlx_available():
    """Check if MLX is available."""
    if sys.platform != "darwin":
        pytest.skip("MLX requires macOS")

    try:
        import mlx.core as mx

        return True
    except ImportError:
        pytest.skip("MLX not installed")


@pytest.fixture(scope="session")
def test_model_path():
    """
    Get path to test model.
    """
    # Check env var first
    env_path = os.environ.get("HEYLOOK_TEST_MODEL")
    if env_path and os.path.isdir(env_path):
        return env_path

    # Check common locations
    common_paths = []

    for path in common_paths:
        if os.path.isdir(path):
            return path

    return None


@pytest.fixture(scope="session")
def loaded_model(mlx_available, test_model_path):
    """
    Load test model for benchmarks.

    Yields (model, tokenizer) tuple.
    Cleans up after tests complete.
    """
    if not test_model_path:
        pytest.skip("No test model available. Set HEYLOOK_TEST_MODEL env var.")

    try:
        import mlx.core as mx
        from mlx_lm.utils import load as lm_load

        model, tokenizer = lm_load(test_model_path)

        yield model, tokenizer

        # Cleanup
        del model
        del tokenizer
        mx.clear_cache()

    except Exception as e:
        pytest.skip(f"Could not load test model: {e}")


@pytest.fixture
def benchmark_prompt():
    """Standard prompt for benchmarking."""
    return "Explain the concept of machine learning in simple terms."


@pytest.fixture
def warmup_iterations():
    """Number of warmup iterations before benchmarking."""
    return 2


@pytest.fixture
def benchmark_iterations():
    """Number of benchmark iterations."""
    return 5
