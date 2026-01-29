# tests/integration/mlx_perf/test_baseline_benchmarks.py
"""
Baseline performance benchmarks for MLX generation.

These tests establish and verify performance baselines:
- Tokens per second (threshold: >10 tok/s for 8-bit model)
- Time to first token (threshold: <500ms)
- End-to-end latency distribution

Run with: cd tests && uv run pytest -m mlx_perf integration/mlx_perf/test_baseline_benchmarks.py -v
"""
import pytest
import sys
import time
from typing import List

pytestmark = [
    pytest.mark.mlx_perf,
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]


class TestTokenThroughput:
    """Tests for tokens per second metrics."""

    @pytest.mark.slow
    def test_generation_throughput_above_threshold(
        self, loaded_model, benchmark_prompt, warmup_iterations, benchmark_iterations
    ):
        """Verify generation achieves minimum throughput."""
        import mlx.core as mx
        from mlx_lm.generate import stream_generate

        model, tokenizer = loaded_model
        min_throughput = 10.0  # tokens/sec threshold for 8-bit model

        # Tokenize prompt
        prompt_tokens = tokenizer.encode(benchmark_prompt)

        # Warmup
        for _ in range(warmup_iterations):
            tokens = []
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=20,
            ):
                tokens.append(response.token)
            mx.synchronize()

        # Benchmark
        throughputs = []
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            token_count = 0

            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=50,
            ):
                token_count += 1

            mx.synchronize()
            elapsed = time.perf_counter() - start

            throughput = token_count / elapsed if elapsed > 0 else 0
            throughputs.append(throughput)

        avg_throughput = sum(throughputs) / len(throughputs)
        min_observed = min(throughputs)

        print(f"\nThroughput: avg={avg_throughput:.1f} tok/s, min={min_observed:.1f} tok/s")

        assert avg_throughput >= min_throughput, (
            f"Average throughput {avg_throughput:.1f} tok/s below threshold {min_throughput} tok/s"
        )


class TestTimeToFirstToken:
    """Tests for time-to-first-token latency."""

    @pytest.mark.slow
    def test_ttft_below_threshold(
        self, loaded_model, benchmark_prompt, warmup_iterations, benchmark_iterations
    ):
        """Verify time to first token is within acceptable range."""
        import mlx.core as mx
        from mlx_lm.generate import stream_generate

        model, tokenizer = loaded_model
        max_ttft_ms = 500  # Maximum acceptable TTFT in milliseconds

        prompt_tokens = tokenizer.encode(benchmark_prompt)

        # Warmup
        for _ in range(warmup_iterations):
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=5,
            ):
                break  # Only need first token
            mx.synchronize()

        # Benchmark TTFT
        ttfts = []
        for _ in range(benchmark_iterations):
            start = time.perf_counter()

            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=5,
            ):
                ttft = (time.perf_counter() - start) * 1000  # Convert to ms
                ttfts.append(ttft)
                break

            mx.synchronize()

        avg_ttft = sum(ttfts) / len(ttfts)
        p95_ttft = sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) >= 20 else max(ttfts)

        print(f"\nTTFT: avg={avg_ttft:.1f}ms, p95={p95_ttft:.1f}ms")

        assert avg_ttft <= max_ttft_ms, (
            f"Average TTFT {avg_ttft:.1f}ms exceeds threshold {max_ttft_ms}ms"
        )


class TestLatencyDistribution:
    """Tests for end-to-end latency consistency."""

    @pytest.mark.slow
    def test_latency_variance_acceptable(
        self, loaded_model, benchmark_prompt, benchmark_iterations
    ):
        """Verify latency variance is within acceptable bounds."""
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        import statistics

        model, tokenizer = loaded_model
        max_cv = 0.3  # Maximum coefficient of variation (30%)

        prompt_tokens = tokenizer.encode(benchmark_prompt)

        # Collect latency samples
        latencies = []
        for _ in range(benchmark_iterations * 2):  # More samples for statistics
            start = time.perf_counter()
            token_count = 0

            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=30,
            ):
                token_count += 1

            mx.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)

        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        cv = std_latency / mean_latency if mean_latency > 0 else 0

        print(f"\nLatency: mean={mean_latency*1000:.1f}ms, std={std_latency*1000:.1f}ms, CV={cv:.2f}")

        assert cv <= max_cv, (
            f"Latency coefficient of variation {cv:.2f} exceeds threshold {max_cv}"
        )


class TestSamplerPerformance:
    """Tests for sampler compilation performance impact."""

    def test_presence_penalty_overhead(self, loaded_model, benchmark_prompt):
        """Verify presence penalty adds minimal overhead."""
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        from heylook_llm.providers.common.samplers import make_sampler, make_logits_processors, make_presence_penalty_processor

        model, tokenizer = loaded_model
        max_overhead_percent = 15  # Maximum acceptable overhead

        prompt_tokens = tokenizer.encode(benchmark_prompt)

        # Baseline without presence penalty
        start = time.perf_counter()
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            max_tokens=50,
        ):
            pass
        mx.synchronize()
        baseline_time = time.perf_counter() - start

        # With presence penalty processor
        processors = [make_presence_penalty_processor(1.5)]

        start = time.perf_counter()
        for response in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            max_tokens=50,
            logits_processors=processors,
        ):
            pass
        mx.synchronize()
        penalty_time = time.perf_counter() - start

        overhead_percent = ((penalty_time - baseline_time) / baseline_time) * 100

        print(f"\nPresence penalty overhead: {overhead_percent:.1f}%")

        assert overhead_percent <= max_overhead_percent, (
            f"Presence penalty overhead {overhead_percent:.1f}% exceeds {max_overhead_percent}%"
        )
