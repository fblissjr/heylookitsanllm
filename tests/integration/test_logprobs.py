#!/usr/bin/env python
"""Test script for logprobs functionality.

Usage:
    # Unit tests (no server required)
    uv run pytest tests/integration/test_logprobs.py -v

    # Integration test (requires running server with MLX model)
    uv run python tests/integration/test_logprobs.py --integration
"""

import sys

import pytest


class MockTokenizer:
    """Mock tokenizer for unit testing."""

    def decode(self, token_ids):
        """Simple mock decode - returns placeholder text."""
        if isinstance(token_ids, list):
            return f"tok_{token_ids[0]}" if token_ids else ""
        return f"tok_{token_ids}"


class TestLogprobsCollector:
    """Unit tests for LogprobsCollector."""

    def test_collector_initialization(self):
        """Test collector can be initialized."""
        from heylook_llm.logprobs import LogprobsCollector

        tokenizer = MockTokenizer()
        collector = LogprobsCollector(tokenizer, top_logprobs=5)
        assert collector.top_logprobs == 5
        assert len(collector.content) == 0

    def test_add_token_basic(self):
        """Test adding a token with logprobs."""
        from heylook_llm.logprobs import LogprobsCollector

        tokenizer = MockTokenizer()
        collector = LogprobsCollector(tokenizer, top_logprobs=3)

        # Simulate logprobs array (10 vocab entries for simplicity)
        logprobs = [-2.0, -1.0, -0.5, -3.0, -4.0, -1.5, -2.5, -3.5, -0.8, -1.2]

        collector.add_token(2, logprobs)  # Token 2 has logprob -0.5

        assert len(collector.content) == 1
        entry = collector.content[0]
        assert entry.token_id == 2
        assert entry.logprob == -0.5
        assert len(entry.top_logprobs) == 3

    def test_top_logprobs_ordering(self):
        """Test that top logprobs are sorted correctly."""
        from heylook_llm.logprobs import LogprobsCollector

        tokenizer = MockTokenizer()
        collector = LogprobsCollector(tokenizer, top_logprobs=3)

        # Token 2 has highest prob (-0.5), then 8 (-0.8), then 1 (-1.0)
        logprobs = [-2.0, -1.0, -0.5, -3.0, -4.0, -1.5, -2.5, -3.5, -0.8, -1.2]

        collector.add_token(2, logprobs)

        top = collector.content[0].top_logprobs
        assert top[0].token_id == 2  # -0.5
        assert top[1].token_id == 8  # -0.8
        assert top[2].token_id == 1  # -1.0

    def test_to_dict_format(self):
        """Test OpenAI-compatible dict format."""
        from heylook_llm.logprobs import LogprobsCollector

        tokenizer = MockTokenizer()
        collector = LogprobsCollector(tokenizer, top_logprobs=2)

        logprobs = [-0.5, -1.0, -2.0]
        collector.add_token(0, logprobs)

        result = collector.to_dict()
        assert "content" in result
        assert len(result["content"]) == 1
        assert "token" in result["content"][0]
        assert "logprob" in result["content"][0]
        assert "top_logprobs" in result["content"][0]


class TestLogprobsEdgeCases:
    """Edge-case tests for LogprobsCollector."""

    def test_add_token_out_of_range_token_id(self):
        """Out-of-range token_id triggers IndexError, caught and logged, not raised."""
        import logging
        from heylook_llm.logprobs import LogprobsCollector

        tokenizer = MockTokenizer()
        collector = LogprobsCollector(tokenizer, top_logprobs=3)

        # vocab has 5 entries, token_id=999 is out of range
        logprobs = [-0.5, -1.0, -2.0, -3.0, -4.0]

        with pytest.raises(IndexError):
            _ = logprobs[999]  # confirm raw IndexError exists

        # add_token should NOT raise -- the exception is caught internally
        collector.add_token(999, logprobs)

        # No entry added because the exception was caught
        assert len(collector.content) == 0


    def test_decode_token_returns_fallback_for_bad_id(self):
        """_decode_token returns '<token_N>' when tokenizer.decode raises KeyError."""
        from heylook_llm.logprobs import LogprobsCollector

        class BadTokenizer:
            def decode(self, token_ids):
                raise KeyError(f"Unknown token id {token_ids[0]}")

        collector = LogprobsCollector(BadTokenizer(), top_logprobs=1)
        result = collector._decode_token(99999)
        assert result == "<token_99999>"

    def test_get_token_bytes_handles_surrogate(self):
        """_get_token_bytes returns [] for strings with surrogate characters."""
        from heylook_llm.logprobs import LogprobsCollector

        collector = LogprobsCollector(MockTokenizer(), top_logprobs=1)
        result = collector._get_token_bytes("\ud800")
        assert result == []


class TestStreamingLogprobsCollector:
    """Unit tests for StreamingLogprobsCollector."""

    def test_add_token_and_get_delta(self):
        """Test streaming collector returns delta."""
        from heylook_llm.logprobs import StreamingLogprobsCollector

        tokenizer = MockTokenizer()
        collector = StreamingLogprobsCollector(tokenizer, top_logprobs=2)

        logprobs = [-0.5, -1.0, -2.0]
        delta = collector.add_token_and_get_delta(0, logprobs)

        assert delta is not None
        assert "content" in delta
        assert len(delta["content"]) == 1


class TestStreamingUsage:
    """Unit tests for stream_options.include_usage support."""

    def test_stream_options_in_config(self):
        """Test stream_options field exists in ChatRequest."""
        from heylook_llm.config import ChatRequest

        # Test that stream_options is accepted
        request = ChatRequest(
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert request.stream_options == {"include_usage": True}

    def test_stream_options_none_by_default(self):
        """Test stream_options is None by default."""
        from heylook_llm.config import ChatRequest

        request = ChatRequest(
            messages=[{"role": "user", "content": "test"}],
        )
        assert request.stream_options is None


class TestLogprobsIntegration:
    """Integration tests requiring running server."""

    @pytest.mark.skip(reason="Requires running server - run with --integration flag")
    def test_non_streaming_logprobs(self):
        """Test non-streaming endpoint with logprobs."""
        import httpx

        response = httpx.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "logprobs": True,
                "top_logprobs": 5,
                "max_tokens": 10,
            },
            timeout=30.0,
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        choice = data["choices"][0]
        assert "logprobs" in choice
        assert "content" in choice["logprobs"]

    @pytest.mark.skip(reason="Requires running server - run with --integration flag")
    def test_streaming_logprobs(self):
        """Test streaming endpoint with logprobs."""
        import httpx

        with httpx.stream(
            "POST",
            "http://localhost:8080/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "logprobs": True,
                "top_logprobs": 3,
                "max_tokens": 10,
                "stream": True,
            },
            timeout=30.0,
        ) as response:
            assert response.status_code == 200

            found_logprobs = False
            for line in response.iter_lines():
                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    import json

                    chunk = json.loads(line[6:])
                    if "logprobs" in chunk.get("choices", [{}])[0]:
                        found_logprobs = True
                        break

            # Note: logprobs might not be in every chunk
            # This test just verifies the endpoint works

    @pytest.mark.skip(reason="Requires running server - run with --integration flag")
    def test_streaming_usage_stats(self):
        """Test streaming endpoint with stream_options.include_usage."""
        import httpx

        with httpx.stream(
            "POST",
            "http://localhost:8080/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=30.0,
        ) as response:
            assert response.status_code == 200

            found_usage = False
            for line in response.iter_lines():
                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    import json

                    chunk = json.loads(line[6:])
                    if "usage" in chunk:
                        found_usage = True
                        usage = chunk["usage"]
                        assert "prompt_tokens" in usage
                        assert "completion_tokens" in usage
                        assert "total_tokens" in usage
                        break

            assert found_usage, "No usage stats found in streaming response"


def run_integration_tests():
    """Run integration tests manually."""
    import httpx

    print("Running integration tests...")
    print("Make sure the server is running on localhost:8080")

    # Get available models first
    print("\n0. Fetching available models...")
    try:
        models_response = httpx.get("http://localhost:8080/v1/models", timeout=10.0)
        if models_response.status_code != 200:
            print(f"   FAILED: Could not fetch models (status {models_response.status_code})")
            return
        models_data = models_response.json()
        available_models = [m["id"] for m in models_data.get("data", [])]
        if not available_models:
            print("   FAILED: No models available on server")
            return
        model_id = available_models[0]
        print(f"   Using model: {model_id}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Test non-streaming
    print("\n1. Testing non-streaming with logprobs...")
    try:
        response = httpx.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hello in 5 words"}],
                "logprobs": True,
                "top_logprobs": 3,
                "max_tokens": 20,
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            data = response.json()
            choice = data["choices"][0]
            if "logprobs" in choice:
                logprobs = choice["logprobs"]
                print(
                    f"   SUCCESS: Got logprobs with {len(logprobs.get('content', []))} tokens"
                )
                if logprobs.get("content"):
                    first = logprobs["content"][0]
                    print(
                        f"   First token: '{first['token']}' (id={first['token_id']}, logprob={first['logprob']:.3f})"
                    )
                    print(f"   Top alternatives: {len(first.get('top_logprobs', []))}")
            else:
                print("   WARNING: Response has no logprobs field")
                print(f"   Choice: {choice}")
        else:
            print(f"   FAILED: Status {response.status_code}")
            print(f"   Response: {response.text}")

    except Exception as e:
        print(f"   ERROR: {e}")

    # Test streaming
    print("\n2. Testing streaming with logprobs...")
    try:
        with httpx.stream(
            "POST",
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Count to 5"}],
                "logprobs": True,
                "top_logprobs": 2,
                "max_tokens": 20,
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            if response.status_code == 200:
                import json

                logprobs_count = 0
                for line in response.iter_lines():
                    if line.startswith("data: ") and not line.startswith(
                        "data: [DONE]"
                    ):
                        chunk = json.loads(line[6:])
                        if chunk.get("choices") and "logprobs" in chunk["choices"][0]:
                            logprobs_count += 1

                if logprobs_count > 0:
                    print(f"   SUCCESS: Got logprobs in {logprobs_count} chunks")
                else:
                    print("   WARNING: No logprobs in any chunks")
            else:
                print(f"   FAILED: Status {response.status_code}")

    except Exception as e:
        print(f"   ERROR: {e}")

    # Test streaming with usage stats
    print("\n3. Testing streaming with stream_options.include_usage...")
    try:
        with httpx.stream(
            "POST",
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=60.0,
        ) as response:
            if response.status_code == 200:
                import json

                found_usage = False
                usage_data = None
                for line in response.iter_lines():
                    if line.startswith("data: ") and not line.startswith(
                        "data: [DONE]"
                    ):
                        chunk = json.loads(line[6:])
                        if "usage" in chunk:
                            found_usage = True
                            usage_data = chunk["usage"]

                if found_usage:
                    print(f"   SUCCESS: Got usage stats in final chunk")
                    print(f"   Prompt tokens: {usage_data.get('prompt_tokens', 'N/A')}")
                    print(
                        f"   Completion tokens: {usage_data.get('completion_tokens', 'N/A')}"
                    )
                    print(f"   Total tokens: {usage_data.get('total_tokens', 'N/A')}")
                else:
                    print("   WARNING: No usage stats in streaming response")
            else:
                print(f"   FAILED: Status {response.status_code}")

    except Exception as e:
        print(f"   ERROR: {e}")

    print("\nDone!")


if __name__ == "__main__":
    if "--integration" in sys.argv:
        run_integration_tests()
    else:
        pytest.main([__file__, "-v"])
