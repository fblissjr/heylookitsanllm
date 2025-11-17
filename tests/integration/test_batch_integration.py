#!/usr/bin/env python3
"""
Integration test for batch text processing.

This test validates the batch endpoint works correctly.
Requires a running heylookllm server with a text-only MLX model.

Usage:
    python tests/test_batch_integration.py
"""

import json
import sys
import time

import requests


def test_batch_endpoint(
    base_url="http://localhost:8080", model_id="gpt-oss-120b-MXFP4-Q8-mlx"
):
    """Test the batch chat completions endpoint."""

    print("=" * 60)
    print("BATCH PROCESSING INTEGRATION TEST")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"Model: {model_id}")
    print()

    # Test 1: Basic batch request
    print("Test 1: Basic batch processing (3 requests)")
    print("-" * 60)

    batch_request = {
        "requests": [
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
            {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "max_tokens": 50,
            },
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "What color is the sky?"}],
                "max_tokens": 50,
            },
        ]
    }

    start_time = time.time()

    try:
        response = requests.post(
            f"{base_url}/v1/batch/chat/completions",
            json=batch_request,
            headers={"Content-Type": "application/json"},
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            print(f"✓ Status: {response.status_code} OK")
            print(f"✓ Response time: {elapsed:.2f}s")
            print(f"✓ Batch size: {len(result['data'])}")

            # Print batch statistics
            stats = result["batch_stats"]
            print(f"\nBatch Statistics:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Elapsed: {stats['elapsed_seconds']:.2f}s")
            print(f"  Throughput: {stats['throughput_req_per_sec']:.2f} req/s")
            print(f"  Throughput: {stats['throughput_tok_per_sec']:.1f} tok/s")
            print(f"  Prefill time: {stats['prefill_time']:.2f}s")
            print(f"  Generation time: {stats['generation_time']:.2f}s")

            # Print individual responses
            print(f"\nResponses:")
            for i, completion in enumerate(result["data"]):
                content = completion["choices"][0]["message"]["content"]
                tokens = completion["usage"]["completion_tokens"]
                print(f"  {i + 1}. ({tokens} tokens) {content[:80]}...")

            print("\n✅ Test 1 PASSED")

        else:
            print(f"✗ Status: {response.status_code}")
            print(f"✗ Error: {response.text}")
            print("\n❌ Test 1 FAILED")
            return False

    except Exception as e:
        print(f"✗ Exception: {e}")
        print("\n❌ Test 1 FAILED")
        return False

    print()

    # Test 2: Error handling - mixed models
    print("Test 2: Error handling - mixed models")
    print("-" * 60)

    bad_request = {
        "requests": [
            {"model": model_id, "messages": [{"role": "user", "content": "Test"}]},
            {
                "model": "different-model",
                "messages": [{"role": "user", "content": "Test"}],
            },
        ]
    }

    try:
        response = requests.post(
            f"{base_url}/v1/batch/chat/completions",
            json=bad_request,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 400:
            print(f"✓ Correctly rejected mixed models: {response.status_code}")
            print(f"✓ Error message: {response.json()['detail']}")
            print("\n✅ Test 2 PASSED")
        else:
            print(f"✗ Expected 400, got {response.status_code}")
            print("\n❌ Test 2 FAILED")
            return False

    except Exception as e:
        print(f"✗ Exception: {e}")
        print("\n❌ Test 2 FAILED")
        return False

    print()

    # Test 3: Error handling - streaming not supported
    print("Test 3: Error handling - streaming request")
    print("-" * 60)

    streaming_request = {
        "requests": [
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            }
        ]
    }

    try:
        response = requests.post(
            f"{base_url}/v1/batch/chat/completions",
            json=streaming_request,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 400:
            print(f"✓ Correctly rejected streaming: {response.status_code}")
            print(f"✓ Error message: {response.json()['detail']}")
            print("\n✅ Test 3 PASSED")
        else:
            print(f"✗ Expected 400, got {response.status_code}")
            print("\n❌ Test 3 FAILED")
            return False

    except Exception as e:
        print(f"✗ Exception: {e}")
        print("\n❌ Test 3 FAILED")
        return False

    print()
    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test batch processing endpoint")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument(
        "--model", default="gpt-oss-120b-MXFP4-Q8-mlx", help="Model ID to test"
    )

    args = parser.parse_args()

    success = test_batch_endpoint(args.url, args.model)
    sys.exit(0 if success else 1)
