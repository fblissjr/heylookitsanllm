#!/usr/bin/env python3
"""
Example: Using batch processing for improved throughput.

This example demonstrates how to use the batch chat completions endpoint
to process multiple requests efficiently.

Performance Benefits:
- 2-4x throughput vs sequential processing
- Efficient variable-length prompt handling
- Optimized Metal memory management

Requirements:
- Running heylookllm server
- Text-only MLX model configured
"""

import requests
import json
import time
from typing import List, Dict


class BatchClient:
    """Simple client for batch processing."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    def batch_complete(
        self,
        model: str,
        prompts: List[str],
        max_tokens: int = 100,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8
    ) -> Dict:
        """
        Process multiple prompts in a single batch.

        Args:
            model: Model ID
            prompts: List of prompt strings
            max_tokens: Max tokens per completion
            completion_batch_size: Max concurrent generations
            prefill_batch_size: Max prefill parallelism

        Returns:
            Dictionary with 'data' (completions) and 'batch_stats' (metrics)
        """
        # Build batch request
        requests_list = [
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            for prompt in prompts
        ]

        batch_request = {
            "requests": requests_list,
            "completion_batch_size": completion_batch_size,
            "prefill_batch_size": prefill_batch_size
        }

        # Send request
        response = requests.post(
            f"{self.base_url}/v1/batch/chat/completions",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()
        return response.json()


def example_1_basic_batch():
    """Example 1: Basic batch processing."""
    print("=" * 60)
    print("Example 1: Basic Batch Processing")
    print("=" * 60)

    client = BatchClient()

    prompts = [
        "What is machine learning in one sentence?",
        "What is deep learning in one sentence?",
        "What is neural network in one sentence?",
        "What is reinforcement learning in one sentence?"
    ]

    print(f"Processing {len(prompts)} prompts...")
    print()

    start = time.time()
    result = client.batch_complete(
        model="qwen-14b",  # Change to your model
        prompts=prompts,
        max_tokens=50
    )
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.2f}s")
    print()

    # Print statistics
    stats = result['batch_stats']
    print(f"Batch Statistics:")
    print(f"  Throughput: {stats['throughput_req_per_sec']:.2f} req/s")
    print(f"  Throughput: {stats['throughput_tok_per_sec']:.1f} tok/s")
    print(f"  Speedup: {stats['total_requests'] / elapsed:.1f}x vs 1 req/s baseline")
    print()

    # Print responses
    print("Responses:")
    for i, completion in enumerate(result['data']):
        content = completion['choices'][0]['message']['content']
        print(f"  {i+1}. {content}")

    print()


def example_2_compare_sequential_vs_batch():
    """Example 2: Compare sequential vs batch processing."""
    print("=" * 60)
    print("Example 2: Sequential vs Batch Comparison")
    print("=" * 60)

    client = BatchClient()

    prompts = [
        "Count to 5",
        "List 3 colors",
        "Name 2 animals",
        "Say hello in 3 languages"
    ]

    # Sequential processing
    print("Processing sequentially...")
    sequential_start = time.time()

    for prompt in prompts:
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "qwen-14b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50
            }
        )
        response.raise_for_status()

    sequential_time = time.time() - sequential_start
    print(f"  Completed in {sequential_time:.2f}s")
    print()

    # Batch processing
    print("Processing in batch...")
    batch_start = time.time()

    result = client.batch_complete(
        model="qwen-14b",
        prompts=prompts,
        max_tokens=50
    )

    batch_time = time.time() - batch_start
    print(f"  Completed in {batch_time:.2f}s")
    print()

    # Compare
    speedup = sequential_time / batch_time
    print(f"Results:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Batch: {batch_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()


def example_3_variable_length_prompts():
    """Example 3: Handling variable-length prompts."""
    print("=" * 60)
    print("Example 3: Variable-Length Prompts")
    print("=" * 60)

    client = BatchClient()

    prompts = [
        "Hi",  # Very short
        "What is the capital of France and what is its population?",  # Medium
        "Explain the difference between machine learning and deep learning, "
        "including examples of when each approach is most appropriate.",  # Long
        "Hello there"  # Short
    ]

    print("Prompt lengths:")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {len(p)} chars: {p[:50]}...")
    print()

    result = client.batch_complete(
        model="qwen-14b",
        prompts=prompts,
        max_tokens=100
    )

    stats = result['batch_stats']
    print(f"Batch handled variable lengths efficiently:")
    print(f"  Throughput: {stats['throughput_req_per_sec']:.2f} req/s")
    print(f"  Total time: {stats['elapsed_seconds']:.2f}s")
    print()


def example_4_tuning_batch_parameters():
    """Example 4: Tuning batch parameters for your model."""
    print("=" * 60)
    print("Example 4: Tuning Batch Parameters")
    print("=" * 60)

    client = BatchClient()

    prompts = ["Tell me a fact about " + topic for topic in [
        "space", "ocean", "mountains", "forests",
        "deserts", "cities", "technology", "history"
    ]]

    print(f"Testing different batch configurations with {len(prompts)} prompts...")
    print()

    configurations = [
        {"completion_batch_size": 8, "prefill_batch_size": 2},
        {"completion_batch_size": 16, "prefill_batch_size": 4},
        {"completion_batch_size": 32, "prefill_batch_size": 8},
    ]

    for config in configurations:
        result = client.batch_complete(
            model="qwen-14b",
            prompts=prompts,
            max_tokens=50,
            **config
        )

        stats = result['batch_stats']
        print(f"Config: completion={config['completion_batch_size']}, "
              f"prefill={config['prefill_batch_size']}")
        print(f"  Time: {stats['elapsed_seconds']:.2f}s")
        print(f"  Throughput: {stats['throughput_req_per_sec']:.2f} req/s")
        print()

    print("Note: Optimal settings depend on model size and hardware.")
    print("Larger models benefit from smaller batch sizes.")
    print()


def example_5_error_handling():
    """Example 5: Error handling."""
    print("=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    client = BatchClient()

    # Example: Handling mixed models (will fail)
    print("Testing mixed models (should fail)...")
    try:
        response = requests.post(
            "http://localhost:8080/v1/batch/chat/completions",
            json={
                "requests": [
                    {"model": "model-a", "messages": [{"role": "user", "content": "Hi"}]},
                    {"model": "model-b", "messages": [{"role": "user", "content": "Hello"}]}
                ]
            }
        )
        response.raise_for_status()
        print("  ✗ Should have failed!")
    except requests.HTTPError as e:
        print(f"  ✓ Correctly rejected: {e.response.json()['detail']}")

    print()

    # Example: Handling streaming request (will fail)
    print("Testing streaming request (should fail)...")
    try:
        response = requests.post(
            "http://localhost:8080/v1/batch/chat/completions",
            json={
                "requests": [
                    {
                        "model": "qwen-14b",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True
                    }
                ]
            }
        )
        response.raise_for_status()
        print("  ✗ Should have failed!")
    except requests.HTTPError as e:
        print(f"  ✓ Correctly rejected: {e.response.json()['detail']}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch processing examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific example (1-5), or all if not specified"
    )

    args = parser.parse_args()

    examples = {
        1: example_1_basic_batch,
        2: example_2_compare_sequential_vs_batch,
        3: example_3_variable_length_prompts,
        4: example_4_tuning_batch_parameters,
        5: example_5_error_handling
    }

    if args.example:
        examples[args.example]()
    else:
        # Run all examples
        for i, example_func in examples.items():
            example_func()
            if i < len(examples):
                input("\nPress Enter to continue to next example...")
                print("\n")
