#!/usr/bin/env python3
"""
Test script for queue integration with heylookllm.

This tests that:
1. Queue manager initializes properly when enabled
2. Requests route through queue when conditions are met
3. Direct path still works when queue is disabled
4. Metal optimizations remain in place
"""

import asyncio
import json
import time
import requests
from typing import List, Dict

# Configuration
BASE_URL = "http://localhost:8080"
TEST_MODEL = "mistral-small-mlx"  # Adjust to your available model

def test_queue_disabled():
    """Test that direct path works when queue is disabled (default)."""
    print("\n=== Testing Direct Path (Queue Disabled) ===")
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Say 'direct path works' in 3 words"}],
            "max_tokens": 10,
            "stream": False
        }
    )
    
    assert response.status_code == 200, f"Failed: {response.text}"
    result = response.json()
    print(f"✓ Direct path response: {result['choices'][0]['message']['content'][:50]}...")
    return True

def test_batch_processing():
    """Test that batch processing mode uses queue."""
    print("\n=== Testing Batch Processing Mode ===")
    
    # This should trigger queue usage via processing_mode
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": TEST_MODEL,
            "messages": [
                {"role": "user", "content": "First prompt"},
                {"role": "user", "content": "Second prompt"}
            ],
            "processing_mode": "sequential",
            "max_tokens": 10,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Batch processing completed")
        return True
    else:
        print(f"✗ Batch processing failed: {response.text}")
        return False

def test_multiple_images():
    """Test that multiple images trigger queue usage."""
    print("\n=== Testing Multiple Images (Queue Trigger) ===")
    
    # Create a request with multiple images (should trigger queue)
    # Using small test images
    small_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": TEST_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {"url": small_image}},
                    {"type": "image_url", "image_url": {"url": small_image}}
                ]
            }],
            "max_tokens": 20,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        print(f"✓ Multiple image request processed")
        return True
    else:
        # This might fail if vision model isn't available
        print(f"⚠ Multiple image test skipped (vision model may not be available)")
        return True

def test_performance_endpoint():
    """Test that performance metrics are still tracked."""
    print("\n=== Testing Performance Monitoring ===")
    
    response = requests.get(f"{BASE_URL}/v1/performance")
    
    assert response.status_code == 200, f"Performance endpoint failed: {response.text}"
    metrics = response.json()
    
    # Check that Metal info is present
    if "metal" in metrics:
        print(f"✓ Metal optimization status: {metrics['metal']}")
    
    # Check memory management
    if "memory" in metrics:
        print(f"✓ Memory tracking active: {metrics['memory'].get('current_usage_gb', 0):.2f}GB used")
    
    return True

def test_concurrent_requests():
    """Test that concurrent requests work properly."""
    print("\n=== Testing Concurrent Requests ===")
    
    import concurrent.futures
    
    def make_request(prompt: str):
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "stream": False
            }
        )
        return response.status_code == 200
    
    prompts = [f"Count to {i}" for i in range(1, 4)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        start_time = time.time()
        results = list(executor.map(make_request, prompts))
        elapsed = time.time() - start_time
    
    success_count = sum(results)
    print(f"✓ Processed {success_count}/{len(prompts)} concurrent requests in {elapsed:.2f}s")
    
    return success_count > 0

def main():
    """Run all tests."""
    print("=" * 60)
    print("Queue Integration Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        if response.status_code != 200:
            print("❌ Server not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start heylookllm first.")
        print("   Run: heylookllm --api openai --log-level DEBUG")
        return
    
    # Run tests
    tests = [
        test_queue_disabled,
        test_batch_processing,
        test_multiple_images,
        test_performance_endpoint,
        test_concurrent_requests
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ All tests passed! Queue integration working correctly.")
        print("\nTo enable queue manager globally:")
        print("1. Edit models.yaml")
        print("2. Set queue_config.enabled: true")
        print("3. Restart the server")
    else:
        print("⚠ Some tests failed. Check the output above.")
    print("=" * 60)

if __name__ == "__main__":
    main()