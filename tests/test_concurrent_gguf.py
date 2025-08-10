#!/usr/bin/env python3
"""Test concurrent requests to GGUF models with mutex protection."""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any

# Server configuration
BASE_URL = "http://localhost:8080"
MODEL = "smol-gguf"  # or any GGUF model you have configured

async def make_request(session: aiohttp.ClientSession, request_id: int, prompt: str) -> Dict[str, Any]:
    """Make a single chat completion request."""

    request_data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "stream": False
    }

    start_time = time.time()

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            elapsed = time.time() - start_time

            result = {
                "request_id": request_id,
                "status": response.status,
                "elapsed_time": elapsed,
                "headers": dict(response.headers)
            }

            if response.status == 200:
                data = await response.json()
                result["response"] = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]
                result["success"] = True
            elif response.status == 503:
                data = await response.json()
                result["error"] = data.get("error", {}).get("message", "Unknown error")
                result["retry_after"] = response.headers.get("Retry-After", "N/A")
                result["success"] = False
                print(f"Request {request_id}: Got 503 (model busy) - Retry-After: {result['retry_after']}s")
            else:
                result["error"] = await response.text()
                result["success"] = False

            return result

    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "success": False
        }

async def test_concurrent_requests(num_requests: int = 3):
    """Test multiple concurrent requests to the GGUF model."""

    print(f"Testing {num_requests} concurrent requests to model: {MODEL}")
    print(f"Server: {BASE_URL}")
    print("-" * 60)

    # Different prompts for variety
    prompts = [
        "What is 2+2?",
        "Name three colors.",
        "What is the capital of France?",
        "How many days in a week?",
        "What comes after Monday?"
    ]

    async with aiohttp.ClientSession() as session:
        # Launch all requests concurrently
        tasks = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            task = make_request(session, i+1, prompt)
            tasks.append(task)
            print(f"Launched request {i+1}: '{prompt}'")

        print("\nWaiting for responses...")
        print("-" * 60)

        # Gather all results
        results = await asyncio.gather(*tasks)

        # Analyze results
        successful = sum(1 for r in results if r.get("success", False))
        rate_limited = sum(1 for r in results if r.get("status") == 503)
        failed = sum(1 for r in results if not r.get("success", False) and r.get("status") != 503)

        print("\nResults:")
        for result in results:
            status = "✓" if result.get("success") else "⚠" if result.get("status") == 503 else "✗"
            print(f"  Request {result['request_id']}: {status} Status={result.get('status')} Time={result['elapsed_time']:.2f}s")
            if result.get("success"):
                print(f"    Response: {result.get('response', '')[:50]}...")
            elif result.get("status") == 503:
                print(f"    Rate limited: {result.get('error', 'Unknown')}")
                print(f"    Retry-After: {result.get('retry_after', 'N/A')}s")
            else:
                print(f"    Error: {result.get('error', 'Unknown')}")

        print("\nSummary:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful}")
        print(f"  Rate limited (503): {rate_limited}")
        print(f"  Failed: {failed}")

        # If we got rate limited, that's actually good - it means the mutex is working!
        if rate_limited > 0:
            print("\n✅ Mutex protection is working! Server properly rate-limits concurrent requests.")
        elif successful == num_requests:
            print("\n⚠️  All requests succeeded - mutex might not be needed or requests were sequential.")

        return results

async def test_with_retry():
    """Test that retries work after getting rate limited."""

    print("\n" + "=" * 60)
    print("Testing retry behavior after rate limit...")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # First request should succeed
        result1 = await make_request(session, 1, "First request")

        # Launch two concurrent requests
        task2 = asyncio.create_task(make_request(session, 2, "Second request"))
        task3 = asyncio.create_task(make_request(session, 3, "Third request"))

        result2, result3 = await asyncio.gather(task2, task3)

        # One should succeed, one should be rate limited
        results = [result1, result2, result3]
        rate_limited = [r for r in results if r.get("status") == 503]

        if rate_limited:
            print(f"\nGot rate limited on {len(rate_limited)} request(s)")
            print("Waiting 2 seconds before retry...")
            await asyncio.sleep(2)

            # Retry the rate-limited request
            print("Retrying...")
            retry_result = await make_request(session, 99, "Retry request")

            if retry_result.get("success"):
                print("✅ Retry succeeded after waiting!")
            else:
                print(f"⚠️  Retry failed: {retry_result.get('error', 'Unknown')}")

if __name__ == "__main__":
    print("=" * 60)
    print("GGUF Concurrent Request Test with Mutex Protection")
    print("=" * 60)
    print("\nMake sure the server is running with a GGUF model loaded.")
    print(f"Expected model: {MODEL}")
    print()

    # Run the concurrent test
    asyncio.run(test_concurrent_requests(3))

    # Test retry behavior
    asyncio.run(test_with_retry())
