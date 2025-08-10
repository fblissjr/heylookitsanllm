#!/usr/bin/env python3
"""Stress test with 10 parallel requests to GGUF models."""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any
import sys

# Server configuration
BASE_URL = "http://localhost:8080"
MODEL = "smol-gguf"  # Change this to your GGUF model

async def make_request(session: aiohttp.ClientSession, request_id: int, prompt: str) -> Dict[str, Any]:
    """Make a single chat completion request."""

    request_data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 20,  # Keep it short for testing
        "temperature": 0.0,  # Deterministic
        "stream": False
    }

    start_time = time.time()

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            elapsed = time.time() - start_time

            result = {
                "request_id": request_id,
                "status": response.status,
                "elapsed_time": elapsed,
            }

            if response.status == 200:
                data = await response.json()
                result["response"] = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]
                result["success"] = True
                print(f"✓ Request {request_id:2d}: SUCCESS in {elapsed:.2f}s")
            elif response.status == 503:
                data = await response.json()
                result["error"] = data.get("error", {}).get("message", "Unknown error")
                result["retry_after"] = response.headers.get("Retry-After", "N/A")
                result["success"] = False
                print(f"⚠ Request {request_id:2d}: RATE LIMITED (503) in {elapsed:.2f}s - Retry after {result['retry_after']}s")
            else:
                result["error"] = await response.text()
                result["success"] = False
                print(f"✗ Request {request_id:2d}: ERROR {response.status} in {elapsed:.2f}s")

            return result

    except asyncio.TimeoutError:
        print(f"✗ Request {request_id:2d}: TIMEOUT after 30s")
        return {
            "request_id": request_id,
            "status": "timeout",
            "error": "Request timed out after 30 seconds",
            "elapsed_time": 30.0,
            "success": False
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Request {request_id:2d}: EXCEPTION in {elapsed:.2f}s - {str(e)[:50]}")
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e),
            "elapsed_time": elapsed,
            "success": False
        }

async def stress_test(num_requests: int = 10):
    """Run parallel stress test."""

    print(f"\n{'='*60}")
    print(f"STRESS TEST: {num_requests} Parallel Requests")
    print(f"{'='*60}")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")

    # Simple prompts that should work
    prompts = [
        "What is 2+2?",
        "Name a color.",
        "Count to 3.",
        "Say hello.",
        "What day comes after Monday?",
        "Name a fruit.",
        "What is 10-5?",
        "Name an animal.",
        "What is the opposite of hot?",
        "Complete: Mary had a little",
    ]

    # Create connector with limited connections
    connector = aiohttp.TCPConnector(limit=num_requests)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Launch all requests at once
        print(f"Launching {num_requests} requests simultaneously...")
        start_time = time.time()

        tasks = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            task = make_request(session, i+1, prompt)
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        successful = sum(1 for r in results if r.get("success", False))
        rate_limited = sum(1 for r in results if r.get("status") == 503)
        failed = sum(1 for r in results if not r.get("success", False) and r.get("status") != 503)
        timeouts = sum(1 for r in results if r.get("status") == "timeout")

        print(f"\n{'='*60}")
        print("RESULTS SUMMARY:")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total requests: {num_requests}")
        print(f"  ✓ Successful: {successful}")
        print(f"  ⚠ Rate limited (503): {rate_limited}")
        print(f"  ✗ Failed: {failed}")
        print(f"  ⏱ Timeouts: {timeouts}")

        if successful > 0:
            avg_time = sum(r["elapsed_time"] for r in results if r.get("success")) / successful
            print(f"\nAverage successful response time: {avg_time:.2f}s")

        # Show individual results
        print(f"\n{'='*60}")
        print("DETAILED RESULTS:")
        print(f"{'='*60}")
        for r in sorted(results, key=lambda x: x["request_id"]):
            status = "✓" if r.get("success") else "⚠" if r.get("status") == 503 else "⏱" if r.get("status") == "timeout" else "✗"
            print(f"{status} Request {r['request_id']:2d}: Status={r.get('status'):8} Time={r['elapsed_time']:.2f}s")
            if r.get("success"):
                response_preview = r.get('response', '')[:50]
                if response_preview:
                    print(f"    Response: {response_preview}...")
            else:
                error = r.get('error', 'Unknown error')[:100]
                print(f"    Error: {error}")

        print(f"\n{'='*60}")

        # Interpretation
        if rate_limited > 0:
            print("\n✅ MUTEX PROTECTION WORKING:")
            print(f"   - {rate_limited} requests were properly rate-limited")
            print("   - Server correctly prevents concurrent GGUF model access")
            print("   - Clients received proper 503 responses with retry headers")

        if successful == num_requests:
            print("\n⚠️  ALL REQUESTS SUCCEEDED:")
            print("   - This might mean requests were processed sequentially")
            print("   - Or the model can handle concurrent requests (unlikely for GGUF)")

        if failed > 0 or timeouts > 0:
            print("\n❌ PROBLEMS DETECTED:")
            if failed > 0:
                print(f"   - {failed} requests failed with errors")
            if timeouts > 0:
                print(f"   - {timeouts} requests timed out")
            print("   - Check server logs for details")

        return results

async def test_sequential_after_parallel():
    """Test that sequential requests work after parallel stress."""

    print(f"\n{'='*60}")
    print("SEQUENTIAL TEST (After Parallel Stress)")
    print(f"{'='*60}\n")

    async with aiohttp.ClientSession() as session:
        # Try 3 sequential requests
        for i in range(3):
            print(f"Sequential request {i+1}...")
            result = await make_request(session, 100+i, f"Count to {i+1}")

            if result.get("success"):
                print(f"  ✓ Success: {result.get('response', '')[:50]}...")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown')[:50]}")

            # Small delay between sequential requests
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    print("="*60)
    print("GGUF MODEL PARALLEL STRESS TEST")
    print("="*60)
    print(f"\nMake sure the server is running with a GGUF model: {MODEL}")
    print("Start server with: heylookllm --host 0.0.0.0 --log-level DEBUG\n")

    # Check if custom number of requests specified
    num_requests = 10
    if len(sys.argv) > 1:
        try:
            num_requests = int(sys.argv[1])
        except:
            pass

    print(f"Running with {num_requests} parallel requests...")
    print("Press Ctrl+C to abort\n")

    try:
        # Run parallel stress test
        asyncio.run(stress_test(num_requests))

        # Test sequential after parallel
        asyncio.run(test_sequential_after_parallel())

    except KeyboardInterrupt:
        print("\n\nTest aborted by user")
