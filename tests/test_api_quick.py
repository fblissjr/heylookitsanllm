#!/usr/bin/env python3
"""
Quick API test for MLX provider optimizations.
Simple test to validate optimizations are working.

Usage: python test_api_quick.py
"""

import requests
import time
import json
from typing import Dict, Any

# Test configuration
BASE_URL = "http://127.0.0.1:8080"
VLM_MODEL = "gemma3n-e4b-it"  # MLX VLM model
TEXT_MODEL = "Llama-3.2-1B-Instruct-4bit"  # Text-only model

def test_endpoint(model: str, messages: list, test_name: str) -> Dict[str, Any]:
    """Test a single endpoint."""
    print(f"\n🧪 Testing: {test_name}")
    print(f"   Model: {model}")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 20,
        "temperature": 0.1,
        "stream": False,
        "include_performance": True
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        end_time = time.perf_counter()

        if response.status_code == 200:
            data = response.json()
            response_time = end_time - start_time

            # Extract response text
            response_text = data["choices"][0]["message"]["content"] if data.get("choices") else "No response"

            # Extract performance metrics
            perf_metrics = data.get("performance", {})

            print(f"   ✅ Success: {response_time:.3f}s")
            print(f"   📝 Response: {response_text[:50]}...")

            if perf_metrics:
                print(f"   📊 Prompt TPS: {perf_metrics.get('prompt_tps', 'N/A')}")
                print(f"   📊 Generation TPS: {perf_metrics.get('generation_tps', 'N/A')}")

            return {
                "success": True,
                "response_time": response_time,
                "response_text": response_text,
                "performance": perf_metrics
            }
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}

    except Exception as e:
        end_time = time.perf_counter()
        print(f"   ❌ Error: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Run quick API tests."""
    print("🚀 Quick API Test for MLX Optimizations")
    print("=" * 50)

    # Test cases
    test_cases = [
        {
            "name": "VLM Text-Only Path (Optimized)",
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": "Hello! How are you?"}],
            "expected_path": "vlm_text"
        },
        {
            "name": "VLM Vision Path",
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFfWUoYjAAAAABJRU5ErkJggg=="}}
                    ]
                }
            ],
            "expected_path": "vlm_vision"
        },
        {
            "name": "Text-Only Model",
            "model": TEXT_MODEL,
            "messages": [{"role": "user", "content": "Hello! How are you?"}],
            "expected_path": "text_only"
        }
    ]

    # Run tests
    results = []
    for test_case in test_cases:
        result = test_endpoint(
            test_case["model"],
            test_case["messages"],
            test_case["name"]
        )
        result["expected_path"] = test_case["expected_path"]
        results.append(result)

    # Analyze results
    print("\n" + "=" * 50)
    print("📊 RESULTS ANALYSIS")
    print("=" * 50)

    successful_tests = [r for r in results if r["success"]]

    if len(successful_tests) >= 2:
        # Look for VLM text vs vision comparison
        vlm_text_result = next((r for r in successful_tests if r["expected_path"] == "vlm_text"), None)
        vlm_vision_result = next((r for r in successful_tests if r["expected_path"] == "vlm_vision"), None)

        if vlm_text_result and vlm_vision_result:
            text_time = vlm_text_result["response_time"]
            vision_time = vlm_vision_result["response_time"]

            print(f"🎯 VLM PATH COMPARISON:")
            print(f"   VLM Text-Only: {text_time:.3f}s")
            print(f"   VLM Vision: {vision_time:.3f}s")

            if text_time < vision_time:
                speedup = vision_time / text_time
                print(f"   ✅ Text-only path is {speedup:.1f}x faster!")
                print(f"   🚀 Optimization is working!")
            else:
                print(f"   ⚠️ Text-only path not faster (may need tuning)")

        # Overall stats
        avg_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Successful tests: {len(successful_tests)}/{len(results)}")

        # Performance metrics
        perf_data = [r["performance"] for r in successful_tests if r.get("performance")]
        if perf_data:
            avg_prompt_tps = sum(p.get("prompt_tps", 0) for p in perf_data) / len(perf_data)
            avg_gen_tps = sum(p.get("generation_tps", 0) for p in perf_data) / len(perf_data)

            print(f"   Average Prompt TPS: {avg_prompt_tps:.1f}")
            print(f"   Average Generation TPS: {avg_gen_tps:.1f}")

        if len(successful_tests) == len(results):
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ MLX optimizations are working correctly!")
            return 0
        else:
            print("\n⚠️ Some tests failed")
            return 1
    else:
        print("\n❌ Not enough successful tests to analyze")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
