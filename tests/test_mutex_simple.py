#!/usr/bin/env python3
"""Simple test to verify mutex protection is working for GGUF models."""

import requests
import json
import time
import threading
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8080"
MODEL = "dolphin-mistral"  # Change to your GGUF model

def make_request(request_id: int, prompt: str) -> Dict[str, Any]:
    """Make a single synchronous request."""
    
    request_data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,  # Very short for quick testing
        "temperature": 0.0,
        "stream": False
    }
    
    start_time = time.time()
    print(f"[Request {request_id}] Starting: '{prompt}'")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"[Request {request_id}] ✓ SUCCESS in {elapsed:.2f}s: {content[:50]}")
            return {"id": request_id, "success": True, "time": elapsed, "response": content}
        else:
            print(f"[Request {request_id}] ✗ ERROR {response.status_code} in {elapsed:.2f}s")
            print(f"  Response: {response.text[:200]}")
            return {"id": request_id, "success": False, "time": elapsed, "status": response.status_code}
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Request {request_id}] ✗ EXCEPTION in {elapsed:.2f}s: {e}")
        return {"id": request_id, "success": False, "time": elapsed, "error": str(e)}

def test_sequential():
    """Test sequential requests (should all work)."""
    print("\n" + "="*60)
    print("SEQUENTIAL TEST (Baseline)")
    print("="*60)
    
    prompts = ["Say hi", "Say bye", "Count to 3"]
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        result = make_request(i, prompt)
        results.append(result)
        time.sleep(0.5)  # Small delay between requests
    
    successes = sum(1 for r in results if r.get("success"))
    print(f"\nSequential Results: {successes}/{len(results)} successful")
    return all(r.get("success") for r in results)

def test_concurrent():
    """Test concurrent requests (should serialize via mutex)."""
    print("\n" + "="*60)
    print("CONCURRENT TEST (With Mutex)")
    print("="*60)
    
    prompts = ["What is 2+2?", "Name a color", "Say hello", "Count to 2", "Name a fruit"]
    results = []
    threads = []
    
    def run_request(req_id, prompt):
        result = make_request(req_id, prompt)
        results.append(result)
    
    # Launch all threads at once
    print(f"Launching {len(prompts)} concurrent requests...")
    start_time = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        thread = threading.Thread(target=run_request, args=(i, prompt))
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    successes = sum(1 for r in results if r.get("success"))
    failures = len(results) - successes
    
    print(f"\n" + "-"*60)
    print(f"Concurrent Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Successful: {successes}/{len(results)}")
    print(f"  Failed: {failures}/{len(results)}")
    
    if successes == len(results):
        print("\n✅ SUCCESS: All requests completed successfully!")
        print("   The mutex is properly serializing requests.")
    elif successes > 0:
        print("\n⚠️  PARTIAL SUCCESS: Some requests succeeded.")
        print("   The mutex may be working but there might be other issues.")
    else:
        print("\n❌ FAILURE: All requests failed!")
        print("   Check server logs for details.")
    
    return successes > 0

def main():
    print("="*60)
    print("GGUF MODEL MUTEX TEST")
    print("="*60)
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL}")
    print("\nMake sure the server is running with:")
    print("heylookllm --host 0.0.0.0 --log-level DEBUG --api openai")
    print()
    
    # Test server connectivity
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=2)
        if response.status_code != 200:
            print("❌ Cannot connect to server!")
            return
        
        models = response.json().get("data", [])
        model_ids = [m["id"] for m in models]
        
        if MODEL not in model_ids:
            print(f"❌ Model '{MODEL}' not found!")
            print(f"Available models: {', '.join(model_ids)}")
            return
            
        print(f"✓ Server is running with {len(models)} models available")
        
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Run tests
    seq_ok = test_sequential()
    
    if seq_ok:
        concurrent_ok = test_concurrent()
        
        if concurrent_ok:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED")
            print("The mutex protection is working correctly!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("⚠️  CONCURRENT TEST ISSUES")
            print("Check server logs for errors")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ SEQUENTIAL TEST FAILED")
        print("Fix basic functionality before testing concurrency")
        print("="*60)

if __name__ == "__main__":
    main()