#!/usr/bin/env python3
"""
Server connectivity test.
Verifies that heylookllm server is running and responding on port 8080.

Usage: python tests/test_server_connectivity.py
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8080"

def test_server_health():
    """Test server health endpoint."""
    print("🔍 Testing server health...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Server health check passed")
            return True
        else:
            print(f"  ❌ Server health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Server health check failed: {e}")
        return False

def test_models_endpoint():
    """Test models endpoint."""
    print("🔍 Testing models endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            model_count = len(models)
            print(f"  ✅ Models endpoint working: {model_count} models available")
            if model_count > 0:
                print(f"  📋 Available models: {[m['id'] for m in models[:3]]}")
            return True
        else:
            print(f"  ❌ Models endpoint failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Models endpoint failed: {e}")
        return False

def test_simple_completion():
    """Test simple completion request."""
    print("🔍 Testing simple completion...")
    
    try:
        # Get available models first
        models_response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        if models_response.status_code != 200:
            print("  ❌ Cannot get models list")
            return False
        
        models = models_response.json().get("data", [])
        if not models:
            print("  ❌ No models available")
            return False
        
        # Use first available model
        model_id = models[0]["id"]
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                response_text = data["choices"][0]["message"]["content"]
                print(f"  ✅ Simple completion working: '{response_text[:30]}...'")
                return True
            else:
                print("  ❌ Completion response missing choices")
                return False
        else:
            print(f"  ❌ Completion failed: HTTP {response.status_code}")
            print(f"  📝 Response: {response.text[:100]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Completion request failed: {e}")
        return False

def main():
    """Run server connectivity tests."""
    print("🔗 Server Connectivity Test")
    print("=" * 40)
    print(f"Testing: {BASE_URL}")
    print("=" * 40)
    
    tests = [
        ("Server Health", test_server_health),
        ("Models Endpoint", test_models_endpoint),
        ("Simple Completion", test_simple_completion),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Connectivity Test Results")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Server is ready for optimization testing!")
        return 0
    else:
        print("❌ Server connectivity issues detected")
        print("\n🔧 Troubleshooting:")
        print("  1. Check server is running: python -m heylook_llm.server --port 8080")
        print("  2. Verify port 8080 is not in use")
        print("  3. Check models.yaml configuration")
        print("  4. Ensure model files exist in modelzoo/")
        return 1

if __name__ == "__main__":
    sys.exit(main())
