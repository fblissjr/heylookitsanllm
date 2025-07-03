#!/usr/bin/env python3
"""
Simple API test to test simple things. Super simple things. Like models not loading.
"""

import requests
import json
import sys

def test_models_endpoint(server_url):
    """Test the models endpoint first."""
    print("🔍 Testing /v1/models endpoint...")

    try:
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            print(f"✅ Models endpoint works")
            print(f"   Available models: {models}")
            return models
        else:
            print(f"❌ Models endpoint failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return []

def test_minimal_request(server_url, model):
    """Test the most minimal possible request."""
    print(f"🧪 Testing minimal request with model: {model}")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "hi"}
        ],
        "max_tokens": 5
    }

    print(f"📤 Sending payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        print(f"📥 Response: HTTP {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Request successful!")
            print(f"   Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print("❌ Request failed")
            try:
                error_data = response.json()
                print(f"   Error data: {json.dumps(error_data, indent=2)}")
            except:
                print(f"   Raw response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def test_empty_model_request(server_url):
    """Test with no model specified."""
    print("🧪 Testing request with no model...")

    payload = {
        "messages": [
            {"role": "user", "content": "hi"}
        ],
        "max_tokens": 5
    }

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        print(f"📥 Response: HTTP {response.status_code}")

        if response.status_code in [200, 422]:  # Either success or validation error
            print("✅ No-model request handled correctly")
            return True
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ No-model request error: {e}")
        return False

def test_invalid_model_request(server_url):
    """Test with invalid model."""
    print("🧪 Testing request with invalid model...")

    payload = {
        "model": "definitely-not-a-real-model",
        "messages": [
            {"role": "user", "content": "hi"}
        ],
        "max_tokens": 5
    }

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        print(f"📥 Response: HTTP {response.status_code}")

        if response.status_code == 404:
            print("✅ Invalid model request handled correctly (404)")
            return True
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Invalid model request error: {e}")
        return False

def main():
    """Run simple API tests."""
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    print("🎯 Simple API Test")
    print("=" * 40)
    print(f"Server: {server_url}")
    print()

    # Test 1: Models endpoint
    models = test_models_endpoint(server_url)
    if not models:
        print("🛑 No models available, cannot test chat completions")
        return False
    print()

    # Test 2: Minimal valid request
    test_model = models[3]
    success = test_minimal_request(server_url, test_model)
    print()

    # Test 3: No model specified
    test_empty_model_request(server_url)
    print()

    # Test 4: Invalid model
    test_invalid_model_request(server_url)
    print()

    print("=" * 40)
    if success:
        print("✅ Basic API functionality works")
    else:
        print("❌ API has issues - check server logs")

    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        sys.exit(1)
