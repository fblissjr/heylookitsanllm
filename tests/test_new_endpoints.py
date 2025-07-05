#!/usr/bin/env python3
"""
Quick test of the new Ollama endpoints
"""

import requests
import json

BASE_URL = "http://localhost:11434"

def test_new_endpoints():
    """Test the newly added Ollama endpoints"""
    
    print("🧪 Testing newly added Ollama endpoints...")
    
    # Test version endpoint
    print("\n1. Testing /api/version")
    try:
        response = requests.get(f"{BASE_URL}/api/version", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Version: {data.get('version', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test show endpoint
    print("\n2. Testing /api/show")
    try:
        payload = {"model": "gemma3n-e4b-it"}
        response = requests.post(f"{BASE_URL}/api/show", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model info received")
            print(f"   📝 Capabilities: {data.get('capabilities', [])}")
            print(f"   📝 Format: {data.get('details', {}).get('format', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test ps endpoint
    print("\n3. Testing /api/ps")
    try:
        response = requests.get(f"{BASE_URL}/api/ps", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"   ✅ Running models: {len(models)}")
            if models:
                print(f"   📝 First model: {models[0].get('name', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test embed endpoint
    print("\n4. Testing /api/embed")
    try:
        payload = {"model": "gemma3n-e4b-it", "input": "Hello world"}
        response = requests.post(f"{BASE_URL}/api/embed", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get('embeddings', [])
            if embeddings and isinstance(embeddings[0], list):
                print(f"   ✅ Embeddings: {len(embeddings[0])}D")
            else:
                print("   ✅ Embeddings endpoint working")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    print("🚀 Quick Test of New Ollama Endpoints")
    print("=" * 40)
    print("Make sure server is running: heylookllm --host 0.0.0.0")
    print("=" * 40)
    
    test_new_endpoints()
    
    print("\n✨ Test complete!")
    print("For comprehensive testing, run: python tests/test_ollama.py")
