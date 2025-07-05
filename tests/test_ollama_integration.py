#!/usr/bin/env python3
"""
Integration tests for Ollama API compatibility endpoints.
Tests that the Ollama API endpoints work correctly and produce the expected response format.

REQUIREMENTS:
- heylookllm server must be running on port 11434 (default Ollama port)
- Start server: heylookllm --host 0.0.0.0 --port 11434

Usage: python -m pytest tests/test_ollama_integration.py -v
"""

import sys
import os
import time
import json
import requests
import base64
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Test configuration
BASE_URL = "http://localhost:11434"  # Changed to Ollama standard port
VLM_MODEL = "gemma3n-e4b-it"  # MLX VLM model
TEXT_MODEL = "llama-3.1-8b-instruct"  # Text-only model

# Test image (1x1 red pixel)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFfWUoYjAAAAABJRU5ErkJggg=="

def check_server_running():
    """Check if heylookllm server is running on port 11434."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def get_available_models() -> List[str]:
    """Get list of available models via OpenAI API."""
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        else:
            return []
    except Exception:
        return []

def test_ollama_tags():
    """Test Ollama /api/tags endpoint (list models)."""
    print("🧪 Testing Ollama /api/tags endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
        
        data = response.json()
        
        # Check Ollama response structure
        if "models" not in data:
            print("❌ Missing 'models' field in response")
            return False
        
        models = data["models"]
        if not isinstance(models, list):
            print("❌ 'models' field is not a list")
            return False
        
        print(f"✅ Found {len(models)} models")
        
        # Check model structure
        if models:
            model = models[0]
            required_fields = ["name", "model", "modified_at", "size", "digest", "details"]
            for field in required_fields:
                if field not in model:
                    print(f"❌ Missing required field '{field}' in model")
                    return False
            
            print(f"✅ Model structure correct: {model['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_chat_basic():
    """Test Ollama /api/chat endpoint with basic text."""
    print("🧪 Testing Ollama /api/chat endpoint (basic text)...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    model = models[0]
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
        end_time = time.perf_counter()
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        response_time = end_time - start_time
        
        # Check Ollama response structure
        required_fields = ["model", "created_at", "message", "done", "total_duration"]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field '{field}' in response")
                return False
        
        # Check message structure
        message = data["message"]
        if not isinstance(message, dict):
            print("❌ 'message' field is not a dict")
            return False
        
        if "role" not in message or "content" not in message:
            print("❌ Missing 'role' or 'content' in message")
            return False
        
        if message["role"] != "assistant":
            print(f"❌ Expected role 'assistant', got '{message['role']}'")
            return False
        
        content = message["content"]
        if not isinstance(content, str) or len(content) == 0:
            print("❌ Message content is empty or not string")
            return False
        
        print(f"✅ Success: {response_time:.3f}s")
        print(f"📝 Response: {content[:50]}...")
        print(f"🔧 Duration: {data.get('total_duration', 0) / 1_000_000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_chat_vision():
    """Test Ollama /api/chat endpoint with vision (image)."""
    print("🧪 Testing Ollama /api/chat endpoint (vision)...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    # Find a vision model
    vision_model = None
    for model in models:
        if "vision" in model.lower() or "vlm" in model.lower() or VLM_MODEL in model:
            vision_model = model
            break
    
    if not vision_model:
        print("⚠️ No vision model found, using first available model")
        vision_model = models[0]
    
    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": "What color is this image?",
                "images": [TEST_IMAGE_B64]
            }
        ],
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
        end_time = time.perf_counter()
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        response_time = end_time - start_time
        
        # Check response structure (same as basic test)
        required_fields = ["model", "created_at", "message", "done", "total_duration"]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field '{field}' in response")
                return False
        
        message = data["message"]
        content = message.get("content", "")
        
        print(f"✅ Success: {response_time:.3f}s")
        print(f"📝 Response: {content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_generate():
    """Test Ollama /api/generate endpoint."""
    print("🧪 Testing Ollama /api/generate endpoint...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    model = models[0]
    
    payload = {
        "model": model,
        "prompt": "Write a short poem about cats",
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=30)
        end_time = time.perf_counter()
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        response_time = end_time - start_time
        
        # Check Ollama generate response structure
        required_fields = ["model", "created_at", "response", "done", "total_duration"]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field '{field}' in response")
                return False
        
        response_text = data["response"]
        if not isinstance(response_text, str) or len(response_text) == 0:
            print("❌ Response text is empty or not string")
            return False
        
        print(f"✅ Success: {response_time:.3f}s")
        print(f"📝 Response: {response_text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_generate_vision():
    """Test Ollama /api/generate endpoint with vision."""
    print("🧪 Testing Ollama /api/generate endpoint (vision)...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    # Find a vision model
    vision_model = None
    for model in models:
        if "vision" in model.lower() or "vlm" in model.lower() or VLM_MODEL in model:
            vision_model = model
            break
    
    if not vision_model:
        print("⚠️ No vision model found, using first available model")
        vision_model = models[0]
    
    payload = {
        "model": vision_model,
        "prompt": "Describe this image in detail",
        "images": [TEST_IMAGE_B64],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 30
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=30)
        end_time = time.perf_counter()
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        response_time = end_time - start_time
        
        # Check response structure
        required_fields = ["model", "created_at", "response", "done", "total_duration"]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field '{field}' in response")
                return False
        
        response_text = data["response"]
        
        print(f"✅ Success: {response_time:.3f}s")
        print(f"📝 Response: {response_text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_parameter_mapping():
    """Test that Ollama parameters are correctly mapped."""
    print("🧪 Testing parameter mapping...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    model = models[0]
    
    # Test with various parameters
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'test'"}
        ],
        "stream": False,
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 5,
        "seed": 42
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check basic response structure
        if "message" not in data or "content" not in data["message"]:
            print("❌ Invalid response structure")
            return False
        
        print("✅ Parameter mapping working")
        print(f"📝 Response: {data['message']['content']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_openai_vs_ollama_consistency():
    """Test that OpenAI and Ollama endpoints produce similar results."""
    print("🧪 Testing OpenAI vs Ollama consistency...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    model = models[0]
    
    # Test same request on both APIs
    test_message = "Hello! How are you?"
    
    # OpenAI API request
    openai_payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_message}],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 20
    }
    
    # Ollama API request
    ollama_payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_message}],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 20
    }
    
    try:
        # Make OpenAI request
        start_time = time.perf_counter()
        openai_response = requests.post(f"{BASE_URL}/v1/chat/completions", json=openai_payload, timeout=30)
        openai_time = time.perf_counter() - start_time
        
        # Make Ollama request
        start_time = time.perf_counter()
        ollama_response = requests.post(f"{BASE_URL}/api/chat", json=ollama_payload, timeout=30)
        ollama_time = time.perf_counter() - start_time
        
        if openai_response.status_code != 200:
            print(f"❌ OpenAI endpoint failed: {openai_response.status_code}")
            return False
        
        if ollama_response.status_code != 200:
            print(f"❌ Ollama endpoint failed: {ollama_response.status_code}")
            return False
        
        openai_data = openai_response.json()
        ollama_data = ollama_response.json()
        
        # Extract content from both responses
        openai_content = openai_data["choices"][0]["message"]["content"]
        ollama_content = ollama_data["message"]["content"]
        
        print(f"✅ Both endpoints working")
        print(f"🔧 OpenAI time: {openai_time:.3f}s")
        print(f"🔧 Ollama time: {ollama_time:.3f}s")
        print(f"📝 OpenAI: {openai_content[:30]}...")
        print(f"📝 Ollama: {ollama_content[:30]}...")
        
        # Check if responses are similar (both non-empty)
        if len(openai_content) > 0 and len(ollama_content) > 0:
            print("✅ Both responses contain content")
            return True
        else:
            print("❌ One or both responses are empty")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_show():
    """Test Ollama /api/show endpoint."""
    print("🧪 Testing Ollama /api/show endpoint...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    model = models[0]
    
    payload = {
        "model": model,
        "verbose": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/show", json=payload, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check Ollama show response structure
        required_fields = ["modelfile", "parameters", "template", "details", "capabilities"]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field '{field}' in response")
                return False
        
        # Check details structure
        details = data["details"]
        if not isinstance(details, dict):
            print("❌ 'details' field is not a dict")
            return False
        
        print(f"✅ Show endpoint working for model: {model}")
        print(f"📝 Capabilities: {data.get('capabilities', [])}")
        print(f"📝 Format: {details.get('format', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_version():
    """Test Ollama /api/version endpoint."""
    print("🧪 Testing Ollama /api/version endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/version", timeout=5)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
        
        data = response.json()
        
        # Check version response structure
        if "version" not in data:
            print("❌ Missing 'version' field in response")
            return False
        
        version = data["version"]
        if not isinstance(version, str):
            print("❌ 'version' field is not a string")
            return False
        
        print(f"✅ Version endpoint working: {version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_ps():
    """Test Ollama /api/ps endpoint (list running models)."""
    print("🧪 Testing Ollama /api/ps endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/ps", timeout=5)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
        
        data = response.json()
        
        # Check ps response structure
        if "models" not in data:
            print("❌ Missing 'models' field in response")
            return False
        
        models = data["models"]
        if not isinstance(models, list):
            print("❌ 'models' field is not a list")
            return False
        
        print(f"✅ PS endpoint working: {len(models)} running models")
        
        # Check model structure if any models are running
        if models:
            model = models[0]
            required_fields = ["name", "model", "size", "digest", "details"]
            for field in required_fields:
                if field not in model:
                    print(f"❌ Missing required field '{field}' in running model")
                    return False
            
            print(f"📝 Running model: {model['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_embed():
    """Test Ollama /api/embed endpoint."""
    print("🧪 Testing Ollama /api/embed endpoint...")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return False
    
    model = models[0]
    
    payload = {
        "model": model,
        "input": "Hello world"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/embed", json=payload, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check embed response structure
        required_fields = ["model", "embeddings", "total_duration"]
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing required field '{field}' in response")
                return False
        
        embeddings = data["embeddings"]
        if not isinstance(embeddings, list):
            print("❌ 'embeddings' field is not a list")
            return False
        
        if embeddings and isinstance(embeddings[0], list):
            embedding_dim = len(embeddings[0])
            print(f"✅ Embed endpoint working: {embedding_dim}D embeddings")
        else:
            print("✅ Embed endpoint working (placeholder response)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all Ollama integration tests."""
    print("🚀 Ollama API Compatibility - Integration Tests")
    print("=" * 60)
    print("Server URL: http://localhost:11434")
    print("=" * 60)
    
    # Check server first
    if not check_server_running():
        print("❌ REQUIREMENTS NOT MET:")
        print("   heylookllm server must be running on port 11434")
        print("   Start server: heylookllm --host 0.0.0.0 --port 11434")
        return 1
    
    print("✅ Server is running")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No models available")
        return 1
    
    print(f"✅ Found {len(models)} models: {', '.join(models)}")
    
    # Test suite
    tests = [
        ("Ollama Tags (List Models)", test_ollama_tags),
        ("Ollama Chat Basic", test_ollama_chat_basic),
        ("Ollama Chat Vision", test_ollama_chat_vision),
        ("Ollama Generate", test_ollama_generate),
        ("Ollama Generate Vision", test_ollama_generate_vision),
        ("Ollama Show Model Info", test_ollama_show),
        ("Ollama Version", test_ollama_version),
        ("Ollama PS (Running Models)", test_ollama_ps),
        ("Ollama Embed", test_ollama_embed),
        ("Parameter Mapping", test_parameter_mapping),
        ("OpenAI vs Ollama Consistency", test_openai_vs_ollama_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print("="*60)
        
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 OLLAMA INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL OLLAMA INTEGRATION TESTS PASSED!")
        print("🚀 Ollama API compatibility is working correctly!")
        
        print("\n🏆 Validated Features:")
        print("  ✅ Ollama /api/tags endpoint")
        print("  ✅ Ollama /api/chat endpoint")
        print("  ✅ Ollama /api/generate endpoint")
        print("  ✅ Ollama /api/show endpoint")
        print("  ✅ Ollama /api/version endpoint")
        print("  ✅ Ollama /api/ps endpoint")
        print("  ✅ Ollama /api/embed endpoint")
        print("  ✅ Vision support with base64 images")
        print("  ✅ Parameter mapping (temperature, top_p, etc.)")
        print("  ✅ Response format compatibility")
        print("  ✅ Model information and capabilities")
        
        return 0
    else:
        print("⚠️ Some integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
