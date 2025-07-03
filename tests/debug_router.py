#!/usr/bin/env python3
"""
Diagnostic script to test the router and provider loading.
Run this to identify router/provider issues.

Usage: python debug_router.py
"""

import sys
import os
import logging
import yaml

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_config_loading():
    """Test configuration loading."""
    print("📋 Testing configuration loading...")

    try:
        from heylook_llm.config import AppConfig

        # Load models.yaml
        if not os.path.exists("models.yaml"):
            print("❌ models.yaml not found in current directory")
            return False, None

        with open("models.yaml", 'r') as f:
            config_data = yaml.safe_load(f)

        app_config = AppConfig(**config_data)

        print(f"✅ Configuration loaded successfully")
        print(f"   Total models: {len(app_config.models)}")

        enabled_models = app_config.get_enabled_models()
        print(f"   Enabled models: {len(enabled_models)}")

        for model in enabled_models:
            print(f"     - {model.id} ({model.provider})")

        return True, app_config

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_router_creation(app_config):
    """Test router creation."""
    print("🔄 Testing router creation...")

    try:
        from heylook_llm.router import ModelRouter

        # Create router without initial model
        router = ModelRouter(
            config_path="models.yaml",
            log_level=logging.DEBUG,
            initial_model_id=None  # Don't load any model initially
        )

        print(f"✅ Router created successfully")
        print(f"   Current provider: {router.current_provider_id}")
        print(f"   Available models: {router.list_available_models()}")

        return True, router

    except Exception as e:
        print(f"❌ Router creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_loading(router, model_id):
    """Test loading a specific model."""
    print(f"🔧 Testing model loading for: {model_id}")

    try:
        # Test get_provider
        provider = router.get_provider(model_id)

        if provider is None:
            print("❌ get_provider returned None")
            return False

        print(f"✅ Provider loaded successfully")
        print(f"   Provider type: {type(provider).__name__}")
        print(f"   Model ID: {provider.model_id}")
        print(f"   Is VLM: {getattr(provider, 'is_vlm', 'Unknown')}")

        # Test create_chat_completion method exists
        if hasattr(provider, 'create_chat_completion'):
            print("✅ create_chat_completion method exists")
        else:
            print("❌ create_chat_completion method missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_completion_creation(router, model_id):
    """Test creating a chat completion generator."""
    print(f"💬 Testing chat completion creation for: {model_id}")

    try:
        provider = router.get_provider(model_id)

        if provider is None:
            print("❌ Provider is None")
            return False

        # Test request
        test_request = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }

        # Try to create generator
        generator = provider.create_chat_completion(test_request)

        if generator is None:
            print("❌ Generator is None")
            return False

        print("✅ Generator created successfully")
        print(f"   Generator type: {type(generator)}")

        # Try to get first response
        try:
            first_response = next(generator)
            print(f"✅ First response received: {type(first_response)}")
            if hasattr(first_response, 'text'):
                print(f"   Response text: {first_response.text[:50]}...")
            return True
        except Exception as gen_error:
            print(f"❌ Generator failed: {gen_error}")
            return False

    except Exception as e:
        print(f"❌ Chat completion creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run router diagnostics."""
    print("🧪 Router and Provider Diagnostic")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    # Test 1: Configuration loading
    success, app_config = test_config_loading()
    if not success:
        print("\n🛑 Cannot continue without valid configuration")
        return False
    print()

    # Test 2: Router creation
    success, router = test_router_creation(app_config)
    if not success:
        print("\n🛑 Cannot continue without working router")
        return False
    print()

    # Test 3: Model loading
    available_models = router.list_available_models()
    if not available_models:
        print("🛑 No models available for testing")
        return False

    test_model = available_models[0]
    success = test_model_loading(router, test_model)
    if not success:
        print(f"\n🛑 Model loading failed for {test_model}")
        return False
    print()

    # Test 4: Chat completion creation
    success = test_chat_completion_creation(router, test_model)
    if not success:
        print(f"\n🛑 Chat completion creation failed for {test_model}")
        return False
    print()

    print("=" * 50)
    print("✅ All router diagnostics passed!")
    print()
    print("💡 If the API is still failing, the issue might be:")
    print("   1. Model path problems (check models.yaml)")
    print("   2. Missing dependencies (check imports)")
    print("   3. Threading issues (check if models load on demand)")
    print("   4. Request format problems (check ComfyUI payload)")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
