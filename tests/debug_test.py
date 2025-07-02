#!/usr/bin/env python3
"""
Debug script to test edge-llm functionality step by step.
Run this to identify where issues occur.
"""

import logging
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_loading():
    """Test that configuration loads correctly."""
    print("=" * 50)
    print("Testing configuration loading...")
    try:
        from edge_llm.config import AppConfig
        import yaml

        with open("models.yaml", 'r') as f:
            config_data = yaml.safe_load(f)

        app_config = AppConfig(**config_data)
        print(f"✓ Configuration loaded successfully")
        print(f"  Found {len(app_config.models)} models:")
        for model in app_config.models:
            status = "enabled" if model.enabled else "disabled"
            print(f"    - {model.id} ({model.provider}) [{status}]")
        return app_config
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return None

def test_model_loading(app_config, model_id=None):
    """Test loading a specific model."""
    print("=" * 50)
    print("Testing model loading...")

    if not model_id:
        enabled_models = [m for m in app_config.models if m.enabled]
        if not enabled_models:
            print("✗ No enabled models found")
            return None
        model_id = enabled_models[0].id

    print(f"Testing model: {model_id}")

    try:
        from edge_llm.router import ModelRouter

        router = ModelRouter("models.yaml", logging.INFO)
        provider = router.get_provider(model_id)
        print(f"✓ Model '{model_id}' loaded successfully")
        print(f"  Provider type: {type(provider).__name__}")
        print(f"  Is VLM: {getattr(provider, 'is_vlm', 'Unknown')}")
        return provider
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return None

def test_text_generation(provider, model_id):
    """Test basic text generation."""
    print("=" * 50)
    print("Testing text generation...")

    try:
        test_request = {
            "messages": [{"role": "user", "content": "Say hello!"}],
            "max_tokens": 20,
            "temperature": 0.1
        }

        print("Generating response...")
        response_text = ""
        token_count = 0

        for response in provider.create_chat_completion(test_request):
            if hasattr(response, 'text'):
                response_text += response.text
                print(response.text, end='', flush=True)
            else:
                # Handle llama.cpp style responses
                content = response.get('choices', [{}])[0].get('delta', {}).get('content', '')
                if content:
                    response_text += content
                    print(content, end='', flush=True)

            token_count += 1
            if token_count > 50:  # Safety limit
                print("\n[Truncated - too many tokens]")
                break

        print(f"\n✓ Text generation successful")
        print(f"  Generated {token_count} tokens")
        print(f"  Full response: {repr(response_text[:100])}")
        return True
    except Exception as e:
        print(f"\n✗ Text generation failed: {e}")
        traceback.print_exc()
        return False

def test_vlm_generation(provider, model_id):
    """Test vision-language generation if supported."""
    print("=" * 50)
    print("Testing VLM generation...")

    if not getattr(provider, 'is_vlm', False):
        print("⚠ Model is not a VLM, skipping vision test")
        return True

    try:
        test_request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
                    ]
                }
            ],
            "max_tokens": 30,
            "temperature": 0.1
        }

        print("Generating VLM response...")
        response_text = ""
        token_count = 0

        for response in provider.create_chat_completion(test_request):
            if hasattr(response, 'text'):
                response_text += response.text
                print(response.text, end='', flush=True)
            token_count += 1
            if token_count > 50:  # Safety limit
                print("\n[Truncated]")
                break

        print(f"\n✓ VLM generation successful")
        print(f"  Generated {token_count} tokens")
        return True
    except Exception as e:
        print(f"\n✗ VLM generation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Edge LLM Debug Script")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test 1: Configuration
    app_config = test_config_loading()
    if not app_config:
        return False

    # Test 2: Model loading
    provider = test_model_loading(app_config)
    if not provider:
        return False

    # Test 3: Text generation
    if not test_text_generation(provider, provider.model_id):
        return False

    # Test 4: VLM generation (if applicable)
    if not test_vlm_generation(provider, provider.model_id):
        return False

    print("=" * 50)
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
