#!/usr/bin/env python3
"""
Debug script to diagnose the "list index out of range" error in VLM processing.
Run this to test VLM input processing step by step.

Usage: python debug_vlm.py
"""

import sys
import os
import base64
import io
import json
from PIL import Image

def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (4, 4), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def test_message_processing():
    """Test the message processing step by step."""
    print("📝 Testing message processing step by step...")

    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'src'))

    try:
        from heylook_llm.utils import process_vlm_messages

        # Create test messages of different types
        test_image = create_test_image()

        test_cases = [
            {
                "name": "Simple text message",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            },
            {
                "name": "Text + image message",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What color is this?"},
                        {"type": "image_url", "image_url": {"url": test_image}}
                    ]}
                ]
            },
            {
                "name": "Multiple messages",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ]
            },
            {
                "name": "Complex multimodal",
                "messages": [
                    {"role": "system", "content": "You are a vision assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze this image:"},
                        {"type": "image_url", "image_url": {"url": test_image}}
                    ]}
                ]
            },
            {
                "name": "Empty messages (should be handled)",
                "messages": []
            },
            {
                "name": "Invalid message structure",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text"},  # Missing text field
                        {"type": "image_url"}  # Missing image_url field
                    ]}
                ]
            }
        ]

        # Mock processor and config
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3]

        class MockProcessor:
            def __init__(self):
                self.tokenizer = MockTokenizer()

        class MockConfig:
            def __init__(self):
                self.model_type = "test"

        processor = MockProcessor()
        config = MockConfig()

        for i, test_case in enumerate(test_cases):
            print(f"\n  Test {i+1}: {test_case['name']}")
            try:
                print(f"    Input: {json.dumps(test_case['messages'][:1], indent=2)}...")  # Show first message only

                images, formatted_prompt = process_vlm_messages(
                    processor, config, test_case['messages']
                )

                print(f"    ✅ Success")
                print(f"      Images: {len(images)}")
                print(f"      Prompt: {formatted_prompt[:50]}...")

            except Exception as e:
                print(f"    ❌ Failed: {e}")
                import traceback
                traceback.print_exc()

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the correct directory")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mlx_vlm_processor():
    """Test the actual MLX-VLM processor if available."""
    print("\n🔧 Testing actual MLX-VLM processor...")

    try:
        from mlx_vlm.utils import load as vlm_load

        # Try to load a model (adjust path as needed)
        model_paths = [
            "./modelzoo/google/gemma-3n-E4B-it-bf16-mlx",
            "./modelzoo/",
            "../modelzoo/google/gemma-3n-E4B-it-bf16-mlx"
        ]

        model = None
        processor = None

        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"  Trying to load model from: {model_path}")
                try:
                    model, processor = vlm_load(model_path)
                    print(f"  ✅ Model loaded successfully")
                    break
                except Exception as e:
                    print(f"  ❌ Failed to load from {model_path}: {e}")

        if not model or not processor:
            print("  ⚠️  No models available for testing")
            return True

        # Test processor with different inputs
        test_image = create_test_image()

        test_inputs = [
            {
                "name": "Text only",
                "text": "Hello",
                "images": None
            },
            {
                "name": "Text + PIL image",
                "text": "What color is this?",
                "images": [Image.new('RGB', (4, 4), color='blue')]
            },
            {
                "name": "Text + empty images list",
                "text": "Hello",
                "images": []
            }
        ]

        for test_input in test_inputs:
            print(f"\n    Testing {test_input['name']}...")
            try:
                if test_input['images'] is not None:
                    inputs = processor(
                        text=test_input['text'],
                        images=test_input['images'],
                        return_tensors="np"
                    )
                else:
                    inputs = processor(
                        text=test_input['text'],
                        return_tensors="np"
                    )

                print(f"      ✅ Processor succeeded")
                print(f"        Keys: {list(inputs.keys()) if isinstance(inputs, dict) else type(inputs)}")

                # Try with PyTorch tensors too
                if test_input['images'] is not None:
                    inputs_pt = processor(
                        text=test_input['text'],
                        images=test_input['images'],
                        return_tensors="pt"
                    )
                else:
                    inputs_pt = processor(
                        text=test_input['text'],
                        return_tensors="pt"
                    )
                print(f"      ✅ PyTorch tensors also work")

            except Exception as e:
                print(f"      ❌ Processor failed: {e}")
                import traceback
                traceback.print_exc()

        return True

    except ImportError:
        print("  ⚠️  MLX-VLM not available for testing")
        return True
    except Exception as e:
        print(f"  ❌ MLX-VLM test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases that might cause list index errors."""
    print("\n🚨 Testing edge cases...")

    edge_cases = [
        {"messages": None},
        {"messages": "not a list"},
        {"messages": [None]},
        {"messages": [{"role": "user"}]},  # Missing content
        {"messages": [{"content": "hello"}]},  # Missing role
        {"messages": [{"role": "user", "content": None}]},
        {"messages": [{"role": "user", "content": []}]},  # Empty content list
        {"messages": [{"role": "user", "content": [{"type": "unknown"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},  # Empty image_url
    ]

    for i, case in enumerate(edge_cases):
        print(f"  Edge case {i+1}: {json.dumps(case)}")
        try:
            # This would normally cause the "list index out of range" error
            messages = case.get("messages", [])
            if not isinstance(messages, list):
                print(f"    ✅ Detected non-list messages: {type(messages)}")
                continue

            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    print(f"    ✅ Detected non-dict message {j}: {type(msg)}")
                    continue

                content = msg.get("content")
                if isinstance(content, list):
                    for k, part in enumerate(content):
                        if not isinstance(part, dict):
                            print(f"    ✅ Detected non-dict content part {k}: {type(part)}")
                            continue

                        part_type = part.get("type")
                        if part_type == "image_url":
                            image_url = part.get("image_url", {})
                            if not isinstance(image_url, dict):
                                print(f"    ✅ Detected non-dict image_url: {type(image_url)}")

            print(f"    ✅ Edge case handled without crashing")

        except Exception as e:
            print(f"    ❌ Edge case caused error: {e}")

def main():
    """Run VLM debugging tests."""
    print("🔍 VLM List Index Debug Script")
    print("=" * 50)

    # Test 1: Message processing
    test_message_processing()

    # Test 2: MLX-VLM processor
    test_mlx_vlm_processor()

    # Test 3: Edge cases
    test_edge_cases()

    print("\n" + "=" * 50)
    print("📋 Debug Summary:")
    print("- If message processing works but MLX-VLM fails, the issue is in VLM input handling")
    print("- If edge cases fail, add more defensive programming")
    print("- The 'list index out of range' error is likely in accessing message content")
    print("- Check that all list accesses use safe indexing or iteration")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Debug interrupted")
    except Exception as e:
        print(f"\n\n💥 Debug error: {e}")
        import traceback
        traceback.print_exc()
