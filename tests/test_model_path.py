#!/usr/bin/env python3
"""
Diagnostic script to check model paths and suggest fixes.

Usage: python check_model_paths.py
"""

import os
import yaml
import sys
from pathlib import Path

def check_model_paths():
    """Check if model paths in models.yaml actually exist."""
    print("🔍 Checking model paths in models.yaml...")
    print("=" * 50)

    # Check if models.yaml exists
    if not os.path.exists("models.yaml"):
        print("❌ models.yaml not found in current directory")
        print("   Make sure you're running this from the heylookllm directory")
        return False

    try:
        with open("models.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading models.yaml: {e}")
        return False

    models = config.get("models", [])
    if not models:
        print("❌ No models found in models.yaml")
        return False

    print(f"Found {len(models)} model(s) in config:")
    print()

    all_good = True
    for model in models:
        model_id = model.get("id", "unknown")
        model_config = model.get("config", {})
        model_path = model_config.get("model_path", "")
        enabled = model.get("enabled", True)

        print(f"📝 Model: {model_id}")
        print(f"   Path: {model_path}")
        print(f"   Enabled: {enabled}")

        if not enabled:
            print("   ⏭️  Skipped (disabled)")
            print()
            continue

        # Convert relative to absolute path for checking
        if model_path.startswith('./'):
            abs_path = os.path.abspath(model_path)
            print(f"   Absolute path: {abs_path}")
        else:
            abs_path = model_path

        # Check if path exists
        if os.path.exists(abs_path):
            print("   ✅ Path exists")

            # Check for key files
            config_file = os.path.join(abs_path, "config.json")
            if os.path.exists(config_file):
                print("   ✅ config.json found")
            else:
                print("   ⚠️  config.json missing")
                all_good = False

            # Check for weight files
            weight_files = list(Path(abs_path).glob("*.safetensors"))
            if weight_files:
                print(f"   ✅ Found {len(weight_files)} weight file(s)")
            else:
                print("   ⚠️  No .safetensors files found")
                all_good = False

        else:
            print("   ❌ Path does not exist")
            all_good = False

            # Suggest possible fixes
            print("   💡 Possible fixes:")
            if model_path.startswith("./modelzoo"):
                print("      1. Check if the modelzoo directory exists")
                print("      2. Verify the exact folder name (case sensitive)")
                print("      3. Try downloading the model:")
                print(f"         huggingface-cli download {model_path.split('/')[-1]} --local-dir {model_path}")
            else:
                print("      1. Use absolute path instead of relative")
                print("      2. Check spelling and case sensitivity")
                print("      3. Ensure the model is downloaded")

        print()

    return all_good

def suggest_fixes():
    """Suggest common fixes for model loading issues."""
    print("🛠️  Common Fixes for Model Loading Issues:")
    print("=" * 50)
    print()

    print("1. **Use absolute paths** in models.yaml:")
    print("   Instead of: modelzoo/mlx-community/model-name")
    print("   Use: ./modelzoo/mlx-community/model-name")
    print()

    print("2. **Download missing models:**")
    print("   huggingface-cli download mlx-community/gemma-3n-E4B-it-bf16-mlx --local-dir ./modelzoo/mlx-community/gemma-3n-E4B-it-bf16-mlx")
    print()

    print("3. **Check directory structure:**")
    print("   heylookllm/")
    print("   ├── models.yaml")
    print("   └── modelzoo/")
    print("       └── mlx-community/")
    print("           └── model-name/")
    print("               ├── config.json")
    print("               └── *.safetensors")
    print()

    print("4. **Test model loading manually:**")
    print("   python -c \"from mlx_vlm import load; model, processor = load('./modelzoo/path/to/model')\"")

def main():
    """Main diagnostic function."""
    print("🩺 heylookllm Model Path Diagnostic")
    print("=" * 50)
    print()

    # Check current directory
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")

    # Check if this looks like the heylookllm directory
    if not os.path.exists("models.yaml"):
        print("⚠️  models.yaml not found in current directory")
        print("   Please run this from your heylookllm directory")
        return False

    print()

    # Check model paths
    paths_ok = check_model_paths()

    print()
    suggest_fixes()

    if paths_ok:
        print("🎉 All model paths look good!")
        return True
    else:
        print("⚠️  Some model paths need attention. See suggestions above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)
