#!/usr/bin/env python3
"""
Debug script to investigate the gemma3n model weight mismatch issue.
This will help us understand what weights are in the model vs what MLX VLM expects.
"""

import sys
import os
from pathlib import Path
import json

def inspect_model_weights(model_path):
    """Inspect the model weights and structure."""
    
    print(f"🔍 Inspecting model at: {model_path}")
    print("=" * 60)
    
    model_path = Path(model_path)
    
    # Check if model directory exists
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    print("📁 Model directory contents:")
    for item in sorted(model_path.iterdir()):
        print(f"  {item.name}")
    
    # Check config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        print(f"\n📄 Config file found: {config_path}")
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            print("📝 Model configuration:")
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                elif isinstance(value, dict) and len(value) < 10:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: <{type(value).__name__} with {len(value) if hasattr(value, '__len__') else '?'} items>")
                    
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    else:
        print(f"❌ No config.json found in {model_path}")
    
    # Look for weight files
    print(f"\n🔍 Looking for weight files...")
    weight_files = []
    for pattern in ["*.safetensors", "*.npz", "*.mlx"]:
        weight_files.extend(model_path.glob(pattern))
    
    if weight_files:
        print(f"Found {len(weight_files)} weight files:")
        for wf in weight_files:
            print(f"  {wf.name} ({wf.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("❌ No weight files found")
    
    # Try to load and inspect weights with MLX
    print(f"\n🔧 Attempting to load weights with MLX...")
    try:
        import mlx.core as mx
        
        # Try to load the weights directly
        weights = {}
        for weight_file in weight_files:
            if weight_file.suffix == '.safetensors':
                print(f"📥 Loading SafeTensors: {weight_file.name}")
                try:
                    import safetensors
                    with safetensors.safe_open(weight_file, framework="mlx") as f:
                        for key in f.keys():
                            weights[key] = f.get_tensor(key)
                except Exception as e:
                    print(f"  ❌ Error loading {weight_file.name}: {e}")
            elif weight_file.suffix == '.npz':
                print(f"📥 Loading NPZ: {weight_file.name}")
                try:
                    loaded = mx.load(str(weight_file))
                    weights.update(loaded)
                except Exception as e:
                    print(f"  ❌ Error loading {weight_file.name}: {e}")
        
        if weights:
            print(f"\n📊 Found {len(weights)} weight tensors:")
            
            # Look for language_model related weights
            lm_weights = [k for k in weights.keys() if 'language_model' in k]
            if lm_weights:
                print(f"\n🎯 Language model weights ({len(lm_weights)}):")
                for w in sorted(lm_weights)[:20]:  # First 20
                    shape = weights[w].shape if hasattr(weights[w], 'shape') else 'unknown'
                    print(f"  {w}: {shape}")
                if len(lm_weights) > 20:
                    print(f"  ... and {len(lm_weights) - 20} more")
            else:
                print(f"❌ No 'language_model' weights found!")
            
            # Check for the specific missing weight
            missing_weight = "language_model.lm_head.weight"
            if missing_weight in weights:
                print(f"\n✅ Found the supposedly missing weight: {missing_weight}")
                print(f"   Shape: {weights[missing_weight].shape}")
            else:
                print(f"\n❌ Confirmed missing: {missing_weight}")
                
                # Look for similar weights
                similar = [k for k in weights.keys() if 'lm_head' in k or 'head' in k]
                if similar:
                    print(f"🔍 Found similar weights:")
                    for s in similar:
                        shape = weights[s].shape if hasattr(weights[s], 'shape') else 'unknown'
                        print(f"  {s}: {shape}")
            
            # Show first 10 weight names for context
            print(f"\n📝 First 10 weight names (for context):")
            for w in sorted(list(weights.keys()))[:10]:
                shape = weights[w].shape if hasattr(weights[w], 'shape') else 'unknown'
                print(f"  {w}: {shape}")
                
        else:
            print(f"❌ No weights could be loaded")
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
    except Exception as e:
        print(f"❌ Error inspecting weights: {e}")
    
    return True

def main():
    """Main inspection function."""
    
    print("🔍 MLX VLM Model Weight Inspector")
    print("=" * 60)
    
    # Default model path
    model_path = "modelzoo/google/gemma-3n-E4B-it-bf16-mlx"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python debug_model_weights.py [model_path]")
            print("")
            print("This script inspects MLX VLM model weights to debug loading issues.")
            print(f"Default model path: {model_path}")
            return 0
        else:
            model_path = sys.argv[1]
    
    print(f"Target model: {model_path}")
    print("=" * 60)
    
    success = inspect_model_weights(model_path)
    
    if success:
        print("\n" + "="*60)
        print("🎯 INSPECTION COMPLETE")
        print("Check the output above for clues about the weight mismatch.")
        print("Look for:")
        print("  1. Whether 'language_model.lm_head.weight' exists")
        print("  2. Similar weight names that might be the correct one")
        print("  3. The overall structure of language_model weights")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️ Inspection failed - check model path")
        return 1

if __name__ == "__main__":
    sys.exit(main())
