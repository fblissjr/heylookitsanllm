#!/usr/bin/env python3
"""
Quick test script to validate the gemma3n-e4b-it model loading fix.
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

import logging
from heylook_llm.providers.mlx_provider_optimized import MLXProvider

def test_gemma3n_loading():
    """Test loading the problematic gemma3n-e4b-it model."""
    
    print("🧪 Testing gemma3n-e4b-it model loading fix...")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Model config for gemma3n-e4b-it
    model_config = {
        "model_path": "modelzoo/google/gemma-3n-E4B-it-bf16-mlx",
        "vision": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.15,
        "repetition_context_size": 25
    }
    
    try:
        print("📥 Creating MLX provider...")
        provider = MLXProvider("gemma3n-e4b-it", model_config, verbose=True)
        
        print("🔄 Loading model with resilient loading...")
        provider.load_model()
        
        print("✅ SUCCESS! Model loaded successfully!")
        print("🎉 The language_model.lm_head.weight error has been fixed!")
        
        # Clean up
        provider.unload()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Main test function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_gemma3n_fix.py")
        print("")
        print("This script tests if the gemma3n-e4b-it model loading fix works.")
        print("It will attempt to load the model using the new resilient loading strategies.")
        return 0
    
    success = test_gemma3n_loading()
    
    if success:
        print("\n" + "="*60)
        print("🎯 FIX VALIDATION SUCCESSFUL!")
        print("🚀 You can now start the server with gemma3n-e4b-it")
        print("✨ Try: heylookllm --host 0.0.0.0 --log-level DEBUG")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️ Fix validation failed - please check the logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
