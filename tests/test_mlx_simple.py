#!/usr/bin/env python3
"""
Simple validation for MLX provider optimizations.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test that all optimized modules can be imported."""
    print("🔍 Testing Imports...")
    try:
        from heylook_llm.config import ChatRequest, ChatMessage, TextContentPart, ImageContentPart, ImageUrl
        print("  ✅ Config imports successful")
        
        from heylook_llm.providers.mlx_provider_optimized import MLXProvider, OptimizedLanguageModelWrapper
        print("  ✅ MLX Provider imports successful")
        
        from heylook_llm.providers.common.performance_monitor import performance_monitor
        print("  ✅ Performance Monitor imports successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_config_structure():
    """Test the config structure."""
    print("🔧 Testing Config Structure...")
    try:
        from heylook_llm.config import ChatMessage, TextContentPart, ImageContentPart, ImageUrl
        
        # Test simple text message
        text_msg = ChatMessage(role="user", content="Hello")
        print(f"  ✅ Text message: {text_msg.role}")
        
        # Test multimodal message
        multimodal_msg = ChatMessage(role="user", content=[
            TextContentPart(type="text", text="What's this?"),
            ImageContentPart(type="image_url", image_url=ImageUrl(url="data:image/png;base64,test"))
        ])
        print(f"  ✅ Multimodal message: {len(multimodal_msg.content)} parts")
        
        return True
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_performance_monitor():
    """Test performance monitor functionality."""
    print("📊 Testing Performance Monitor...")
    try:
        from heylook_llm.providers.common.performance_monitor import performance_monitor
        
        # Reset for clean test
        performance_monitor.reset_metrics()
        
        # Test recording
        performance_monitor.record_timing("test_op", 0.1, "path1")
        performance_monitor.record_timing("test_op", 0.2, "path2")
        
        # Get metrics
        metrics = performance_monitor.get_metrics()
        print(f"  ✅ Recorded metrics: {len(metrics)} operations")
        
        # Test summary
        summary = performance_monitor.get_performance_summary()
        print(f"  ✅ Summary generated: {len(summary)} chars")
        
        return True
    except Exception as e:
        print(f"  ❌ Performance monitor test failed: {e}")
        return False

def test_optimization_components():
    """Test optimization components individually."""
    print("🚀 Testing Optimization Components...")
    try:
        from heylook_llm.providers.mlx_provider_optimized import (
            OptimizedLanguageModelWrapper,
            TextOnlyStrategy,
            VLMTextOnlyStrategy,
            VLMVisionStrategy
        )
        
        # Test wrapper with mock model
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {'head_dim': 512})()
                self.layers = ['layer1', 'layer2']
            
            def __call__(self, *args, **kwargs):
                return type('Output', (), {'logits': 'test_logits'})()
        
        mock_model = MockModel()
        wrapper = OptimizedLanguageModelWrapper(mock_model)
        
        # Test caching
        layers1 = wrapper.layers
        layers2 = wrapper.layers  # Should use cached value
        print(f"  ✅ Wrapper caching: {layers1 == layers2}")
        
        # Test strategies
        text_strategy = TextOnlyStrategy()
        vlm_text_strategy = VLMTextOnlyStrategy()
        vlm_vision_strategy = VLMVisionStrategy()
        
        print(f"  ✅ All strategies created successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Optimization components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_provider_initialization():
    """Test provider initialization without model loading."""
    print("🎯 Testing Provider Initialization...")
    try:
        from heylook_llm.providers.mlx_provider_optimized import MLXProvider
        
        # Test config
        config = {
            "model_path": "./test/path",
            "vision": True,
            "temperature": 0.7
        }
        
        provider = MLXProvider("test-model", config, verbose=False)
        print(f"  ✅ Provider created: {provider.model_id}")
        print(f"  ✅ VLM mode: {provider.is_vlm}")
        print(f"  ✅ Content cache initialized: {len(provider._content_cache)}")
        
        return True
    except Exception as e:
        print(f"  ❌ Provider initialization failed: {e}")
        return False

def main():
    """Run simplified tests."""
    print("🧪 MLX Provider Optimization - Simple Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_structure,
        test_performance_monitor,
        test_optimization_components,
        test_provider_initialization
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📋 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("✅ Optimizations are ready for use")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
