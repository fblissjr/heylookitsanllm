# test_resilient_loader.py
"""Test suite for the resilient model loader."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.heylook_llm.providers.common.resilient_loader import (
    ResilientModelLoader,
    ModelLoadingError,
    load_model_resilient,
    resilient_loader
)


class TestResilientModelLoader:
    """Test the resilient model loader functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.loader = ResilientModelLoader(verbose=True)
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = str(Path(self.temp_dir) / "test_model")
        
    def test_text_model_loading_success(self):
        """Test successful text model loading."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.lm_load') as mock_lm_load:
            mock_lm_load.return_value = (mock_model, mock_processor)
            
            model, processor = self.loader.load_model(self.model_path, is_vlm=False)
            
            assert model is mock_model
            assert processor is mock_processor
            mock_lm_load.assert_called_once_with(self.model_path)
    
    def test_text_model_loading_failure(self):
        """Test text model loading failure."""
        with patch('src.heylook_llm.providers.common.resilient_loader.lm_load') as mock_lm_load:
            mock_lm_load.side_effect = Exception("Test error")
            
            with pytest.raises(ModelLoadingError):
                self.loader.load_model(self.model_path, is_vlm=False)
    
    def test_vlm_model_skip_audio_success(self):
        """Test VLM model loading with skip_audio=True success."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            mock_vlm_load.return_value = (mock_model, mock_processor)
            
            model, processor = self.loader.load_model(self.model_path, is_vlm=True)
            
            assert model is mock_model
            assert processor is mock_processor
            # Should have tried skip_audio=True first
            mock_vlm_load.assert_called_with(self.model_path, skip_audio=True)
    
    def test_vlm_model_skip_audio_fallback(self):
        """Test VLM model loading with skip_audio fallback."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            # First call (skip_audio=True) raises TypeError
            # Second call (fallback) succeeds
            mock_vlm_load.side_effect = [
                TypeError("skip_audio parameter not supported"),
                (mock_model, mock_processor)
            ]
            
            model, processor = self.loader.load_model(self.model_path, is_vlm=True)
            
            assert model is mock_model
            assert processor is mock_processor
            # Should have called twice: once with skip_audio, once without
            assert mock_vlm_load.call_count == 2
    
    def test_vlm_model_audiomodel_error_handling(self):
        """Test VLM model loading with AudioModel error."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            with patch.object(self.loader, '_load_vlm_monkey_patched') as mock_monkey_patch:
                # First few strategies fail with AudioModel error
                mock_vlm_load.side_effect = AttributeError("module 'mlx_vlm.models.qwen2_5_vl' has no attribute 'AudioModel'")
                
                # Monkey patch strategy succeeds
                mock_monkey_patch.return_value = (mock_model, mock_processor)
                
                model, processor = self.loader.load_model(self.model_path, is_vlm=True)
                
                assert model is mock_model
                assert processor is mock_processor
                mock_monkey_patch.assert_called_once()
    
    def test_strategy_caching(self):
        """Test that successful strategies are cached."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            mock_vlm_load.return_value = (mock_model, mock_processor)
            
            # First load - should try strategies
            model1, processor1 = self.loader.load_model(self.model_path, is_vlm=True)
            
            # Second load - should use cached strategy
            model2, processor2 = self.loader.load_model(self.model_path, is_vlm=True)
            
            # Both should succeed
            assert model1 is mock_model
            assert model2 is mock_model
            
            # Should have cached the successful strategy
            assert len(self.loader._successful_strategies) == 1
    
    def test_loading_stats(self):
        """Test loading statistics collection."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            mock_vlm_load.return_value = (mock_model, mock_processor)
            
            # Load a model
            self.loader.load_model(self.model_path, is_vlm=True)
            
            # Get stats
            stats = self.loader.get_loading_stats()
            
            assert stats['total_models_loaded'] == 1
            assert 'skip_audio_explicit' in stats['strategy_usage']
            assert 'known_quirks' in stats
    
    def test_convenience_function(self):
        """Test the convenience function."""
        mock_model = Mock()
        mock_processor = Mock()
        
        with patch('src.heylook_llm.providers.common.resilient_loader.lm_load') as mock_lm_load:
            mock_lm_load.return_value = (mock_model, mock_processor)
            
            model, processor = load_model_resilient(
                model_path=self.model_path,
                is_vlm=False,
                verbose=True
            )
            
            assert model is mock_model
            assert processor is mock_processor
    
    def test_qwen25_vl_specific_handling(self):
        """Test specific handling for qwen2.5-vl models."""
        mock_model = Mock()
        mock_processor = Mock()
        
        # Mock the config loading to return qwen2.5-vl model type
        mock_config = {"model_type": "qwen2_5_vl"}
        
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            with patch('src.heylook_llm.providers.common.resilient_loader.get_model_path') as mock_get_path:
                with patch('src.heylook_llm.providers.common.resilient_loader.load_config') as mock_load_config:
                    
                    # Setup mocks
                    mock_get_path.return_value = Path(self.model_path)
                    mock_load_config.return_value = mock_config
                    
                    # First calls fail with AudioModel error
                    mock_vlm_load.side_effect = [
                        AttributeError("module 'mlx_vlm.models.qwen2_5_vl' has no attribute 'AudioModel'"),
                        AttributeError("module 'mlx_vlm.models.qwen2_5_vl' has no attribute 'AudioModel'"),
                        AttributeError("module 'mlx_vlm.models.qwen2_5_vl' has no attribute 'AudioModel'"),
                        # Monkey patch strategy works
                        (mock_model, mock_processor)
                    ]
                    
                    model, processor = self.loader.load_model(self.model_path, is_vlm=True)
                    
                    assert model is mock_model
                    assert processor is mock_processor
    
    def test_all_strategies_fail(self):
        """Test behavior when all loading strategies fail."""
        with patch('src.heylook_llm.providers.common.resilient_loader.vlm_load') as mock_vlm_load:
            # Make all strategies fail
            mock_vlm_load.side_effect = Exception("All strategies failed")
            
            with pytest.raises(ModelLoadingError, match="All loading strategies failed"):
                self.loader.load_model(self.model_path, is_vlm=True)
    
    def test_global_resilient_loader_instance(self):
        """Test that the global resilient loader instance works."""
        assert resilient_loader is not None
        assert isinstance(resilient_loader, ResilientModelLoader)
        
        # Should be able to get stats even if no models loaded
        stats = resilient_loader.get_loading_stats()
        assert isinstance(stats, dict)
        assert 'total_models_loaded' in stats


if __name__ == "__main__":
    # Run a simple test
    print("Testing resilient loader...")
    
    # Test the convenience function with mocked loading
    with patch('src.heylook_llm.providers.common.resilient_loader.lm_load') as mock_lm_load:
        mock_lm_load.return_value = (Mock(), Mock())
        
        try:
            model, processor = load_model_resilient(
                model_path="test_model",
                is_vlm=False,
                verbose=True
            )
            print("✅ Convenience function test passed")
        except Exception as e:
            print(f"❌ Convenience function test failed: {e}")
    
    # Test stats collection
    stats = resilient_loader.get_loading_stats()
    print(f"✅ Stats collection test passed: {stats}")
    
    print("Basic tests completed!")
