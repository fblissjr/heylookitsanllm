import unittest
from unittest.mock import MagicMock, patch, call
import os
import yaml
import tempfile
import logging

# Add src to path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from heylook_llm.router import ModelRouter
from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider

# Mock Provider class for testing
class MockProvider(BaseProvider):
    def __init__(self, model_id, model_config, is_debug):
        self.model_id = model_id
        self.model_config = model_config
        self.is_debug = is_debug
        self.unload = MagicMock()

    def load_model(self):
        pass

    def create_chat_completion(self, request):
        pass

    def get_model_path(self):
        return self.model_config.get("model_path")

@patch('heylook_llm.router.MLXProvider', new=MockProvider)
@patch('heylook_llm.router.LlamaCppProvider', new=MockProvider)
class TestModelRouter(unittest.TestCase):

    def setUp(self):
        """Set up a temporary config file for tests."""
        self.config_data = {
            'default_model': 'model1-mlx',
            'max_loaded_models': 2,
            'models': [
                {
                    'id': 'model1-mlx',
                    'provider': 'mlx',
                    'enabled': True,
                    'config': {'model_path': '/fake/path/model1'}
                },
                {
                    'id': 'model2-llama',
                    'provider': 'llama_cpp',
                    'enabled': True,
                    'config': {'model_path': '/fake/path/model2'}
                },
                {
                    'id': 'model3-mlx',
                    'provider': 'mlx',
                    'enabled': True,
                    'config': {'model_path': '/fake/path/model3'}
                }
            ]
        }
        # Create a temporary YAML file
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
        yaml.dump(self.config_data, self.temp_config_file)
        self.temp_config_file.close()
        self.config_path = self.temp_config_file.name

    def tearDown(self):
        """Clean up the temporary config file."""
        os.unlink(self.config_path)

    def test_initialization(self):
        """Test that the router initializes correctly without loading a model."""
        # Override config to ensure no model is pre-warmed by disabling all models
        for model in self.config_data['models']:
            model['enabled'] = False
        self.config_data['default_model'] = None
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

        router = ModelRouter(config_path=self.config_path, log_level=logging.INFO, initial_model_id=None)
        self.assertEqual(len(router.list_available_models()), 0)
        self.assertEqual(len(router.providers), 0)

    def test_get_provider_loads_and_caches(self):
        """Test that get_provider loads a model and caches it."""
        router = ModelRouter(config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None)
        
        # Load first model
        provider1 = router.get_provider('model1-mlx')
        self.assertIsNotNone(provider1)
        self.assertEqual(provider1.model_id, 'model1-mlx')
        self.assertEqual(len(router.providers), 1)
        # The mock provider doesn't have load_model, the router calls the constructor
        
        # Get the same model again, should be a cache hit
        provider1_cached = router.get_provider('model1-mlx')
        self.assertIs(provider1, provider1_cached)
        self.assertEqual(len(router.providers), 1)


    def test_lru_eviction(self):
        """Test that the least recently used model is evicted when the cache is full."""
        router = ModelRouter(config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None)
        
        # Load model 1
        provider1 = router.get_provider('model1-mlx')
        self.assertIn('model1-mlx', router.providers)
        
        # Load model 2
        provider2 = router.get_provider('model2-llama')
        self.assertIn('model1-mlx', router.providers)
        self.assertIn('model2-llama', router.providers)
        self.assertEqual(len(router.providers), 2) # Cache is now full

        # Load model 3, which should evict model 1
        provider3 = router.get_provider('model3-mlx')
        self.assertIn('model2-llama', router.providers)
        self.assertIn('model3-mlx', router.providers)
        self.assertNotIn('model1-mlx', router.providers)
        self.assertEqual(len(router.providers), 2)
        
        # Check that model 1's unload method was called
        provider1.unload.assert_called_once()
        provider2.unload.assert_not_called()

    def test_hot_swapping(self):
        """Test switching between already loaded models."""
        router = ModelRouter(config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None)

        # Load model 1 and 2
        provider1 = router.get_provider('model1-mlx')
        provider2 = router.get_provider('model2-llama')
        
        # Access model 1 again (should be a cache hit)
        router.get_provider('model1-mlx')
        
        # Access model 2 again (should be a cache hit)
        router.get_provider('model2-llama')
        
        self.assertEqual(len(router.providers), 2)

    def test_max_loaded_models_one(self):
        """Test behavior with max_loaded_models set to 1."""
        self.config_data['max_loaded_models'] = 1
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
            
        router = ModelRouter(config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None)
        
        # Load model 1
        provider1 = router.get_provider('model1-mlx')
        self.assertIn('model1-mlx', router.providers)
        self.assertEqual(len(router.providers), 1)
        
        # Load model 2, should evict model 1
        provider2 = router.get_provider('model2-llama')
        self.assertNotIn('model1-mlx', router.providers)
        self.assertIn('model2-llama', router.providers)
        self.assertEqual(len(router.providers), 1)
        provider1.unload.assert_called_once()

if __name__ == '__main__':
    unittest.main()
