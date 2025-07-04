import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from heylook_llm.providers.llama_cpp_provider import LlamaCppProvider

class TestLlamaCppProvider(unittest.TestCase):

    def setUp(self):
        """Mock the imported modules and classes."""
        self.mock_llama_chat_format = MagicMock()
        self.mock_llama_cpp = MagicMock()

        # Mock the classes themselves
        self.MockLlama = MagicMock()
        self.MockLlamaRAMCache = MagicMock()
        self.MockJinja2ChatFormatter = MagicMock()
        self.MockLlava15ChatHandler = MagicMock()

        # Assign the mock classes to the mock modules
        self.mock_llama_cpp.Llama = self.MockLlama
        self.mock_llama_cpp.LlamaRAMCache = self.MockLlamaRAMCache
        self.mock_llama_chat_format.Jinja2ChatFormatter = self.MockJinja2ChatFormatter
        self.mock_llama_chat_format.Llava15ChatHandler = self.MockLlava15ChatHandler

        # Patch the sys.modules dictionary
        self.module_patcher = patch.dict('sys.modules', {
            'llama_cpp': self.mock_llama_cpp,
            'llama_cpp.llama_chat_format': self.mock_llama_chat_format,
        })
        self.module_patcher.start()

    def tearDown(self):
        self.module_patcher.stop()

    def test_jinja2_chat_format(self):
        """Test that Jinja2ChatFormatter is used when a template is provided."""
        model_config = {
            'model_path': '/fake/model.gguf',
            'chat_format_template': '/fake/template.jinja2'
        }
        with patch('builtins.open', unittest.mock.mock_open(read_data='template_content')):
            provider = LlamaCppProvider('test-model', model_config, verbose=False)
            

        self.MockJinja2ChatFormatter.assert_called_once_with(template='template_content')
        formatter_instance = self.MockJinja2ChatFormatter.return_value
        formatter_instance.to_chat_handler.assert_called_once()
        self.MockLlama.assert_called_once()
        _, kwargs = self.MockLlama.call_args
        self.assertEqual(kwargs['chat_handler'], formatter_instance.to_chat_handler.return_value)

    def test_vision_chat_handler(self):
        """Test that Llava15ChatHandler is used for vision models."""
        model_config = {
            'model_path': '/fake/model.gguf',
            'mmproj_path': '/fake/mmproj.gguf'
        }
        provider = LlamaCppProvider('test-model', model_config, verbose=False)
        

        self.MockLlava15ChatHandler.assert_called_once_with(clip_model_path='/fake/mmproj.gguf', verbose=False)
        self.MockLlama.assert_called_once()
        _, kwargs = self.MockLlama.call_args
        self.assertEqual(kwargs['chat_handler'], self.MockLlava15ChatHandler.return_value)

    def test_named_chat_format(self):
        """Test that a named chat_format is passed to Llama."""
        model_config = {
            'model_path': '/fake/model.gguf',
            'chat_format': 'llama-3'
        }
        provider = LlamaCppProvider('test-model', model_config, verbose=False)
        

        self.MockLlama.assert_called_once()
        _, kwargs = self.MockLlama.call_args
        self.assertEqual(kwargs['chat_format'], 'llama-3')
        self.assertIsNone(kwargs.get('chat_handler'))

    def test_gguf_autodetection(self):
        """Test that the provider falls back to GGUF auto-detection."""
        model_config = {
            'model_path': '/fake/model.gguf'
        }
        provider = LlamaCppProvider('test-model', model_config, verbose=False)
        

        self.MockLlama.assert_called_once()
        _, kwargs = self.MockLlama.call_args
        self.assertIsNone(kwargs.get('chat_format'))
        self.assertIsNone(kwargs.get('chat_handler'))

if __name__ == '__main__':
    unittest.main()
