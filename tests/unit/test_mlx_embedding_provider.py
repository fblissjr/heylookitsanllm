"""Tests for MLXEmbeddingProvider.

Covers:
- Provider initialization and config parsing
- get_embeddings returns correct shapes
- Task prefix prepending
- create_chat_completion raises NotImplementedError
- get_tokenizer returns tokenizer from provider
"""

import pytest
from unittest.mock import MagicMock, patch
import mlx.core as mx

from heylook_llm.providers.base import BaseProvider


class TestMLXEmbeddingProviderInit:
    """Test provider initialization and config parsing."""

    def test_provider_inherits_base(self):
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider
        assert issubclass(MLXEmbeddingProvider, BaseProvider)

    def test_provider_init_stores_config(self):
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        config = {
            "model_path": "/fake/path",
            "max_length": 512,
        }
        provider = MLXEmbeddingProvider("test-model", config, verbose=False)
        assert provider.model_id == "test-model"
        assert provider.config["model_path"] == "/fake/path"

    def test_chat_completion_raises(self):
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        provider = MLXEmbeddingProvider("test-model", {"model_path": "/fake"}, verbose=False)
        with pytest.raises(NotImplementedError, match="embedding-only"):
            list(provider.create_chat_completion(MagicMock()))


class TestTaskPrefixes:
    """Test task prefix prepending for different task types."""

    def test_query_prefix(self):
        from heylook_llm.providers.mlx_embedding_provider import apply_task_prefix

        result = apply_task_prefix("hello world", task="query")
        assert result == "task: search result | query: hello world"

    def test_document_prefix(self):
        from heylook_llm.providers.mlx_embedding_provider import apply_task_prefix

        result = apply_task_prefix("some document text", task="document")
        assert result == "title: none | text: some document text"

    def test_code_retrieval_prefix(self):
        from heylook_llm.providers.mlx_embedding_provider import apply_task_prefix

        result = apply_task_prefix("find auth function", task="code_retrieval")
        assert result == "task: code retrieval | query: find auth function"

    def test_clustering_prefix(self):
        from heylook_llm.providers.mlx_embedding_provider import apply_task_prefix

        result = apply_task_prefix("cluster this", task="clustering")
        assert result == "task: clustering | query: cluster this"

    def test_no_task_no_prefix(self):
        from heylook_llm.providers.mlx_embedding_provider import apply_task_prefix

        result = apply_task_prefix("raw text", task=None)
        assert result == "raw text"

    def test_unknown_task_no_prefix(self):
        from heylook_llm.providers.mlx_embedding_provider import apply_task_prefix

        result = apply_task_prefix("raw text", task="unknown_task")
        assert result == "raw text"


class TestGetEmbeddings:
    """Test get_embeddings with a mocked model."""

    @pytest.fixture
    def provider_with_tiny_model(self):
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider
        from heylook_llm.models.embedding_gemma import EmbeddingGemmaModel, EmbeddingGemmaModelArgs

        args = EmbeddingGemmaModelArgs(
            model_type="gemma3_text",
            hidden_size=32,
            num_hidden_layers=1,
            intermediate_size=64,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=128,
            dense_out_features=[64, 32],
        )
        model = EmbeddingGemmaModel(args)
        mx.eval(model.parameters())

        provider = MLXEmbeddingProvider(
            "test-model",
            {"model_path": "/fake", "max_length": 32},
            verbose=False,
        )
        provider.model = model

        # Create a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        def mock_call(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # Return fake token IDs (all within vocab_size=128)
            result = {
                "input_ids": [[1, 2, 3, 4, 5]] * len(texts),
                "attention_mask": [[1, 1, 1, 1, 1]] * len(texts),
            }
            return result

        mock_tokenizer.side_effect = mock_call
        mock_tokenizer.__call__ = mock_call
        provider.tokenizer = mock_tokenizer

        return provider

    def test_single_text_returns_list_of_lists(self, provider_with_tiny_model):
        result = provider_with_tiny_model.get_embeddings(["hello world"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 32  # output dim

    def test_batch_texts_returns_correct_count(self, provider_with_tiny_model):
        result = provider_with_tiny_model.get_embeddings(["a", "b", "c"])
        assert len(result) == 3

    def test_embeddings_are_normalized(self, provider_with_tiny_model):
        result = provider_with_tiny_model.get_embeddings(["test"])
        import math
        norm = math.sqrt(sum(x * x for x in result[0]))
        assert abs(norm - 1.0) < 1e-4

    def test_task_prefix_applied(self, provider_with_tiny_model):
        """When task is specified, prefix should be applied to text before tokenization."""
        captured_texts = []
        original_fn = provider_with_tiny_model.tokenizer.__call__

        def capturing_call(texts, **kwargs):
            captured_texts.append(texts)
            return original_fn(texts, **kwargs)

        provider_with_tiny_model.tokenizer = MagicMock(side_effect=capturing_call)
        provider_with_tiny_model.tokenizer.pad_token_id = 0
        provider_with_tiny_model.get_embeddings(["hello"], task="query")
        assert len(captured_texts) == 1
        assert captured_texts[0] == ["task: search result | query: hello"]


class TestGetTokenizer:
    """Test that get_tokenizer works with the embedding provider."""

    def test_returns_tokenizer(self):
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        provider = MLXEmbeddingProvider("test", {"model_path": "/fake"}, verbose=False)
        mock_tok = MagicMock(spec=[])  # empty spec so no auto-generated attrs
        mock_tok.decode = MagicMock()
        provider.processor = mock_tok
        # BaseProvider.get_tokenizer checks for decode attr -> returns processor
        assert provider.get_tokenizer() is mock_tok
