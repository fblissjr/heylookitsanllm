"""MLX Embedding Provider for EmbeddingGemma models.

Loads EmbeddingGemma (or compatible sentence-transformer style) models
and produces contextual embeddings via the full transformer forward pass
with bidirectional attention.

Supports task prefixes from the sentence-transformers config for optimal
embedding quality on different tasks (search, code retrieval, clustering, etc).
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import orjson

from ..config import ChatRequest
from .base import BaseProvider

logger = logging.getLogger(__name__)

# Task prefix mappings from EmbeddingGemma's config_sentence_transformers.json
TASK_PREFIXES = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "code_retrieval": "task: code retrieval | query: ",
    "clustering": "task: clustering | query: ",
    "classification": "task: classification | query: ",
    "sentence_similarity": "task: sentence similarity | query: ",
    "summarization": "task: summarization | query: ",
}


def apply_task_prefix(text: str, task: Optional[str] = None) -> str:
    """Prepend a task-specific prefix to the input text.

    Args:
        text: The raw input text.
        task: Task type key (e.g. "query", "document", "code_retrieval").
            If None or unknown, returns text unchanged.

    Returns:
        Prefixed text or original text if no matching task.
    """
    if task is None:
        return text
    prefix = TASK_PREFIXES.get(task)
    if prefix is None:
        return text
    return prefix + text


class MLXEmbeddingProvider(BaseProvider):
    """Embedding-only provider using EmbeddingGemma on MLX.

    Does NOT support chat completion -- only embedding extraction.
    """

    def __init__(self, model_id: str, config: Dict, verbose: bool):
        super().__init__(model_id, config, verbose)
        self.model = None
        self.tokenizer = None
        self.processor = None  # For BaseProvider.get_tokenizer() compatibility
        self.max_length = config.get("max_length", 2048)

    def load_model(self):
        """Load EmbeddingGemma model and tokenizer from local path or HF hub."""
        from ..models.embedding_gemma import EmbeddingGemmaModel, EmbeddingGemmaModelArgs

        model_path = self.config["model_path"]

        # Resolve model path
        if os.path.isdir(model_path):
            local_path = Path(model_path)
        else:
            # Try HF hub download
            from huggingface_hub import snapshot_download
            local_path = Path(snapshot_download(model_path))

        logger.info("loading embedding model from %s", local_path)

        # Load config
        config_path = local_path / "config.json"
        with open(config_path, "rb") as f:
            model_config = orjson.loads(f.read())

        # Detect dense projection layer dimensions from 2_Dense, 3_Dense dirs
        dense_out_features = []
        for dense_dir in sorted(local_path.glob("*_Dense")):
            dense_config_path = dense_dir / "config.json"
            if dense_config_path.exists():
                with open(dense_config_path, "rb") as f:
                    dc = orjson.loads(f.read())
                dense_out_features.append(dc["out_features"])

        # Handle sliding_window_pattern field name variants
        swp = model_config.get(
            "sliding_window_pattern",
            model_config.get("_sliding_window_pattern", 6),
        )

        args = EmbeddingGemmaModelArgs(
            model_type=model_config["model_type"],
            hidden_size=model_config["hidden_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            intermediate_size=model_config["intermediate_size"],
            num_attention_heads=model_config["num_attention_heads"],
            num_key_value_heads=model_config["num_key_value_heads"],
            head_dim=model_config["head_dim"],
            vocab_size=model_config["vocab_size"],
            rms_norm_eps=model_config.get("rms_norm_eps", 1e-6),
            rope_theta=model_config.get("rope_theta", 1_000_000.0),
            rope_local_base_freq=model_config.get("rope_local_base_freq", 10_000.0),
            sliding_window=model_config.get("sliding_window", 512),
            sliding_window_pattern=swp,
            max_position_embeddings=model_config.get("max_position_embeddings", 2048),
            query_pre_attn_scalar=model_config.get("query_pre_attn_scalar", 256),
            dense_out_features=dense_out_features if dense_out_features else [3072, 768],
        )

        # Instantiate model
        model = EmbeddingGemmaModel(args)

        # Load transformer weights
        weights = {}
        for sf_path in local_path.glob("*.safetensors"):
            weights.update(mx.load(str(sf_path)))

        # Load dense projection weights
        for i, dense_dir in enumerate(sorted(local_path.glob("*_Dense"))):
            sf_path = dense_dir / "model.safetensors"
            if sf_path.exists():
                dense_weights = mx.load(str(sf_path))
                for key, value in dense_weights.items():
                    # Map 'linear.weight' -> 'dense_layers.{i}.weight'
                    new_key = key.replace("linear.", f"dense_layers.{i}.")
                    weights[new_key] = value

        # Sanitize and load weights
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        self.model = model

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(local_path))
        self.processor = self.tokenizer  # For BaseProvider.get_tokenizer()

        logger.info(
            "loaded embedding model: %d layers, hidden_size=%d, output_dim=%d",
            args.num_hidden_layers,
            args.hidden_size,
            args.dense_out_features[-1] if args.dense_out_features else args.hidden_size,
        )

    def create_chat_completion(self, request: ChatRequest) -> Generator:
        """Not supported -- this is an embedding-only provider."""
        raise NotImplementedError(
            f"{self.model_id} is an embedding-only provider, "
            "chat completions are not supported"
        )

    def get_embeddings(
        self,
        texts: List[str],
        task: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.
            task: Optional task type for prefix (e.g. "query", "document",
                "code_retrieval", "clustering").

        Returns:
            List of embedding vectors as Python float lists.
        """
        # Apply task prefixes
        prefixed = [apply_task_prefix(t, task=task) for t in texts]

        # Tokenize
        encoded = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # Forward pass
        embeddings = self.model(input_ids, attention_mask=attention_mask)
        mx.eval(embeddings)

        # Convert to Python lists
        return embeddings.tolist()

    def unload(self):
        """Release model resources."""
        if self.model is not None:
            self.model = None
            self.tokenizer = None
            self.processor = None
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass
