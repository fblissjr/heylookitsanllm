# src/heylook_llm/hidden_states.py
"""
Hidden states extraction for LLM models.

This module provides functionality to extract raw hidden states from specific
layers of LLM models, enabling use as a text encoder backend for image
generation models like Z-Image.

Key differences from embeddings:
- Returns full sequence [seq_len, hidden_dim], not pooled
- Extracts from specific layer (default: -2, second-to-last)
- Filters out padding tokens via attention mask
- Supports base64 encoding for efficient transfer
"""
import base64
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class HiddenStatesRequest(BaseModel):
    """Request model for hidden states extraction."""

    input: Union[str, List[str]] = Field(
        ..., description="Text(s) to encode (with chat template already applied)"
    )
    model: str = Field(..., description="Model ID to use for extraction")
    layer: int = Field(
        -2, description="Which layer to extract from (-2 for second-to-last)"
    )
    max_length: int = Field(512, description="Maximum sequence length")
    return_attention_mask: bool = Field(
        False, description="Whether to return the attention mask"
    )
    encoding_format: str = Field(
        "float", description="Output format: 'float' (nested list) or 'base64'"
    )


class HiddenStatesResponse(BaseModel):
    """Response model for hidden states extraction."""

    hidden_states: Union[List[List[float]], str] = Field(
        ..., description="Hidden states as nested list or base64 string"
    )
    shape: List[int] = Field(..., description="Shape of hidden states [seq_len, hidden_dim]")
    model: str = Field(..., description="Model ID used")
    layer: int = Field(..., description="Layer extracted from")
    dtype: str = Field(..., description="Data type of the hidden states")
    encoding_format: Optional[str] = Field(
        None, description="Encoding format used (only present for base64)"
    )
    attention_mask: Optional[List[int]] = Field(
        None, description="Attention mask if requested"
    )


def encode_hidden_states_base64(hidden_states: np.ndarray) -> str:
    """
    Encode hidden states as base64 string.

    Args:
        hidden_states: numpy array of shape [seq_len, hidden_dim]

    Returns:
        Base64-encoded string of float32 data
    """
    # Ensure float32 and C-contiguous for consistent encoding
    data = np.ascontiguousarray(hidden_states, dtype=np.float32)
    return base64.b64encode(data.tobytes()).decode("ascii")


class HiddenStatesExtractor:
    """Base class for extracting hidden states from models."""

    def extract(
        self, texts: List[str], layer: int = -2, max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Extract hidden states from texts.

        Args:
            texts: List of texts to process
            layer: Which layer to extract (-2 for second-to-last)
            max_length: Maximum sequence length

        Returns:
            List of dicts with hidden_states, shape, dtype per text
        """
        raise NotImplementedError


class MLXHiddenStatesExtractor(HiddenStatesExtractor):
    """Extract hidden states from MLX models."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

        # Handle different processor types to get tokenizer
        if hasattr(processor, "_tokenizer"):
            # MLX TokenizerWrapper case
            self.tokenizer = processor._tokenizer
        elif hasattr(processor, "tokenizer"):
            # VLM processor case
            self.tokenizer = processor.tokenizer
        else:
            # Direct tokenizer case
            self.tokenizer = processor

        # Get model parts (layers, embed function)
        self._layers, self._embed_fn, self._norm_fn = self._get_model_parts()

    def _get_model_parts(self):
        """
        Get the layers, embedding function, and final norm from the model.

        Handles both VLM models (model.language_model) and text-only models.
        """
        model = self.model

        # Check for VLM structure
        if hasattr(model, "language_model"):
            text_model = model.language_model
        else:
            text_model = model

        # Get the actual model object (handles wrapper patterns)
        if hasattr(text_model, "model"):
            inner = text_model.model
        else:
            inner = text_model

        # Get layers
        if hasattr(inner, "layers"):
            layers = inner.layers
        else:
            raise ValueError(f"Cannot find layers in model structure: {type(model)}")

        # Get embedding function
        embed_fn = None
        for attr in ["embed_tokens", "tok_embeddings", "wte", "embedding"]:
            if hasattr(inner, attr):
                embed_fn = getattr(inner, attr)
                break

        if embed_fn is None:
            raise ValueError(f"Cannot find embedding function in model: {type(model)}")

        # Get final layer norm (optional, for after last layer)
        norm_fn = None
        for attr in ["norm", "ln_f", "final_layer_norm"]:
            if hasattr(inner, attr):
                norm_fn = getattr(inner, attr)
                break

        return layers, embed_fn, norm_fn

    def extract(
        self,
        texts: List[str],
        layer: int = -2,
        max_length: int = 512,
        return_attention_mask: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract hidden states from MLX models.

        Args:
            texts: List of texts to process
            layer: Which layer to extract (-2 for second-to-last)
            max_length: Maximum sequence length
            return_attention_mask: Whether to include attention mask in output

        Returns:
            List of dicts with hidden_states, shape, attention_mask, dtype per text
        """
        import mlx.core as mx

        results = []

        for text in texts:
            # Tokenize
            try:
                if callable(self.tokenizer):
                    inputs = self.tokenizer(
                        text,
                        return_tensors="np",
                        padding=False,
                        truncation=True,
                        max_length=max_length,
                    )
                else:
                    if hasattr(self.tokenizer, "encode"):
                        input_ids = self.tokenizer.encode(text)
                        if len(input_ids) > max_length:
                            input_ids = input_ids[:max_length]
                        inputs = {"input_ids": np.array([input_ids])}
                    else:
                        raise ValueError(f"Tokenizer not supported: {type(self.tokenizer)}")
            except Exception as e:
                logging.error(f"Tokenization failed: {e}")
                if hasattr(self.tokenizer, "encode"):
                    input_ids = self.tokenizer.encode(text)
                    if len(input_ids) > max_length:
                        input_ids = input_ids[:max_length]
                    inputs = {"input_ids": np.array([input_ids])}
                else:
                    raise

            # Convert to MLX
            input_ids = mx.array(inputs["input_ids"])

            # Get attention mask (all 1s since we don't pad single sequences)
            seq_len = input_ids.shape[1]
            attention_mask = [1] * seq_len

            # Extract hidden states from target layer
            hidden_states = self._extract_from_layer(input_ids, layer)

            # Evaluate to ensure computation is done
            mx.eval(hidden_states)

            # Get shape and dtype info
            shape = [hidden_states.shape[1], hidden_states.shape[2]]  # [seq_len, hidden_dim]
            dtype_str = str(hidden_states.dtype)

            # Convert to numpy for output
            # MLX bfloat16 doesn't convert directly to numpy - convert to float32 first
            hidden_np = np.array(hidden_states[0].astype(mx.float32))  # Remove batch dimension

            result = {
                "hidden_states": hidden_np,
                "shape": shape,
                "dtype": dtype_str,
            }

            if return_attention_mask:
                result["attention_mask"] = attention_mask

            results.append(result)

        return results

    def _extract_from_layer(self, input_ids, layer_idx: int):
        """
        Extract hidden states from a specific layer.

        Args:
            input_ids: MLX array of shape [batch, seq_len]
            layer_idx: Layer index (can be negative)

        Returns:
            Hidden states of shape [batch, seq_len, hidden_dim]
        """
        import mlx.core as mx

        # Get initial embeddings
        hidden = self._embed_fn(input_ids)

        # Normalize layer index
        n_layers = len(self._layers)
        if layer_idx < 0:
            target_idx = n_layers + layer_idx
        else:
            target_idx = layer_idx

        if target_idx < 0 or target_idx >= n_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range for model with {n_layers} layers"
            )

        # Pass through layers up to target
        for i, layer_module in enumerate(self._layers):
            # MLX transformer layers typically take hidden states and return hidden states
            # Some may also take/return cache, mask, etc. - try common signatures
            try:
                hidden = layer_module(hidden)
            except TypeError:
                # Try with mask argument
                try:
                    hidden = layer_module(hidden, mask=None)
                except TypeError:
                    # Try with cache argument
                    hidden, _ = layer_module(hidden, cache=None)

            if i == target_idx:
                return hidden

        # Should not reach here if layer_idx is valid
        return hidden


class LlamaCppHiddenStatesExtractor(HiddenStatesExtractor):
    """Extract hidden states from llama.cpp models."""

    def __init__(self, model):
        self.model = model

    def extract(
        self,
        texts: List[str],
        layer: int = -2,
        max_length: int = 512,
        return_attention_mask: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract hidden states from llama.cpp models.

        Note: llama-cpp-python does not expose intermediate layer hidden states.
        Only final layer embeddings are available via create_embedding().
        See: https://github.com/abetlen/llama-cpp-python/issues/1695
        """
        raise NotImplementedError(
            "Hidden state extraction from llama.cpp is not supported. "
            "The llama-cpp-python library only provides access to final layer embeddings, "
            "not intermediate layer hidden states. "
            "Please use an MLX model (e.g., Qwen3-4B-mxfp4-mlx) for this functionality. "
            "See: https://github.com/abetlen/llama-cpp-python/issues/1695"
        )


def create_hidden_states_extractor(
    provider_type: str, model: Any, processor: Any = None
) -> HiddenStatesExtractor:
    """
    Factory function to create the appropriate hidden states extractor.

    Args:
        provider_type: Type of provider ('mlx' or 'llama_cpp')
        model: The loaded model
        processor: The processor/tokenizer (required for MLX models)

    Returns:
        Appropriate HiddenStatesExtractor instance
    """
    if provider_type == "mlx":
        if processor is None:
            raise ValueError("MLX models require a processor/tokenizer")
        return MLXHiddenStatesExtractor(model, processor)
    elif provider_type in ["llama_cpp", "gguf"]:
        return LlamaCppHiddenStatesExtractor(model)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


async def create_hidden_states(
    request: HiddenStatesRequest,
    router: Any,
) -> HiddenStatesResponse:
    """
    Create hidden states for the given input text(s).

    Args:
        request: The hidden states request
        router: The model router instance

    Returns:
        HiddenStatesResponse with extracted hidden states
    """
    try:
        # Ensure input is a list
        texts = request.input if isinstance(request.input, list) else [request.input]

        # Load the model
        provider = router.get_provider(request.model)

        # Determine provider type
        provider_class_name = provider.__class__.__name__
        if "MLX" in provider_class_name:
            provider_type = "mlx"
        elif "LlamaCpp" in provider_class_name:
            provider_type = "llama_cpp"
        else:
            # Fallback: try to infer from model file extension
            model_path = provider.config.get("model_path", "")
            if ".gguf" in model_path.lower():
                provider_type = "llama_cpp"
            else:
                provider_type = "mlx"

        # Create the appropriate extractor
        if provider_type == "mlx":
            processor = getattr(provider, "processor", None)
            if processor is None:
                raise ValueError(
                    f"MLX provider for model {request.model} has no processor/tokenizer"
                )
            extractor = create_hidden_states_extractor(
                provider_type, provider.model, processor
            )
        else:
            extractor = create_hidden_states_extractor(provider_type, provider.model)

        # Extract hidden states
        results = extractor.extract(
            texts,
            layer=request.layer,
            max_length=request.max_length,
            return_attention_mask=request.return_attention_mask,
        )

        # For now, return first result (single input case)
        # TODO: Support batch responses if needed
        result = results[0]

        # Format hidden states based on encoding format
        if request.encoding_format == "base64":
            hidden_states_output = encode_hidden_states_base64(result["hidden_states"])
            encoding_format = "base64"
        else:
            hidden_states_output = result["hidden_states"].tolist()
            encoding_format = None

        return HiddenStatesResponse(
            hidden_states=hidden_states_output,
            shape=result["shape"],
            model=request.model,
            layer=request.layer,
            dtype=result["dtype"],
            encoding_format=encoding_format,
            attention_mask=result.get("attention_mask"),
        )

    except NotImplementedError as e:
        # Re-raise NotImplementedError for proper error handling in API layer
        raise
    except Exception as e:
        logging.error(f"Error extracting hidden states: {e}")
        raise
