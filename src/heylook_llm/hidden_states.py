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
from typing import Any, Dict, List, Optional, Tuple, Union

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


# --- Structured Hidden States Models ---


class StructuredHiddenStatesRequest(BaseModel):
    """Request for structured hidden states with server-side chat template application."""

    model: str = Field(..., description="Model ID to use for extraction")
    user_prompt: str = Field(..., description="User message content")
    system_prompt: Optional[str] = Field(None, description="System prompt content")
    thinking_content: Optional[str] = Field(
        None, description="Pre-filled thinking block content"
    )
    assistant_content: Optional[str] = Field(
        None, description="Pre-filled assistant response content"
    )
    enable_thinking: bool = Field(
        True, description="Control chat template thinking mode (Qwen3)"
    )
    layer: int = Field(
        -2, description="Which layer to extract from (-2 for second-to-last)"
    )
    max_length: int = Field(512, description="Maximum sequence length")
    encoding_format: str = Field(
        "float", description="Output format: 'float' (nested list) or 'base64'"
    )
    return_token_boundaries: bool = Field(
        False, description="Return token indices for each section"
    )
    return_formatted_prompt: bool = Field(
        False, description="Return the formatted prompt string (for debugging)"
    )


class TokenBoundary(BaseModel):
    """Token boundary information for a prompt section."""

    start: int = Field(..., description="Start token index (inclusive)")
    end: int = Field(..., description="End token index (exclusive)")


class StructuredHiddenStatesResponse(BaseModel):
    """Response for structured hidden states extraction with token boundaries."""

    hidden_states: Union[List[List[float]], str] = Field(
        ..., description="Hidden states as nested list or base64 string"
    )
    shape: List[int] = Field(
        ..., description="Shape of hidden states [seq_len, hidden_dim]"
    )
    model: str = Field(..., description="Model ID used")
    layer: int = Field(..., description="Layer extracted from")
    dtype: str = Field(..., description="Data type of the hidden states")
    encoding_format: Optional[str] = Field(
        None, description="Encoding format used (only present for base64)"
    )

    # Token boundary fields
    token_boundaries: Optional[Dict[str, TokenBoundary]] = Field(
        None,
        description="Token boundaries for each section: system, user, think, assistant",
    )
    token_counts: Optional[Dict[str, int]] = Field(
        None, description="Token count for each section"
    )
    formatted_prompt: Optional[str] = Field(
        None, description="The formatted prompt string (for debugging)"
    )


# --- Chat Template Tokenizer for Boundary Tracking ---


class ChatTemplateTokenizer:
    """
    Handles chat template application with token boundary tracking.

    Builds prompts incrementally and tracks where each section (system, user,
    think, assistant) starts and ends in the token sequence. This enables
    research use cases like ablation studies on prompt sections.
    """

    # Qwen3 special token IDs
    THINK_START_TOKEN = 151667  # <think>
    THINK_END_TOKEN = 151668  # </think>
    IM_START_TOKEN = 151644  # <|im_start|>
    IM_END_TOKEN = 151645  # <|im_end|>

    def __init__(self, tokenizer, config: Optional[Dict] = None):
        """
        Initialize with a tokenizer.

        Args:
            tokenizer: HuggingFace-style tokenizer with encode() and
                       apply_chat_template() methods
            config: Optional model config dict
        """
        self.tokenizer = tokenizer
        self.config = config or {}

    def build_prompt_with_boundaries(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        thinking_content: Optional[str] = None,
        assistant_content: Optional[str] = None,
        enable_thinking: bool = True,
    ) -> Tuple[str, List[int], Dict[str, Dict[str, int]]]:
        """
        Build chat prompt and track token boundaries for each section.

        Args:
            user_prompt: Required user message
            system_prompt: Optional system message
            thinking_content: Optional pre-filled thinking block
            assistant_content: Optional pre-filled assistant response
            enable_thinking: Whether to enable thinking mode in template

        Returns:
            Tuple of:
                - formatted_prompt: The complete chat template string
                - token_ids: List of token IDs for the prompt
                - token_boundaries: Dict mapping section names to {"start": int, "end": int}
        """
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Build assistant message if we have pre-filled content
        if thinking_content or assistant_content:
            assistant_msg = ""
            if thinking_content:
                assistant_msg += f"<think>\n{thinking_content}\n</think>\n"
            if assistant_content:
                assistant_msg += assistant_content
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        # Apply chat template
        formatted_prompt = self._apply_chat_template(messages, enable_thinking)

        # Tokenize full prompt
        full_tokens = self._encode(formatted_prompt)

        # Compute token boundaries
        boundaries = self._compute_boundaries(
            formatted_prompt,
            full_tokens,
            system_prompt,
            user_prompt,
            thinking_content,
            assistant_content,
        )

        return formatted_prompt, full_tokens, boundaries

    def _apply_chat_template(
        self, messages: List[Dict[str, str]], enable_thinking: bool
    ) -> str:
        """Apply chat template with enable_thinking support."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking parameter
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(text)
        elif callable(self.tokenizer):
            result = self.tokenizer(text, return_tensors=None, add_special_tokens=False)
            return result["input_ids"]
        else:
            raise ValueError(f"Tokenizer {type(self.tokenizer)} not supported")

    def _compute_boundaries(
        self,
        formatted_prompt: str,
        full_tokens: List[int],
        system_prompt: Optional[str],
        user_prompt: str,
        thinking_content: Optional[str],
        assistant_content: Optional[str],
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute token boundaries by incremental tokenization.

        Strategy: Tokenize progressively longer prefixes of the prompt
        and use the token counts to determine section boundaries.
        """
        boundaries = {}
        total_tokens = len(full_tokens)

        # Build sections incrementally to find boundaries
        # Qwen3 format: <|im_start|>role\ncontent<|im_end|>\n
        current_pos = 0

        if system_prompt:
            # System section: <|im_start|>system\n{content}<|im_end|>\n
            system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            system_tokens = self._encode(system_text)
            boundaries["system"] = {"start": current_pos, "end": len(system_tokens)}
            current_pos = len(system_tokens)

        # User section
        user_start = current_pos
        if system_prompt:
            # Tokenize system + user together for accurate boundary
            prefix_with_user = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            )
            prefix_tokens = self._encode(prefix_with_user)
            user_end = len(prefix_tokens)
        else:
            user_text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            user_tokens = self._encode(user_text)
            user_end = len(user_tokens)

        boundaries["user"] = {"start": user_start, "end": user_end}
        current_pos = user_end

        # Assistant section with thinking
        if thinking_content or assistant_content:
            # Find think section boundaries using special token IDs
            if thinking_content:
                think_start = current_pos
                think_end = total_tokens  # Default to end

                # Search for </think> token (ID 151668) in the token sequence
                for i, tok_id in enumerate(full_tokens[current_pos:], start=current_pos):
                    if tok_id == self.THINK_END_TOKEN:
                        think_end = i + 1  # Include the </think> token
                        break

                boundaries["think"] = {"start": think_start, "end": think_end}
                current_pos = think_end

            if assistant_content:
                boundaries["assistant"] = {"start": current_pos, "end": total_tokens}

        return boundaries


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

        # Create attention mask - this is CRITICAL for correct hidden state values!
        # Without a proper causal mask, attention computes incorrectly and values
        # can be ~30x smaller than expected.
        try:
            from mlx_lm.models.base import create_attention_mask
            mask = create_attention_mask(hidden, cache=None)
        except ImportError:
            # Fallback: create a simple causal mask manually
            seq_len = hidden.shape[1]
            # Create causal mask: lower triangular with -inf for masked positions
            mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
            mask = mask.reshape(1, 1, seq_len, seq_len)

        # Pass through layers up to target
        for i, layer_module in enumerate(self._layers):
            # MLX transformer layers take (hidden, mask, cache)
            # We must pass the attention mask for correct computation
            try:
                hidden = layer_module(hidden, mask=mask, cache=None)
            except TypeError:
                # Fallback for layers with different signatures
                try:
                    hidden = layer_module(hidden, mask=mask)
                except TypeError:
                    try:
                        hidden = layer_module(hidden)
                    except TypeError:
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

        # Apply model config defaults for hidden states parameters
        layer = request.layer
        max_length = request.max_length
        if hasattr(provider, 'config') and isinstance(provider.config, dict):
            # Use model config defaults if request uses default values
            if layer == -2:  # Default value in request
                layer = provider.config.get('default_hidden_layer', -2)
            if max_length == 512:  # Default value in request
                max_length = provider.config.get('default_max_length', 512)

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

        # Extract hidden states (using possibly model-config-overridden values)
        results = extractor.extract(
            texts,
            layer=layer,
            max_length=max_length,
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
            layer=layer,  # Use actual layer (may be from model config)
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


async def create_structured_hidden_states(
    request: StructuredHiddenStatesRequest,
    router: Any,
) -> StructuredHiddenStatesResponse:
    """
    Create structured hidden states with server-side chat template application.

    This function:
    1. Accepts chat components separately (user_prompt, system_prompt, etc.)
    2. Applies the chat template server-side with token boundary tracking
    3. Extracts hidden states from the specified layer
    4. Returns token boundaries for each section (system, user, think, assistant)

    Args:
        request: The structured hidden states request
        router: The model router instance

    Returns:
        StructuredHiddenStatesResponse with hidden states and token boundaries
    """
    try:
        # Load the model
        provider = router.get_provider(request.model)

        # Only MLX models are supported for structured hidden states
        provider_class_name = provider.__class__.__name__
        if "MLX" not in provider_class_name:
            raise NotImplementedError(
                "Structured hidden states extraction only supported for MLX models. "
                f"Provider {provider_class_name} is not supported."
            )

        # Get processor/tokenizer
        processor = getattr(provider, "processor", None)
        if processor is None:
            raise ValueError(
                f"MLX provider for model {request.model} has no processor/tokenizer"
            )

        # Get the tokenizer from processor
        if hasattr(processor, "_tokenizer"):
            tokenizer = processor._tokenizer
        elif hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor

        # Apply model config defaults
        layer = request.layer
        max_length = request.max_length
        if hasattr(provider, "config") and isinstance(provider.config, dict):
            if layer == -2:
                layer = provider.config.get("default_hidden_layer", -2)
            if max_length == 512:
                max_length = provider.config.get("default_max_length", 512)

        # Build prompt with boundary tracking
        template_tokenizer = ChatTemplateTokenizer(tokenizer, provider.config)
        formatted_prompt, token_ids, token_boundaries = (
            template_tokenizer.build_prompt_with_boundaries(
                user_prompt=request.user_prompt,
                system_prompt=request.system_prompt,
                thinking_content=request.thinking_content,
                assistant_content=request.assistant_content,
                enable_thinking=request.enable_thinking,
            )
        )

        # Extract hidden states using existing extractor
        extractor = MLXHiddenStatesExtractor(provider.model, processor)
        results = extractor.extract(
            [formatted_prompt],
            layer=layer,
            max_length=max_length,
        )
        result = results[0]

        # Format output
        if request.encoding_format == "base64":
            hidden_states_output = encode_hidden_states_base64(result["hidden_states"])
            encoding_format = "base64"
        else:
            hidden_states_output = result["hidden_states"].tolist()
            encoding_format = None

        # Build response
        response_data = {
            "hidden_states": hidden_states_output,
            "shape": result["shape"],
            "model": request.model,
            "layer": layer,
            "dtype": result["dtype"],
            "encoding_format": encoding_format,
        }

        # Add optional fields
        if request.return_token_boundaries:
            # Convert dict boundaries to TokenBoundary objects
            response_data["token_boundaries"] = {
                section: TokenBoundary(**bounds)
                for section, bounds in token_boundaries.items()
            }
            response_data["token_counts"] = {
                section: bounds["end"] - bounds["start"]
                for section, bounds in token_boundaries.items()
            }
            response_data["token_counts"]["total"] = result["shape"][0]

        if request.return_formatted_prompt:
            response_data["formatted_prompt"] = formatted_prompt

        return StructuredHiddenStatesResponse(**response_data)

    except NotImplementedError:
        raise
    except Exception as e:
        logging.error(f"Error extracting structured hidden states: {e}")
        raise
