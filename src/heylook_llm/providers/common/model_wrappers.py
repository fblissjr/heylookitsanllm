# src/heylook_llm/providers/common/model_wrappers.py
"""
Wrapper modules for MLX model interop.

LanguageModelLogitsWrapper enables VLM language models to work with mlx-lm's
text generation pipeline by extracting logits from the VLM output format.

The wrapper is a transparent nn.Module that must not interfere with:
- mx.fast.scaled_dot_product_attention
- mx.fast.rms_norm / mx.fast.layer_norm
- mx.fast.rope
- nn.QuantizedLinear
- Future mx.compile wrapping
"""

import logging

import mlx.nn as nn


class LanguageModelLogitsWrapper(nn.Module):
    """
    Wrapper for VLM language models to extract logits directly.

    Why this exists:
    - Provides direct logits extraction for mlx-lm compatibility
    - Caches frequently accessed attributes
    - Enables VLM models to work with mlx-lm's text generation pipeline
    """

    def __init__(self, language_model):
        super().__init__()
        # Use object.__setattr__ to avoid triggering __getattr__ during initialization
        object.__setattr__(self, 'language_model', language_model)
        object.__setattr__(self, '_cached_layers', None)
        object.__setattr__(self, '_cached_config', None)
        object.__setattr__(self, '_cached_head_dim', None)
        object.__setattr__(self, '_cache_populated', False)

    def _populate_cache(self):
        """Populate attribute cache on first access."""
        if not self._cache_populated:
            # Cache layers - check both common attribute locations
            if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'layers'):
                self._cached_layers = self.language_model.model.layers
            elif hasattr(self.language_model, 'layers'):
                self._cached_layers = self.language_model.layers

            # Cache config
            if hasattr(self.language_model, 'config'):
                self._cached_config = self.language_model.config
            elif hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'config'):
                self._cached_config = self.language_model.model.config

            # Cache head dimension if available
            if self._cached_config and hasattr(self._cached_config, 'head_dim'):
                self._cached_head_dim = self._cached_config.head_dim
            elif self._cached_config and hasattr(self._cached_config, 'hidden_size'):
                self._cached_head_dim = self._cached_config.hidden_size

            self._cache_populated = True

    def __call__(self, *args, **kwargs):
        """Direct logits extraction - the core optimization."""
        # Direct logits extraction avoids creating intermediate objects
        try:
            result = self.language_model(*args, **kwargs)
            # Check if result has logits attribute
            if hasattr(result, 'logits'):
                return result.logits
            # If result is already logits (tensor), return it
            elif hasattr(result, 'shape'):
                return result
            # Otherwise try to extract logits
            else:
                # Log for debugging
                logging.debug(f"LanguageModelLogitsWrapper: Unexpected result type: {type(result)}")
                return result
        except Exception as e:
            logging.error(f"LanguageModelLogitsWrapper error: {e}")
            raise

    @property
    def layers(self):
        """Cached layers property."""
        if self._cached_layers is None:
            self._populate_cache()
        return self._cached_layers

    @property
    def config(self):
        """Cached config property."""
        if self._cached_config is None:
            self._populate_cache()
        return self._cached_config

    @property
    def head_dim(self):
        """Cached head dimension property."""
        if self._cached_head_dim is None:
            self._populate_cache()
        return self._cached_head_dim

    def __getattr__(self, name):
        """Fast forwarding for any other attributes."""
        # Only forward if language_model exists to avoid recursion
        if 'language_model' in self.__dict__:
            return getattr(self.language_model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
