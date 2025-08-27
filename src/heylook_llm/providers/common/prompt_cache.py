# src/heylook_llm/providers/common/prompt_cache.py
"""
Cross-request prompt cache management for MLX models.

Based on mlx-lm's prompt caching implementation but adapted for 
persistent caching across requests.
"""

import logging
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass, field
import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
from .cache_helpers import make_cache


@dataclass
class PromptCache:
    """Persistent prompt cache that survives across requests."""
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)
    
    def __str__(self):
        """String representation for debugging."""
        return f"PromptCache(tokens={len(self.tokens)}, model={self.model_key[0]})"


class PromptCacheManager:
    """
    Manages prompt caches across multiple models and requests.
    
    This allows reusing cached KV states from previous requests when
    prompts share common prefixes (like system prompts or conversation history).
    """
    
    def __init__(self, max_cache_entries: int = 10):
        self._caches = {}  # model_id -> PromptCache
        self._max_entries = max_cache_entries
        self._access_order = []  # LRU tracking
        
    def get_or_create_cache(self, model_id: str, model: Any, cache_config: dict = None) -> PromptCache:
        """Get existing cache or create new one for model."""
        if model_id not in self._caches:
            # Create new cache with configuration
            cache_config = cache_config or {}
            cache = PromptCache(
                cache=make_cache(model, cache_config),
                model_key=(model_id, None),
                tokens=[]
            )
            self._caches[model_id] = cache
            self._update_lru(model_id)
            logging.debug(f"Created new prompt cache for {model_id} with config: {cache_config}")
        else:
            self._update_lru(model_id)
            logging.debug(f"Reusing existing prompt cache for {model_id} with {len(self._caches[model_id].tokens)} tokens")
            
        return self._caches[model_id]
    
    def _update_lru(self, model_id: str):
        """Update LRU order."""
        if model_id in self._access_order:
            self._access_order.remove(model_id)
        self._access_order.append(model_id)
        
        # Evict if needed
        while len(self._access_order) > self._max_entries:
            evicted = self._access_order.pop(0)
            del self._caches[evicted]
            logging.debug(f"Evicted prompt cache for {evicted} (LRU)")
    
    def invalidate_cache(self, model_id: str):
        """Invalidate cache for a specific model."""
        if model_id in self._caches:
            del self._caches[model_id]
            if model_id in self._access_order:
                self._access_order.remove(model_id)
            logging.debug(f"Invalidated prompt cache for {model_id}")
    
    def clear_all(self):
        """Clear all caches."""
        self._caches.clear()
        self._access_order.clear()
        logging.debug("Cleared all prompt caches")
        
    def get_cache_info(self) -> dict:
        """Get information about cached prompts."""
        return {
            model_id: {
                "tokens_cached": len(cache.tokens),
                "cache_layers": len(cache.cache)
            }
            for model_id, cache in self._caches.items()
        }


def find_common_prefix_length(cached_tokens: List[int], new_tokens: List[int]) -> int:
    """
    Find the length of the common prefix between cached and new tokens.
    
    Args:
        cached_tokens: Previously cached token sequence
        new_tokens: New token sequence to process
        
    Returns:
        Length of common prefix
    """
    min_len = min(len(cached_tokens), len(new_tokens))
    
    for i in range(min_len):
        if cached_tokens[i] != new_tokens[i]:
            return i
    
    return min_len


def process_prompt_with_cache(
    prompt_cache: PromptCache,
    new_tokens: List[int],
    model: Any,
    cache_config: dict = None
) -> Tuple[List[int], PromptCache]:
    """
    Process a prompt using cached KV states when possible.
    
    Args:
        prompt_cache: The current cache state
        new_tokens: New tokenized prompt
        model: The model (for cache recreation if needed)
        
    Returns:
        Tuple of (tokens_to_process, updated_cache)
    """
    
    if not prompt_cache.tokens:
        # Empty cache, process all tokens
        prompt_cache.tokens = new_tokens
        logging.debug(f"Empty cache, processing all {len(new_tokens)} tokens")
        return new_tokens, prompt_cache
    
    # Find common prefix
    common_len = find_common_prefix_length(prompt_cache.tokens, new_tokens)
    
    # Leave at least one token to process
    common_len = min(common_len, len(new_tokens) - 1)
    
    if common_len == 0:
        # No common prefix, reset cache
        logging.debug("No common prefix found, resetting cache")
        cache_config = cache_config or {}
        prompt_cache.cache = make_cache(model, cache_config)
        prompt_cache.tokens = new_tokens
        return new_tokens, prompt_cache
    
    # Check if we can reuse the cache
    cache_len = len(prompt_cache.tokens)
    
    if common_len == cache_len:
        # Cache is a prefix of the new prompt
        tokens_to_process = new_tokens[common_len:]
        prompt_cache.tokens = new_tokens
        logging.debug(f"Reusing {common_len} cached tokens, processing {len(tokens_to_process)} new tokens")
        return tokens_to_process, prompt_cache
    
    elif common_len < cache_len:
        # Need to trim the cache
        if can_trim_prompt_cache(prompt_cache.cache):
            num_to_trim = cache_len - common_len
            trimmed = trim_prompt_cache(prompt_cache.cache, num_to_trim)
            
            if trimmed == num_to_trim:
                # Successfully trimmed
                prompt_cache.tokens = new_tokens
                tokens_to_process = new_tokens[common_len:]
                logging.debug(f"Trimmed {num_to_trim} tokens from cache, reusing {common_len}, processing {len(tokens_to_process)}")
                return tokens_to_process, prompt_cache
        
        # Can't trim or trim failed, reset cache
        logging.debug(f"Cannot trim cache, resetting")
        cache_config = cache_config or {}
        prompt_cache.cache = make_cache(model, cache_config)
        prompt_cache.tokens = new_tokens
        return new_tokens, prompt_cache
    
    # Should not reach here
    logging.warning("Unexpected cache state, resetting")
    cache_config = cache_config or {}
    prompt_cache.cache = make_cache(model, cache_config)
    prompt_cache.tokens = new_tokens
    return new_tokens, prompt_cache


# Global cache manager instance
_global_cache_manager = PromptCacheManager()

def get_global_cache_manager() -> PromptCacheManager:
    """Get the global prompt cache manager instance."""
    return _global_cache_manager