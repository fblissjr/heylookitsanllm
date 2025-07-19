# src/heylook_llm/optimizations/mlx_optimizations.py
"""
MLX-specific optimizations for faster LLM inference.

Based on MLX best practices and performance features.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List
import logging


class MLXOptimizations:
    """MLX-specific performance optimizations."""
    
    @staticmethod
    def optimize_memory_usage():
        """Optimize MLX memory management."""
        # Set memory growth strategy
        mx.set_memory_limit_policy("grow_only")
        
        # Enable unified memory (shares between CPU/GPU efficiently)
        mx.set_default_device(mx.Device.gpu)
        
        # Clear caches before major operations
        mx.clear_cache()
    
    @staticmethod
    def enable_fast_math():
        """Enable fast math operations (slight accuracy tradeoff for speed)."""
        # MLX uses fast math by default on Metal
        # But we can ensure optimal settings
        mx.set_default_dtype(mx.float16)  # Use FP16 when possible
    
    @staticmethod
    def optimize_generation_loop(model, max_tokens: int):
        """Optimize the generation loop with MLX-specific features."""
        
        # Pre-allocate KV cache tensors to avoid reallocation
        if hasattr(model, 'allocate_kv_cache'):
            model.allocate_kv_cache(max_tokens)
        
        # Compile the model's forward pass if possible
        # MLX JIT compilation for repeated operations
        if hasattr(mx, 'compile'):
            model.forward = mx.compile(model.forward)
    
    @staticmethod
    def batch_optimize_attention(attention_scores: mx.array, batch_size: int) -> mx.array:
        """Optimize batched attention computation."""
        # Use MLX's optimized operations
        # Fuse operations when possible
        return mx.fast.softmax(attention_scores, axis=-1)
    
    @staticmethod
    def optimize_model_loading(model_path: str):
        """Optimize model loading with MLX features."""
        # Use memory mapping for large models
        # This allows the OS to manage memory more efficiently
        weights = mx.load(model_path, mmap=True)
        
        # Lazy loading - weights are only loaded when accessed
        return weights
    
    @staticmethod
    def enable_graph_optimization():
        """Enable MLX graph optimizations."""
        # MLX automatically optimizes computation graphs
        # But we can hint at optimization opportunities
        mx.set_default_stream(mx.gpu)  # Ensure GPU stream
    
    @staticmethod
    def optimize_kv_cache(cache_size: int, n_heads: int, head_dim: int):
        """Pre-allocate and optimize KV cache."""
        # Pre-allocate cache to avoid dynamic allocation
        k_cache = mx.zeros((cache_size, n_heads, head_dim), dtype=mx.float16)
        v_cache = mx.zeros((cache_size, n_heads, head_dim), dtype=mx.float16)
        
        # Keep on GPU to avoid transfers
        k_cache = mx.array(k_cache, mx.gpu)
        v_cache = mx.array(v_cache, mx.gpu)
        
        return k_cache, v_cache


class MLXModelOptimizer:
    """Optimize MLX models for inference."""
    
    def __init__(self, model):
        self.model = model
    
    def quantize_model(self, bits: int = 4):
        """Quantize model to reduce memory and increase speed."""
        # MLX supports various quantization schemes
        if hasattr(self.model, 'quantize'):
            self.model.quantize(bits=bits, group_size=64)
            logging.info(f"Model quantized to {bits} bits")
    
    def enable_flash_attention(self):
        """Enable Flash Attention if available."""
        # Check if model supports flash attention
        if hasattr(self.model, 'enable_flash_attention'):
            self.model.enable_flash_attention()
            logging.info("Flash Attention enabled")
    
    def optimize_for_generation(self, max_length: int = 2048):
        """Optimize model specifically for text generation."""
        # Pre-compute positional encodings
        if hasattr(self.model, 'precompute_positions'):
            self.model.precompute_positions(max_length)
        
        # Enable KV cache
        if hasattr(self.model, 'enable_kv_cache'):
            self.model.enable_kv_cache()
        
        # Set inference mode
        self.model.eval()
    
    def fuse_operations(self):
        """Fuse operations for better performance."""
        # MLX can fuse certain operations
        if hasattr(self.model, 'fuse'):
            self.model.fuse()
            logging.info("Model operations fused")


class MLXBatchedGeneration:
    """Optimized batched generation for MLX."""
    
    @staticmethod
    def prepare_batch_kv_cache(batch_size: int, max_length: int, n_layers: int, 
                              n_heads: int, head_dim: int) -> Tuple[List, List]:
        """Pre-allocate KV cache for batch generation."""
        k_caches = []
        v_caches = []
        
        for _ in range(n_layers):
            # Pre-allocate on GPU
            k = mx.zeros((batch_size, max_length, n_heads, head_dim), 
                        dtype=mx.float16, device=mx.gpu)
            v = mx.zeros((batch_size, max_length, n_heads, head_dim), 
                        dtype=mx.float16, device=mx.gpu)
            k_caches.append(k)
            v_caches.append(v)
        
        return k_caches, v_caches
    
    @staticmethod
    def optimized_sampling(logits: mx.array, temperature: float = 1.0, 
                          top_k: int = 50) -> mx.array:
        """Optimized sampling with MLX operations."""
        # Scale by temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k filtering with MLX optimized ops
        if top_k > 0:
            # Get top-k indices and values efficiently
            top_k_values, top_k_indices = mx.topk(logits, k=top_k, axis=-1)
            
            # Create sparse representation
            sparse_logits = mx.full_like(logits, float('-inf'))
            sparse_logits = mx.scatter(sparse_logits, top_k_indices, top_k_values, axis=-1)
            logits = sparse_logits
        
        # Softmax with fast implementation
        probs = mx.fast.softmax(logits, axis=-1)
        
        # Sample
        return mx.random.categorical(probs)


class MLXVisionOptimizations:
    """MLX optimizations specific to vision models."""
    
    @staticmethod
    def optimize_image_preprocessing(batch_size: int = 4):
        """Optimize image preprocessing pipeline."""
        # Pre-allocate buffers for image batches
        # Standard size for many vision models
        buffer = mx.zeros((batch_size, 3, 336, 336), dtype=mx.float16)
        return buffer
    
    @staticmethod
    def parallel_encode_images(images: List[mx.array], vision_encoder) -> mx.array:
        """Encode multiple images in parallel."""
        # Stack images for batch processing
        if len(images) == 1:
            return vision_encoder(images[0])
        
        # Batch encode
        stacked = mx.stack(images)
        return vision_encoder(stacked)
    
    @staticmethod
    def optimize_vision_attention(image_features: mx.array, 
                                 text_features: mx.array) -> mx.array:
        """Optimize cross-attention between vision and text."""
        # Use MLX's optimized matmul
        attention = mx.fast.matmul(text_features, image_features.transpose(0, 2, 1))
        
        # Scaled dot-product attention
        scale = 1.0 / (text_features.shape[-1] ** 0.5)
        attention = attention * scale
        
        # Fast softmax
        attention = mx.fast.softmax(attention, axis=-1)
        
        return attention


def apply_mlx_optimizations(model, config: dict):
    """Apply all MLX optimizations to a model."""
    
    # Memory optimizations
    MLXOptimizations.optimize_memory_usage()
    
    # Enable fast math
    MLXOptimizations.enable_fast_math()
    
    # Model-specific optimizations
    optimizer = MLXModelOptimizer(model)
    
    # Quantization (if not already quantized)
    if config.get('quantize', False):
        optimizer.quantize_model(bits=config.get('quantize_bits', 4))
    
    # Flash attention
    optimizer.enable_flash_attention()
    
    # Generation optimizations
    max_length = config.get('max_length', 2048)
    optimizer.optimize_for_generation(max_length)
    
    # Fuse operations
    optimizer.fuse_operations()
    
    # Graph optimizations
    MLXOptimizations.enable_graph_optimization()
    
    logging.info("MLX optimizations applied")
    
    return model


# Specific optimizations for the models you're using
def optimize_qwen_vlm(model, processor):
    """Specific optimizations for Qwen VLM models."""
    
    # Qwen uses specific attention patterns
    if hasattr(model, 'config'):
        # Enable sliding window attention if available
        if hasattr(model.config, 'use_sliding_window'):
            model.config.use_sliding_window = True
        
        # Optimize RoPE (Rotary Position Embeddings)
        if hasattr(model, 'rotary_embedding'):
            # Pre-compute RoPE for common sequence lengths
            seq_lengths = [512, 1024, 2048, 4096]
            for seq_len in seq_lengths:
                model.rotary_embedding.precompute(seq_len)
    
    # Vision-specific optimizations
    if hasattr(model, 'vision_encoder'):
        # Enable NHWC format for better Metal performance
        model.vision_encoder.use_nhwc_layout = True
    
    return model


def optimize_gemma_vlm(model, processor):
    """Specific optimizations for Gemma VLM models."""
    
    # Gemma uses GQA (Grouped Query Attention)
    if hasattr(model, 'enable_gqa_optimization'):
        model.enable_gqa_optimization()
    
    # Optimize for Metal backend
    if hasattr(model, 'optimize_for_metal'):
        model.optimize_for_metal()
    
    return model