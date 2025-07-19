# src/heylook_llm/providers/common/batch_vision.py
"""
Batch vision processing for parallel image encoding.

This module provides optimized batch processing for vision models,
enabling parallel image loading and encoding to reduce latency.
"""

import asyncio
import concurrent.futures
import time
from typing import List, Union, Tuple, Any
from PIL import Image
import mlx.core as mx
import logging

from ...utils import load_image


class BatchVisionProcessor:
    """Handles parallel image processing for vision models."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch vision processor.
        
        Args:
            max_workers: Maximum number of parallel workers for image loading
        """
        self.max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def load_images_parallel(self, image_urls: List[str]) -> List[Image.Image]:
        """
        Load multiple images in parallel.
        
        Args:
            image_urls: List of image URLs/paths/base64 strings
            
        Returns:
            List of PIL Image objects
        """
        if not image_urls:
            return []
        
        start_time = time.time()
        logging.info(f"[BATCH VISION] Starting parallel load of {len(image_urls)} images")
        
        if len(image_urls) == 1:
            # Single image, no need for parallelization
            return [load_image(image_urls[0])]
        
        # Load images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(load_image, url) for url in image_urls]
            images = []
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    image = future.result()
                    images.append((i, image))
                except Exception as e:
                    logging.error(f"Failed to load image {i}: {e}")
                    # Create placeholder for failed image
                    images.append((i, Image.new('RGB', (64, 64), color='red')))
            
            # Sort by original index to maintain order
            images.sort(key=lambda x: x[0])
            sorted_images = [img for _, img in images]
            
            load_time = time.time() - start_time
            # Calculate total size info
            total_pixels = sum(img.width * img.height for img in sorted_images)
            sizes_summary = ", ".join([f"{img.size}" for img in sorted_images[:3]])
            if len(sorted_images) > 3:
                sizes_summary += f", ... ({len(sorted_images)} total)"
            
            logging.info(f"[BATCH VISION] Loaded {len(sorted_images)} images in {load_time*1000:.1f}ms | "
                       f"Sizes: [{sizes_summary}] | Total pixels: {total_pixels:,}")
            
            return sorted_images
    
    async def load_images_async(self, image_urls: List[str]) -> List[Image.Image]:
        """
        Load multiple images asynchronously.
        
        Args:
            image_urls: List of image URLs/paths/base64 strings
            
        Returns:
            List of PIL Image objects
        """
        loop = asyncio.get_event_loop()
        
        if not image_urls:
            return []
        
        if len(image_urls) == 1:
            # Single image
            return [await loop.run_in_executor(self._executor, load_image, image_urls[0])]
        
        # Create tasks for parallel loading
        tasks = [
            loop.run_in_executor(self._executor, load_image, url)
            for url in image_urls
        ]
        
        # Wait for all images to load
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        images = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Failed to load image {i}: {result}")
                images.append(Image.new('RGB', (64, 64), color='red'))
            else:
                images.append(result)
        
        return images
    
    def preprocess_images_batch(self, images: List[Image.Image], processor: Any) -> mx.array:
        """
        Preprocess multiple images in batch.
        
        Args:
            images: List of PIL images
            processor: Vision processor with preprocess method
            
        Returns:
            Batched preprocessed images as MLX array
        """
        if not images:
            return mx.array([])
        
        start_time = time.time()
        logging.info(f"[BATCH PREPROCESS] Starting preprocessing of {len(images)} images")
        
        # Process each image
        processed_images = []
        resize_count = 0
        
        for i, img in enumerate(images):
            original_size = img.size
            # Use processor's preprocess method
            if hasattr(processor, 'preprocess'):
                processed = processor.preprocess(img)
                # Check if processor resized (we can't easily track this without modifying processor)
                logging.debug(f"[BATCH PREPROCESS] Image {i+1}/{len(images)} preprocessed using model processor")
            else:
                # Fallback to basic preprocessing
                processed = self._basic_preprocess(img)
                if original_size != (336, 336):  # Default size
                    resize_count += 1
            processed_images.append(processed)
        
        # Stack into batch
        batch_array = mx.stack(processed_images)
        preprocess_time = time.time() - start_time
        
        logging.info(f"[BATCH PREPROCESS] Preprocessed {len(images)} images in {preprocess_time*1000:.1f}ms | "
                   f"Batch shape: {batch_array.shape} | Resized: {resize_count}/{len(images)}")
        
        return batch_array
    
    def _basic_preprocess(self, image: Image.Image, size: int = 336) -> mx.array:
        """Basic image preprocessing fallback."""
        original_size = image.size
        # Resize to square
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        
        if original_size != (size, size):
            logging.info(f"[IMAGE RESIZE] Resized image from {original_size} to {size}x{size} for model processing")
        
        # Convert to array and normalize
        import numpy as np
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to MLX array
        return mx.array(img_array)
    
    def encode_images_batch(self, images: List[Image.Image], model: Any, processor: Any) -> mx.array:
        """
        Encode multiple images in batch using the vision model.
        
        Args:
            images: List of PIL images
            model: Vision model with encode_image method
            processor: Vision processor
            
        Returns:
            Batch of encoded image features
        """
        if not images:
            return mx.array([])
        
        # Preprocess images
        batch_images = self.preprocess_images_batch(images, processor)
        
        # Encode in batch if model supports it
        if hasattr(model, 'encode_images_batch'):
            # Model supports native batch encoding
            return model.encode_images_batch(batch_images)
        elif hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'forward'):
            # Use vision encoder directly for batch
            return model.vision_encoder(batch_images)
        else:
            # Fallback to sequential encoding
            encoded = []
            for i in range(batch_images.shape[0]):
                if hasattr(model, 'encode_image'):
                    enc = model.encode_image(batch_images[i:i+1])
                else:
                    # Try vision encoder
                    enc = model.vision_encoder(batch_images[i:i+1])
                encoded.append(enc)
            
            return mx.concatenate(encoded, axis=0)
    
    def __del__(self):
        """Clean up executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


def create_batch_vision_processor(max_workers: int = 4) -> BatchVisionProcessor:
    """
    Factory function to create a batch vision processor.
    
    Args:
        max_workers: Maximum number of parallel workers
        
    Returns:
        BatchVisionProcessor instance
    """
    return BatchVisionProcessor(max_workers=max_workers)