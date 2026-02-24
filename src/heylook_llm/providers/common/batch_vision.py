# src/heylook_llm/providers/common/batch_vision.py
"""
Batch vision processing for parallel image encoding.

This module provides optimized batch processing for vision models,
enabling parallel image loading and encoding to reduce latency.
"""

import concurrent.futures
import time
from typing import List
from PIL import Image
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
    
    def __del__(self):
        """Clean up executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


