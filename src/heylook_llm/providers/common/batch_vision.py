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
        
        # Submission order IS marker order: the caller lines this list up
        # with the image markers the template renders, so results are
        # collected in the order they were submitted, never as they finish.
        # (Until 2026-09-24 this enumerated as_completed(), so the "sort by
        # original index" sorted by completion order and a multi-image
        # request could hand the model its images shuffled.)
        futures = [self._executor.submit(load_image, url) for url in image_urls]
        images = [f.result() for f in futures]

        load_time = time.time() - start_time
        total_pixels = sum(img.width * img.height for img in images)
        sizes_summary = ", ".join([f"{img.size}" for img in images[:3]])
        if len(images) > 3:
            sizes_summary += f", ... ({len(images)} total)"
        logging.info(f"[BATCH VISION] Loaded {len(images)} images in {load_time*1000:.1f}ms | "
                     f"Sizes: [{sizes_summary}] | Total pixels: {total_pixels:,}")
        return images
    
    def __del__(self):
        """Clean up executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


