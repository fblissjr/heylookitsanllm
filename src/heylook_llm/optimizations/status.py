# src/heylook_llm/optimizations/status.py
"""
Centralized optimization status reporting.
"""

import logging
from . import fast_json, fast_image


def log_all_optimization_status():
    """Log the status of all optimizations at server startup."""
    
    logging.info("=" * 60)
    logging.info("PERFORMANCE OPTIMIZATIONS STATUS")
    logging.info("=" * 60)
    
    # JSON optimizations
    fast_json.log_status()
    
    # Image optimizations  
    fast_image.log_status()
    
    # Get summary
    json_status = fast_json.get_status()
    image_status = fast_image.get_status()
    
    # Log summary
    active_optimizations = []
    
    if json_status["orjson_available"]:
        active_optimizations.append("orjson (JSON)")
    
    if image_status["xxhash_available"]:
        active_optimizations.append("xxHash")
        
    if image_status["turbojpeg_available"]:
        active_optimizations.append("TurboJPEG")
    
    if active_optimizations:
        logging.info(f"Active optimizations: {', '.join(active_optimizations)}")
    else:
        logging.info("No performance optimization libraries detected")
    
    logging.info("=" * 60)


def get_optimization_summary():
    """Get a summary of all optimization statuses."""
    return {
        "json": fast_json.get_status(),
        "image": fast_image.get_status()
    }