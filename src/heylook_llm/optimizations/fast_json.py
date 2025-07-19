# src/heylook_llm/optimizations/fast_json.py
"""
Fast JSON processing with orjson.

Provides drop-in replacement for json module with 3-10x speedups.
"""

import logging
from typing import Any, Union

# Try to import orjson
try:
    import orjson
    HAS_ORJSON = True
    
    def dumps(obj: Any, **kwargs) -> str:
        """Fast JSON encoding with orjson."""
        # orjson returns bytes, convert to str
        options = 0
        if kwargs.get('indent'):
            options |= orjson.OPT_INDENT_2
        if kwargs.get('sort_keys'):
            options |= orjson.OPT_SORT_KEYS
        
        return orjson.dumps(obj, option=options).decode('utf-8')
    
    def loads(s: Union[str, bytes], **kwargs) -> Any:
        """Fast JSON decoding with orjson."""
        if isinstance(s, str):
            s = s.encode('utf-8')
        return orjson.loads(s)
    
    def dump(obj: Any, fp, **kwargs):
        """Fast JSON encoding to file with orjson."""
        data = dumps(obj, **kwargs)
        fp.write(data)
    
    def load(fp, **kwargs):
        """Fast JSON decoding from file with orjson."""
        return loads(fp.read(), **kwargs)
    
    # Don't log at import time - will be logged when checked
    pass
    
except ImportError:
    import json
    HAS_ORJSON = False
    
    # Fallback to standard json
    dumps = json.dumps
    loads = json.loads
    dump = json.dump
    load = json.load


def get_status():
    """Get the status of JSON optimization."""
    return {
        "orjson_available": HAS_ORJSON,
        "speedup": "3-10x" if HAS_ORJSON else "1x"
    }


def log_status():
    """Log the JSON optimization status."""
    if HAS_ORJSON:
        logging.info("orjson available - using fast JSON operations (3-10x speedup)")
    else:
        logging.info("orjson not available - using standard json library")


def install_fast_json():
    """Replace json module with fast implementation globally."""
    if not HAS_ORJSON:
        logging.warning("Cannot install fast JSON - orjson not available")
        return
    
    import sys
    import json as json_module
    
    # Replace json module functions
    json_module.dumps = dumps
    json_module.loads = loads
    json_module.dump = dump
    json_module.load = load
    
    # Also update JSONEncoder/Decoder for compatibility
    class FastJSONEncoder(json_module.JSONEncoder):
        def encode(self, o):
            return dumps(o)
        
        def iterencode(self, o, _one_shot=False):
            yield dumps(o)
    
    json_module.JSONEncoder = FastJSONEncoder
    
    logging.info("Installed fast JSON implementation globally")


# Benchmark function for testing
def benchmark_json(data: Any, iterations: int = 1000):
    """Benchmark JSON performance."""
    import time
    import json as std_json
    
    # Standard json
    start = time.time()
    for _ in range(iterations):
        s = std_json.dumps(data)
        std_json.loads(s)
    std_time = time.time() - start
    
    # Fast json
    start = time.time()
    for _ in range(iterations):
        s = dumps(data)
        loads(s)
    fast_time = time.time() - start
    
    speedup = std_time / fast_time if fast_time > 0 else 0
    
    return {
        "standard_json_time": std_time,
        "fast_json_time": fast_time,
        "speedup": speedup,
        "using_orjson": HAS_ORJSON
    }