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

except ImportError:
    import json
    HAS_ORJSON = False

    dumps = json.dumps
    loads = json.loads


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


