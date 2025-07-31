# Fast performance optimizations module

from . import fast_json
from . import fast_image
from . import status

# Conditionally import MLX-dependent modules
try:
    from . import mlx_metal_tuning
    __all__ = ['fast_json', 'fast_image', 'mlx_metal_tuning', 'status']
except ImportError:
    __all__ = ['fast_json', 'fast_image', 'status']