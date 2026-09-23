# src/heylook_llm/monitoring_api.py
"""Monitoring and discovery routes: system metrics, the performance profile,
server capabilities, and the prompt-cache clear. Split out of api.py in
v1.79.67; the resource-snapshot loop in api.py shares the metrics collector
through get_metrics_collector()."""
import threading

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from heylook_llm.auth import require_admin_token
from heylook_llm.config import (
    CacheClearRequest,
    CacheClearResponse,
    SystemMetricsResponse,
)
from heylook_llm.perf_collector import get_perf_collector
from heylook_llm.router import ModelRouter
from heylook_llm.system_metrics import SystemMetricsCollector

monitoring_router = APIRouter(tags=["Monitoring"])


# The system metrics collector: created on first use, shared by the metrics
# route and api.py's resource-snapshot loop (double-checked locking).
_metrics_collector: SystemMetricsCollector | None = None
_metrics_collector_lock = threading.Lock()


def get_metrics_collector(router: ModelRouter) -> SystemMetricsCollector:
    """Get or create the system metrics collector (thread-safe)."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_collector_lock:
            # Double-check after acquiring lock
            if _metrics_collector is None:
                _metrics_collector = SystemMetricsCollector(router, cache_ttl_seconds=30.0)
    return _metrics_collector


@monitoring_router.get("/v1/system/metrics",
    summary="Get System Metrics",
    description="""
Get current system resource and model metrics for monitoring dashboards.

**Returns:**
- System metrics: RAM usage, CPU percentage
- Per-model metrics: Context usage, memory, active requests
- Cached for 30 seconds to minimize polling overhead

**Use Cases:**
- Build monitoring dashboards
- Track context window usage during conversations
- Monitor system resource consumption
- Alert on high memory/context usage

**Polling:**
- Recommended poll interval: 5-10 seconds
- Backend caches metrics for 30 seconds
    """,
    response_model=SystemMetricsResponse,
    response_description="Current system and model metrics",
)
async def get_system_metrics(request: Request, force_refresh: bool = False):
    """
    Get current system and model metrics.

    Args:
        force_refresh: If true, bypass cache and collect fresh metrics
    """
    router = request.app.state.router_instance
    collector = get_metrics_collector(router)
    return collector.collect(force_refresh=force_refresh)


@monitoring_router.get("/v1/performance/profile/{time_range}",
    summary="Performance Profile",
    description="Aggregated performance profiling data for the Performance applet. "
                "Valid time_range values: 1h, 6h, 24h, 7d.",
)
async def get_performance_profile(time_range: str):
    """Return aggregated performance profile from in-memory ring buffer."""
    valid_ranges = {"1h", "6h", "24h", "7d"}
    if time_range not in valid_ranges:
        raise HTTPException(status_code=400, detail=f"Invalid time_range. Must be one of: {', '.join(sorted(valid_ranges))}")
    return get_perf_collector().build_profile(time_range)


@monitoring_router.get("/v1/capabilities",
    summary="Get Server Capabilities",
    description="""
Get detailed information about server capabilities and optimization options.

**Returns:**
- Available performance optimizations
- Supported features and endpoints
- Optimal usage recommendations
- API extensions

**Use this endpoint to:**
- Check which optimizations are active
- Get recommendations for best performance
- Understand server limits and capabilities

**Client Integration:**
Clients should query this endpoint on startup to discover:
1. Preferred image format
2. Request patterns
3. Available performance features

Generation is serialized through ONE process-wide queue, so concurrent
requests wait their turn whatever model they name.
    """,
    response_description="Server capabilities and optimization details",
)
async def get_capabilities(request: Request):
    """Get server capabilities and optimization options."""
    from heylook_llm import __version__ as server_version
    from heylook_llm.optimizations.status import get_optimization_summary

    # Get optimization status
    optimizations = get_optimization_summary()

    # Check Metal availability
    try:
        import mlx.core as mx
        has_metal = mx.metal.is_available()
        if has_metal:
            device_info = mx.device_info()
            metal_info = {
                "available": True,
                "device_name": device_info.get("name", "Unknown"),
                "max_recommended_working_set_size": device_info.get("max_recommended_working_set_size", 0)
            }
        else:
            metal_info = {"available": False}
    except Exception:
        metal_info = {"available": False}

    capabilities = {
        "server_version": server_version,
        "optimizations": optimizations,
        "metal": metal_info,
        "endpoints": {
            # ONE inference wire (v1.79.66). Images ride it as content blocks;
            # there is no server-side resize anywhere any more.
            "messages": {
                "endpoint": "/v1/messages",
                "description": "Anthropic Messages-conformant wire: top-level system, "
                               "typed content blocks with nested `source`, Anthropic "
                               "stop_reason vocabulary. No server-side resize -- "
                               "clients downscale before sending.",
                "supports_base64": True,
                "server_side_image_resize": False,
            },
        },
        "features": {
            "streaming": True,
            "model_caching": {
                "enabled": True,
                "eviction_policy": "LRU",
            },
            "vision_models": True,
            "max_image_size": "No hard limit; clients downscale before sending",
            "supported_image_formats": ["JPEG", "PNG", "WEBP", "BMP", "GIF"]
        },
        "recommendations": {
            "image_format": {
                "preferred": "JPEG",
                "quality": 85,
                "note": "JPEG with quality 85 provides best size/quality tradeoff"
            },
            "request_pattern": {
                "use_streaming": "For responses > 100 tokens",
                "reuse_connection": "Keep-alive recommended",
            }
        },
    }

    return capabilities


@monitoring_router.post("/v1/cache/clear",
    summary="Clear Prompt Caches",
    dependencies=[Depends(require_admin_token)],
    description="""
Clear prompt caches for a specific model or all models.

**Use Cases:**
- Free memory by clearing unused caches
- Reset cache state when switching contexts
- Troubleshooting cache-related issues

**Note:** This clears in-memory caches. The next request will rebuild the cache.
    """,
    response_model=CacheClearResponse,
    response_description="Number of caches cleared",
)
async def clear_caches(request: Request, body: CacheClearRequest = Body(default=CacheClearRequest())):
    """Clear the prefix cache of one loaded model, or of every loaded model.
    Each provider owns its cache (MLX: mlx-vlm's prefix cache per model), so
    this asks the providers; a model that is not loaded has no cache."""
    router: ModelRouter = request.app.state.router_instance
    loaded = router.get_loaded_models()
    targets = [body.model] if body.model else list(loaded)
    count = 0
    for model_id in targets:
        provider = loaded.get(model_id)
        clear = getattr(provider, "clear_cache", None)
        if clear is not None and clear():
            count += 1
    return CacheClearResponse(deleted_count=count)
