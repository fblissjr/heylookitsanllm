# src/heylook_llm/system_metrics.py
"""System metrics collector for monitoring RAM, CPU, and model context usage."""

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Optional

from .config import ModelMetrics, SystemMetricsResponse, SystemResourceMetrics

if TYPE_CHECKING:
    from .router import ModelRouter

logger = logging.getLogger(__name__)

# Try to import psutil, graceful fallback if unavailable
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed - system metrics will return zeros")


def _empty_system_metrics() -> SystemResourceMetrics:
    """Return zeroed system metrics for fallback cases."""
    return SystemResourceMetrics(
        ram_used_gb=0.0,
        ram_available_gb=0.0,
        ram_total_gb=0.0,
        cpu_percent=0.0,
    )


def _empty_model_metrics() -> ModelMetrics:
    """Return zeroed model metrics for fallback cases."""
    return ModelMetrics(
        context_used=0,
        context_capacity=0,
        context_percent=0.0,
        memory_mb=0.0,
        requests_active=0,
    )


class SystemMetricsCollector:
    """Collects system and model metrics with caching."""

    def __init__(self, router: "ModelRouter", cache_ttl_seconds: float = 30.0):
        """
        Initialize the metrics collector.

        Args:
            router: ModelRouter instance to get loaded model info
            cache_ttl_seconds: How long to cache metrics (default 30s)
        """
        self.router = router
        self.cache_ttl = cache_ttl_seconds
        self._cached_metrics: Optional[SystemMetricsResponse] = None
        self._cache_time: float = 0

    def _get_system_metrics(self) -> SystemResourceMetrics:
        """Collect system-wide RAM and CPU metrics."""
        if not PSUTIL_AVAILABLE:
            return _empty_system_metrics()

        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)  # Non-blocking, uses cached value

            return SystemResourceMetrics(
                ram_used_gb=round(mem.used / (1024 ** 3), 2),
                ram_available_gb=round(mem.available / (1024 ** 3), 2),
                ram_total_gb=round(mem.total / (1024 ** 3), 2),
                cpu_percent=round(cpu, 1),
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return _empty_system_metrics()

    def _get_model_metrics(self) -> Dict[str, ModelMetrics]:
        """Collect metrics from all loaded models via their providers."""
        model_metrics: Dict[str, ModelMetrics] = {}

        for model_id, provider in self.router.get_loaded_models().items():
            try:
                metrics = provider.get_metrics()
                if metrics:
                    model_metrics[model_id] = metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics from model {model_id}: {e}")
                model_metrics[model_id] = _empty_model_metrics()

        return model_metrics

    def collect(self, force_refresh: bool = False) -> SystemMetricsResponse:
        """
        Collect system and model metrics, using cache if available.

        Args:
            force_refresh: If True, bypass cache and collect fresh metrics

        Returns:
            SystemMetricsResponse with current metrics
        """
        now = time.time()

        # Return cached metrics if still valid
        if not force_refresh and self._cached_metrics is not None:
            if (now - self._cache_time) < self.cache_ttl:
                return self._cached_metrics

        # Collect fresh metrics
        timestamp = datetime.now(timezone.utc).isoformat()
        system_metrics = self._get_system_metrics()
        model_metrics = self._get_model_metrics()

        metrics = SystemMetricsResponse(
            timestamp=timestamp,
            system=system_metrics,
            models=model_metrics,
        )

        # Update cache
        self._cached_metrics = metrics
        self._cache_time = now

        return metrics

    def invalidate_cache(self) -> None:
        """Invalidate the metrics cache, forcing fresh collection on next call."""
        self._cached_metrics = None
        self._cache_time = 0
