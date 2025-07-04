# src/heylook_llm/providers/common/performance_monitor.py
"""
Performance monitoring for MLX providers.

Why this exists:
- Provides timing decorators for profiling hot paths
- Tracks path-specific performance (VLM vs text paths)
- Enables before/after performance comparisons
- Data-centric approach to optimization validation
"""
import time
import logging
import functools
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class PerformanceMetrics:
    """Container for performance metrics with flexible data structure."""
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_measurement(self, duration: float):
        """Add a new timing measurement."""
        self.total_time += duration
        self.count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)
    
    @property
    def avg_time(self) -> float:
        """Average time across all measurements."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def recent_avg(self) -> float:
        """Average of recent measurements."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)


class PerformanceMonitor:
    """
    Thread-safe performance monitor for MLX providers.
    
    Tracks timing data for different operation types and paths.
    Designed for elegant simplicity while providing actionable insights.
    """
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.lock = Lock()
        self.enabled = True
    
    def time_operation(self, operation_name: str, path_info: Optional[str] = None):
        """
        Decorator to time operations with optional path information.
        
        Args:
            operation_name: Base name of the operation
            path_info: Optional path information (e.g., 'vlm_text', 'vlm_vision', 'text_only')
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Build metric key
                metric_key = operation_name
                if path_info:
                    metric_key = f"{operation_name}_{path_info}"
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    with self.lock:
                        self.metrics[metric_key].add_measurement(duration)
                        
                        # Log slow operations (data-driven thresholds)
                        if duration > 1.0:  # Log operations over 1 second
                            logging.warning(f"Slow operation: {metric_key} took {duration:.3f}s")
                        elif duration > 0.5:  # Debug log for moderately slow operations
                            logging.debug(f"Operation timing: {metric_key} took {duration:.3f}s")
            
            return wrapper
        return decorator
    
    def record_timing(self, operation_name: str, duration: float, path_info: Optional[str] = None):
        """Directly record a timing measurement."""
        if not self.enabled:
            return
        
        metric_key = operation_name
        if path_info:
            metric_key = f"{operation_name}_{path_info}"
        
        with self.lock:
            self.metrics[metric_key].add_measurement(duration)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics as a dictionary."""
        with self.lock:
            return {
                name: {
                    'avg_time': metrics.avg_time,
                    'recent_avg': metrics.recent_avg,
                    'min_time': metrics.min_time if metrics.count > 0 else 0.0,
                    'max_time': metrics.max_time,
                    'count': metrics.count,
                    'total_time': metrics.total_time
                }
                for name, metrics in self.metrics.items()
            }
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        with self.lock:
            if not self.metrics:
                return "No performance data recorded"
            
            summary_lines = ["Performance Summary:", "=" * 50]
            
            # Group by operation type
            operation_groups = defaultdict(list)
            for metric_name in self.metrics:
                base_name = metric_name.split('_')[0]
                operation_groups[base_name].append(metric_name)
            
            for operation_type, metric_names in operation_groups.items():
                summary_lines.append(f"\n{operation_type.title()} Operations:")
                
                for metric_name in sorted(metric_names):
                    metrics = self.metrics[metric_name]
                    if metrics.count > 0:
                        summary_lines.append(
                            f"  {metric_name}: {metrics.avg_time:.3f}s avg "
                            f"({metrics.count} calls, recent: {metrics.recent_avg:.3f}s)"
                        )
            
            return "\n".join(summary_lines)
    
    def compare_paths(self, base_operation: str) -> Dict[str, float]:
        """
        Compare performance across different paths for the same operation.
        
        Returns relative performance (1.0 = baseline, <1.0 = faster, >1.0 = slower)
        """
        with self.lock:
            path_metrics = {}
            baseline_time = None
            
            for metric_name, metrics in self.metrics.items():
                if metric_name.startswith(base_operation) and metrics.count > 0:
                    path_metrics[metric_name] = metrics.avg_time
                    if baseline_time is None:
                        baseline_time = metrics.avg_time
            
            if not path_metrics or baseline_time is None:
                return {}
            
            # Return relative performance
            return {
                path: time_val / baseline_time 
                for path, time_val in path_metrics.items()
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self.lock:
            self.metrics.clear()
    
    def enable(self):
        """Enable performance monitoring."""
        self.enabled = True
    
    def disable(self):
        """Disable performance monitoring."""
        self.enabled = False


# Global performance monitor instance
# Single instance for simplicity while maintaining extensibility
performance_monitor = PerformanceMonitor()


def time_mlx_operation(operation_name: str, path_info: Optional[str] = None):
    """
    Convenience decorator for timing MLX operations.
    
    Usage:
        @time_mlx_operation("generation", "vlm_text")
        def generate_text(self, ...):
            ...
    """
    return performance_monitor.time_operation(operation_name, path_info)


def log_performance_summary():
    """Log the current performance summary."""
    summary = performance_monitor.get_performance_summary()
    logging.info(f"MLX Performance Summary:\n{summary}")


def get_path_performance_comparison(operation: str) -> Dict[str, float]:
    """Get performance comparison for different paths."""
    return performance_monitor.compare_paths(operation)
