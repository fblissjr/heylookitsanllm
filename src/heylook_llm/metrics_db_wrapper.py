# src/heylook_llm/metrics_db_wrapper.py
"""Wrapper for optional metrics database functionality."""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import DuckDB and related functionality
try:
    import duckdb
    DUCKDB_AVAILABLE = True
    from .metrics_db import MetricsDB, init_metrics_db as _init_metrics_db, get_metrics_db as _get_metrics_db
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.info("DuckDB not installed. Analytics features will be disabled.")
    logger.info("Install with: pip install heylookitsanllm[analytics]")
    
    # Define dummy implementations
    class MetricsDB:
        """Dummy MetricsDB when duckdb is not available."""
        def __init__(self, *args, **kwargs):
            self.enabled = False
            
        def log_request(self, metrics: Dict[str, Any]):
            pass
            
        def log_model_switch(self, from_model: str, to_model: str, unload_time: float, load_time: float):
            pass
            
        def execute_query(self, query: str, limit: int = 1000) -> Dict[str, Any]:
            return {"error": "Analytics not available. Install with: pip install heylookitsanllm[analytics]"}
            
        def get_slow_requests(self, threshold_ms: int = 1000, limit: int = 20):
            return None
            
        def get_performance_trends(self, hours: int = 24):
            return None
            
        def get_model_comparison(self):
            return None
            
        def get_error_analysis(self):
            return None
            
        def export_dashboard_data(self) -> Dict[str, Any]:
            return {}
            
        def get_request_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
            return None
            
        def search_requests(self, *args, **kwargs):
            return []
    
    _metrics_db_instance = None
    
    def _init_metrics_db(db_path: Optional[str] = None):
        """Dummy init when duckdb is not available."""
        global _metrics_db_instance
        _metrics_db_instance = MetricsDB()
        logger.info("Metrics database disabled (DuckDB not installed)")
        
    def _get_metrics_db() -> Optional[MetricsDB]:
        """Return dummy instance when duckdb is not available."""
        return _metrics_db_instance


# Export the wrapped functions
init_metrics_db = _init_metrics_db
get_metrics_db = _get_metrics_db

__all__ = ['MetricsDB', 'init_metrics_db', 'get_metrics_db', 'DUCKDB_AVAILABLE']