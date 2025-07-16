# src/heylook_llm/metrics_db.py
"""DuckDB-based metrics and logging system for performance analysis."""

import duckdb
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging


class MetricsDB:
    """
    DuckDB-based metrics storage for performance tracking and analysis.
    
    Features:
    - Async write buffering to avoid blocking requests
    - Rich analytics queries for performance debugging
    - Request tracing with detailed timing breakdowns
    """
    
    def __init__(self, db_path: str = "~/.heylook_llm/metrics.duckdb"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection pool for reads
        self.conn = duckdb.connect(str(self.db_path))
        
        # Write buffer for async logging
        self.write_buffer = []
        self.write_lock = threading.Lock()
        
        # Initialize schema
        self._init_schema()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()
    
    def _init_schema(self):
        """Create tables for metrics storage."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                timestamp TIMESTAMP DEFAULT current_timestamp,
                request_id VARCHAR,
                model VARCHAR,
                provider VARCHAR,
                request_type VARCHAR,
                
                -- Request details
                num_images INTEGER DEFAULT 0,
                num_messages INTEGER,
                max_tokens INTEGER,
                temperature FLOAT,
                
                -- Timing breakdown (all in milliseconds)
                total_time_ms INTEGER,
                queue_time_ms INTEGER,
                model_load_time_ms INTEGER,
                image_processing_ms INTEGER,
                token_generation_ms INTEGER,
                first_token_ms INTEGER,
                
                -- Token counts
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                tokens_per_second FLOAT,
                
                -- Resource usage
                memory_used_gb FLOAT,
                gpu_utilization FLOAT,
                
                -- Error tracking
                success BOOLEAN DEFAULT true,
                error_type VARCHAR,
                error_message VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_switches (
                timestamp TIMESTAMP DEFAULT current_timestamp,
                from_model VARCHAR,
                to_model VARCHAR,
                unload_time_ms INTEGER,
                load_time_ms INTEGER,
                memory_freed_gb FLOAT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_summary (
                timestamp TIMESTAMP DEFAULT current_timestamp,
                model VARCHAR,
                operation VARCHAR,
                path VARCHAR,
                avg_time_ms FLOAT,
                min_time_ms FLOAT,
                max_time_ms FLOAT,
                p95_time_ms FLOAT,
                count INTEGER
            )
        """)
    
    def log_request(self, metrics: Dict[str, Any]):
        """Log request metrics (non-blocking)."""
        with self.write_lock:
            self.write_buffer.append(('request_logs', metrics))
    
    def log_model_switch(self, from_model: str, to_model: str, unload_time: float, load_time: float):
        """Log model switching event."""
        with self.write_lock:
            self.write_buffer.append(('model_switches', {
                'from_model': from_model,
                'to_model': to_model,
                'unload_time_ms': int(unload_time * 1000),
                'load_time_ms': int(load_time * 1000)
            }))
    
    def _write_loop(self):
        """Background thread that flushes write buffer to disk."""
        write_conn = duckdb.connect(str(self.db_path))
        
        while True:
            time.sleep(1)  # Flush every second
            
            if self.write_buffer:
                with self.write_lock:
                    to_write = self.write_buffer[:]
                    self.write_buffer.clear()
                
                try:
                    for table, data in to_write:
                        columns = ', '.join(data.keys())
                        placeholders = ', '.join(['?' for _ in data])
                        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                        write_conn.execute(query, list(data.values()))
                    write_conn.commit()
                except Exception as e:
                    logging.error(f"Failed to write metrics: {e}")
    
    # Analytics queries
    def get_slow_requests(self, threshold_ms: int = 1000, limit: int = 20) -> duckdb.DuckDBPyRelation:
        """Get slowest requests above threshold."""
        return self.conn.execute(f"""
            SELECT 
                timestamp,
                model,
                request_type,
                total_time_ms,
                prompt_tokens + completion_tokens as total_tokens,
                error_message
            FROM request_logs
            WHERE total_time_ms > {threshold_ms}
            ORDER BY total_time_ms DESC
            LIMIT {limit}
        """)
    
    def get_performance_trends(self, hours: int = 24) -> duckdb.DuckDBPyRelation:
        """Get performance trends over time."""
        return self.conn.execute(f"""
            SELECT 
                time_bucket(INTERVAL '15 minutes', timestamp) as time_bucket,
                model,
                request_type,
                COUNT(*) as requests,
                AVG(total_time_ms) as avg_time_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_time_ms,
                AVG(tokens_per_second) as avg_tps
            FROM request_logs
            WHERE timestamp > NOW() - INTERVAL '{hours} hours'
              AND success = true
            GROUP BY 1, 2, 3
            ORDER BY 1 DESC
        """)
    
    def get_model_comparison(self) -> duckdb.DuckDBPyRelation:
        """Compare performance across models."""
        return self.conn.execute("""
            SELECT 
                model,
                request_type,
                COUNT(*) as requests,
                AVG(total_time_ms) as avg_time_ms,
                AVG(first_token_ms) as avg_first_token_ms,
                AVG(tokens_per_second) as avg_tps,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as errors
            FROM request_logs
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY 1, 2
            ORDER BY 3 DESC
        """)
    
    def get_error_analysis(self) -> duckdb.DuckDBPyRelation:
        """Analyze errors by type and model."""
        return self.conn.execute("""
            SELECT 
                model,
                error_type,
                COUNT(*) as count,
                MAX(timestamp) as last_seen,
                SUBSTRING(error_message, 1, 100) as sample_error
            FROM request_logs
            WHERE success = false
              AND timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY 1, 2, 5
            ORDER BY 3 DESC
        """)
    
    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export data for dashboard visualization."""
        return {
            'performance_trends': self.get_performance_trends().df(),
            'slow_requests': self.get_slow_requests().df(),
            'model_comparison': self.get_model_comparison().df(),
            'errors': self.get_error_analysis().df()
        }


# Global instance
metrics_db = None

def init_metrics_db(db_path: Optional[str] = None):
    """Initialize the global metrics database."""
    global metrics_db
    metrics_db = MetricsDB(db_path) if db_path else MetricsDB()
    logging.info(f"Initialized metrics database at {metrics_db.db_path}")

def get_metrics_db() -> Optional[MetricsDB]:
    """Get the global metrics database instance."""
    return metrics_db