# src/heylook_llm/metrics_db.py
"""DuckDB-based metrics and logging system for performance analysis."""

import duckdb
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
from .analytics_config import analytics_config, StorageLevel

logger = logging.getLogger(__name__)


class MetricsDB:
    """
    DuckDB-based metrics storage for performance tracking and analysis.

    Features:
    - Async write buffering to avoid blocking requests
    - Rich analytics queries for performance debugging
    - Request tracing with detailed timing breakdowns
    """

    def __init__(self, db_path: Optional[str] = None):
        self.enabled = analytics_config.enabled
        if not self.enabled:
            logger.info("MetricsDB disabled - analytics not enabled")
            return

        # Use configured path or default
        self.db_path = Path(db_path or analytics_config.db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing MetricsDB at {self.db_path}")

        # Connection pool for reads
        self.conn = duckdb.connect(str(self.db_path))

        # Write buffer for async logging
        self.write_buffer = []
        self.write_lock = threading.Lock()
        self.running = True

        # Initialize schema
        self._init_schema()

        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

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
                error_message VARCHAR,
                
                -- Full content storage (when enabled)
                messages JSON,
                response_text VARCHAR
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
        if not self.enabled:
            return

        # Filter metrics based on storage level
        filtered_metrics = self._filter_metrics(metrics)
        if filtered_metrics:
            with self.write_lock:
                self.write_buffer.append(('request_logs', filtered_metrics))

    def log_model_switch(self, from_model: str, to_model: str, unload_time: float, load_time: float):
        """Log model switching event."""
        if not self.enabled or not analytics_config.should_log_metrics():
            return

        with self.write_lock:
            self.write_buffer.append(('model_switches', {
                'from_model': from_model,
                'to_model': to_model,
                'unload_time_ms': int(unload_time * 1000),
                'load_time_ms': int(load_time * 1000)
            }))

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter metrics based on storage level configuration."""
        storage_level = analytics_config.storage_level

        if storage_level == StorageLevel.NONE:
            return None

        # Copy metrics to avoid modifying original
        filtered = metrics.copy()

        # Always remove image data unless explicitly enabled
        if not analytics_config.log_images and 'messages' in filtered:
            # Remove image data from messages
            filtered['messages'] = self._strip_images(filtered['messages'])

        if storage_level == StorageLevel.BASIC:
            # Only keep basic metrics
            keep_fields = ['timestamp', 'request_id', 'model', 'provider', 'request_type',
                          'total_time_ms', 'first_token_ms', 'tokens_per_second',
                          'prompt_tokens', 'completion_tokens', 'success']
            filtered = {k: v for k, v in filtered.items() if k in keep_fields}

        elif storage_level == StorageLevel.REQUESTS:
            # Keep request metadata but not content
            if 'messages' in filtered:
                # Only keep message count and types
                filtered['message_count'] = len(filtered['messages'])
                del filtered['messages']
            if 'response_text' in filtered:
                filtered['response_length'] = len(filtered['response_text'])
                del filtered['response_text']

        elif storage_level == StorageLevel.FULL:
            # Keep everything, but maybe anonymize
            if analytics_config.anonymize_content:
                filtered = self._anonymize_content(filtered)

        return filtered

    def _strip_images(self, messages: Any) -> Any:
        """Remove image data from messages."""
        if not isinstance(messages, list):
            return messages

        cleaned = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_copy = msg.copy()
                if 'content' in msg_copy and isinstance(msg_copy['content'], list):
                    # Handle multi-part content
                    cleaned_content = []
                    for part in msg_copy['content']:
                        if isinstance(part, dict) and part.get('type') == 'image_url':
                            # Replace with placeholder
                            cleaned_content.append({
                                'type': 'image_url',
                                'image_url': {'url': '[IMAGE_REMOVED]'}
                            })
                        else:
                            cleaned_content.append(part)
                    msg_copy['content'] = cleaned_content
                cleaned.append(msg_copy)
            else:
                cleaned.append(msg)
        return cleaned

    def _anonymize_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive content in the data."""
        # Simple anonymization - can be extended
        anonymized = data.copy()

        # Hash any potential user identifiers
        if 'user_id' in anonymized:
            anonymized['user_id'] = hashlib.sha256(str(anonymized['user_id']).encode()).hexdigest()[:8]

        # Redact potential PII in messages (basic implementation)
        # This could be extended with more sophisticated PII detection

        return anonymized

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
                        # Convert messages to JSON if present
                        processed_data = data.copy()
                        if 'messages' in processed_data and processed_data['messages'] is not None:
                            processed_data['messages'] = json.dumps(processed_data['messages'])
                        
                        columns = ', '.join(processed_data.keys())
                        placeholders = ', '.join(['?' for _ in processed_data])
                        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                        write_conn.execute(query, list(processed_data.values()))
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
        if not self.enabled:
            return {}

        return {
            'performance_trends': self.get_performance_trends().df(),
            'slow_requests': self.get_slow_requests().df(),
            'model_comparison': self.get_model_comparison().df(),
            'errors': self.get_error_analysis().df()
        }

    def _cleanup_loop(self):
        """Background thread that cleans up old data based on retention policy."""
        while self.running:
            time.sleep(3600)  # Run every hour

            try:
                # Clean up old data
                retention_date = datetime.now() - timedelta(days=analytics_config.retention_days)
                self.conn.execute(f"""
                    DELETE FROM request_logs
                    WHERE timestamp < '{retention_date.isoformat()}'
                """)
                self.conn.execute(f"""
                    DELETE FROM model_switches
                    WHERE timestamp < '{retention_date.isoformat()}'
                """)

                # Check database size
                db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
                if db_size_mb > analytics_config.max_db_size_mb:
                    logger.warning(f"Database size ({db_size_mb:.1f}MB) exceeds limit ({analytics_config.max_db_size_mb}MB)")
                    # Could implement more aggressive cleanup here

                logger.info(f"Cleaned up analytics data older than {retention_date}")
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def execute_query(self, query: str, limit: int = 1000) -> Dict[str, Any]:
        """Execute arbitrary SQL query (with safety limits)."""
        if not self.enabled:
            return {"error": "Analytics not enabled"}

        try:
            # Add LIMIT if not present (safety measure)
            query_lower = query.lower()
            if 'limit' not in query_lower and query_lower.strip().startswith('select'):
                query = f"{query.rstrip(';')} LIMIT {limit}"

            result = self.conn.execute(query)
            df = result.df()

            return {
                "columns": list(df.columns),
                "data": df.values.tolist(),
                "row_count": len(df)
            }
        except Exception as e:
            return {"error": str(e)}

    def get_request_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get full request details by ID (for replay functionality)."""
        if not self.enabled:
            return None

        try:
            # Execute query and get description
            query_result = self.conn.execute("""
                SELECT * FROM request_logs
                WHERE request_id = ?
            """, [request_id])
            
            # Get column names from the result
            cols = [desc[0] for desc in query_result.description]
            
            # Fetch the row
            result = query_result.fetchone()
            
            if result:
                # Convert to dict
                return dict(zip(cols, result))
            return None
        except Exception as e:
            logger.error(f"Error fetching request {request_id}: {e}")
            return None

    def search_requests(self,
                       model: Optional[str] = None,
                       text_search: Optional[str] = None,
                       min_tokens: Optional[int] = None,
                       max_tokens: Optional[int] = None,
                       success_only: bool = True,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Search for requests based on various criteria."""
        if not self.enabled:
            return []

        conditions = []
        params = []

        if model:
            conditions.append("model = ?")
            params.append(model)

        if text_search and analytics_config.storage_level == StorageLevel.FULL:
            conditions.append("(messages::TEXT ILIKE ? OR response_text ILIKE ?)")
            params.extend([f"%{text_search}%", f"%{text_search}%"])

        if min_tokens:
            conditions.append("(prompt_tokens + completion_tokens) >= ?")
            params.append(min_tokens)

        if max_tokens:
            conditions.append("(prompt_tokens + completion_tokens) <= ?")
            params.append(max_tokens)

        if success_only:
            conditions.append("success = true")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT
                request_id,
                timestamp,
                model,
                request_type,
                prompt_tokens + completion_tokens as total_tokens,
                total_time_ms,
                tokens_per_second
            FROM request_logs
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        try:
            result = self.conn.execute(query, params)
            return [dict(zip([desc[0] for desc in result.description], row))
                   for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error searching requests: {e}")
            return []


# Global instance
metrics_db = None

def init_metrics_db(db_path: Optional[str] = None):
    """Initialize the global metrics database."""
    global metrics_db
    metrics_db = MetricsDB(db_path)
    if metrics_db.enabled:
        logger.info(f"Initialized metrics database at {metrics_db.db_path}")
    else:
        logger.info("Metrics database disabled")

def get_metrics_db() -> Optional[MetricsDB]:
    """Get the global metrics database instance."""
    return metrics_db
