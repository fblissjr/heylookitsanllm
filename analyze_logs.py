#!/usr/bin/env python3
"""
Analyze logs and metrics from heylookitsanllm.
"""

import duckdb
import pandas as pd
from pathlib import Path
import json

def analyze_metrics(db_path="logs/analytics.db"):
    """Analyze metrics from the database."""
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run the server with analytics enabled first.")
        return
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get overview
    print("\n=== Request Overview ===")
    overview = conn.execute("""
        SELECT 
            COUNT(*) as total_requests,
            COUNT(DISTINCT model) as unique_models,
            AVG(total_time_ms) as avg_time_ms,
            AVG(tokens_per_second) as avg_tokens_per_sec,
            SUM(completion_tokens) as total_tokens_generated
        FROM request_logs
        WHERE success = true
    """).fetchone()
    
    if overview[0] > 0:
        print(f"Total Requests: {overview[0]}")
        print(f"Unique Models: {overview[1]}")
        print(f"Avg Response Time: {overview[2]:.2f}ms")
        print(f"Avg Tokens/sec: {overview[3]:.2f}")
        print(f"Total Tokens Generated: {overview[4]}")
    
    # Get model performance
    print("\n=== Model Performance ===")
    model_stats = conn.execute("""
        SELECT 
            model,
            COUNT(*) as requests,
            AVG(tokens_per_second) as avg_tps,
            AVG(total_time_ms) as avg_time_ms,
            AVG(first_token_ms) as avg_first_token_ms
        FROM request_logs
        WHERE success = true
        GROUP BY model
        ORDER BY requests DESC
    """).fetchdf()
    
    if not model_stats.empty:
        print(model_stats.to_string())
    
    # Get recent errors
    print("\n=== Recent Errors ===")
    errors = conn.execute("""
        SELECT 
            timestamp,
            model,
            error_type,
            error_message
        FROM request_logs
        WHERE success = false
        ORDER BY timestamp DESC
        LIMIT 5
    """).fetchdf()
    
    if not errors.empty:
        print(errors.to_string())
    
    # Export to CSV for further analysis
    print("\n=== Exporting Data ===")
    all_requests = conn.execute("SELECT * FROM request_logs").fetchdf()
    all_requests.to_csv("logs/request_logs.csv", index=False)
    print(f"Exported {len(all_requests)} requests to logs/request_logs.csv")
    
    conn.close()

if __name__ == "__main__":
    analyze_metrics()
