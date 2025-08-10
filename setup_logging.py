#!/usr/bin/env python3
"""
Setup script to configure logging and analytics for heylookitsanllm.
Run this to enable persistent logging to files and analytics.
"""

import os
import json
from pathlib import Path

def setup_analytics():
    """Configure analytics with DuckDB storage."""
    
    # Create analytics config
    analytics_config = {
        "enabled": True,
        "storage_level": "full",  # Store full requests and responses
        "db_path": "logs/analytics.db",
        "retention_days": 30,
        "max_db_size_mb": 5000,  # 5GB max
        "log_images": False,  # Don't store image data to save space
        "anonymize_content": False,
        "export_formats": ["json", "csv", "parquet"]
    }
    
    # Save to config file
    with open("analytics_config.json", "w") as f:
        json.dump(analytics_config, f, indent=2)
    
    print("✓ Created analytics_config.json")
    
    # Also create .env file for environment variables
    env_content = """# HeylookLLM Configuration
HEYLOOK_ANALYTICS_ENABLED=true
HEYLOOK_ANALYTICS_STORAGE_LEVEL=full
HEYLOOK_ANALYTICS_DB_PATH=logs/analytics.db
HEYLOOK_ANALYTICS_RETENTION_DAYS=30
HEYLOOK_ANALYTICS_MAX_DB_SIZE_MB=5000
HEYLOOK_ANALYTICS_LOG_IMAGES=false
HEYLOOK_ANALYTICS_ANONYMIZE_CONTENT=false
HEYLOOK_ANALYTICS_EXPORT_FORMATS=json,csv,parquet
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✓ Created .env file with analytics settings")

def setup_file_logging():
    """Create a logging configuration that saves to files."""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a startup script with file logging
    startup_script = """#!/usr/bin/env python3
\"\"\"
Enhanced server startup with file logging.
Use this instead of 'heylookllm' to get persistent logs.
\"\"\"

import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Set up file logging
log_file = logs_dir / f"heylookllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=100*1024*1024,  # 100MB per file
    backupCount=10
)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

print(f"Logging to: {log_file}")
print(f"Analytics database: logs/analytics.db")

# Now import and run the server
from heylook_llm.server import main
main()
"""
    
    with open("start_with_logging.py", "w") as f:
        f.write(startup_script)
    
    # Make it executable
    os.chmod("start_with_logging.py", 0o755)
    
    print("✓ Created start_with_logging.py")
    
def create_analysis_scripts():
    """Create helper scripts for analyzing logs."""
    
    analyze_script = """#!/usr/bin/env python3
\"\"\"
Analyze logs and metrics from heylookitsanllm.
\"\"\"

import duckdb
import pandas as pd
from pathlib import Path
import json

def analyze_metrics(db_path="logs/analytics.db"):
    \"\"\"Analyze metrics from the database.\"\"\"
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run the server with analytics enabled first.")
        return
    
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get overview
    print("\\n=== Request Overview ===")
    overview = conn.execute(\"\"\"
        SELECT 
            COUNT(*) as total_requests,
            COUNT(DISTINCT model) as unique_models,
            AVG(total_time_ms) as avg_time_ms,
            AVG(tokens_per_second) as avg_tokens_per_sec,
            SUM(completion_tokens) as total_tokens_generated
        FROM request_logs
        WHERE success = true
    \"\"\").fetchone()
    
    if overview[0] > 0:
        print(f"Total Requests: {overview[0]}")
        print(f"Unique Models: {overview[1]}")
        print(f"Avg Response Time: {overview[2]:.2f}ms")
        print(f"Avg Tokens/sec: {overview[3]:.2f}")
        print(f"Total Tokens Generated: {overview[4]}")
    
    # Get model performance
    print("\\n=== Model Performance ===")
    model_stats = conn.execute(\"\"\"
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
    \"\"\").fetchdf()
    
    if not model_stats.empty:
        print(model_stats.to_string())
    
    # Get recent errors
    print("\\n=== Recent Errors ===")
    errors = conn.execute(\"\"\"
        SELECT 
            timestamp,
            model,
            error_type,
            error_message
        FROM request_logs
        WHERE success = false
        ORDER BY timestamp DESC
        LIMIT 5
    \"\"\").fetchdf()
    
    if not errors.empty:
        print(errors.to_string())
    
    # Export to CSV for further analysis
    print("\\n=== Exporting Data ===")
    all_requests = conn.execute("SELECT * FROM request_logs").fetchdf()
    all_requests.to_csv("logs/request_logs.csv", index=False)
    print(f"Exported {len(all_requests)} requests to logs/request_logs.csv")
    
    conn.close()

if __name__ == "__main__":
    analyze_metrics()
"""
    
    with open("analyze_logs.py", "w") as f:
        f.write(analyze_script)
    
    os.chmod("analyze_logs.py", 0o755)
    
    print("✓ Created analyze_logs.py")

def main():
    print("Setting up logging and analytics for heylookitsanllm...")
    print()
    
    setup_analytics()
    setup_file_logging()
    create_analysis_scripts()
    
    print()
    print("Setup complete! To use:")
    print()
    print("1. Start server with file logging:")
    print("   python start_with_logging.py --api openai --log-level DEBUG")
    print()
    print("2. Or set environment and use regular command:")
    print("   export HEYLOOK_ANALYTICS_ENABLED=true")
    print("   heylookllm --api openai --log-level DEBUG")
    print()
    print("3. Analyze logs after running:")
    print("   python analyze_logs.py")
    print()
    print("Logs will be saved to:")
    print("  - Text logs: logs/heylookllm_*.log")
    print("  - Analytics DB: logs/analytics.db")

if __name__ == "__main__":
    main()