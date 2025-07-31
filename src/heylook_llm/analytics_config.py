# src/heylook_llm/analytics_config.py
"""
Configuration system for analytics and metrics collection.
Opt-in by default, with environment variable and config file support.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

class StorageLevel(Enum):
    """Different levels of data storage for analytics"""
    NONE = "none"  # No storage
    BASIC = "basic"  # Basic metrics only (counters, timings)
    REQUESTS = "requests"  # Store request metadata (no content)
    FULL = "full"  # Store full requests and responses
    
class AnalyticsConfig:
    """Configuration for analytics and metrics collection"""
    
    def __init__(self):
        self.enabled = False
        self.storage_level = StorageLevel.NONE
        self.db_path = "analytics.db"
        self.retention_days = 30
        self.max_db_size_mb = 1000
        self.log_images = False
        self.anonymize_content = False
        self.export_formats = ["json", "csv"]
        
        # Auto-create .env.example if it doesn't exist
        self._create_env_example()
        
        # Load configuration
        self._load_config()
    
    def _create_env_example(self):
        """Create example .env file with all available options"""
        env_example_path = Path(".env.example")
        if not env_example_path.exists():
            env_content = """# HeylookLLM Analytics Configuration
# Copy this file to .env and uncomment/modify as needed

# Enable analytics and metrics collection (default: false)
# HEYLOOK_ANALYTICS_ENABLED=true

# Storage level for analytics data (default: none)
# Options: none, basic, requests, full
# - none: No data collection
# - basic: Only counters and timing metrics
# - requests: Request metadata without content
# - full: Complete requests and responses
# HEYLOOK_ANALYTICS_STORAGE_LEVEL=basic

# Path to DuckDB database file (default: analytics.db)
# HEYLOOK_ANALYTICS_DB_PATH=analytics.db

# Data retention period in days (default: 30)
# HEYLOOK_ANALYTICS_RETENTION_DAYS=30

# Maximum database size in MB (default: 1000)
# HEYLOOK_ANALYTICS_MAX_DB_SIZE_MB=1000

# Log image data (default: false)
# Warning: This can significantly increase storage usage
# HEYLOOK_ANALYTICS_LOG_IMAGES=false

# Anonymize content before storage (default: false)
# Useful for privacy-sensitive deployments
# HEYLOOK_ANALYTICS_ANONYMIZE_CONTENT=false

# Export formats for data (comma-separated, default: json,csv)
# HEYLOOK_ANALYTICS_EXPORT_FORMATS=json,csv,parquet
"""
            env_example_path.write_text(env_content)
            logger.info(f"Created {env_example_path} with example configuration")
    
    def _load_config(self):
        """Load configuration from environment variables and config files"""
        # Check if analytics is enabled
        self.enabled = os.getenv("HEYLOOK_ANALYTICS_ENABLED", "false").lower() == "true"
        
        if not self.enabled:
            logger.info("Analytics disabled (set HEYLOOK_ANALYTICS_ENABLED=true to enable)")
            return
        
        # Load storage level
        storage_level_str = os.getenv("HEYLOOK_ANALYTICS_STORAGE_LEVEL", "none").lower()
        try:
            self.storage_level = StorageLevel(storage_level_str)
        except ValueError:
            logger.warning(f"Invalid storage level: {storage_level_str}, using 'none'")
            self.storage_level = StorageLevel.NONE
        
        # Load other settings
        self.db_path = os.getenv("HEYLOOK_ANALYTICS_DB_PATH", "analytics.db")
        self.retention_days = int(os.getenv("HEYLOOK_ANALYTICS_RETENTION_DAYS", "30"))
        self.max_db_size_mb = int(os.getenv("HEYLOOK_ANALYTICS_MAX_DB_SIZE_MB", "1000"))
        self.log_images = os.getenv("HEYLOOK_ANALYTICS_LOG_IMAGES", "false").lower() == "true"
        self.anonymize_content = os.getenv("HEYLOOK_ANALYTICS_ANONYMIZE_CONTENT", "false").lower() == "true"
        
        # Parse export formats
        export_formats_str = os.getenv("HEYLOOK_ANALYTICS_EXPORT_FORMATS", "json,csv")
        self.export_formats = [fmt.strip() for fmt in export_formats_str.split(",")]
        
        # Also check for config file
        config_path = Path("analytics_config.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)
                    # File config takes precedence over env vars
                    self._merge_config(file_config)
            except Exception as e:
                logger.error(f"Failed to load analytics config file: {e}")
        
        logger.info(f"Analytics enabled with storage level: {self.storage_level.value}")
        logger.info(f"Database path: {self.db_path}, retention: {self.retention_days} days")
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge configuration from file"""
        if "enabled" in file_config:
            self.enabled = file_config["enabled"]
        if "storage_level" in file_config:
            try:
                self.storage_level = StorageLevel(file_config["storage_level"])
            except ValueError:
                pass
        if "db_path" in file_config:
            self.db_path = file_config["db_path"]
        if "retention_days" in file_config:
            self.retention_days = file_config["retention_days"]
        if "max_db_size_mb" in file_config:
            self.max_db_size_mb = file_config["max_db_size_mb"]
        if "log_images" in file_config:
            self.log_images = file_config["log_images"]
        if "anonymize_content" in file_config:
            self.anonymize_content = file_config["anonymize_content"]
        if "export_formats" in file_config:
            self.export_formats = file_config["export_formats"]
    
    def should_log_request(self) -> bool:
        """Check if requests should be logged"""
        return self.enabled and self.storage_level in [StorageLevel.REQUESTS, StorageLevel.FULL]
    
    def should_log_content(self) -> bool:
        """Check if request/response content should be logged"""
        return self.enabled and self.storage_level == StorageLevel.FULL
    
    def should_log_metrics(self) -> bool:
        """Check if basic metrics should be logged"""
        return self.enabled and self.storage_level != StorageLevel.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "enabled": self.enabled,
            "storage_level": self.storage_level.value,
            "db_path": self.db_path,
            "retention_days": self.retention_days,
            "max_db_size_mb": self.max_db_size_mb,
            "log_images": self.log_images,
            "anonymize_content": self.anonymize_content,
            "export_formats": self.export_formats
        }
    
    def save_to_file(self, path: str = "analytics_config.json"):
        """Save current configuration to file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved analytics configuration to {path}")

# Global instance
analytics_config = AnalyticsConfig()