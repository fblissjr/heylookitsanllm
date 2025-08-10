#!/usr/bin/env python3
"""
Server startup with file logging.
Use this instead of 'heylookllm' to get persistent logs.
"""

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
