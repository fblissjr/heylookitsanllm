# src/heylook_llm/server.py
import os

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys

import uvicorn

from heylook_llm.router import ModelRouter

# Try to use uvloop for better async performance
try:
    import asyncio

    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False


def get_api_endpoints(app_instance):
    """Extract API endpoint paths from app routes for display."""
    endpoints = []
    for route in app_instance.routes:
        if hasattr(route, "path") and route.path.startswith("/v1/"):
            endpoints.append(route.path)
    return sorted(set(endpoints))


def _log_disk_usage(args):
    """Log disk usage of log directory at startup."""
    from pathlib import Path

    parts = []

    # Log directory
    if args.file_log_level:
        log_dir = Path(args.log_dir)
        if log_dir.exists():
            total = sum(f.stat().st_size for f in log_dir.iterdir() if f.is_file())
            size_mb = total / (1024 * 1024)
            parts.append(f"log dir: {size_mb:.1f}MB")
    else:
        parts.append("file logging: disabled")

    if parts:
        logging.info(f"Disk usage -- {', '.join(parts)}")


def main():
    """
    The main entry point for the heylookllm command-line tool.
    This script parses arguments, initializes the ModelRouter,
    and launches the uvicorn server process.
    """
    if sys.platform != "darwin":
        print(
            "Error: heylookitsanllm requires macOS with Apple Silicon.\n"
            "MLX only supports Apple Silicon Macs.\n"
            "See: https://github.com/fblissjr/heylookitsanllm"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Hey Look It's an LLM Server")

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Import command
    import_parser = subparsers.add_parser(
        "import", help="Import models from directories or HF cache"
    )
    import_parser.add_argument(
        "--folder", type=str, help="Path to folder containing models to import"
    )
    import_parser.add_argument(
        "--hf-cache", action="store_true", help="Scan HuggingFace cache for models"
    )
    import_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models.toml",
        help="Output file for generated configuration (default: models.toml)",
    )
    from heylook_llm.model_service import get_available_profiles

    preset_names = get_available_profiles()
    import_parser.add_argument(
        "--preset",
        "--profile",
        dest="profile",
        choices=preset_names,
        help="Record a preset name as the model's default_preset "
             "(applied at request time). --profile is accepted as an alias "
             "for --preset. See src/heylook_llm/data/presets/*.toml for the "
             "canonical set.",
    )
    import_parser.add_argument(
        "--override",
        action="append",
        help="Override specific settings (e.g., --override temperature=0.5 --override max_tokens=256)",
    )
    import_parser.add_argument(
        "--chat-template",
        dest="chat_template",
        help="Chat template source policy recorded on the imported model. "
             "'auto' (default): HF picks the right one; 'jinja': force-load "
             "chat_template.jinja from the model dir; 'tokenizer_config': "
             "force the embedded template; 'chat_template_json': force the "
             "processor-side chat_template.json; or an absolute path to a "
             ".jinja file. Import wizard auto-sets 'jinja' when "
             "chat_template.jinja is present; pass this flag to override.",
    )
    import_parser.add_argument(
        "--merge",
        action="store_true",
        help="Show instructions for merging with existing models.toml",
    )
    import_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    import_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively customize sampler and KV cache settings for each discovered model",
    )

    # Service command - manage background service (macOS/Linux)
    service_parser = subparsers.add_parser(
        "service", help="Manage heylookllm as a background service (macOS/Linux)"
    )
    service_parser.add_argument(
        "action",
        choices=["install", "uninstall", "start", "stop", "restart", "status"],
        help="Service action to perform",
    )
    service_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for local only, use 0.0.0.0 for LAN access)",
    )
    service_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    service_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level (default: INFO)",
    )
    service_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files (default: ./logs)",
    )
    service_parser.add_argument(
        "--system-wide",
        action="store_true",
        help="Linux only: Install as system-wide service (requires sudo)",
    )

    # Add server arguments to main parser for backwards compatibility
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the server on"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level",
    )
    parser.add_argument(
        "--file-log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="File logging level (if not set, file logging is disabled)",
    )
    parser.add_argument(
        "--log-dir", default="logs", help="Directory for log files (default: logs)"
    )
    parser.add_argument(
        "--log-rotate-mb",
        type=int,
        default=100,
        help="Max size in MB per log file before rotation (default: 100)",
    )
    parser.add_argument(
        "--log-rotate-count",
        type=int,
        default=10,
        help="Number of rotated log files to keep (default: 10)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional ID of a model to load on startup.",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=str,
        default=None,
        help="Max prompt cache memory (e.g. '2G', '512M'). Default: unlimited.",
    )

    args = parser.parse_args()

    # Handle import command
    if args.command == "import":
        # Set up logging for import
        import logging as log_module

        log_level = getattr(log_module, args.log_level.upper())
        log_module.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        from heylook_llm.model_importer import import_models

        import_models(args)
        return

    # Handle service command
    if args.command == "service":
        # Set up logging for service management
        import logging as log_module

        log_level = getattr(log_module, args.log_level.upper())
        log_module.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        from heylook_llm.service_manager import manage_service

        sys.exit(manage_service(args))

    # Set up logging with separate console and file handlers
    import logging.handlers
    from datetime import datetime
    from pathlib import Path

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_level = getattr(logging, args.log_level.upper())
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if args.file_log_level:
        # Create log directory if it doesn't exist
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        log_file = (
            log_dir / f"heylookllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Set up rotating file handler
        file_level = getattr(logging, args.file_log_level.upper())
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=args.log_rotate_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=args.log_rotate_count,
        )
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"File logging enabled: {log_file} (level: {args.file_log_level})")
    else:
        logging.info("File logging disabled (use --file-log-level to enable)")

    # Log uvloop status
    if HAS_UVLOOP:
        logging.info("Using uvloop for improved async performance")
    else:
        logging.info("uvloop not available - using standard asyncio event loop")

    # Set process-wide wired memory limit for Metal.
    # mlx-lm does this at server startup (server.py main()) to keep model weights
    # wired between requests. Without this, the OS can page out model weights
    # between requests, causing memory churn and slower time-to-first-token.
    # The per-generation wired_limit() context manager still applies on top of this.
    import mlx.core as mx

    if mx.metal.is_available():
        wired_limit = int(mx.device_info()["max_recommended_working_set_size"])
        mx.set_wired_limit(wired_limit)
        logging.info(
            f"Set wired memory limit: {wired_limit / (1024**3):.1f} GB "
            f"(max recommended working set)"
        )

    # Configure prompt cache byte budget if specified
    if args.prompt_cache_bytes:
        from heylook_llm.providers.common.prompt_cache import get_global_cache_manager

        def _parse_size(s: str) -> int:
            """Parse size string like '2G', '512M', '1024K' to bytes."""
            s = s.strip().upper()
            multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
            if s[-1] in multipliers:
                return int(float(s[:-1]) * multipliers[s[-1]])
            return int(s)

        cache_bytes = _parse_size(args.prompt_cache_bytes)
        mgr = get_global_cache_manager()
        mgr.set_byte_budget(cache_bytes)
        logging.info(f"Prompt cache byte budget: {cache_bytes / (1024**3):.2f} GB")

    # Log all optimization statuses
    from heylook_llm.optimizations.status import log_all_optimization_status

    log_all_optimization_status()

    # Initialize the router and store it in the app's state
    # Pass "models" without extension - router will try .toml first, then .yaml
    router = ModelRouter(
        config_path="models",
        log_level=console_level,  # Use console log level for router
        initial_model_id=args.model_id,
    )

    # Check if any providers are available
    from heylook_llm.router import HAS_MLX

    if not HAS_MLX:
        logging.error("No model providers available! Please install MLX:")
        logging.error("  uv sync --extra mlx")
        sys.exit(1)

    # Use the app from api.py directly (single source of truth for endpoints)
    from heylook_llm.api import app

    # Attach router to app state
    app.state.router_instance = router

    # Log disk usage of persistent storage
    _log_disk_usage(args)

    # Display startup info with dynamic endpoint discovery
    endpoints = get_api_endpoints(app)
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Available endpoints: {', '.join(endpoints)}")

    # LAN hardening hint (S1.6): when bound to a non-loopback address, the
    # server is reachable by every host on the local network. Plain HTTP is
    # fine on a trusted home LAN but gets upgraded to HTTPS via a reverse
    # proxy (Caddy / nginx), not uvicorn-native TLS. See docs/lan_setup.md.
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.info(
            "Listening on http://%s:%s -- LAN-reachable. For HTTPS, put Caddy "
            "or nginx in front (see docs/lan_setup.md).",
            args.host,
            args.port,
        )
    if os.environ.get("HEYLOOK_ADMIN_TOKEN", "").strip():
        logging.info(
            "HEYLOOK_ADMIN_TOKEN set -- admin endpoints require "
            "X-Heylook-Admin-Token header."
        )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
