# src/heylook_llm/server.py
import os
# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import uvicorn
import argparse
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from heylook_llm.router import ModelRouter
from heylook_llm.api import app

# Try to use uvloop for better async performance
try:
    import uvloop
    import asyncio
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False

def create_openai_only_app():
    """Create app with only OpenAI API endpoints"""
    from heylook_llm.api import list_models, create_chat_completion, create_batch_chat_completion, get_capabilities, performance_metrics, data_query, data_summary, create_embeddings_endpoint, extract_hidden_states_endpoint, restart_server, reload_models
    from heylook_llm.api_multipart import create_chat_multipart
    from heylook_llm.stt_api import stt_router

    openai_app = FastAPI(title="OpenAI-Compatible LLM Server", version="1.0.0")

    # Add CORS middleware
    openai_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add OpenAI endpoints
    openai_app.get("/v1/models")(list_models)
    openai_app.post("/v1/chat/completions")(create_chat_completion)
    openai_app.post("/v1/batch/chat/completions")(create_batch_chat_completion)
    openai_app.post("/v1/embeddings")(create_embeddings_endpoint)
    openai_app.post("/v1/hidden_states")(extract_hidden_states_endpoint)
    openai_app.get("/v1/capabilities")(get_capabilities)
    openai_app.get("/v1/performance")(performance_metrics)
    openai_app.post("/v1/chat/completions/multipart")(create_chat_multipart)
    openai_app.post("/v1/data/query")(data_query)
    openai_app.get("/v1/data/summary")(data_summary)

    # Add STT endpoints
    openai_app.include_router(stt_router)

    # Add admin endpoints
    openai_app.post("/v1/admin/restart")(restart_server)
    openai_app.post("/v1/admin/reload")(reload_models)

    @openai_app.get("/")
    async def root():
        return {
            "message": "OpenAI-Compatible LLM Server",
            "endpoints": {
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
                "batch_chat": "/v1/batch/chat/completions",
                "embeddings": "/v1/embeddings",
                "hidden_states": "/v1/hidden_states",
                "capabilities": "/v1/capabilities",
                "performance": "/v1/performance",
                "multipart": "/v1/chat/completions/multipart",
                "data_query": "/v1/data/query",
                "data_summary": "/v1/data/summary"
            }
        }

    return openai_app


def main():
    """
    The main entry point for the heylookllm command-line tool.
    This script parses arguments, initializes the ModelRouter,
    and launches the uvicorn server process.
    """
    parser = argparse.ArgumentParser(description="Hey Look It's an LLM Server")

    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import models from directories or HF cache')
    import_parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing models to import"
    )
    import_parser.add_argument(
        "--hf-cache",
        action="store_true",
        help="Scan HuggingFace cache for models"
    )
    import_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models.toml",
        help="Output file for generated configuration (default: models.toml)"
    )
    import_parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "performance", "max_quality", "background", "memory", "interactive"],
        help="Apply a predefined profile for model defaults (max_quality=no quantization, performance=high throughput, background=24/7 low resource)"
    )
    import_parser.add_argument(
        "--override",
        action="append",
        help="Override specific settings (e.g., --override temperature=0.5 --override max_tokens=256)"
    )
    import_parser.add_argument(
        "--merge",
        action="store_true",
        help="Show instructions for merging with existing models.yaml"
    )
    import_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    # Service command - manage background service (macOS/Linux)
    service_parser = subparsers.add_parser(
        'service',
        help='Manage heylookllm as a background service (macOS/Linux)'
    )
    service_parser.add_argument(
        "action",
        choices=["install", "uninstall", "start", "stop", "restart", "status"],
        help="Service action to perform"
    )
    service_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for local only, use 0.0.0.0 for LAN access)"
    )
    service_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    service_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level (default: INFO)"
    )
    service_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files (default: ./logs)"
    )
    service_parser.add_argument(
        "--system-wide",
        action="store_true",
        help="Linux only: Install as system-wide service (requires sudo)"
    )

    # Add server arguments to main parser for backwards compatibility
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Console logging level")
    parser.add_argument("--file-log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="File logging level (if not set, file logging is disabled)")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files (default: logs)")
    parser.add_argument("--log-rotate-mb", type=int, default=100,
                       help="Max size in MB per log file before rotation (default: 100)")
    parser.add_argument("--log-rotate-count", type=int, default=10,
                       help="Number of rotated log files to keep (default: 10)")
    parser.add_argument("--model-id", type=str, default=None, help="Optional ID of a model to load on startup.")

    args = parser.parse_args()

    # Handle import command
    if args.command == 'import':
        # Set up logging for import
        import logging as log_module
        log_level = getattr(log_module, args.log_level.upper())
        log_module.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        from heylook_llm.model_importer import import_models
        import_models(args)
        return

    # Handle service command
    if args.command == 'service':
        # Set up logging for service management
        import logging as log_module
        log_level = getattr(log_module, args.log_level.upper())
        log_module.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        from heylook_llm.service_manager import manage_service
        sys.exit(manage_service(args))

    # Default port if not provided
    if args.port is None:
        args.port = 8080

    # Set up logging with separate console and file handlers
    import logging.handlers
    from pathlib import Path
    from datetime import datetime

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_level = getattr(logging, args.log_level.upper())
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if args.file_log_level:
        # Create log directory if it doesn't exist
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        log_file = log_dir / f"heylookllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Set up rotating file handler
        file_level = getattr(logging, args.file_log_level.upper())
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=args.log_rotate_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=args.log_rotate_count
        )
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    # Log all optimization statuses
    from heylook_llm.optimizations.status import log_all_optimization_status
    log_all_optimization_status()

    # Initialize metrics database (will auto-detect if enabled)
    from heylook_llm.metrics_db_wrapper import init_metrics_db
    init_metrics_db()

    # Initialize the router and store it in the app's state
    # Pass "models" without extension - router will try .toml first, then .yaml
    router = ModelRouter(
        config_path="models",
        log_level=console_level,  # Use console log level for router
        initial_model_id=args.model_id
    )

    # Check if any providers are available
    from heylook_llm.router import HAS_MLX, HAS_LLAMA_CPP
    if not HAS_MLX and not HAS_LLAMA_CPP:
        logging.error("No model providers available! Please install at least one provider:")
        logging.error("  - For MLX models: uv sync --extra mlx")
        logging.error("  - For GGUF models: uv sync --extra llama-cpp")
        logging.error("  - For both: uv sync --extra all")
        sys.exit(1)

    # Create and configure the app
    selected_app = create_openai_only_app()
    print(f"Starting API server on {args.host}:{args.port}")
    print("Available endpoints: /v1/models, /v1/chat/completions, /v1/batch/chat/completions, /v1/embeddings, /v1/hidden_states, /v1/audio/transcriptions, /v1/stt/models, /v1/capabilities, /v1/performance, /v1/chat/completions/multipart, /v1/data/query")

    selected_app.state.router_instance = router

    uvicorn.run(
        selected_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()
