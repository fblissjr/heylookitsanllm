# src/heylook_llm/server.py
import uvicorn
import argparse
import logging
from heylook_llm.router import ModelRouter
from heylook_llm.api import app

def main():
    """
    The main entry point for the heylookllm command-line tool.
    This script parses arguments, initializes the ModelRouter,
    and launches the uvicorn server process.
    """
    parser = argparse.ArgumentParser(description="Edge LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--model-id", type=str, default=None, help="Optional ID of a model to load on startup.")
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the router and store it in the app's state
    router = ModelRouter(
        config_path="models.yaml",
        log_level=log_level,
        initial_model_id=args.model_id
    )
    app.state.router_instance = router

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()
