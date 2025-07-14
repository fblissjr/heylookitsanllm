# src/heylook_llm/server.py
import os
# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import uvicorn
import argparse
import logging
from fastapi import FastAPI
from heylook_llm.router import ModelRouter
from heylook_llm.api import app

def create_openai_only_app():
    """Create app with only OpenAI API endpoints"""
    from heylook_llm.api import list_models, create_chat_completion

    openai_app = FastAPI(title="OpenAI-Compatible LLM Server", version="1.0.0")

    # Add OpenAI endpoints
    openai_app.get("/v1/models")(list_models)
    openai_app.post("/v1/chat/completions")(create_chat_completion)

    @openai_app.get("/")
    async def root():
        return {
            "message": "OpenAI-Compatible LLM Server",
            "endpoints": {
                "models": "/v1/models",
                "chat": "/v1/chat/completions"
            }
        }

    return openai_app

def create_ollama_only_app():
    """Create app with only Ollama API endpoints"""
    from heylook_llm.api import ollama_chat, ollama_generate, ollama_tags, ollama_show, ollama_version, ollama_ps, ollama_embed

    ollama_app = FastAPI(title="Ollama-Compatible LLM Server", version="1.0.0")

    # Add Ollama endpoints
    ollama_app.post("/api/chat")(ollama_chat)
    ollama_app.post("/api/generate")(ollama_generate)
    ollama_app.get("/api/tags")(ollama_tags)
    ollama_app.post("/api/show")(ollama_show)
    ollama_app.get("/api/version")(ollama_version)
    ollama_app.get("/api/ps")(ollama_ps)
    ollama_app.post("/api/embed")(ollama_embed)

    @ollama_app.get("/")
    async def root():
        return {
            "message": "Ollama-Compatible LLM Server",
            "endpoints": {
                "models": "/api/tags",
                "chat": "/api/chat",
                "generate": "/api/generate",
                "show": "/api/show",
                "version": "/api/version",
                "ps": "/api/ps",
                "embed": "/api/embed"
            }
        }

    return ollama_app

def main():
    """
    The main entry point for the heylookllm command-line tool.
    This script parses arguments, initializes the ModelRouter,
    and launches the uvicorn server process.
    """
    parser = argparse.ArgumentParser(description="Hey Look It's an LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 11434, Ollama standard)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--model-id", type=str, default=None, help="Optional ID of a model to load on startup.")
    parser.add_argument("--api", default="both", choices=["openai", "ollama", "both"],
                       help="Which API to serve: openai, ollama, or both (default: both) - selecting ollama only will run on port 11434 unless specified explicitly")
    args = parser.parse_args()

    # Check if --port was explicitly provided
    port_provided = "--port" in sys.argv or any(arg.startswith("--port=") for arg in sys.argv)

    # Set to 11434 if only ollama and no port was provided
    if args.api == "ollama" and not port_provided:
        args.port = 11434
    elif args.port is None:
        args.port = 8080

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the router and store it in the app's state
    router = ModelRouter(
        config_path="models.yaml",
        log_level=log_level,
        initial_model_id=args.model_id
    )

    # Choose which app to run based on API selection
    if args.api == "openai":
        selected_app = create_openai_only_app()
        print(f"Starting OpenAI-compatible API server on {args.host}:{args.port}")
        print(f"Available endpoints: /v1/models, /v1/chat/completions")
    elif args.api == "ollama":
        selected_app = create_ollama_only_app()
        print(f"Starting Ollama-compatible API server on {args.host}:{args.port}")
        print(f"Available endpoints: /api/tags, /api/chat, /api/generate, /api/show, /api/version, /api/ps, /api/embed")
    else:  # both
        selected_app = app
        print(f"Starting combined API server on {args.host}:{args.port}")
        print(f"OpenAI endpoints: /v1/models, /v1/chat/completions")
        print(f"Ollama endpoints: /api/tags, /api/chat, /api/generate, /api/show, /api/version, /api/ps, /api/embed")

    selected_app.state.router_instance = router

    uvicorn.run(
        selected_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()
