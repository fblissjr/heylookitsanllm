# src/heylook_llm/server.py
import uvicorn
import argparse

def main():
    """
    The main entry point for the heylookllm command-line tool.
    This script's ONLY job is to parse uvicorn-related arguments
    and launch the uvicorn server process.
    The application itself will parse its own arguments inside its lifespan.
    """
    parser = argparse.ArgumentParser(description="Edge LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    # We add other args here so they appear in `heylookllm --help`
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--model-id", type=str, default=None, help="Optional ID of a model to load on startup.")

    args = parser.parse_args()

    uvicorn.run(
        "heylook_llm.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=True
    )

if __name__ == "__main__":
    main()
