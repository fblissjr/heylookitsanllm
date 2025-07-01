# src/server.py
import uvicorn
import argparse

def main():
    """The main entry point for the edge-llm command-line tool."""
    parser = argparse.ArgumentParser(description="Edge LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Why: We pass the application as a string. Uvicorn's reloader process can
    # now correctly find `src.api` because `src` is defined as a package root.
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=True,
        reload_dirs=["src"]
    )

if __name__ == "__main__":
    main()
