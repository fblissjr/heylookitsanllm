# src/edge_llm/server.py
import uvicorn
import argparse

def main():
    """The main entry point for the edge-llm command-line tool."""
    parser = argparse.ArgumentParser(description="Edge LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    uvicorn.run(
        "edge_llm.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=True
    )

if __name__ == "__main__":
    main()
