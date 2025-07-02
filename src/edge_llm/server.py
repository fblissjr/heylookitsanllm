# server.py
import uvicorn
import argparse

def main():
    """The main entry point for the edge-llm command-line tool."""
    parser = argparse.ArgumentParser(description="Edge LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    from edge_llm.api import app
    uvicorn.run(app, host=host, port=port, reload=True, log_level=log_level)


if __name__ == "__main__":
    main()
