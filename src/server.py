# src/server.py
import uvicorn, argparse, logging

def parse_args():
    parser = argparse.ArgumentParser(description="Edge LLM Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Why: This command starts the FastAPI application using the uvicorn server.
    # The string "src.api:app" tells uvicorn where to find the FastAPI instance.
    uvicorn.run("src.api:app", host=args.host, port=args.port, log_level=args.log_level.lower(), reload=True)
