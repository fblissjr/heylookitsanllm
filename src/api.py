# src/api.py
import json, uuid, time, logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from .router import ModelRouter
from .config import ChatRequest

# Why: The router instance is now managed via the app's state to be
# accessible throughout the application lifecycle.
@asynccontextmanager
async def lifespan(app: FastAPI):
    log_level = getattr(logging, app.state.cli_args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
    app.state.router_instance = ModelRouter(config_path="models.yaml", log_level=log_level)
    yield
    app.state.router_instance = None

app = FastAPI(title="Edge LLM Server", version="1.0", lifespan=lifespan)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    # Why: We access the router via the request's app state.
    router = request.app.state.router_instance
    body = await request.json()
    chat_request = ChatRequest(**body)

    try:
        provider = router.get_provider(chat_request.model)
        generator = provider.create_chat_completion(chat_request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if chat_request.stream:
        # Pass the provider's class name to handle different generator outputs
        return StreamingResponse(stream_response_generator(generator, chat_request.model, type(provider).__name__), media_type="text/event-stream")
    else:
        return await non_stream_response(generator, chat_request.model, type(provider).__name__)


def stream_response_generator(generator, model_id, provider_name, log_level):
    request_id, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())

    if provider_name == "MLXProvider":
        # MLX generator yields (text_chunk, from_draft_bool, usage_object)
        for text_chunk, from_draft, _ in generator:
            if log_level <= logging.DEBUG: print("✅" if from_draft else "🔥", end="", flush=True)
            response = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model_id, "choices": [{"delta": {"content": text_chunk}}]}
            yield f"data: {json.dumps(response)}\n\n"
    else: # LlamaCppProvider
        # llama-cpp-python yields OpenAI-compatible chunks directly
        for chunk in generator:
            chunk['id'] = request_id; chunk['created'] = created; chunk['model'] = model_id
            yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"

async def non_stream_response(generator, model_id, provider_name):
    # Consumes the generator to create a single response object.
    full_response_chunks = list(generator)
    if not full_response_chunks: return JSONResponse(content={"error": "No content generated"}, status_code=500)

    if provider_name == "MLXProvider":
        full_text = "".join(chunk[0] for chunk in full_response_chunks)
        usage = {"prompt_tokens": 0, "completion_tokens": len(full_response_chunks), "total_tokens": 0}
    else: # LlamaCppProvider
        full_text = "".join(c['choices'][0]['delta'].get('content', '') for c in full_response_chunks if c['choices'])
        final_chunk = full_response_chunks[-1]
        usage = final_chunk.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    return {
        "id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()),
        "model": model_id, "choices": [{"message": {"role": "assistant", "content": full_text}}],
        "usage": usage,
    }
