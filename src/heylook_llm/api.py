# src/heylook_llm/api.py
import json, uuid, time, logging, argparse, sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

from heylook_llm.router import ModelRouter
from heylook_llm.config import ChatRequest, PerformanceMetrics, ChatCompletionResponse

def _parse_app_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--model-id", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

@asynccontextmanager
async def lifespan(app: FastAPI):
    args = _parse_app_args()
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    app.state.router_instance = ModelRouter(
        config_path="models.yaml",
        log_level=log_level,
        initial_model_id=args.model_id
    )
    yield
    logging.info("Server shut down.")

app = FastAPI(title="Edge LLM Server", version="1.0.0", lifespan=lifespan)

@app.get("/v1/models")
async def list_models(request: Request):
    router = request.app.state.router_instance
    models_data = [{"id": model_id, "object": "model", "owned_by": "user"} for model_id in router.list_available_models()]
    return {"object": "list", "data": models_data}

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    router = request.app.state.router_instance
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
    except Exception as e:
        logging.error(f"Request validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))

    try:
        provider = router.get_provider(chat_request.model)
        generator = provider.create_chat_completion(chat_request)
    except Exception as e:
        logging.error(f"Failed to get provider or create generator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if chat_request.stream:
        return StreamingResponse(
            stream_response_generator(generator, chat_request.model),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_response(generator, chat_request)

def stream_response_generator(generator, model_id):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    for chunk in generator:
        text_chunk = ""
        if hasattr(chunk, 'text'): # MLX
            text_chunk = chunk.text
        elif isinstance(chunk, dict): # Llama.cpp
            text_chunk = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')

        if text_chunk:
            response = {"id": request_id, "object": "chat.completion.chunk", "created": created_time, "model": model_id, "choices": [{"delta": {"content": text_chunk}}]}
            yield f"data: {json.dumps(response)}\n\n"
    yield "data: [DONE]\n\n"

async def non_stream_response(generator, chat_request: ChatRequest):
    full_text = ""
    last_usage = None
    prompt_tokens = 0
    gen_tokens = 0
    for chunk in generator:
        if hasattr(chunk, 'text'): # MLX
            full_text += chunk.text
            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            gen_tokens = getattr(chunk, 'generation_tokens', gen_tokens)
        elif isinstance(chunk, dict): # Llama.cpp
            full_text += chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
            if usage := chunk.get("usage"):
                last_usage = usage

    usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if last_usage: # from llama.cpp
        usage_dict.update(last_usage)
        usage_dict["total_tokens"] = usage_dict.get("prompt_tokens", 0) + usage_dict.get("completion_tokens", 0)
    else: # from mlx
        usage_dict["prompt_tokens"] = prompt_tokens
        usage_dict["completion_tokens"] = gen_tokens
        usage_dict["total_tokens"] = prompt_tokens + gen_tokens

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=chat_request.model,
        choices=[{"message": {"role": "assistant", "content": full_text}}],
        usage=usage_dict
    )
