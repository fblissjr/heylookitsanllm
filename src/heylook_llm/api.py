# src/heylook_llm/api.py
import json, uuid, time, logging, argparse, sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

from heylook_llm.router import ModelRouter
from heylook_llm.config import ChatRequest, PerformanceMetrics, ChatCompletionResponse



@asynccontextmanager
async def lifespan(app: FastAPI):
    # The router is now initialized in server.py and passed in app.state
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
        logging.warning(f"Request validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    try:
        provider = router.get_provider(chat_request.model)
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")
            logging.debug(f"ChatRequest: {chat_request.model_dump_json(indent=2)}")
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
        return await non_stream_response(generator, chat_request, router)

def stream_response_generator(generator, model_id):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    for chunk in generator:
        if not chunk.text:
            continue
        response = {
            "id": request_id, 
            "object": "chat.completion.chunk", 
            "created": created_time, 
            "model": model_id, 
            "choices": [{"delta": {"content": chunk.text}}]
        }
        yield f"data: {json.dumps(response)}\n\n"
    yield "data: [DONE]\n\n"

async def non_stream_response(generator, chat_request: ChatRequest, router):
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    
    for chunk in generator:
        full_text += chunk.text
        prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
        completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)

    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=chat_request.model,
        choices=[{"message": {"role": "assistant", "content": full_text}}],
        usage=usage_dict
    )
    
    # Optional debug logging
    if router.log_level <= logging.DEBUG:
        logging.debug(f"Full non-stream response: {response.model_dump_json(indent=2)}")
        
    return response
