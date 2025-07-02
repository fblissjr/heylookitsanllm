# src/edge_llm/api.py

import json, uuid, time, logging, argparse, sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from rich.console import Console

from edge_llm.router import ModelRouter
from edge_llm.config import ChatRequest, PerformanceMetrics, ChatCompletionResponse

router_instance = None
console = Console()

def _parse_app_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_known_args(sys.argv[1:])[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    args = _parse_app_args()
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    app.state.router_instance = ModelRouter(config_path="models.yaml", log_level=log_level)
    yield
    app.state.router_instance = None

app = FastAPI(title="Edge LLM Server", version="1.0.0", lifespan=lifespan)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    router = request.app.state.router_instance
    body = await request.json()
    chat_request = ChatRequest(**body)
    try:
        provider = router.get_provider(chat_request.model)
        generator = provider.create_chat_completion(chat_request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    log_level = router.log_level

    if chat_request.stream:
        return StreamingResponse(stream_response_generator(chat_request, generator, log_level), media_type="text/event-stream")
    else:
        return await non_stream_response(chat_request, generator, log_level)

def _log_performance(start_time, first_token_time, tokens_generated, final_result):
    end_time = time.time()
    total_time = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else total_time
    gen_time = end_time - (first_token_time or start_time)
    tok_per_sec = (tokens_generated - 1) / gen_time if gen_time > 0 and tokens_generated > 1 else 0.0

    prompt_tps = getattr(final_result, 'prompt_tps', 0.0)
    peak_memory = getattr(final_result, 'peak_memory', 0.0)

    logging.info(
        f"\n--- Request Performance ---\n"
        f"  - Time to First Token: {ttft:.3f} s\n"
        f"  - Prompt TPS:          {prompt_tps:.2f} tok/s\n"
        f"  - Generation TPS:      {tok_per_sec:.2f} tok/s\n"
        f"  - Peak Memory:         {peak_memory:.3f} GB\n"
        f"---------------------------\n"
    )
    return PerformanceMetrics(prompt_tps=prompt_tps, generation_tps=tok_per_sec, peak_memory_gb=peak_memory)

def stream_response_generator(request: ChatRequest, generator, log_level: int):
    request_id, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    start_time, first_token_time, last_result = time.time(), None, None
    tokens_generated = 0

    try:
        for result in generator:
            if first_token_time is None: first_token_time = time.time()
            last_result = result

            if hasattr(result, 'text'):
                text_chunk = result.text
                from_draft = getattr(result, 'from_draft', False)
                if log_level <= logging.DEBUG: console.print("[green]✅[/green]" if from_draft else "[red]🔥[/red]", end="")
            else:
                text_chunk = result['choices'][0]['delta'].get('content', '')

            if text_chunk:
                tokens_generated += 1
                response = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"delta": {"content": text_chunk}}]}
                yield f"data: {json.dumps(response)}\n\n"

    finally:
        if log_level <= logging.INFO and hasattr(last_result, 'prompt_tps'):
            _log_performance(start_time, first_token_time, tokens_generated, last_result)
        if log_level <= logging.DEBUG: console.print()
        yield "data: [DONE]\n\n"

async def non_stream_response(request: ChatRequest, generator, log_level: int):
    start_time = time.time()
    full_text, last_result = "", None
    tokens_generated = 0

    for result in generator:
        tokens_generated += 1
        last_result = result
        if hasattr(result, 'text'):
            full_text += result.text
        else:
            full_text += result['choices'][0]['delta'].get('content', '')

    perf_metrics = None
    if request.include_performance and hasattr(last_result, 'prompt_tps'):
        perf_metrics = _log_performance(start_time, None, tokens_generated, last_result)

    prompt_tokens = getattr(last_result, 'prompt_tokens', 0) if hasattr(last_result, 'prompt_tokens') else last_result.get("usage", {}).get("prompt_tokens", 0)
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": tokens_generated, "total_tokens": prompt_tokens + tokens_generated}

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}", object="chat.completion", created=int(time.time()),
        model=request.model, choices=[{"message": {"role": "assistant", "content": full_text}}],
        usage=usage, performance=perf_metrics
    )
