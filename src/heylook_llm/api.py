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
    app.state.router_instance = None
    logging.info("Server shut down.")

app = FastAPI(title="Edge LLM Server", version="1.0.0", lifespan=lifespan)

@app.get("/v1/models")
async def list_models(request: Request):
    """List available models in OpenAI format."""
    router = request.app.state.router_instance
    try:
        available_models = router.list_available_models()

        # Format in OpenAI-compatible structure
        models_data = []
        for model_id in available_models:
            model_config = router.app_config.get_model_config(model_id)
            models_data.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "heylookllm",
                "permission": [],
                "root": model_id,
                "parent": None,
                # Add metadata about the model
                "provider": model_config.provider if model_config else "unknown",
                "vision": getattr(model_config.config, 'vision', False) if model_config else False
            })

        return {
            "object": "list",
            "data": models_data
        }
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    router = request.app.state.router_instance
    body = await request.json()

    try:
        chat_request = ChatRequest(**body)
    except Exception as e:
        # Handle Pydantic validation errors gracefully
        logging.error(f"Request validation failed: {e}")
        error_detail = str(e)

        # Provide more helpful error messages
        if "Messages list cannot be empty" in error_detail:
            error_detail = "Messages list cannot be empty. Please provide at least one message."
        elif "Input should be greater than 0" in error_detail and "max_tokens" in error_detail:
            error_detail = "max_tokens must be greater than 0."
        elif "Value error" in error_detail:
            error_detail = f"Invalid request: {error_detail}"

        raise HTTPException(status_code=422, detail=error_detail)

    try:
        # Get the provider - this is where the error occurs
        provider = router.get_provider(chat_request.model)

        # Double-check that we got a valid provider
        if provider is None:
            available_models = router.list_available_models()
            raise ValueError(f"Failed to load provider for model '{chat_request.model}'. Available models: {available_models}")

        # Create the generator
        generator = provider.create_chat_completion(chat_request.model_dump())

        # Validate that generator is not None
        if generator is None:
            raise ValueError(f"Provider for model '{chat_request.model}' returned None generator")

    except ValueError as e:
        # Model not found or provider creation failed
        logging.error(f"Model/provider error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Other errors during provider/generator creation
        logging.error(f"Failed to create generator: {e}")
        import traceback
        logging.debug(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    log_level = router.log_level

    try:
        if chat_request.stream:
            return StreamingResponse(
                stream_response_generator(chat_request, generator, provider, log_level),
                media_type="text/event-stream"
            )
        else:
            return await non_stream_response(chat_request, generator, provider, log_level)
    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def _log_performance(start_time, first_token_time, tokens_generated, final_result):
    """Log performance metrics based on the final result object."""
    try:
        end_time = time.time()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        gen_time = end_time - (first_token_time or start_time)
        tok_per_sec = (tokens_generated - 1) / gen_time if gen_time > 0 and tokens_generated > 1 else 0.0

        # Extract metrics from the result object (works for both MLX and llama.cpp)
        prompt_tps = 0.0
        peak_memory = 0.0

        if final_result:
            if hasattr(final_result, 'prompt_tps'):  # MLX GenerationResponse
                prompt_tps = getattr(final_result, 'prompt_tps', 0.0)
                peak_memory = getattr(final_result, 'peak_memory', 0.0)
            elif isinstance(final_result, dict):  # llama.cpp style response
                usage = final_result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                prompt_tps = prompt_tokens / max(ttft, 0.001) if prompt_tokens > 0 else 0.0
                peak_memory = 0.0  # llama.cpp doesn't provide this

        logging.info(
            f"\n--- Request Performance ---\n"
            f"  - Time to First Token: {ttft:.3f} s\n"
            f"  - Prompt TPS:          {prompt_tps:.2f} tok/s\n"
            f"  - Generation TPS:      {tok_per_sec:.2f} tok/s\n"
            f"  - Peak Memory:         {peak_memory:.3f} GB\n"
            f"---------------------------\n"
        )
        return PerformanceMetrics(prompt_tps=prompt_tps, generation_tps=tok_per_sec, peak_memory_gb=peak_memory)

    except Exception as e:
        logging.warning(f"Performance logging failed: {e}")
        # Return minimal metrics to avoid further errors
        return PerformanceMetrics(prompt_tps=0.0, generation_tps=0.0, peak_memory_gb=0.0)

def stream_response_generator(chat_request: ChatRequest, generator, provider, log_level: int):
    """Handles streaming responses, yielding data in the Server-Sent Events (SSE) format."""
    request_id, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    start_time, first_token_time, last_result = time.time(), None, None
    tokens_generated = 0

    try:
        for result in generator:
            if first_token_time is None:
                first_token_time = time.time()
            last_result = result

        # Handle different response types from different providers
        if hasattr(result, 'text'):  # MLXProvider (GenerationResponse)
            text_chunk = result.text
            if log_level <= logging.DEBUG and text_chunk:
                print(text_chunk, end="", flush=True)
        else:  # LlamaCppProvider
            text_chunk = result.get('choices', [{}])[0].get('delta', {}).get('content', '')
            if log_level <= logging.DEBUG and text_chunk:
                print(text_chunk, end="", flush=True)

            if text_chunk:
                tokens_generated += 1
                response = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": chat_request.model or "unknown",
                    "choices": [{"delta": {"content": text_chunk}}]
                }
                yield f"data: {json.dumps(response)}\n\n"

    except Exception as e:
        logging.error(f"Error in stream generation: {e}")
        error_response = {
            "id": request_id,
            "object": "error",
            "error": {"message": str(e), "type": "generation_error"}
        }
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        # Log performance if we have meaningful results
        if log_level <= logging.INFO and last_result and tokens_generated > 0:
            _log_performance(start_time, first_token_time, tokens_generated, last_result)
        yield "data: [DONE]\n\n"

async def non_stream_response(chat_request: ChatRequest, generator, provider, log_level: int):
    """Handle non-streaming chat completion requests."""
    start_time = time.time()
    full_text, last_result = "", None
    tokens_generated = 0

    try:
        for result in generator:
            tokens_generated += 1
            last_result = result

            if hasattr(result, 'text'):  # MLXProvider (GenerationResponse)
                full_text += result.text
            else:  # LlamaCppProvider
                content = result.get('choices', [{}])[0].get('delta', {}).get('content', '')
                if content:
                    full_text += content

        # Extract usage information
        if hasattr(last_result, 'prompt_tokens'):  # MLXProvider
            prompt_tokens = getattr(last_result, 'prompt_tokens', 0)
        else:  # LlamaCppProvider
            prompt_tokens = last_result.get("usage", {}).get("prompt_tokens", 0)

        # Build performance metrics if requested
        perf_metrics = None
        if chat_request.include_performance and last_result:
            perf_metrics = _log_performance(start_time, None, tokens_generated, last_result)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens_generated,
            "total_tokens": prompt_tokens + tokens_generated
        }

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=chat_request.model or "unknown",
            choices=[{"message": {"role": "assistant", "content": full_text}}],
            usage=usage,
            performance=perf_metrics
        )

    except Exception as e:
        logging.error(f"Error in non-stream generation: {e}")
        # Build performance metrics if requested
        perf_metrics = None
        if chat_request.include_performance and last_result:
            perf_metrics = _log_performance(start_time, None, tokens_generated, last_result)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens_generated,
            "total_tokens": prompt_tokens + tokens_generated
        }

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=chat_request.model or "unknown",
            choices=[{"message": {"role": "assistant", "content": full_text}}],
            usage=usage,
            performance=perf_metrics
        )

    except Exception as e:
        logging.error(f"Error in non-stream generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
