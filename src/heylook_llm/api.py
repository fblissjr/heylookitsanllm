# src/heylook_llm/api.py
import json, uuid, time, logging, argparse, sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

from heylook_llm.router import ModelRouter
from heylook_llm.config import ChatRequest, PerformanceMetrics, ChatCompletionResponse
from heylook_llm.middleware.ollama import OllamaTranslator
from heylook_llm.utils import sanitize_request_for_debug, sanitize_dict_for_debug



@asynccontextmanager
async def lifespan(app: FastAPI):
    # The router is now initialized in server.py and passed in app.state
    yield
    logging.info("Server shut down.")

app = FastAPI(title="Hey Look It's an LLM Server", version="1.0.0", lifespan=lifespan)

# Initialize Ollama translator
ollama_translator = OllamaTranslator()

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
            logging.debug(f"ChatRequest: {sanitize_request_for_debug(chat_request)}")
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

# =============================================================================
# Ollama API Compatibility Endpoints
# =============================================================================

@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Ollama chat completions endpoint - translates to OpenAI format"""
    router = request.app.state.router_instance
    request_start_time = time.time()

    try:
        body = await request.json()
        logging.debug(f"Received Ollama chat request: {sanitize_dict_for_debug(body)}")

        # Translate Ollama request to OpenAI format
        openai_request_data = ollama_translator.translate_ollama_chat_to_openai(body)
        chat_request = ChatRequest(**openai_request_data)

        # Use existing OpenAI logic
        provider = router.get_provider(chat_request.model)
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching Ollama request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        generator = provider.create_chat_completion(chat_request)

        # Get non-streaming response (we don't support streaming yet)
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in generator:
            full_text += chunk.text
            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)

        # Create OpenAI-style response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_request.model,
            "choices": [{"message": {"role": "assistant", "content": full_text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        # Translate back to Ollama format
        ollama_response = ollama_translator.translate_openai_to_ollama_chat(
            openai_response,
            body.get("model", "default"),
            request_start_time
        )

        logging.debug(f"Returning Ollama chat response: {json.dumps(ollama_response, indent=2)}")
        return JSONResponse(content=ollama_response)

    except Exception as e:
        logging.error(f"Error in Ollama chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def ollama_generate(request: Request):
    """Ollama generate endpoint - translates to OpenAI format"""
    router = request.app.state.router_instance
    request_start_time = time.time()

    try:
        body = await request.json()
        logging.debug(f"Received Ollama generate request: {sanitize_dict_for_debug(body)}")

        # Translate Ollama request to OpenAI format
        openai_request_data = ollama_translator.translate_ollama_generate_to_openai(body)
        chat_request = ChatRequest(**openai_request_data)

        # Use existing OpenAI logic
        provider = router.get_provider(chat_request.model)
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching Ollama request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        generator = provider.create_chat_completion(chat_request)

        # Get non-streaming response
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in generator:
            full_text += chunk.text
            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)

        # Create OpenAI-style response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_request.model,
            "choices": [{"message": {"role": "assistant", "content": full_text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        # Translate back to Ollama format
        ollama_response = ollama_translator.translate_openai_to_ollama_generate(
            openai_response,
            body.get("model", "default"),
            request_start_time
        )

        logging.debug(f"Returning Ollama generate response: {json.dumps(ollama_response, indent=2)}")
        return JSONResponse(content=ollama_response)

    except Exception as e:
        logging.error(f"Error in Ollama generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tags")
async def ollama_tags(request: Request):
    """Ollama list models endpoint - translates to OpenAI format"""
    router = request.app.state.router_instance

    try:
        # Get models from router
        models_data = [{"id": model_id, "object": "model", "owned_by": "user"} for model_id in router.list_available_models()]
        openai_response = {"object": "list", "data": models_data}

        # Translate to Ollama format
        ollama_response = ollama_translator.translate_openai_models_to_ollama(openai_response)

        logging.debug(f"Returning Ollama tags response: {json.dumps(ollama_response, indent=2)}")
        return JSONResponse(content=ollama_response)

    except Exception as e:
        logging.error(f"Error in Ollama tags endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/show")
async def ollama_show(request: Request):
    """Ollama show model information endpoint"""
    router = request.app.state.router_instance

    try:
        body = await request.json()
        model_name = body.get("model")
        verbose = body.get("verbose", False)

        if not model_name:
            raise HTTPException(status_code=422, detail="Missing required parameter 'model'")

        # Check if model exists
        available_models = router.list_available_models()
        if model_name not in available_models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        # Get model config
        model_config = router.app_config.get_model_config(model_name)

        # Build Ollama show response
        show_response = {
            "modelfile": f"# Modelfile for {model_name}\nFROM {model_config.config.model_path}\n",
            "parameters": "",  # We don't expose internal parameters
            "template": "",    # We don't use custom templates
            "details": {
                "parent_model": "",
                "format": "mlx" if model_config.provider == "mlx" else "gguf",
                "family": "unknown",
                "families": [],
                "parameter_size": "unknown",
                "quantization_level": "unknown"
            },
            "model_info": {
                "general.architecture": "unknown",
                "general.file_type": 0,
                "general.parameter_count": 0
            },
            "capabilities": ["completion"]
        }

        # Add vision capability if it's a vision model
        if hasattr(model_config.config, 'vision') and getattr(model_config.config, 'vision', False):
            show_response["capabilities"].append("vision")

        # Add verbose info if requested
        if verbose:
            show_response["model_info"].update({
                "tokenizer.ggml.tokens": [],
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.token_type": []
            })

        logging.debug(f"Returning Ollama show response for {model_name}: {json.dumps(show_response, indent=2)}")
        return JSONResponse(content=show_response)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama show endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/version")
async def ollama_version():
    """Ollama version endpoint"""
    return JSONResponse(content={"version": "0.5.1"})

@app.get("/api/ps")
async def ollama_ps(request: Request):
    """Ollama list running models endpoint"""
    router = request.app.state.router_instance

    try:
        # Get currently loaded models from router cache
        running_models = []

        # Check which models are currently loaded in the router cache
        for model_id in router.providers.keys():
            try:
                model_config = router.app_config.get_model_config(model_id)
                if model_config:
                    running_model = {
                        "name": f"{model_id}:latest",
                        "model": f"{model_id}:latest",
                        "size": 0,  # We don't track actual size
                        "digest": "unknown",
                        "details": {
                            "parent_model": "",
                            "format": "mlx" if model_config.provider == "mlx" else "gguf",
                            "family": "unknown",
                            "families": [],
                            "parameter_size": "unknown",
                            "quantization_level": "unknown"
                        },
                        "expires_at": "2024-12-31T23:59:59Z",  # Placeholder
                        "size_vram": 0
                    }
                    running_models.append(running_model)
            except Exception as e:
                logging.warning(f"Error getting config for loaded model {model_id}: {e}")
                continue

        return JSONResponse(content={"models": running_models})

    except Exception as e:
        logging.error(f"Error in Ollama ps endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embed")
async def ollama_embed(request: Request):
    """Ollama generate embeddings endpoint"""
    try:
        body = await request.json()
        model_name = body.get("model")
        input_text = body.get("input")

        if not model_name or not input_text:
            raise HTTPException(status_code=422, detail="Missing required parameters 'model' and 'input'")

        # For now, return a placeholder response since we don't have embedding models
        # In the future, this could be extended to support actual embedding generation
        placeholder_embedding = [0.0] * 384  # Common embedding dimension

        embed_response = {
            "model": model_name,
            "embeddings": [placeholder_embedding] if isinstance(input_text, str) else [placeholder_embedding] * len(input_text),
            "total_duration": 1000000,  # 1ms in nanoseconds
            "load_duration": 0,
            "prompt_eval_count": len(input_text) if isinstance(input_text, str) else sum(len(t) for t in input_text)
        }

        return JSONResponse(content=embed_response)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama embed endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/create")
async def ollama_create(request: Request):
    """Ollama create model endpoint - Not implemented yet"""
    try:
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            raise HTTPException(status_code=422, detail="Missing required parameter 'model'")

        # For now, return a not implemented response
        # This would need to integrate with the model creation system
        raise HTTPException(status_code=501, detail="Model creation not implemented yet")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama create endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/copy")
async def ollama_copy(request: Request):
    """Ollama copy model endpoint - Not implemented yet"""
    try:
        body = await request.json()
        source = body.get("source")
        destination = body.get("destination")

        if not source or not destination:
            raise HTTPException(status_code=422, detail="Missing required parameters 'source' and 'destination'")

        # For now, return a not implemented response
        raise HTTPException(status_code=501, detail="Model copying not implemented yet")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama copy endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete")
async def ollama_delete(request: Request):
    """Ollama delete model endpoint - Not implemented yet"""
    try:
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            raise HTTPException(status_code=422, detail="Missing required parameter 'model'")

        # For now, return a not implemented response
        raise HTTPException(status_code=501, detail="Model deletion not implemented yet")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama delete endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pull")
async def ollama_pull(request: Request):
    """Ollama pull model endpoint - Not implemented yet"""
    try:
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            raise HTTPException(status_code=422, detail="Missing required parameter 'model'")

        # For now, return a not implemented response
        raise HTTPException(status_code=501, detail="Model pulling not implemented yet")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama pull endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/push")
async def ollama_push(request: Request):
    """Ollama push model endpoint - Not implemented yet"""
    try:
        body = await request.json()
        model_name = body.get("model")

        if not model_name:
            raise HTTPException(status_code=422, detail="Missing required parameter 'model'")

        # For now, return a not implemented response
        raise HTTPException(status_code=501, detail="Model pushing not implemented yet")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in Ollama push endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.head("/api/blobs/{digest}")
async def ollama_blob_exists(digest: str):
    """Ollama check blob exists endpoint - Not implemented yet"""
    # For now, return not found
    raise HTTPException(status_code=404, detail="Blob management not implemented yet")

@app.post("/api/blobs/{digest}")
async def ollama_blob_upload(digest: str, request: Request):
    """Ollama upload blob endpoint - Not implemented yet"""
    # For now, return not implemented
    raise HTTPException(status_code=501, detail="Blob management not implemented yet")

@app.get("/")
async def root():
    """Root endpoint showing available APIs"""
    return {
        "message": "Edge LLM Server",
        "version": "1.0.0",
        "apis": {
            "openai": {
                "models": "/v1/models",
                "chat": "/v1/chat/completions"
            },
            "ollama": {
                "core": {
                    "models": "/api/tags",
                    "chat": "/api/chat",
                    "generate": "/api/generate",
                    "show": "/api/show",
                    "version": "/api/version",
                    "ps": "/api/ps",
                    "embed": "/api/embed"
                },
                "model_management": {
                    "create": "/api/create (501 - not implemented)",
                    "copy": "/api/copy (501 - not implemented)",
                    "delete": "/api/delete (501 - not implemented)",
                    "pull": "/api/pull (501 - not implemented)",
                    "push": "/api/push (501 - not implemented)"
                },
                "blob_management": {
                    "check": "/api/blobs/{digest} (HEAD, 404 - not implemented)",
                    "upload": "/api/blobs/{digest} (POST, 501 - not implemented)"
                }
            }
        },
        "compatibility": {
            "implemented_endpoints": 7,
            "total_ollama_endpoints": 14,
            "coverage": "50%"
        }
    }
