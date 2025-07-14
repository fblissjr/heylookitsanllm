# src/heylook_llm/api.py
import json, uuid, time, logging, argparse, sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

from heylook_llm.router import ModelRouter
from heylook_llm.config import ChatRequest, PerformanceMetrics, ChatCompletionResponse
from heylook_llm.middleware.ollama import OllamaTranslator
from heylook_llm.utils import sanitize_request_for_debug, sanitize_dict_for_debug, log_request_start, log_request_stage, log_request_complete, log_full_request_details, log_request_summary, log_response_summary
from heylook_llm.api_batch import BatchChatRequests, create_batch_chat_completions



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
    request_id = f"req-{uuid.uuid4()}"

    request_start_time = time.time()
    
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
        # Add start time for processing time calculation
        chat_request._start_time = request_start_time
    except Exception as e:
        logging.warning(f"Request validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    # Check if this is a batch processing request
    if chat_request.processing_mode and chat_request.processing_mode != "conversation":
        # Use batch processor for non-conversation modes
        from heylook_llm.batch_processor import BatchProcessor, ProcessingMode
        
        logging.info(f"Processing batch request with mode: {chat_request.processing_mode}")
        batch_processor = BatchProcessor(router)
        
        # Convert to batch request format
        from heylook_llm.batch_processor import BatchChatRequest
        batch_request = BatchChatRequest(
            model=chat_request.model,
            messages=chat_request.messages,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            top_k=chat_request.top_k,
            min_p=chat_request.min_p,
            repetition_penalty=chat_request.repetition_penalty,
            repetition_context_size=chat_request.repetition_context_size,
            max_tokens=chat_request.max_tokens,
            stream=False,  # Batch doesn't support streaming
            seed=chat_request.seed,
            processing_mode=ProcessingMode(chat_request.processing_mode),
            return_individual=chat_request.return_individual if chat_request.return_individual is not None else True,
            include_timing=chat_request.include_timing if chat_request.include_timing is not None else False
        )
        
        # Process batch and return
        batch_response = await batch_processor.process_batch_request(batch_request)
        return batch_response.model_dump()

    # Standard processing for conversation mode or no processing_mode specified
    # Start real-time logging
    log_request_start(request_id, chat_request.model)

    # Analyze request for image metadata
    request_dict = chat_request.model_dump()
    from heylook_llm.utils import _analyze_images_in_request
    image_stats = _analyze_images_in_request(request_dict)
    
    # Log enhanced request summary
    log_request_summary(
        request_id, 
        chat_request.model, 
        has_images=image_stats['count'] > 0,
        image_count=image_stats['count'],
        total_image_size=image_stats['total_size']
    )
    
    # Log full request details if DEBUG level
    if router.log_level <= logging.DEBUG:
        log_full_request_details(request_id, chat_request)

    try:
        log_request_stage(request_id, "routing")
        provider = router.get_provider(chat_request.model)

        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        log_request_stage(request_id, "generating")
        generator = provider.create_chat_completion(chat_request)

    except Exception as e:
        logging.error(f"Failed to get provider or create generator: {e}", exc_info=True)
        log_request_complete(request_id, success=False, error_msg=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if chat_request.stream:
        return StreamingResponse(
            stream_response_generator(generator, chat_request.model, request_id),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_response(generator, chat_request, router, request_id, request_start_time)

def stream_response_generator(generator, model_id, request_id):
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    token_count = 0

    log_request_stage(request_id, "streaming")

    for chunk in generator:
        if not chunk.text:
            continue

        token_count += 1

        # Update token count periodically
        if token_count % 10 == 0:  # Update every 10 tokens for streaming
            from heylook_llm.utils import log_token_update
            log_token_update(request_id, token_count)

        response = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [{"delta": {"content": chunk.text}}]
        }
        yield f"data: {json.dumps(response)}\n\n"

    # Log completion
    log_request_complete(request_id, success=True)
    yield "data: [DONE]\n\n"

async def non_stream_response(generator, chat_request: ChatRequest, router, request_id, request_start_time):
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    token_count = 0

    log_request_stage(request_id, "processing_response")

    for chunk in generator:
        full_text += chunk.text
        token_count += 1

        # Update token count periodically for long responses
        if token_count % 25 == 0:
            from heylook_llm.utils import log_token_update
            log_token_update(request_id, token_count)

        prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
        completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)

    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens or token_count,  # Fallback to our count
        "total_tokens": (prompt_tokens or 0) + (completion_tokens or token_count)
    }

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=chat_request.model or "unknown",
        choices=[{"message": {"role": "assistant", "content": full_text}}],
        usage=usage_dict
    )

    # Calculate processing time
    processing_time = time.time() - request_start_time
    
    # Log response summary
    log_response_summary(
        request_id, 
        len(full_text), 
        token_count=completion_tokens or token_count,
        processing_time=processing_time
    )
    
    # Log full response details if DEBUG level
    if router.log_level <= logging.DEBUG:
        log_full_request_details(request_id, chat_request, full_text)
        logging.debug(f"Full non-stream response: {response.model_dump_json(indent=2)}")

    # Log successful completion
    log_request_complete(request_id, success=True)
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
        
        # Analyze request for image metadata before translation
        from heylook_llm.utils import _analyze_images_in_dict
        image_stats = _analyze_images_in_dict(body)
        
        # Enhanced logging with image info
        logging.info(f"📥 OLLAMA CHAT REQUEST | Images: {image_stats['count']} ({image_stats['total_size']})")
        if router.log_level <= logging.DEBUG:
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

        # Enhanced response logging
        processing_time = time.time() - request_start_time
        response_length = len(full_text)
        logging.info(f"📤 OLLAMA CHAT RESPONSE | {response_length} chars | {processing_time:.2f}s")
        
        if router.log_level <= logging.DEBUG:
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
        
        # Analyze request for image metadata before translation
        from heylook_llm.utils import _analyze_images_in_dict
        image_stats = _analyze_images_in_dict(body)
        
        # Enhanced logging with image info
        logging.info(f"📥 OLLAMA GENERATE REQUEST | Images: {image_stats['count']} ({image_stats['total_size']})")
        if router.log_level <= logging.DEBUG:
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

        # Enhanced response logging
        processing_time = time.time() - request_start_time
        response_length = len(full_text)
        logging.info(f"📤 OLLAMA GENERATE RESPONSE | {response_length} chars | {processing_time:.2f}s")
        
        if router.log_level <= logging.DEBUG:
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
