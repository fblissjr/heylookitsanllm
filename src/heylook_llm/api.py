# src/heylook_llm/api.py
import uuid, time, logging, argparse, sys, asyncio
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, AsyncGenerator

# Use fast JSON implementation
from heylook_llm.optimizations import fast_json as json

# Custom OpenAPI documentation
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from heylook_llm.router import ModelRouter
from heylook_llm.config import ChatRequest, PerformanceMetrics, ChatCompletionResponse
from heylook_llm.providers.common.performance_monitor import performance_monitor
from heylook_llm.middleware.ollama import OllamaTranslator
from heylook_llm.utils import sanitize_request_for_debug, sanitize_dict_for_debug, log_request_start, log_request_stage, log_request_complete, log_full_request_details, log_request_summary, log_response_summary



@asynccontextmanager
async def lifespan(app: FastAPI):
    # The router is now initialized in server.py and passed in app.state
    yield
    logging.info("Server shut down.")

app = FastAPI(
    title="HeylookLLM - High-Performance Local LLM Server",
    version="1.0.1",
    description="A unified, high-performance API server for local LLM inference with dual OpenAI and Ollama compatibility",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "OpenAI API",
            "description": "OpenAI-compatible endpoints for maximum compatibility with existing tools and libraries"
        },
        {
            "name": "Ollama API", 
            "description": "Ollama-compatible endpoints for drop-in replacement of Ollama server"
        },
        {
            "name": "Monitoring",
            "description": "Performance monitoring and server status endpoints"
        }
    ]
)

# Initialize Ollama translator
ollama_translator = OllamaTranslator()

@app.get("/v1/models",
    summary="List Available Models",
    description="""
List all language models currently available on this server.

**Use this endpoint to:**
- Discover which models are loaded and ready for inference
- Verify a specific model is available before making requests
- Get model IDs for use in completion requests

**Returns:**
- Model IDs (e.g., "qwen2.5-coder-1.5b-instruct-4bit")
- OpenAI-compatible model objects
- Only shows models marked as `enabled: true` in models.yaml
    """,
    response_description="List of available models in OpenAI-compatible format",
    tags=["OpenAI API"]
)
async def list_models(request: Request):
    """Get the list of available models in OpenAI format."""
    router = request.app.state.router_instance
    models_data = [{"id": model_id, "object": "model", "owned_by": "user"} for model_id in router.list_available_models()]
    return {"object": "list", "data": models_data}

@app.post("/v1/chat/completions",
    summary="Create Chat Completion",
    description="""
Generate text completions from chat messages using the specified model.

**Key Features:**
- 🚀 Automatic model loading with LRU cache (max 2 models)
- 📸 Vision model support with base64 images
- 🌊 Streaming responses (Server-Sent Events)
- 🎯 Batch processing for multiple prompts
- 🎲 Reproducible generation with seed parameter
- ⚡ Metal-optimized inference for MLX models

**Special Parameters:**
- `processing_mode`: Control batch behavior ("sequential", "parallel", "conversation")
- `return_individual`: Get separate responses for batch requests
- `include_timing`: Add performance metrics to response
- `stream`: Enable token-by-token streaming

**Performance Notes:**
- First request to a model includes loading time (~2-30s depending on size)
- Subsequent requests use cached model for instant inference
- Vision models process images in parallel for efficiency
    """,
    response_model=ChatCompletionResponse,
    response_description="Chat completion with generated text and token usage",
    tags=["OpenAI API"]
)
async def create_chat_completion(request: Request, chat_request: ChatRequest):
    router = request.app.state.router_instance
    request_id = f"req-{uuid.uuid4()}"

    request_start_time = time.time()
    
    try:
        # Add start time for processing time calculation
        chat_request._start_time = request_start_time
    except Exception as e:
        logging.warning(f"Request validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    # Check if this is a batch processing request
    if chat_request.processing_mode and chat_request.processing_mode != "conversation":
        # Use batch processor for non-conversation modes
        from heylook_llm.batch_processor import BatchProcessor, ProcessingMode
        
        logging.info(f"[API] Processing batch request with mode: {chat_request.processing_mode}")
        logging.info(f"[API] Number of messages in request: {len(chat_request.messages)}")
        
        # Check for any resize parameters in the request
        has_resize_params = any([chat_request.resize_max, chat_request.resize_width, chat_request.resize_height])
        if has_resize_params:
            logging.info(f"[API] Resize parameters found: max={chat_request.resize_max}, "
                       f"width={chat_request.resize_width}, height={chat_request.resize_height}, "
                       f"quality={chat_request.image_quality}, preserve_alpha={chat_request.preserve_alpha}")
            
            # Apply resizing to images in messages
            from heylook_llm.utils_resize import process_image_url_with_resize
            
            for msg in chat_request.messages:
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if hasattr(part, 'type') and part.type == 'image_url' and hasattr(part, 'image_url'):
                            original_url = part.image_url.url
                            resized_url = process_image_url_with_resize(
                                original_url,
                                resize_max=chat_request.resize_max,
                                resize_width=chat_request.resize_width,
                                resize_height=chat_request.resize_height,
                                image_quality=chat_request.image_quality or 85,
                                preserve_alpha=chat_request.preserve_alpha or False
                            )
                            if resized_url != original_url:
                                part.image_url.url = resized_url
                                logging.info(f"[API] Image resized before processing")
        
        for idx, msg in enumerate(chat_request.messages):
            if isinstance(msg.content, list):
                parts_info = []
                for part in msg.content:
                    if hasattr(part, 'type'):
                        if part.type == 'text':
                            parts_info.append(f"text: '{part.text[:30]}...'")
                        elif part.type == 'image_url':
                            parts_info.append("image")
                logging.info(f"[API] Message {idx}: {msg.role} - [{', '.join(parts_info)}]")
            else:
                logging.info(f"[API] Message {idx}: {msg.role} - {str(msg.content)[:50]}...")
        
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
    # Check for resize parameters in standard mode too
    has_resize_params = any([chat_request.resize_max, chat_request.resize_width, chat_request.resize_height])
    if has_resize_params:
        logging.info(f"[API] Resize parameters found in standard mode: max={chat_request.resize_max}, "
                   f"width={chat_request.resize_width}, height={chat_request.resize_height}")
        
        # Apply resizing to images
        from heylook_llm.utils_resize import process_image_url_with_resize
        
        for msg in chat_request.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if hasattr(part, 'type') and part.type == 'image_url' and hasattr(part, 'image_url'):
                        original_url = part.image_url.url
                        resized_url = process_image_url_with_resize(
                            original_url,
                            resize_max=chat_request.resize_max,
                            resize_width=chat_request.resize_width,
                            resize_height=chat_request.resize_height,
                            image_quality=chat_request.image_quality or 85,
                            preserve_alpha=chat_request.preserve_alpha or False
                        )
                        if resized_url != original_url:
                            part.image_url.url = resized_url
    
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
    
    # Log detailed image processing info if images are present
    if image_stats['count'] > 0:
        logging.info(f"[IMAGE PROCESSING] Request {request_id[:8]} contains {image_stats['count']} base64 images | "
                   f"Total size: {image_stats['total_size']} | Avg: {image_stats['avg_size']} | "
                   f"Processing via: {'BATCH' if chat_request.processing_mode else 'STANDARD'} API")
        if image_stats['sizes']:
            logging.info(f"[IMAGE PROCESSING] Individual sizes: {', '.join(image_stats['sizes'])}")
    
    # Log full request details if DEBUG level
    if router.log_level <= logging.DEBUG:
        log_full_request_details(request_id, chat_request)

    try:
        log_request_stage(request_id, "routing")
        # Run CPU-bound operations in thread pool
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)

        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        log_request_stage(request_id, "generating")
        # Run model generation in thread pool
        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request)

    except Exception as e:
        logging.error(f"Failed to get provider or create generator: {e}", exc_info=True)
        log_request_complete(request_id, success=False, error_msg=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if chat_request.stream:
        return StreamingResponse(
            stream_response_generator_async(generator, chat_request.model, request_id),
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

async def stream_response_generator_async(generator, model_id, request_id):
    """Async streaming response generator that runs generation in thread pool."""
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    token_count = 0

    log_request_stage(request_id, "streaming")

    # Create async wrapper for the synchronous generator
    async def chunk_generator():
        loop = asyncio.get_event_loop()
        
        # Convert the synchronous generator to async by running in thread
        def get_next_chunk():
            try:
                return next(generator)
            except StopIteration:
                return None
        
        while True:
            chunk = await loop.run_in_executor(None, get_next_chunk)
            if chunk is None:
                break
            yield chunk

    async for chunk in chunk_generator():
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

    # Process generation in thread pool to avoid blocking event loop
    def consume_generator():
        nonlocal full_text, prompt_tokens, completion_tokens, token_count
        for chunk in generator:
            full_text += chunk.text
            token_count += 1

            # Update token count periodically for long responses
            if token_count % 25 == 0:
                from heylook_llm.utils import log_token_update
                log_token_update(request_id, token_count)

            prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
            completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)
    
    await asyncio.to_thread(consume_generator)

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

@app.post("/api/chat",
    summary="Ollama Chat API",
    description="""
Ollama-compatible chat endpoint for seamless migration from Ollama.

**Translation Process:**
1. Receives Ollama-format request
2. Translates to OpenAI format internally
3. Uses same model routing and inference pipeline
4. Translates response back to Ollama format

**Supports:**
- ✅ All Ollama chat parameters
- ✅ Streaming responses
- ✅ Image inputs (base64)
- ✅ Same models as OpenAI endpoints

**Compatible with:**
- Ollama CLI
- Ollama Python/JS libraries
- Continue.dev
- Open WebUI
- Any Ollama-compatible tool
    """,
    response_description="Ollama-format chat response",
    tags=["Ollama API"]
)
async def ollama_chat(request: Request):
    """Ollama chat completions endpoint - translates to OpenAI format"""
    router = request.app.state.router_instance
    request_start_time = time.time()

    try:
        body = await request.json()
        
        # Analyze request for image metadata before translation
        from heylook_llm.utils import _analyze_images_in_dict
        image_stats = _analyze_images_in_dict(body)
        
        # Log request details including image info
        logging.info(f"📥 OLLAMA CHAT REQUEST | Images: {image_stats['count']} ({image_stats['total_size']})")
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Received Ollama chat request: {sanitize_dict_for_debug(body)}")

        # Translate Ollama request to OpenAI format
        openai_request_data = ollama_translator.translate_ollama_chat_to_openai(body)
        chat_request = ChatRequest(**openai_request_data)

        # Use existing OpenAI logic with thread pool
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching Ollama request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request)

        # Get non-streaming response in thread pool
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        def consume_generator():
            nonlocal full_text, prompt_tokens, completion_tokens
            for chunk in generator:
                full_text += chunk.text
                prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
                completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)
        
        await asyncio.to_thread(consume_generator)

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

        # Log response details
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
        
        # Log request details including image info
        logging.info(f"📥 OLLAMA GENERATE REQUEST | Images: {image_stats['count']} ({image_stats['total_size']})")
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Received Ollama generate request: {sanitize_dict_for_debug(body)}")

        # Translate Ollama request to OpenAI format
        openai_request_data = ollama_translator.translate_ollama_generate_to_openai(body)
        chat_request = ChatRequest(**openai_request_data)

        # Use existing OpenAI logic with thread pool
        provider = await asyncio.to_thread(router.get_provider, chat_request.model)
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Dispatching Ollama request to provider: {provider.__class__.__name__} for model '{chat_request.model}'")

        generator = await asyncio.to_thread(provider.create_chat_completion, chat_request)

        # Get non-streaming response in thread pool
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        def consume_generator():
            nonlocal full_text, prompt_tokens, completion_tokens
            for chunk in generator:
                full_text += chunk.text
                prompt_tokens = getattr(chunk, 'prompt_tokens', prompt_tokens)
                completion_tokens = getattr(chunk, 'generation_tokens', completion_tokens)
        
        await asyncio.to_thread(consume_generator)

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

        # Log response details
        processing_time = time.time() - request_start_time
        response_length = len(full_text)
        logging.info(f"📤 OLLAMA GENERATE RESPONSE | {response_length} chars | {processing_time:.2f}s")
        
        if router.log_level <= logging.DEBUG:
            logging.debug(f"Returning Ollama generate response: {json.dumps(ollama_response, indent=2)}")
        return JSONResponse(content=ollama_response)

    except Exception as e:
        logging.error(f"Error in Ollama generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tags",
    summary="List Models (Ollama Format)",
    description="""
Get available models in Ollama-compatible format.

**Equivalent to:** `ollama list`

**Returns:**
- Model names and tags
- Model sizes
- Modification times
- Model families

**Use Cases:**
- Check available models before running
- Verify model installation
- Get model metadata for Ollama clients
    """,
    response_description="List of models in Ollama format",
    tags=["Ollama API"]
)
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

@app.get("/v1/performance",
    summary="Get Performance Metrics",
    description="""
Retrieve detailed performance metrics from all model providers.

**Metrics Include:**
- 📊 Token generation rates (tokens/second)
- ⏱️ Time to first token (TTFT)
- 🚀 Model loading times
- 💾 Memory usage statistics
- 📈 Request processing times
- 🎯 Cache hit rates

**Per-Model Statistics:**
- Request count
- Average/peak token generation speed
- Total tokens generated
- Error rates

**Use Cases:**
- Performance monitoring dashboards
- Capacity planning
- Model comparison
- Troubleshooting slow inference
- Identifying bottlenecks
    """,
    response_description="Detailed performance metrics and summaries",
    tags=["Monitoring"]
)
async def performance_metrics():
    """Get performance metrics from all providers."""
    return {
        "metrics": performance_monitor.get_metrics(),
        "summary": performance_monitor.get_performance_summary()
    }


@app.get("/v1/capabilities",
    summary="Get Server Capabilities",
    description="""
Get detailed information about server capabilities and optimization options.

**Returns:**
- Available performance optimizations
- Supported features and endpoints
- Optimal usage recommendations
- API extensions

**Use this endpoint to:**
- Discover fast endpoints (like multipart upload)
- Check which optimizations are active
- Get recommendations for best performance
- Understand server limits and capabilities

**Client Integration:**
Clients should query this endpoint on startup to discover:
1. Whether to use `/v1/chat/completions/multipart` for images
2. Recommended batch sizes
3. Optimal request patterns
4. Available performance features
    """,
    response_description="Server capabilities and optimization details",
    tags=["Monitoring"]
)
async def get_capabilities(request: Request):
    """Get server capabilities and optimization options."""
    from heylook_llm.optimizations.status import get_optimization_summary
    
    # Get optimization status
    optimizations = get_optimization_summary()
    
    # Check Metal availability
    try:
        import mlx.core as mx
        has_metal = mx.metal.is_available()
        if has_metal:
            device_info = mx.metal.device_info()
            metal_info = {
                "available": True,
                "device_name": device_info.get("name", "Unknown"),
                "max_recommended_working_set_size": device_info.get("max_recommended_working_set_size", 0)
            }
        else:
            metal_info = {"available": False}
    except:
        metal_info = {"available": False}
    
    capabilities = {
        "server_version": "1.0.1",
        "optimizations": optimizations,
        "metal": metal_info,
        "endpoints": {
            "fast_vision": {
                "available": True,
                "endpoint": "/v1/chat/completions/multipart",
                "description": "Raw image upload endpoint - 57ms faster per image",
                "benefits": {
                    "time_saved_per_image_ms": 57,
                    "bandwidth_reduction_percent": 33,
                    "supports_parallel_processing": True
                },
                "usage": {
                    "method": "POST",
                    "content_type": "multipart/form-data",
                    "fields": {
                        "model": "Model ID (required)",
                        "messages": "JSON string of messages with __RAW_IMAGE__ placeholders",
                        "images": "Image files (multiple allowed)",
                        "resize_max": "Max dimension to resize to (optional)",
                        "resize_width": "Specific width to resize to (optional)",
                        "resize_height": "Specific height to resize to (optional)",
                        "image_quality": "JPEG quality 1-100 (default: 85)",
                        "preserve_alpha": "Keep transparency, output PNG (default: false)"
                    }
                },
                "image_processing": {
                    "auto_resize": "Resize images to reduce tokens",
                    "preserve_originals": "Default behavior keeps original dimensions",
                    "format_conversion": "Automatically handles JPEG/PNG conversion",
                    "alpha_support": "Preserves transparency when requested"
                }
            },
            "standard_vision": {
                "endpoint": "/v1/chat/completions",
                "description": "Standard endpoint with base64 images",
                "supports_base64": True
            },
            "batch_processing": {
                "available": True,
                "processing_modes": ["sequential", "parallel", "conversation"],
                "description": "Process multiple prompts in one request"
            }
        },
        "features": {
            "streaming": True,
            "model_caching": {
                "enabled": True,
                "cache_size": 2,
                "eviction_policy": "LRU"
            },
            "vision_models": True,
            "concurrent_requests": True,
            "max_image_size": "No hard limit, auto-resized as needed",
            "supported_image_formats": ["JPEG", "PNG", "WEBP", "BMP", "GIF"]
        },
        "recommendations": {
            "vision_models": {
                "use_multipart": optimizations["image"]["turbojpeg_available"] or optimizations["image"]["xxhash_available"],
                "reason": "Multipart endpoint is faster and uses less bandwidth"
            },
            "batch_size": {
                "optimal": 4,
                "max": 8,
                "note": "Depends on model size and available memory"
            },
            "image_format": {
                "preferred": "JPEG",
                "quality": 85,
                "note": "JPEG with quality 85 provides best size/quality tradeoff"
            },
            "request_pattern": {
                "use_streaming": "For responses > 100 tokens",
                "reuse_connection": "Keep-alive recommended",
                "concurrent_requests": "Safe with different models"
            }
        },
        "limits": {
            "max_tokens": 4096,
            "max_images_per_request": 10,
            "max_request_size_mb": 100,
            "timeout_seconds": 300
        }
    }
    
    return capabilities

# Import and register multipart endpoint
from heylook_llm.api_multipart import create_chat_multipart
app.post("/v1/chat/completions/multipart", 
    summary="Create Chat Completion with Raw Images (Fast)",
    description="""
High-performance vision endpoint that accepts raw image uploads instead of base64.

**🚀 Performance Benefits:**
- ⚡ 57ms faster per image (no base64 encoding/decoding)
- 📉 33% bandwidth reduction
- 🔄 Parallel image processing
- 💾 Smart image caching with xxHash

**How to Use:**
1. Send images as multipart form files
2. Include messages as JSON string with `__RAW_IMAGE__` placeholders
3. Images are injected into messages in order

**Example Request:**
```python
files = [
    ('images', ('img1.jpg', image1_bytes, 'image/jpeg')),
    ('images', ('img2.jpg', image2_bytes, 'image/jpeg'))
]
data = {
    'model': 'llava-1.5-7b-hf-4bit',
    'messages': json.dumps([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these images"},
            {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}},
            {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}}
        ]
    }])
}
response = requests.post(url + '/multipart', files=files, data=data)
```

**Perfect for:**
- ComfyUI integration
- Batch image processing
- Real-time vision applications
- Network-constrained environments
    """,
    tags=["OpenAI API"],
    responses={
        200: {
            "description": "Successful completion",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1677652288,
                        "model": "llava-1.5-7b-hf-4bit",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The first image shows a cat, while the second shows a dog."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 156,
                            "completion_tokens": 23,
                            "total_tokens": 179
                        }
                    }
                }
            }
        }
    }
)(create_chat_multipart)

@app.get("/",
    summary="Server Information",
    description="Get server information and available endpoints",
    tags=["Monitoring"]
)
async def root():
    """Root endpoint showing server info and available APIs"""
    return {
        "name": "HeylookLLM",
        "version": "1.0.1",
        "description": "High-performance local LLM server with dual API compatibility",
        "documentation": {
            "interactive": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "apis": {
            "openai": {
                "description": "OpenAI-compatible API for maximum compatibility",
                "endpoints": {
                    "list_models": {"method": "GET", "path": "/v1/models"},
                    "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                    "chat_multipart": {"method": "POST", "path": "/v1/chat/completions/multipart", "note": "⚡ Fast vision endpoint"},
                    "performance": {"method": "GET", "path": "/v1/performance"},
                    "capabilities": {"method": "GET", "path": "/v1/capabilities", "note": "Discover optimizations"}
                }
            },
            "ollama": {
                "description": "Ollama-compatible API for drop-in replacement",
                "endpoints": {
                    "core": {
                        "list_models": {"method": "GET", "path": "/api/tags"},
                        "chat": {"method": "POST", "path": "/api/chat"},
                        "generate": {"method": "POST", "path": "/api/generate"},
                        "show_model": {"method": "POST", "path": "/api/show"},
                        "version": {"method": "GET", "path": "/api/version"},
                        "running_models": {"method": "GET", "path": "/api/ps"},
                        "embeddings": {"method": "POST", "path": "/api/embed"}
                    },
                    "model_management": {
                        "note": "Not implemented - use models.yaml configuration",
                        "create": "/api/create",
                        "copy": "/api/copy",
                        "delete": "/api/delete",
                        "pull": "/api/pull",
                        "push": "/api/push"
                    }
                }
            }
        },
        "features": {
            "model_providers": ["MLX (Apple Silicon)", "llama.cpp (GGUF)"],
            "vision_models": True,
            "streaming": True,
            "batch_processing": True,
            "model_caching": "LRU (max 2 models)",
            "performance_optimizations": ["Metal acceleration", "Async processing", "Smart caching"]
        },
        "quick_start": {
            "1": "List models: GET /v1/models",
            "2": "Chat: POST /v1/chat/completions",
            "3": "Vision (fast): POST /v1/chat/completions/multipart"
        }
    }


def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="""
# HeylookLLM API Documentation 🚀

A unified, high-performance API server for local LLM inference with dual OpenAI and Ollama compatibility.

## 🎯 Key Features

### Dual API Support
- **OpenAI API**: Full compatibility with OpenAI clients and libraries
- **Ollama API**: Drop-in replacement for Ollama server

### Model Support
- **MLX Models**: Optimized for Apple Silicon with Metal acceleration
- **GGUF Models**: Support via llama.cpp for broad compatibility
- **Vision Models**: Process images with vision-language models

### Performance Features
- 🚀 **Smart Model Caching**: LRU cache keeps 2 models in memory
- ⚡ **Fast Vision Endpoint**: `/v1/chat/completions/multipart` - 57ms faster per image
- 🔄 **Async Processing**: Non-blocking request handling
- 📊 **Performance Monitoring**: Real-time metrics at `/v1/performance`

## 🚦 Quick Start

### 1. Check Available Models
```bash
curl http://localhost:8080/v1/models
```

### 2. Generate Text
```bash
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen2.5-coder-1.5b-instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Process Images (Fast)
```python
import requests

files = [('images', ('image.jpg', image_bytes, 'image/jpeg'))]
data = {
    'model': 'llava-1.5-7b-hf-4bit',
    'messages': json.dumps([{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}}
        ]
    }])
}
response = requests.post('http://localhost:8080/v1/chat/completions/multipart', 
                        files=files, data=data)
```

## 📚 Client Libraries

### OpenAI Python SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen2.5-coder-1.5b-instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Ollama Python SDK
```python
import ollama
ollama.Client(host="http://localhost:8080").chat(
    model="qwen2.5-coder-1.5b-instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🛠️ Configuration

Models are configured in `models.yaml`. The server automatically loads models on demand and manages memory with LRU eviction.

## 📈 Performance Optimization

Install with performance extras for maximum speed:
```bash
pip install heylookllm[performance]
```

This enables:
- ⚡ uvloop for faster async
- 🚀 orjson for 10x faster JSON
- 🖼️ TurboJPEG for fast image processing
- #️⃣ xxHash for ultra-fast caching
        """,
        routes=app.routes,
    )
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8080",
            "description": "Default server (OpenAI-compatible)"
        },
        {
            "url": "http://localhost:11434", 
            "description": "Ollama-compatible port"
        }
    ]
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "GitHub Repository",
        "url": "https://github.com/yourusername/heylookitsanllm"
    }
    
    # Enhanced component schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    # Add example schemas
    openapi_schema["components"]["examples"] = {
        "simple_text_request": {
            "summary": "Simple text completion",
            "value": {
                "model": "qwen2.5-coder-1.5b-instruct-4bit",
                "messages": [
                    {"role": "user", "content": "Write a hello world in Python"}
                ],
                "max_tokens": 256
            }
        },
        "vision_request": {
            "summary": "Vision model request",
            "value": {
                "model": "llava-1.5-7b-hf-4bit",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        ]
                    }
                ],
                "max_tokens": 512
            }
        },
        "streaming_request": {
            "summary": "Streaming response",
            "value": {
                "model": "qwen2.5-coder-1.5b-instruct-4bit",
                "messages": [{"role": "user", "content": "Tell me a story"}],
                "stream": True,
                "max_tokens": 1024
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override the default OpenAPI function
app.openapi = custom_openapi
