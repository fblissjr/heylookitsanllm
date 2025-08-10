# src/heylook_llm/api.py
import uuid, time, logging, argparse, sys, asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Body, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
from heylook_llm.metrics_db_wrapper import get_metrics_db, init_metrics_db
from heylook_llm.analytics_config import analytics_config



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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
- Automatic model loading with LRU cache (max 2 models)
- Vision model support with base64 images
- Streaming responses (Server-Sent Events)
- Batch processing for multiple prompts
- Reproducible generation with seed parameter
- Metal-optimized inference for MLX models

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

    except RuntimeError as e:
        # Check if this is a MODEL_BUSY error
        if "MODEL_BUSY" in str(e):
            logging.warning(f"Model busy for request {request_id[:8]}: {e}")
            log_request_complete(request_id, success=False, error_msg="Model busy")
            
            # Return 503 Service Unavailable with retry headers
            # This tells OpenAI client to retry automatically
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "The server is currently processing another request. Please retry in a moment.",
                        "type": "server_error",
                        "code": "model_overloaded"
                    }
                },
                headers={
                    "Retry-After": "1",  # Suggest retry after 1 second
                    "X-RateLimit-Limit": "1",  # We can handle 1 concurrent request
                    "X-RateLimit-Remaining": "0",  # No capacity right now
                    "X-RateLimit-Reset": str(int(time.time() + 1))  # Reset in 1 second
                }
            )
        else:
            # Other runtime errors
            logging.error(f"Runtime error: {e}", exc_info=True)
            log_request_complete(request_id, success=False, error_msg=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logging.error(f"Failed to get provider or create generator: {e}", exc_info=True)
        log_request_complete(request_id, success=False, error_msg=str(e))

        # Log error to metrics database
        db = get_metrics_db()
        if db and db.enabled:
            try:
                metrics = {
                    'timestamp': datetime.now(),
                    'request_id': request_id,
                    'model': chat_request.model,
                    'provider': 'mlx' if 'mlx' in chat_request.model else 'llama_cpp',
                    'request_type': 'chat_completion',
                    'total_time_ms': int((time.time() - request_start_time) * 1000),
                    'success': False,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                db.log_request(metrics)
            except Exception as log_e:
                logging.error(f"Failed to log error metrics: {log_e}")

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

    # Log to metrics database if enabled
    db = get_metrics_db()
    if db and db.enabled:
        try:
            metrics = {
                'timestamp': datetime.now(),
                'request_id': request_id,
                'model': chat_request.model,
                'provider': 'mlx' if 'mlx' in chat_request.model else 'llama_cpp',
                'request_type': 'chat_completion',
                'total_time_ms': int(processing_time * 1000),
                'first_token_ms': 0,  # TODO: track this properly
                'prompt_tokens': prompt_tokens or 0,
                'completion_tokens': completion_tokens or token_count,
                'tokens_per_second': (completion_tokens or token_count) / processing_time if processing_time > 0 else 0,
                'success': True
            }

            # Add request/response content based on storage level
            if analytics_config.should_log_content():
                # Convert Pydantic models to dictionaries for DuckDB storage
                metrics['messages'] = [msg.dict() for msg in chat_request.messages]
                metrics['response_text'] = full_text

            db.log_request(metrics)
        except Exception as e:
            logging.error(f"Failed to log metrics: {e}")

    return response

@app.post("/v1/admin/restart",
    summary="Restart Server",
    description="""
Restart the server to reload configuration and code changes.

**WARNING**: This will interrupt all active connections!

**Security Note**: This endpoint should be disabled in production or protected with authentication.
    """,
    tags=["Admin"]
)
async def restart_server(request: Request, background_tasks: BackgroundTasks):
    """Restart the server process."""
    import os
    import sys
    import signal
    
    def restart():
        """Restart the current process."""
        try:
            # Give time for response to be sent
            import time
            time.sleep(0.5)
            
            # Use exec to replace the current process
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            logging.error(f"Failed to restart: {e}")
            # Fallback: just exit and rely on process manager to restart
            os._exit(1)
    
    # Schedule restart in background
    background_tasks.add_task(restart)
    
    return {
        "status": "restarting",
        "message": "Server will restart in 0.5 seconds"
    }

@app.post("/v1/admin/reload",
    summary="Reload Models",
    description="""
Reload model configuration and clear model cache without restarting the server.

This will:
- Clear the loaded model cache
- Reload models.yaml configuration
- Keep the server running
    """,
    tags=["Admin"]
)
async def reload_models(request: Request):
    """Reload model configuration without restarting."""
    try:
        router = request.app.state.router_instance
        
        # Clear model cache
        router.clear_cache()
        
        # Reload configuration
        router.reload_config()
        
        return {
            "status": "success",
            "message": "Model configuration reloaded",
            "cache_cleared": True,
            "models_available": router.list_available_models()
        }
    except Exception as e:
        logging.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings",
    summary="Create Embeddings",
    description="""
Generate embeddings for text using the specified model.

**Key Features:**
- Extract actual model embeddings (not hallucinated numbers)
- Support for both text-only and vision models
- Multiple pooling strategies (mean, cls, last, max)
- Optional dimension truncation
- Batch processing support

**Use Cases:**
- Text similarity search
- Semantic clustering
- Cross-modal alignment
- Prompt interpolation
- Document retrieval

**Request Body:**
- `input` (string | array[string]): Text(s) to embed
- `model` (string): Model ID to use
- `dimensions` (integer, optional): Truncate to N dimensions
- `encoding_format` (string, optional): "float" or "base64"
- `user` (string, optional): User identifier
    """,
    response_description="Embeddings in OpenAI-compatible format",
    tags=["OpenAI API"],
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "examples": {
                        "single": {
                            "summary": "Single embedding",
                            "value": {
                                "object": "list",
                                "data": [{
                                    "object": "embedding",
                                    "embedding": [0.0234, -0.1567, 0.8901],
                                    "index": 0
                                }],
                                "model": "dolphin-mistral",
                                "usage": {"prompt_tokens": 10, "total_tokens": 10}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_embeddings_endpoint(
    request: Request,
    embedding_request: dict = Body(...)
):
    """Create embeddings for the given input text(s)."""
    from heylook_llm.embeddings import EmbeddingRequest, create_embeddings
    
    try:
        # Parse request
        req = EmbeddingRequest(**embedding_request)
        
        # Get router
        router = request.app.state.router_instance
        
        # Create embeddings
        response = await create_embeddings(req, router)
        
        return response.model_dump()
        
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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


@app.get("/v1/data/summary",
    summary="Get Analytics Summary",
    description="""
Get pre-computed analytics summary for dashboards.

**Note:** Analytics must be enabled via HEYLOOK_ANALYTICS_ENABLED=true

**Returns:**
- total_requests: Total number of requests
- avg_response_time: Average response time in ms
- tokens_per_second: Average tokens per second
- error_rate: Error rate as percentage
- model_usage: Request count by model
- recent_activity: Time-series data for charts
    """,
    response_description="Analytics summary data",
    tags=["Monitoring"]
)
async def data_summary():
    """Get analytics summary for dashboard."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        return {
            "error": "Analytics not enabled",
            "message": "Set HEYLOOK_ANALYTICS_ENABLED=true to enable analytics"
        }

    db = get_metrics_db()
    if not db:
        return {"error": "Metrics database not initialized"}

    try:
        # Get summary metrics
        summary_query = """
            SELECT
                COUNT(*) as total_requests,
                AVG(total_time_ms) as avg_response_time,
                AVG(tokens_per_second) as avg_tokens_per_second,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as error_rate
            FROM request_logs
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        """

        summary_result = db.execute_query(summary_query)
        if "error" in summary_result:
            return summary_result

        # Get model usage
        model_query = """
            SELECT
                model,
                COUNT(*) as count
            FROM request_logs
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            GROUP BY model
            ORDER BY count DESC
        """

        model_result = db.execute_query(model_query)

        # Get time series data for charts
        timeseries_query = """
            SELECT
                DATE_TRUNC('minute', timestamp) as time,
                COUNT(*) as requests,
                AVG(total_time_ms) as avg_response_time
            FROM request_logs
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY time
            ORDER BY time
        """

        timeseries_result = db.execute_query(timeseries_query)

        # Format response
        summary_data = summary_result.get("data", [[0, 0, 0, 0]])[0] if summary_result.get("data") else [0, 0, 0, 0]

        return {
            "total_requests": int(summary_data[0]) if len(summary_data) > 0 else 0,
            "avg_response_time": float(summary_data[1]) if len(summary_data) > 1 else 0,
            "tokens_per_second": float(summary_data[2]) if len(summary_data) > 2 else 0,
            "error_rate": float(summary_data[3]) if len(summary_data) > 3 else 0,
            "model_usage": [
                {"name": row[0], "value": row[1]}
                for row in model_result.get("data", [])
            ],
            "time_series": [
                {"time": row[0], "requests": row[1], "responseTime": row[2]}
                for row in timeseries_result.get("data", [])
            ]
        }

    except Exception as e:
        logging.error(f"Error getting analytics summary: {e}")
        return {"error": str(e)}


@app.post("/v1/data/query",
    summary="Execute Analytics Query",
    description="""
Execute SQL queries against the analytics database.

**Note:** Analytics must be enabled via HEYLOOK_ANALYTICS_ENABLED=true

**Request body:**
- query: SQL query to execute
- limit: Maximum rows to return (default: 1000)

**Returns:**
- columns: List of column names
- data: Query results as list of rows
- row_count: Number of rows returned

**Available tables:**
- request_logs: Detailed request/response metrics
- model_switches: Model loading/unloading events
- performance_summary: Aggregated performance data

**Example queries:**
- `SELECT * FROM request_logs WHERE total_time_ms > 1000`
- `SELECT model, AVG(tokens_per_second) FROM request_logs GROUP BY model`
    """,
    response_description="Query results with columns and data",
    tags=["Monitoring"]
)
async def data_query(request: Request):
    """Execute SQL query against analytics database."""
    try:
        body = await request.json()
        query = body.get("query", "")
        limit = body.get("limit", 1000)

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        # Check if analytics is enabled
        if not analytics_config.enabled:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Analytics not enabled",
                    "message": "Set HEYLOOK_ANALYTICS_ENABLED=true to enable analytics"
                }
            )

        # Get metrics database
        db = get_metrics_db()
        if not db:
            raise HTTPException(
                status_code=503,
                detail="Metrics database not initialized"
            )

        # Execute query
        result = db.execute_query(query, limit)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/data/request/{request_id}",
    summary="Get Request Details",
    description="""
Get detailed information about a specific request by ID.

**Note:** Analytics must be enabled with storage_level=full for complete data

**Returns:**
- Full request details including messages
- Response content
- Timing breakdown
- Token counts
- Error information (if any)
    """,
    response_description="Complete request details",
    tags=["Monitoring"]
)
async def get_request_details(request_id: str):
    """Get detailed information about a specific request."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        # Get request details
        request_data = db.get_request_by_id(request_id)

        if not request_data:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

        # Convert to JSON-serializable format
        # Handle messages which might be stored as JSON string
        messages = request_data.get("messages")
        if isinstance(messages, str):
            try:
                import json
                messages = json.loads(messages)
            except:
                messages = None

        result = {
            "request_id": request_data.get("request_id"),
            "timestamp": str(request_data.get("timestamp")) if request_data.get("timestamp") else None,
            "model": request_data.get("model"),
            "provider": request_data.get("provider"),
            "request_type": request_data.get("request_type"),
            "messages": messages,
            "response_text": request_data.get("response_text"),
            "num_images": request_data.get("num_images", 0),
            "num_messages": request_data.get("num_messages"),
            "max_tokens": request_data.get("max_tokens"),
            "temperature": request_data.get("temperature"),
            "timing": {
                "total_ms": request_data.get("total_time_ms"),
                "queue_ms": request_data.get("queue_time_ms"),
                "model_load_ms": request_data.get("model_load_time_ms"),
                "image_processing_ms": request_data.get("image_processing_ms"),
                "token_generation_ms": request_data.get("token_generation_ms"),
                "first_token_ms": request_data.get("first_token_ms")
            },
            "tokens": {
                "prompt": request_data.get("prompt_tokens"),
                "completion": request_data.get("completion_tokens"),
                "total": (request_data.get("prompt_tokens", 0) or 0) + (request_data.get("completion_tokens", 0) or 0),
                "per_second": request_data.get("tokens_per_second")
            },
            "resources": {
                "memory_gb": request_data.get("memory_used_gb"),
                "gpu_utilization": request_data.get("gpu_utilization")
            },
            "status": {
                "success": request_data.get("success"),
                "error_type": request_data.get("error_type"),
                "error_message": request_data.get("error_message")
            }
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching request details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/replay/{request_id}",
    summary="Replay Request",
    description="""
Replay a previous request with optional parameter modifications.

**Note:** Analytics must be enabled for request history

**Request body (optional):**
- model: Override the original model
- temperature: Override temperature
- max_tokens: Override max tokens
- system_message: Add/override system message

**Returns:**
- Original request details
- Modified parameters
- New response
- Comparison metrics
    """,
    response_description="Replay results with comparison",
    tags=["Monitoring"]
)
async def replay_request(request_id: str, request: Request):
    """Replay a previous request with modifications."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        # Get original request
        original_request = db.get_request_by_id(request_id)

        if not original_request:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

        # Parse messages if stored as JSON string
        messages = original_request.get("messages")
        if isinstance(messages, str):
            import json
            messages = json.loads(messages)

        if not messages:
            raise HTTPException(status_code=400, detail="Original request has no messages")

        # Get modification parameters
        body = await request.json() if request.body else {}

        # Build modified request
        modified_model = body.get("model", original_request.get("model"))
        modified_temperature = body.get("temperature", original_request.get("temperature", 0.7))
        modified_max_tokens = body.get("max_tokens", original_request.get("max_tokens", 1000))

        # Add system message if provided
        if "system_message" in body:
            # Check if first message is already a system message
            if messages[0].get("role") == "system":
                messages[0]["content"] = body["system_message"]
            else:
                messages.insert(0, {"role": "system", "content": body["system_message"]})

        # Create chat request
        chat_request = ChatRequest(
            model=modified_model,
            messages=messages,
            temperature=modified_temperature,
            max_tokens=modified_max_tokens,
            stream=False
        )

        # Get router from app state
        router = request.app.state.router

        # Record start time
        start_time = time.time()

        # Process the request using the same flow as chat completions
        provider = router.get_provider(chat_request.model)
        generator = provider.create_chat_completion(chat_request)

        # Collect the response
        full_text = ""
        token_count = 0
        for chunk in generator:
            if chunk.text:
                full_text += chunk.text
                token_count += 1

        # Calculate processing time
        processing_time = time.time() - start_time

        # Build comparison result
        result = {
            "original": {
                "request_id": original_request.get("request_id"),
                "model": original_request.get("model"),
                "temperature": original_request.get("temperature"),
                "max_tokens": original_request.get("max_tokens"),
                "response_text": original_request.get("response_text"),
                "total_time_ms": original_request.get("total_time_ms"),
                "tokens_per_second": original_request.get("tokens_per_second")
            },
            "replay": {
                "model": modified_model,
                "temperature": modified_temperature,
                "max_tokens": modified_max_tokens,
                "response_text": full_text,
                "total_time_ms": int(processing_time * 1000),
                "tokens_per_second": token_count / processing_time if processing_time > 0 else 0,
                "total_tokens": token_count
            },
            "modifications": {
                "model_changed": modified_model != original_request.get("model"),
                "temperature_changed": modified_temperature != original_request.get("temperature"),
                "max_tokens_changed": modified_max_tokens != original_request.get("max_tokens"),
                "system_message_added": "system_message" in body
            }
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error replaying request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/eval/create",
    summary="Create Evaluation Set",
    description="""
Create a new evaluation set for model testing.

**Request body:**
- name: Evaluation set name
- description: Optional description
- prompts: Array of evaluation prompts
  - prompt: The test prompt
  - expected_contains: Optional array of strings that should appear in response
  - expected_format: Optional format validation (json, code, markdown)
  - tags: Optional tags for categorization

**Returns:**
- eval_id: Unique evaluation set ID
- created_at: Creation timestamp
- prompt_count: Number of test prompts
    """,
    response_description="Created evaluation set details",
    tags=["Evaluation"]
)
async def create_eval_set(request: Request):
    """Create a new evaluation set."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        body = await request.json()

        # Validate required fields
        if not body.get("name"):
            raise HTTPException(status_code=400, detail="Evaluation set name is required")

        if not body.get("prompts") or not isinstance(body.get("prompts"), list):
            raise HTTPException(status_code=400, detail="Prompts array is required")

        import uuid
        import json
        from datetime import datetime

        eval_id = str(uuid.uuid4())

        # Store evaluation set
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sets (
                eval_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                prompts TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.conn.execute("""
            INSERT INTO evaluation_sets (eval_id, name, description, prompts)
            VALUES (?, ?, ?, ?)
        """, [
            eval_id,
            body.get("name"),
            body.get("description", ""),
            json.dumps(body.get("prompts"))
        ])

        return {
            "eval_id": eval_id,
            "name": body.get("name"),
            "description": body.get("description", ""),
            "prompt_count": len(body.get("prompts")),
            "created_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating evaluation set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/eval/run",
    summary="Run Evaluation",
    description="""
Run an evaluation set against one or more models.

**Request body:**
- eval_id: Evaluation set ID
- models: Array of model IDs to test
- iterations: Number of iterations per prompt (default: 1)
- parameters: Optional generation parameters
  - temperature: Override temperature
  - max_tokens: Override max tokens

**Returns:**
- run_id: Unique run ID
- status: Run status
- progress: Progress information
    """,
    response_description="Evaluation run details",
    tags=["Evaluation"]
)
async def run_evaluation(request: Request, background_tasks: BackgroundTasks):
    """Run an evaluation set against models."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        body = await request.json()

        # Validate required fields
        if not body.get("eval_id"):
            raise HTTPException(status_code=400, detail="eval_id is required")

        if not body.get("models") or not isinstance(body.get("models"), list):
            raise HTTPException(status_code=400, detail="Models array is required")

        # Get evaluation set
        result = db.conn.execute("""
            SELECT prompts FROM evaluation_sets WHERE eval_id = ?
        """, [body.get("eval_id")]).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Evaluation set not found")

        import json
        prompts = json.loads(result[0])

        import uuid
        run_id = str(uuid.uuid4())

        # Create evaluation run table
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                eval_id TEXT NOT NULL,
                models TEXT NOT NULL,
                iterations INTEGER DEFAULT 1,
                parameters TEXT,
                status TEXT DEFAULT 'running',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                results TEXT,
                progress INTEGER DEFAULT 0,
                total INTEGER
            )
        """)

        # Insert run record
        db.conn.execute("""
            INSERT INTO evaluation_runs (run_id, eval_id, models, iterations, parameters, total)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            run_id,
            body.get("eval_id"),
            json.dumps(body.get("models")),
            body.get("iterations", 1),
            json.dumps(body.get("parameters", {})),
            len(prompts) * len(body.get("models")) * body.get("iterations", 1)
        ])

        # Run evaluation in background
        background_tasks.add_task(
            _run_evaluation_async,
            run_id,
            body.get("eval_id"),
            prompts,
            body.get("models"),
            body.get("iterations", 1),
            body.get("parameters", {})
        )

        return {
            "run_id": run_id,
            "eval_id": body.get("eval_id"),
            "models": body.get("models"),
            "status": "running",
            "progress": {
                "completed": 0,
                "total": len(prompts) * len(body.get("models")) * body.get("iterations", 1)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_evaluation_async(run_id, eval_id, prompts, models, iterations, parameters):
    """Run evaluation asynchronously."""
    db = get_metrics_db()
    if not db:
        return

    import json
    import time
    results = {}
    completed = 0

    try:
        for model_id in models:
            results[model_id] = []

            for prompt_data in prompts:
                prompt_results = []

                for i in range(iterations):
                    try:
                        # Build request
                        messages = [{"role": "user", "content": prompt_data.get("prompt")}]

                        # Run completion
                        start_time = time.time()

                        # Use existing chat completion logic
                        from .router import router
                        response = await router.create_chat_completion(
                            model=model_id,
                            messages=messages,
                            temperature=parameters.get("temperature", 0.7),
                            max_tokens=parameters.get("max_tokens", 1000),
                            stream=False
                        )

                        end_time = time.time()
                        response_text = response.choices[0].message.content

                        # Check expected contains
                        passed_checks = []
                        if prompt_data.get("expected_contains"):
                            for expected in prompt_data.get("expected_contains", []):
                                passed_checks.append({
                                    "type": "contains",
                                    "expected": expected,
                                    "passed": expected.lower() in response_text.lower()
                                })

                        prompt_results.append({
                            "iteration": i + 1,
                            "response": response_text,
                            "time_ms": int((end_time - start_time) * 1000),
                            "tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                            "checks": passed_checks,
                            "passed": all(check.get("passed", True) for check in passed_checks)
                        })

                    except Exception as e:
                        prompt_results.append({
                            "iteration": i + 1,
                            "error": str(e),
                            "passed": False
                        })

                    completed += 1

                    # Update progress
                    db.conn.execute("""
                        UPDATE evaluation_runs
                        SET progress = ?
                        WHERE run_id = ?
                    """, [completed, run_id])

                results[model_id].append({
                    "prompt": prompt_data.get("prompt"),
                    "tags": prompt_data.get("tags", []),
                    "results": prompt_results,
                    "summary": {
                        "avg_time_ms": sum(r.get("time_ms", 0) for r in prompt_results if "time_ms" in r) / len(prompt_results) if prompt_results else 0,
                        "success_rate": sum(1 for r in prompt_results if r.get("passed", False)) / len(prompt_results) if prompt_results else 0
                    }
                })

        # Update run as completed
        db.conn.execute("""
            UPDATE evaluation_runs
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                results = ?
            WHERE run_id = ?
        """, [json.dumps(results), run_id])

    except Exception as e:
        # Mark as failed
        db.conn.execute("""
            UPDATE evaluation_runs
            SET status = 'failed',
                completed_at = CURRENT_TIMESTAMP,
                results = ?
            WHERE run_id = ?
        """, [json.dumps({"error": str(e)}), run_id])


@app.get("/v1/eval/run/{run_id}",
    summary="Get Evaluation Run Status",
    description="""
Get the status and results of an evaluation run.

**Returns:**
- run_id: Run ID
- status: Current status (running, completed, failed)
- progress: Progress information
- results: Results by model (when completed)
    """,
    response_description="Evaluation run status and results",
    tags=["Evaluation"]
)
async def get_evaluation_run(run_id: str):
    """Get evaluation run status and results."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        result = db.conn.execute("""
            SELECT eval_id, models, iterations, parameters, status,
                   started_at, completed_at, results, progress, total
            FROM evaluation_runs
            WHERE run_id = ?
        """, [run_id]).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Evaluation run not found")

        import json

        response = {
            "run_id": run_id,
            "eval_id": result[0],
            "models": json.loads(result[1]),
            "iterations": result[2],
            "parameters": json.loads(result[3]) if result[3] else {},
            "status": result[4],
            "started_at": str(result[5]) if result[5] else None,
            "completed_at": str(result[6]) if result[6] else None,
            "progress": {
                "completed": result[8] or 0,
                "total": result[9] or 0
            }
        }

        # Include results if completed
        if result[4] == "completed" and result[7]:
            response["results"] = json.loads(result[7])

        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting evaluation run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/eval/list",
    summary="List Evaluation Sets",
    description="""
List all available evaluation sets.

**Returns:**
Array of evaluation sets with:
- eval_id: Unique ID
- name: Evaluation set name
- description: Description
- prompt_count: Number of prompts
- created_at: Creation timestamp
    """,
    response_description="List of evaluation sets",
    tags=["Evaluation"]
)
async def list_evaluation_sets():
    """List all evaluation sets."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        # Create table if not exists
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sets (
                eval_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                prompts TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        results = db.conn.execute("""
            SELECT eval_id, name, description, prompts, created_at
            FROM evaluation_sets
            ORDER BY created_at DESC
        """).fetchall()

        import json

        eval_sets = []
        for row in results:
            prompts = json.loads(row[3])
            eval_sets.append({
                "eval_id": row[0],
                "name": row[1],
                "description": row[2] or "",
                "prompt_count": len(prompts),
                "created_at": str(row[4]) if row[4] else None
            })

        return {"evaluation_sets": eval_sets}

    except Exception as e:
        logging.error(f"Error listing evaluation sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/performance/profile/{time_range}",
    summary="Get Performance Profile",
    description="""
Get detailed performance profiling data for the specified time range.

**Path parameters:**
- time_range: Time range (1h, 6h, 24h, 7d)

**Returns:**
- Timing breakdowns by operation type
- Resource utilization over time
- Bottleneck analysis
- Performance trends
    """,
    response_description="Performance profiling data",
    tags=["Monitoring"]
)
async def get_performance_profile(time_range: str):
    """Get performance profiling data."""
    # Check if analytics is enabled
    if not analytics_config.enabled:
        raise HTTPException(
            status_code=503,
            detail="Analytics not enabled. Set HEYLOOK_ANALYTICS_ENABLED=true"
        )

    db = get_metrics_db()
    if not db:
        raise HTTPException(status_code=500, detail="Metrics database not initialized")

    try:
        # Convert time range to hours
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}
        hours = hours_map.get(time_range, 1)

        # Get timing breakdown
        timing_breakdown = db.conn.execute("""
            SELECT
                CASE
                    WHEN queue_time_ms > 0 THEN 'queue'
                    WHEN model_load_time_ms > 0 THEN 'model_load'
                    WHEN image_processing_ms > 0 THEN 'image_processing'
                    WHEN token_generation_ms > 0 THEN 'token_generation'
                    ELSE 'other'
                END as operation,
                AVG(CASE
                    WHEN queue_time_ms > 0 THEN queue_time_ms
                    WHEN model_load_time_ms > 0 THEN model_load_time_ms
                    WHEN image_processing_ms > 0 THEN image_processing_ms
                    WHEN token_generation_ms > 0 THEN token_generation_ms
                    ELSE total_time_ms
                END) as avg_time_ms,
                COUNT(*) as count
            FROM request_logs
            WHERE timestamp > datetime('now', '-' || ? || ' hours')
            GROUP BY operation
        """, [hours]).fetchall()

        # Get resource utilization over time
        resource_timeline = db.conn.execute("""
            SELECT
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                AVG(memory_used_gb) as avg_memory_gb,
                AVG(gpu_utilization) as avg_gpu_percent,
                AVG(tokens_per_second) as avg_tps,
                COUNT(*) as requests
            FROM request_logs
            WHERE timestamp > datetime('now', '-' || ? || ' hours')
            GROUP BY hour
            ORDER BY hour
        """, [hours]).fetchall()

        # Get bottleneck analysis
        bottlenecks = db.conn.execute("""
            SELECT
                model,
                AVG(total_time_ms) as avg_total_ms,
                AVG(queue_time_ms) as avg_queue_ms,
                AVG(model_load_time_ms) as avg_load_ms,
                AVG(image_processing_ms) as avg_image_ms,
                AVG(token_generation_ms) as avg_generation_ms,
                AVG(first_token_ms) as avg_first_token_ms,
                COUNT(*) as request_count
            FROM request_logs
            WHERE timestamp > datetime('now', '-' || ? || ' hours')
            GROUP BY model
            ORDER BY avg_total_ms DESC
        """, [hours]).fetchall()

        # Get performance trends
        trends = db.conn.execute("""
            WITH hourly_stats AS (
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    AVG(total_time_ms) as avg_response_time,
                    AVG(tokens_per_second) as avg_tps,
                    COUNT(*) as requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors
                FROM request_logs
                WHERE timestamp > datetime('now', '-' || ? || ' hours')
                GROUP BY hour
            )
            SELECT
                hour,
                avg_response_time,
                avg_tps,
                requests,
                errors,
                LAG(avg_response_time) OVER (ORDER BY hour) as prev_response_time,
                LAG(avg_tps) OVER (ORDER BY hour) as prev_tps
            FROM hourly_stats
            ORDER BY hour
        """, [hours]).fetchall()

        return {
            "time_range": time_range,
            "timing_breakdown": [
                {
                    "operation": row[0],
                    "avg_time_ms": row[1],
                    "count": row[2],
                    "percentage": 0  # Will calculate client-side
                }
                for row in timing_breakdown
            ],
            "resource_timeline": [
                {
                    "timestamp": row[0],
                    "memory_gb": row[1],
                    "gpu_percent": row[2],
                    "tokens_per_second": row[3],
                    "requests": row[4]
                }
                for row in resource_timeline
            ],
            "bottlenecks": [
                {
                    "model": row[0],
                    "avg_total_ms": row[1],
                    "breakdown": {
                        "queue": row[2] or 0,
                        "model_load": row[3] or 0,
                        "image_processing": row[4] or 0,
                        "token_generation": row[5] or 0,
                        "first_token": row[6] or 0
                    },
                    "request_count": row[7]
                }
                for row in bottlenecks
            ],
            "trends": [
                {
                    "hour": row[0],
                    "response_time_ms": row[1],
                    "tokens_per_second": row[2],
                    "requests": row[3],
                    "errors": row[4],
                    "response_time_change": (row[1] - row[5]) / row[5] * 100 if row[5] else 0,
                    "tps_change": (row[2] - row[6]) / row[6] * 100 if row[6] else 0
                }
                for row in trends
            ]
        }

    except Exception as e:
        logging.error(f"Error getting performance profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/batch/process",
    summary="Batch Process Prompts",
    description="""
Process multiple prompts in batch with progress tracking.

**Request body:**
- prompts: Array of prompt objects containing:
  - id: Unique identifier for the prompt
  - content: The prompt text
  - model: Model to use (optional, uses default if not specified)
  - temperature: Temperature override (optional)
  - max_tokens: Max tokens override (optional)
  - metadata: Any additional metadata (optional)
- defaults: Default parameters for all prompts
  - model: Default model to use
  - temperature: Default temperature
  - max_tokens: Default max tokens
- batch_config: Batch processing configuration
  - parallelism: Number of concurrent requests (default: 3)
  - retry_failed: Whether to retry failed requests (default: true)
  - max_retries: Maximum retry attempts (default: 2)

**Returns:**
- batch_id: Unique batch processing ID
- status: Current batch status
- progress: Progress information
    """,
    response_description="Batch processing status",
    tags=["Batch Processing"]
)
async def batch_process(request: Request, background_tasks: BackgroundTasks):
    """Process multiple prompts in batch."""
    try:
        body = await request.json()

        # Validate required fields
        if not body.get("prompts") or not isinstance(body.get("prompts"), list):
            raise HTTPException(status_code=400, detail="Prompts array is required")

        # Get defaults
        defaults = body.get("defaults", {})
        if not defaults.get("model"):
            # Try to get from first prompt or use default
            first_model = next((p.get("model") for p in body["prompts"] if p.get("model")), None)
            if not first_model:
                raise HTTPException(status_code=400, detail="Default model is required")
            defaults["model"] = first_model

        # Get batch config
        batch_config = body.get("batch_config", {})
        parallelism = min(batch_config.get("parallelism", 3), 10)  # Cap at 10
        retry_failed = batch_config.get("retry_failed", True)
        max_retries = batch_config.get("max_retries", 2)

        import uuid
        from datetime import datetime

        batch_id = str(uuid.uuid4())

        # Store batch status in memory (in production, use Redis or DB)
        if not hasattr(app.state, "batch_jobs"):
            app.state.batch_jobs = {}

        app.state.batch_jobs[batch_id] = {
            "batch_id": batch_id,
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "total": len(body["prompts"]),
            "completed": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }

        # Start background processing
        background_tasks.add_task(
            _process_batch_async,
            batch_id,
            body["prompts"],
            defaults,
            parallelism,
            retry_failed,
            max_retries
        )

        return {
            "batch_id": batch_id,
            "status": "processing",
            "progress": {
                "total": len(body["prompts"]),
                "completed": 0,
                "failed": 0,
                "percent": 0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/batch/{batch_id}",
    summary="Get Batch Status",
    description="""Get the status and results of a batch processing job.""",
    response_description="Batch processing status and results",
    tags=["Batch Processing"]
)
async def get_batch_status(batch_id: str):
    """Get batch processing status."""
    if not hasattr(app.state, "batch_jobs") or batch_id not in app.state.batch_jobs:
        raise HTTPException(status_code=404, detail="Batch not found")

    batch_job = app.state.batch_jobs[batch_id]

    return {
        "batch_id": batch_id,
        "status": batch_job["status"],
        "created_at": batch_job["created_at"],
        "progress": {
            "total": batch_job["total"],
            "completed": batch_job["completed"],
            "failed": batch_job["failed"],
            "percent": (batch_job["completed"] / batch_job["total"] * 100) if batch_job["total"] > 0 else 0
        },
        "results": batch_job["results"] if batch_job["status"] == "completed" else [],
        "errors": batch_job["errors"]
    }


async def _process_batch_async(batch_id, prompts, defaults, parallelism, retry_failed, max_retries):
    """Process batch requests asynchronously."""
    import asyncio
    import aiohttp
    from datetime import datetime

    batch_job = app.state.batch_jobs[batch_id]

    async def process_prompt(session, prompt, attempt=0):
        """Process a single prompt."""
        prompt_id = prompt.get("id", f"prompt_{prompts.index(prompt)}")

        try:
            # Merge defaults with prompt-specific settings
            model = prompt.get("model", defaults.get("model"))
            temperature = prompt.get("temperature", defaults.get("temperature", 0.7))
            max_tokens = prompt.get("max_tokens", defaults.get("max_tokens", 1000))

            # Build request
            request_data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt["content"]}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            start_time = datetime.utcnow()

            # Make request to local server
            async with session.post(
                "http://localhost:8080/v1/chat/completions",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = datetime.utcnow()

                    return {
                        "prompt_id": prompt_id,
                        "status": "success",
                        "model": model,
                        "response": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                        "metadata": prompt.get("metadata", {})
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except Exception as e:
            error_msg = str(e)

            # Retry logic
            if retry_failed and attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                return await process_prompt(session, prompt, attempt + 1)

            return {
                "prompt_id": prompt_id,
                "status": "failed",
                "error": error_msg,
                "attempts": attempt + 1,
                "metadata": prompt.get("metadata", {})
            }

    async def process_batch():
        """Process all prompts with controlled parallelism."""
        connector = aiohttp.TCPConnector(limit=parallelism)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []

            for i in range(0, len(prompts), parallelism):
                batch = prompts[i:i + parallelism]
                batch_tasks = [process_prompt(session, prompt) for prompt in batch]

                # Process batch and update progress
                results = await asyncio.gather(*batch_tasks)

                for result in results:
                    if result["status"] == "success":
                        batch_job["completed"] += 1
                        batch_job["results"].append(result)
                    else:
                        batch_job["failed"] += 1
                        batch_job["errors"].append(result)

                # Update progress
                app.state.batch_jobs[batch_id] = batch_job

    try:
        await process_batch()
        batch_job["status"] = "completed"
        batch_job["completed_at"] = datetime.utcnow().isoformat()
    except Exception as e:
        batch_job["status"] = "failed"
        batch_job["error"] = str(e)

    # Final update
    app.state.batch_jobs[batch_id] = batch_job


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
