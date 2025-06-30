import argparse
import base64
import gc
import io
import json
import logging
import platform
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import mlx.core as mx
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
# Import directly from the correct location
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load as vlm_load
from mlx_vlm.utils import stream_generate as vlm_stream_generate
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field

# --- Server Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Global State ---
model_provider = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_provider
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    model_provider = ModelProvider(args)
    yield
    logging.info("Server is shutting down.")

app = FastAPI(
    title="Unified MLX VLM/LLM Server",
    description="An OpenAI-compatible API server for MLX models.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Image Utilities ---
def _load_image_universal(source_str: str, timeout: int = 10) -> Image.Image:
    if source_str.startswith("data:image"):
        try: header, encoded = source_str.split(",", 1); return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception as e: raise ValueError(f"Failed to decode/load Base64 image: {e}") from e
    elif source_str.startswith(("http://", "https://")):
        try: response = requests.get(source_str, stream=True, timeout=timeout); response.raise_for_status(); return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e: raise ValueError(f"Failed to load image from URL: {e}") from e
    else:
        image_path = Path(source_str)
        if not image_path.is_file(): raise FileNotFoundError(f"Image path does not exist: {source_str}")
        return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

# --- Model Management ---
class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model_key, self.model, self.processor = None, None, None
        if self.cli_args.model: self.load(self.cli_args.model, self.cli_args.adapter_path)

    def load(self, model_path: str, adapter_path: Optional[str] = None):
        new_key = (model_path, adapter_path)
        if self.model_key == new_key: return self.model, self.processor
        if self.model: logging.info(f"Unloading model {self.model_key[0]}"); del self.model, self.processor; gc.collect(), mx.clear_cache()

        processor_config = {"trust_remote_code": self.cli_args.trust_remote_code, "use_fast": True}
        logging.info(f"Loading model: {model_path}")
        self.model, self.processor = vlm_load(model_path, adapter_path=adapter_path, processor_config=processor_config)
        self.model_key = new_key
        return self.model, self.processor

    @property
    def current_model_id(self) -> Optional[str]: return self.model_key[0] if self.model_key else None

# --- Pydantic Models ---
class ImageUrl(BaseModel): url: str
class TextContentPart(BaseModel): type: Literal["text"]; text: str
class ImageContentPart(BaseModel): type: Literal["image_url"]; image_url: ImageUrl
ContentPart = Union[TextContentPart, ImageContentPart]
class ChatMessage(BaseModel): role: Literal["system", "user", "assistant"]; content: Union[str, List[ContentPart]]
class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(512, gt=0)
    stream: bool = False

# --- API Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    model_id = request.model or model_provider.current_model_id
    if not model_id: raise HTTPException(status_code=400, detail="No model specified or pre-loaded.")
    model, processor = model_provider.load(model_id)

    images, system_prompt, remaining_messages = [], None, []
    for msg in request.messages:
        if msg.role == "system": system_prompt = msg.content
        else: remaining_messages.append(msg)

    messages_for_prompt = request.messages
    text_content_messages = []
    for msg in messages_for_prompt:
        if isinstance(msg.content, list):
            text_parts = []
            for part in msg.content:
                if part.type == "text": text_parts.append(part.text)
                elif part.type == "image_url": images.append(_load_image_universal(part.image_url.url))
            text_content_messages.append({"role": msg.role, "content": "".join(text_parts)})
        else: text_content_messages.append({"role": msg.role, "content": msg.content})

    formatted_prompt = apply_chat_template(processor, model.config, text_content_messages, num_images=len(images), add_generation_prompt=True)
    gen_kwargs = request.model_dump(exclude={"model", "messages", "stream"})

    generator = vlm_stream_generate(model, processor, formatted_prompt, images, **gen_kwargs)

    if request.stream:
        return StreamingResponse(stream_response_generator(generator, model_id), media_type="text/event-stream")
    else:
        return await non_stream_response(generator, model_id)

# --- Response Generators ---
def stream_response_generator(generator, model_id):
    request_id, created_time = f"chatcmpl-{uuid.uuid4()}", int(time.time())
    try:
        # Correctly iterate over the single VLMGenerationResult object
        for result in generator:
            response = {"id": request_id, "object": "chat.completion.chunk", "created": created_time, "model": model_id, "choices": [{"index": 0, "delta": {"content": result.text}, "finish_reason": None}]}
            yield f"data: {json.dumps(response)}\n\n"
    finally:
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

async def non_stream_response(generator, model_id):
    full_text, last_response = "", None
    # Correctly iterate over the single VLMGenerationResult object
    for result in generator:
        full_text += result.text
        last_response = result

    usage = {"prompt_tokens": last_response.prompt_tokens, "completion_tokens": last_response.generation_tokens, "total_tokens": last_response.prompt_tokens + last_response.generation_tokens}
    return {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}], "usage": usage}

# --- Server Startup ---
def parse_args():
    parser = argparse.ArgumentParser(description="Unified MLX Server")
    parser.add_argument("--model", type=str, help="Default model path or HF repo.")
    parser.add_argument("--adapter-path", type=str)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    uvicorn.run("unified_server:app", host=cli_args.host, port=cli_args.port, log_level=cli_args.log_level.lower(), reload=True)
