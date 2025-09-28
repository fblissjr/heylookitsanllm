"""Speech-to-Text API endpoints for heylookitsanllm."""
import logging
import time
import uuid
from typing import Optional, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create STT router
stt_router = APIRouter(prefix="/v1", tags=["Speech-to-Text"])


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    model: str = Field(..., description="Model ID to use for transcription")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en')")
    prompt: Optional[str] = Field(None, description="Optional prompt to guide transcription")
    response_format: Optional[str] = Field("json", description="Response format: json, text, srt, verbose_json, vtt")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str = Field(..., description="Transcribed text")
    language: Optional[str] = Field(None, description="Detected language")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    model: Optional[str] = Field(None, description="Model used for transcription")


@stt_router.post("/audio/transcriptions",
    summary="Transcribe Audio",
    description="""
Transcribe audio file to text using the specified STT model.

**Supported formats:** WAV, MP3, M4A, FLAC, OGG, WebM
**Max file size:** 25MB
**Models:** parakeet-tdt-v3 (CoreML)

This endpoint is compatible with OpenAI's Whisper API format.
    """,
    response_model=TranscriptionResponse,
    responses={
        200: {"description": "Successful transcription"},
        400: {"description": "Invalid audio format or parameters"},
        413: {"description": "File too large"},
        500: {"description": "Model loading or transcription error"}
    }
)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(..., description="Model ID to use"),
    language: Optional[str] = Form(None, description="Language code"),
    prompt: Optional[str] = Form(None, description="Optional prompt"),
    response_format: Optional[str] = Form("json", description="Response format"),
    temperature: Optional[float] = Form(0.0, description="Sampling temperature")
):
    """Transcribe audio file to text."""
    start_time = time.time()

    # Validate file size (25MB limit)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is 25MB.")

    # Validate audio format
    allowed_formats = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed formats: {', '.join(allowed_formats)}"
        )

    try:
        logger.info(f"Transcribing audio with model {model}, format: {file_ext}, size: {len(contents)} bytes")

        # Get the router and load the STT model
        router = request.app.state.router_instance
        stt_provider = router.get_stt_provider(model)

        # Transcribe the audio
        transcription = stt_provider.transcribe(contents, stream=False)

        duration = time.time() - start_time

        # Format response based on requested format
        if response_format == "text":
            return transcription
        elif response_format in ["srt", "vtt"]:
            # Would generate proper subtitle format
            return f"1\n00:00:00,000 --> 00:00:{duration:.0f},000\n{transcription}\n"
        else:
            # JSON formats
            response = TranscriptionResponse(
                text=transcription,
                language=language or "en",
                duration=duration,
                model=model
            )

            if response_format == "verbose_json":
                # Add more detailed information
                return {
                    **response.dict(),
                    "task": "transcribe",
                    "segments": [{
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": duration,
                        "text": transcription,
                        "temperature": temperature,
                        "avg_logprob": -0.5,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.01
                    }],
                    "words": []  # Would include word-level timestamps if available
                }

            return response

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@stt_router.post("/audio/translations",
    summary="Translate Audio",
    description="""
Transcribe and translate audio to English.

Currently uses the same model as transcription with translation capabilities.
    """,
    response_model=TranscriptionResponse
)
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0)
):
    """Translate audio to English text."""
    # For now, this calls transcribe with translation flag
    # Real implementation would use a translation-capable model
    return await transcribe_audio(
        file=file,
        model=model,
        language="en",  # Force English output
        prompt=prompt,
        response_format=response_format,
        temperature=temperature
    )


@stt_router.websocket("/stt/stream")
async def stream_transcription(websocket):
    """WebSocket endpoint for real-time streaming transcription."""
    await websocket.accept()

    try:
        # Initialize streaming session
        session_id = str(uuid.uuid4())
        logger.info(f"Started streaming transcription session {session_id}")

        while True:
            # Receive audio chunks
            data = await websocket.receive_bytes()

            # Process audio chunk
            # This would feed into the streaming transcription pipeline
            partial_result = f"[Partial {len(data)} bytes]"

            # Send partial transcription
            await websocket.send_json({
                "type": "partial",
                "text": partial_result,
                "timestamp": time.time()
            })

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
    finally:
        await websocket.close()


@stt_router.get("/stt/models",
    summary="List STT Models",
    description="Get list of available speech-to-text models"
)
async def list_stt_models():
    """List available STT models."""
    # This would query the model router for STT models
    return {
        "models": [
            {
                "id": "parakeet-tdt-v3",
                "name": "Parakeet TDT v3 (0.6B)",
                "provider": "coreml",
                "languages": ["en"],
                "capabilities": {
                    "transcribe": True,
                    "translate": False,
                    "streaming": True,
                    "word_timestamps": False
                }
            }
        ]
    }