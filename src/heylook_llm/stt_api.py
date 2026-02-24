"""Speech-to-Text API endpoints for heylookitsanllm."""
import logging
import time
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create STT router
stt_router = APIRouter(prefix="/v1", tags=["Speech-to-Text"])


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    model: str = Field(..., description="Model ID to use for transcription")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en')")
    prompt: Optional[str] = Field(None, description="Optional prompt to guide transcription")
    response_format: Optional[str] = Field("json", description="Response format: json, text")
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
**Models:** parakeet-tdt-v3 (MLX)

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
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB.")

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

        # Default: JSON format
        return TranscriptionResponse(
            text=transcription,
            language=language or "en",
            duration=duration,
            model=model
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
