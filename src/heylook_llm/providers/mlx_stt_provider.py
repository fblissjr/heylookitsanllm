"""MLX STT Provider for Parakeet models.

Uses the parakeet-mlx library for efficient speech-to-text on Apple Silicon.
Reference: https://github.com/ml-explore/parakeet-mlx
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Generator, Union
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
parakeet_mlx = None
MLX_STT_AVAILABLE = None

def _check_dependencies():
    """Check and import MLX STT dependencies when needed."""
    global parakeet_mlx, MLX_STT_AVAILABLE

    if MLX_STT_AVAILABLE is not None:
        return MLX_STT_AVAILABLE

    try:
        import parakeet_mlx as _parakeet
        parakeet_mlx = _parakeet
        MLX_STT_AVAILABLE = True
        logger.info("MLX STT dependencies loaded successfully")
    except ImportError as e:
        MLX_STT_AVAILABLE = False
        logger.warning(f"MLX STT dependencies not available: {e}")
        logger.warning("Install with: uv pip install parakeet-mlx")
    except Exception as e:
        MLX_STT_AVAILABLE = False
        logger.error(f"Error loading MLX STT dependencies: {e}")

    return MLX_STT_AVAILABLE


class MLXSTTProvider:
    """Provider for MLX-based Speech-to-Text models (Parakeet)."""

    def __init__(self, model_id: str, config: Dict, verbose: bool = False):
        """Initialize the MLX STT provider.

        Args:
            model_id: Unique identifier for the model
            config: Model configuration from models.yaml
            verbose: Enable verbose logging
        """
        # Check dependencies
        if not _check_dependencies():
            raise RuntimeError("MLX STT dependencies not available. Install with: uv pip install parakeet-mlx")

        self.model_id = model_id
        self.config = config
        self.verbose = verbose

        # Model path (HuggingFace repo or local path)
        self.model_path = config.get("model_path", "mlx-community/parakeet-tdt-0.6b-v3")

        # Model settings
        self.chunk_duration = config.get("chunk_duration", 120)  # seconds
        self.overlap_duration = config.get("overlap_duration", 15)  # seconds
        self.use_local_attention = config.get("use_local_attention", False)
        self.local_attention_context = config.get("local_attention_context", 256)
        self.fp32 = config.get("fp32", False)  # Use fp32 instead of bf16

        # Cache directory for HuggingFace models
        cache_dir = config.get("cache_dir")
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser()
        else:
            self.cache_dir = None

        # Model instance (loaded lazily)
        self.model = None

        logger.info(f"Initialized MLX STT provider for {model_id}")

    def load_model(self):
        """Load the Parakeet model."""
        logger.info(f"Loading MLX model from {self.model_path}")

        # Load model using parakeet-mlx
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)

        self.model = parakeet_mlx.from_pretrained(self.model_path, **kwargs)

        # Configure for fp32 if requested
        if self.fp32:
            import mlx.core as mx
            mx.set_default_device(mx.gpu)
            # The model will use fp32 automatically if configured

        # Configure local attention if requested
        if self.use_local_attention:
            self.model.encoder.set_attention_model(
                "rel_pos_local_attn",  # NeMo naming convention
                (self.local_attention_context, self.local_attention_context),
            )
            logger.info(f"Using local attention with context size {self.local_attention_context}")

        logger.info(f"Model {self.model_id} loaded successfully")

    def transcribe(
        self,
        audio_data: Union[bytes, np.ndarray],
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Transcribe audio to text.

        Args:
            audio_data: Audio data as bytes or numpy array
            stream: Whether to stream partial results

        Returns:
            Transcribed text or generator of partial results
        """
        if self.model is None:
            raise RuntimeError(f"Model {self.model_id} not loaded. Call load_model() first.")

        # Process audio
        if isinstance(audio_data, bytes):
            # Decode audio bytes to numpy array
            audio = self._decode_audio(audio_data)
        else:
            audio = audio_data

        if stream:
            return self._transcribe_streaming(audio)
        else:
            return self._transcribe_batch(audio)

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to numpy array."""
        import io
        import soundfile as sf
        from parakeet_mlx.audio import load_audio

        # Save to temporary file (parakeet-mlx uses audiofile which needs a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Load audio using parakeet's loader (handles resampling)
            audio = load_audio(tmp_path, self.model.preprocessor_config.sample_rate)
            return audio
        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)

    def _transcribe_batch(self, audio: np.ndarray) -> str:
        """Transcribe entire audio at once."""
        start_time = time.time()

        # Transcribe with optional chunking
        if self.chunk_duration > 0:
            result = self.model.transcribe(
                audio,
                chunk_duration=self.chunk_duration,
                overlap_duration=self.overlap_duration
            )
        else:
            # Transcribe without chunking
            result = self.model.transcribe(audio)

        if self.verbose:
            elapsed = time.time() - start_time
            audio_duration = len(audio) / self.model.preprocessor_config.sample_rate
            rtf = elapsed / audio_duration
            logger.info(f"Transcription took {elapsed:.2f}s for {audio_duration:.1f}s audio, RTF={rtf:.3f}")

            # Log sentence-level alignments if available
            if hasattr(result, 'sentences') and result.sentences:
                logger.info(f"Found {len(result.sentences)} sentences with alignments")

        return result.text

    def _transcribe_streaming(self, audio: np.ndarray) -> Generator[str, None, None]:
        """Stream partial transcription results."""
        # Create streaming context
        with self.model.transcribe_stream(
            context_size=(256, 256),  # (left_context, right_context)
        ) as transcriber:
            # Process audio in chunks
            sample_rate = self.model.preprocessor_config.sample_rate
            chunk_size = sample_rate  # 1 second chunks

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                transcriber.add_audio(chunk)

                # Yield current transcription
                result = transcriber.result
                if result and result.text:
                    yield result.text

    def transcribe_with_timestamps(self, audio_data: Union[bytes, np.ndarray]) -> Dict[str, Any]:
        """Transcribe audio with word-level timestamps.

        Returns:
            Dictionary with text and detailed alignment information
        """
        if self.model is None:
            raise RuntimeError(f"Model {self.model_id} not loaded. Call load_model() first.")

        # Process audio
        if isinstance(audio_data, bytes):
            audio = self._decode_audio(audio_data)
        else:
            audio = audio_data

        # Get transcription with alignments
        result = self.model.transcribe(audio)

        # Format response with timestamps
        response = {
            "text": result.text,
            "sentences": []
        }

        if hasattr(result, 'sentences'):
            for sent in result.sentences:
                sentence_data = {
                    "text": sent.text,
                    "start": sent.start,
                    "end": sent.end,
                    "duration": sent.duration,
                    "tokens": []
                }

                if hasattr(sent, 'tokens'):
                    for token in sent.tokens:
                        sentence_data["tokens"].append({
                            "text": token.text,
                            "start": token.start,
                            "end": token.end,
                            "duration": token.duration
                        })

                response["sentences"].append(sentence_data)

        return response

    def unload(self):
        """Unload model from memory."""
        self.model = None

        # Force MLX memory cleanup
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except:
            pass

        logger.info(f"Model {self.model_id} unloaded")