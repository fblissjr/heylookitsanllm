"""CoreML STT Provider for Parakeet TDT v3 model integration.

Based on the Parakeet TDT v3 RNNT implementation from:
https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/asr

RNNT decoder logic adapted from:
/Users/fredbliss/workspace/mobius/models/stt/parakeet-tdt-v3-0.6b/coreml/speech_to_text_streaming_infer_rnnt.py
"""
from __future__ import annotations

import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Generator, Union, List, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Lazy imports - only import when the provider is actually instantiated
ct = None
librosa = None
torch = None
np = None
COREML_AVAILABLE = None

def _check_dependencies():
    """Check and import CoreML dependencies when needed."""
    global ct, librosa, torch, np, COREML_AVAILABLE

    if COREML_AVAILABLE is not None:
        return COREML_AVAILABLE

    try:
        import numpy as _np
        np = _np

        import coremltools as _ct
        ct = _ct

        import librosa as _librosa
        librosa = _librosa

        import torch as _torch
        torch = _torch

        COREML_AVAILABLE = True
        logger.info("CoreML STT dependencies loaded successfully")
    except ImportError as e:
        COREML_AVAILABLE = False
        logger.warning(f"CoreML dependencies not available: {e}")
        logger.warning("Install with: uv pip install -e .[stt]")
    except Exception as e:
        COREML_AVAILABLE = False
        logger.error(f"Error loading CoreML dependencies: {e}")

    return COREML_AVAILABLE


class CoreMLStreamingDecoder:
    """RNNT greedy decoder using CoreML models.

    Adapted from mobius/models/stt/parakeet-tdt-v3-0.6b/coreml/speech_to_text_streaming_infer_rnnt.py
    """

    def __init__(
        self,
        decoder_model,
        joint_model,
        vocab_size: int = 8192,
        blank_id: int = 8191,  # vocab_size - 1
        num_layers: int = 2,
        hidden_size: int = 640,
        durations: Tuple[int, ...] = (0, 1, 2, 3, 4),
        max_symbols: int = 10
    ):
        self.decoder_model = decoder_model
        self.joint_model = joint_model
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.durations = np.array(durations, dtype=np.int32)
        self.max_symbols = max_symbols

    def decode(self, encoder_output: np.ndarray, encoder_length: int) -> List[int]:
        """Greedy RNNT decoding.

        Args:
            encoder_output: Encoder output [1, D, T]
            encoder_length: Valid encoder timesteps

        Returns:
            List of token IDs
        """
        batch_size = 1  # Single batch for now
        _, _, max_time = encoder_output.shape

        # Initialize state
        labels = np.array([[self.blank_id]], dtype=np.int32)
        time_idx = 0

        # LSTM states
        h_in = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)
        c_in = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)

        # Results storage
        transcript = []

        # Main decoding loop
        while time_idx < encoder_length:
            # Get encoder frame
            encoder_frame = encoder_output[:, :, time_idx:time_idx+1]

            # Inner loop for symbols at this timestep
            for _ in range(self.max_symbols):
                # Get decoder output
                decoder_out = self.decoder_model.predict({
                    "targets": labels,
                    "target_length": np.array([1], dtype=np.int32),
                    "h_in": h_in,
                    "c_in": c_in,
                })
                h_in = decoder_out["h_out"]
                c_in = decoder_out["c_out"]
                decoder_output = decoder_out["decoder"]

                # Get joint predictions
                # Note: The joint model expects full encoder sequence [1, 1024, 188]
                # but we're processing frame by frame. Pad the encoder frame.
                encoder_padded = np.zeros((1, encoder_output.shape[1], encoder_output.shape[2]), dtype=np.float32)
                encoder_padded[:, :, time_idx:time_idx+1] = encoder_frame

                joint_out = self.joint_model.predict({
                    "encoder": encoder_padded,
                    "decoder": decoder_output,
                })
                # Extract logits for current position
                logits = joint_out["logits"][:, time_idx:time_idx+1, :, :]  # [1, 1, 1, vocab_size + durations]

                # Get best label
                token_logits = logits[0, 0, 0, :self.vocab_size]
                label = int(np.argmax(token_logits))

                # Get duration prediction
                duration_logits = logits[0, 0, 0, self.vocab_size:self.vocab_size + len(self.durations)]
                duration_idx = int(np.argmax(duration_logits))
                duration = self.durations[duration_idx]

                if label == self.blank_id:
                    # Advance time
                    if duration == 0:
                        duration = 1
                    time_idx += duration
                    labels = np.array([[self.blank_id]], dtype=np.int32)
                    break
                else:
                    # Add token to transcript
                    transcript.append(label)
                    labels = np.array([[label]], dtype=np.int32)

        return transcript


class CoreMLSTTProvider:
    """Provider for CoreML-based Speech-to-Text models (Parakeet TDT)."""

    def __init__(self, model_id: str, config: Dict, verbose: bool = False):
        """Initialize the CoreML STT provider.

        Args:
            model_id: Unique identifier for the model
            config: Model configuration from models.yaml
            verbose: Enable verbose logging
        """
        # Check dependencies only when provider is actually created
        if not _check_dependencies():
            raise RuntimeError("CoreML dependencies not available. Install with: uv pip install -e .[stt]")

        self.model_id = model_id
        self.config = config
        self.verbose = verbose
        # Expand tilde in model path
        model_path_str = config.get("model_path", "")
        if model_path_str.startswith("~"):
            model_path_str = os.path.expanduser(model_path_str)
        self.model_path = Path(model_path_str)

        # Model components (loaded lazily)
        self.preprocessor = None
        self.encoder = None
        self.decoder = None
        self.joint = None
        self.mel_encoder = None  # Fused mel+encoder if available

        # Model parameters from metadata
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_audio_seconds = config.get("max_audio_seconds", 15)
        self.vocab_size = config.get("vocab_size", 8192)
        self.blank_id = config.get("blank_id", 8191)  # vocab_size - 1
        # Note: The actual decoder expects different dimensions than the encoder
        # Decoder uses 2 layers x 640 hidden size (from metadata.json)
        self.num_layers = config.get("num_layers", 2)
        self.hidden_size = config.get("hidden_size", 640)

        # Load tokenizer if available
        self.tokenizer = self._load_tokenizer()

        # Compute units for CoreML (CPU_ONLY, CPU_AND_NE, ALL)
        self.compute_units = getattr(ct.ComputeUnit, config.get("compute_units", "ALL"))

        # Decoder helper
        self.decoder_helper = None

        logger.info(f"Initialized CoreML STT provider for {model_id}")

    def _load_tokenizer(self) -> Dict[int, str]:
        """Load tokenizer from vocab file."""
        # Check for vocab.txt file extracted from the NEMO model
        vocab_path = Path("~/Storage/nvidia_parakeet-tdt-0.6b-v3/ddbad8f2a94d4186963950e53809d27f_vocab.txt").expanduser()

        if not vocab_path.exists():
            logger.warning(f"Tokenizer not found at {vocab_path}, using placeholder")
            return {}

        tokenizer = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    tokenizer[idx] = token

        logger.info(f"Loaded tokenizer with {len(tokenizer)} tokens")
        return tokenizer

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text using the loaded tokenizer."""
        if not self.tokenizer:
            # Fallback to simple representation
            return " ".join(str(t) for t in tokens)

        pieces = []
        for token in tokens:
            if token in self.tokenizer:
                piece = self.tokenizer[token]
                # Handle sentencepiece format
                if piece.startswith("##"):
                    # Continuation of previous word
                    pieces.append(piece[2:])
                else:
                    # New word
                    if pieces:
                        pieces.append(" ")
                    pieces.append(piece)

        return "".join(pieces).strip()

    def load_model(self):
        """Load CoreML model components."""
        if not self.model_path.exists():
            logger.error(f"Model path does not exist: {self.model_path}")
            raise RuntimeError(f"Model path not found: {self.model_path}")

        # Load metadata if available
        metadata_path = self.model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Update parameters from metadata
                self.vocab_size = metadata.get("vocab_size", self.vocab_size)
                self.blank_id = self.vocab_size - 1  # Always last token

        # The models are directly in the model_path with parakeet_ prefix
        # Try fused mel+encoder first for better performance
        mel_encoder_path = self.model_path / "parakeet_mel_encoder.mlpackage"
        if mel_encoder_path.exists():
            logger.info(f"Loading fused MelEncoder from {mel_encoder_path}")
            self.mel_encoder = ct.models.MLModel(
                str(mel_encoder_path),
                compute_units=self.compute_units
            )
        else:
            # Load separate components
            preprocessor_path = self.model_path / "parakeet_preprocessor.mlpackage"
            encoder_path = self.model_path / "parakeet_encoder.mlpackage"

            if preprocessor_path.exists():
                logger.info(f"Loading Preprocessor from {preprocessor_path}")
                self.preprocessor = ct.models.MLModel(
                    str(preprocessor_path),
                    compute_units=self.compute_units
                )

            if encoder_path.exists():
                logger.info(f"Loading Encoder from {encoder_path}")
                self.encoder = ct.models.MLModel(
                    str(encoder_path),
                    compute_units=self.compute_units
                )

        # Load decoder and joint
        decoder_path = self.model_path / "parakeet_decoder.mlpackage"
        joint_path = self.model_path / "parakeet_joint.mlpackage"

        if decoder_path.exists():
            logger.info(f"Loading Decoder from {decoder_path}")
            self.decoder = ct.models.MLModel(
                str(decoder_path),
                compute_units=self.compute_units
            )

        if joint_path.exists():
            logger.info(f"Loading Joint from {joint_path}")
            self.joint = ct.models.MLModel(
                str(joint_path),
                compute_units=self.compute_units
            )

        # Check if we have at least the minimum required models
        if not (self.mel_encoder or (self.preprocessor and self.encoder)):
            raise RuntimeError("Failed to load encoder components (neither fused mel_encoder nor separate preprocessor+encoder found)")

        if not self.decoder:
            logger.warning("Decoder not loaded - transcription may fail")
        if not self.joint:
            logger.warning("Joint model not loaded - transcription may fail")

        # Initialize decoder helper
        if self.decoder and self.joint:
            self.decoder_helper = CoreMLStreamingDecoder(
                self.decoder,
                self.joint,
                vocab_size=self.vocab_size,
                blank_id=self.blank_id,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size
            )

        logger.info(f"Model {self.model_id} loaded successfully")

    def transcribe(self, audio_data: Union[bytes, np.ndarray], stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Transcribe audio to text.

        Args:
            audio_data: Audio data as bytes or numpy array
            stream: Whether to stream partial results

        Returns:
            Transcribed text or generator of partial results
        """
        # Process audio
        if isinstance(audio_data, bytes):
            # Decode audio bytes to numpy array
            audio = self._decode_audio(audio_data)
        else:
            audio = audio_data

        # Resample if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        # Pad or truncate to max length
        max_samples = self.sample_rate * self.max_audio_seconds
        if len(audio) > max_samples:
            logger.warning(f"Audio truncated from {len(audio)/self.sample_rate:.1f}s to {self.max_audio_seconds}s")
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        if stream:
            return self._transcribe_streaming(audio)
        else:
            return self._transcribe_batch(audio)

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to numpy array."""
        import io
        import soundfile as sf

        # Read audio from bytes
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        return audio.astype(np.float32)

    def _transcribe_batch(self, audio: np.ndarray) -> str:
        """Transcribe entire audio at once."""
        if not self.decoder_helper:
            return "Decoder not available - please check model loading"

        start_time = time.time()

        # Get encoder output
        if self.mel_encoder:
            # Use fused mel+encoder
            audio_signal = audio.reshape(1, -1).astype(np.float32)
            audio_length = np.array([len(audio)], dtype=np.int32)

            result = self.mel_encoder.predict({
                "audio_signal": audio_signal,
                "audio_length": audio_length
            })
            encoder_out = result["encoder"]
            encoder_length = int(result["encoder_length"][0])
        else:
            # Separate preprocessing and encoding
            audio_signal = audio.reshape(1, -1).astype(np.float32)
            audio_length = np.array([len(audio)], dtype=np.int32)

            mel_result = self.preprocessor.predict({
                "audio_signal": audio_signal,
                "audio_length": audio_length
            })
            mel = mel_result["mel"]
            mel_length = mel_result.get("mel_length", audio_length)

            enc_result = self.encoder.predict({
                "mel": mel,
                "mel_length": mel_length
            })
            encoder_out = enc_result["encoder"]
            encoder_length = int(enc_result["encoder_length"][0])

        # Greedy decoding
        token_ids = self.decoder_helper.decode(encoder_out, encoder_length)
        transcript = self._tokens_to_text(token_ids)

        if self.verbose:
            elapsed = time.time() - start_time
            audio_duration = len(audio) / self.sample_rate
            rtf = elapsed / audio_duration
            logger.info(f"Transcription took {elapsed:.2f}s for {audio_duration:.1f}s audio, RTF={rtf:.3f}")
            logger.info(f"Tokens: {token_ids[:50]}..." if len(token_ids) > 50 else f"Tokens: {token_ids}")

        return transcript

    def _transcribe_streaming(self, audio: np.ndarray) -> Generator[str, None, None]:
        """Stream partial transcription results."""
        # For now, implement simple chunked processing
        chunk_size = self.sample_rate * 2  # 2-second chunks

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Process chunk (simplified - real streaming would maintain state)
            partial = self._transcribe_batch(chunk)
            if partial:
                yield partial

    def unload(self):
        """Unload models from memory."""
        self.preprocessor = None
        self.encoder = None
        self.decoder = None
        self.joint = None
        self.mel_encoder = None
        self.decoder_helper = None
        logger.info(f"Model {self.model_id} unloaded")