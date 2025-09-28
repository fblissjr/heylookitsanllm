"""CoreML STT Provider for Parakeet TDT v3 model integration."""
from __future__ import annotations

import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Generator, Union
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

        # Model parameters
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_audio_seconds = config.get("max_audio_seconds", 15)
        self.vocab_size = config.get("vocab_size", 128)
        self.blank_id = config.get("blank_id", 127)
        self.num_layers = config.get("num_layers", 12)
        self.hidden_size = config.get("hidden_size", 320)

        # Compute units for CoreML (CPU_ONLY, CPU_AND_NE, ALL)
        self.compute_units = getattr(ct.ComputeUnit, config.get("compute_units", "ALL"))

        logger.info(f"Initialized CoreML STT provider for {model_id}")

    def load_model(self):
        """Load CoreML model components."""
        if not self.model_path.exists():
            # Try to export models if not present
            self._export_models()

        # Load individual components
        component_dir = self.model_path / "parakeet_coreml"

        # Try fused mel+encoder first for better performance
        mel_encoder_path = component_dir / "MelEncoder.mlpackage"
        if mel_encoder_path.exists():
            logger.info(f"Loading fused MelEncoder from {mel_encoder_path}")
            self.mel_encoder = ct.models.MLModel(
                str(mel_encoder_path),
                compute_units=self.compute_units
            )
        else:
            # Load separate components
            preprocessor_path = component_dir / "Preprocessor.mlpackage"
            encoder_path = component_dir / "Encoder.mlpackage"

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
        decoder_path = component_dir / "Decoder.mlpackage"
        joint_path = component_dir / "Joint.mlpackage"

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

        logger.info(f"Model {self.model_id} loaded successfully")

    def _export_models(self):
        """Export Parakeet models to CoreML format if not present."""
        logger.info("Attempting to export Parakeet models to CoreML...")

        # Import export script
        try:
            from convert_parakeet import main as convert_main

            # Set up arguments for conversion
            class Args:
                command = "convert"
                model_id = "nvidia/parakeet-tdt-0.6b-v3"
                nemo_path = None
                output_dir = str(self.model_path / "parakeet_coreml")
                device = "cpu"

            convert_main(Args())
            logger.info("Models exported successfully")

        except Exception as e:
            logger.error(f"Failed to export models: {e}")
            raise RuntimeError(f"Could not export CoreML models: {e}")

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

        return audio

    def _transcribe_batch(self, audio: np.ndarray) -> str:
        """Transcribe entire audio at once."""
        start_time = time.time()

        # Get encoder output
        if self.mel_encoder:
            # Use fused mel+encoder
            encoder_out = self.mel_encoder.predict({
                "audio": audio.reshape(1, -1).astype(np.float32)
            })["encoder_out"]
        else:
            # Separate preprocessing and encoding
            mel = self.preprocessor.predict({
                "audio": audio.reshape(1, -1).astype(np.float32)
            })["mel"]

            encoder_out = self.encoder.predict({
                "mel": mel
            })["encoder_out"]

        # Greedy decoding
        transcript = self._greedy_decode(encoder_out)

        if self.verbose:
            elapsed = time.time() - start_time
            rtf = elapsed / self.max_audio_seconds
            logger.info(f"Transcription took {elapsed:.2f}s, RTF={rtf:.3f}")

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

    def _greedy_decode(self, encoder_out: np.ndarray) -> str:
        """Perform greedy decoding using RNNT algorithm."""
        # Initialize decoder states
        batch_size = 1
        h_state = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)
        c_state = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)

        # Start with blank token
        labels = np.array([[self.blank_id]], dtype=np.int32)

        transcript_ids = []
        time_steps = encoder_out.shape[1]

        for t in range(time_steps):
            # Get encoder output for this timestep
            enc_t = encoder_out[:, t:t+1, :]

            # Decode loop (simplified)
            for _ in range(10):  # Max symbols per timestep
                # Run decoder
                dec_out = self.decoder.predict({
                    "targets": labels,
                    "target_lengths": np.array([1], dtype=np.int32),
                    "h_in": h_state,
                    "c_in": c_state
                })

                # Update states
                h_state = dec_out["h_out"]
                c_state = dec_out["c_out"]
                decoder_out = dec_out["decoder_out"]

                # Joint network
                joint_out = self.joint.predict({
                    "encoder_out": enc_t,
                    "decoder_out": decoder_out
                })

                logits = joint_out["joint_out"]
                pred_id = np.argmax(logits[0, 0, :])

                if pred_id == self.blank_id:
                    break  # Move to next time step
                else:
                    transcript_ids.append(pred_id)
                    labels = np.array([[pred_id]], dtype=np.int32)

        # Convert IDs to text (simplified - would need actual tokenizer)
        return self._ids_to_text(transcript_ids)

    def _ids_to_text(self, ids: list) -> str:
        """Convert token IDs to text."""
        # This would use the actual tokenizer/vocabulary
        # For now, return a placeholder
        return " ".join(str(i) for i in ids)

    def unload(self):
        """Unload models from memory."""
        self.preprocessor = None
        self.encoder = None
        self.decoder = None
        self.joint = None
        self.mel_encoder = None
        logger.info(f"Model {self.model_id} unloaded")