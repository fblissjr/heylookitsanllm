#!/usr/bin/env python3
"""Test script for STT integration with heylookitsanllm."""
import io
import time
import logging
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_test_audio(duration_seconds=3, sample_rate=16000):
    """Generate a simple test audio file (sine wave)."""
    import wave

    # Generate a simple sine wave
    frequency = 440  # A4 note
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    audio = np.sin(frequency * 2 * np.pi * t) * 0.5

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer.getvalue()


def test_transcription_endpoint(base_url="http://localhost:8080"):
    """Test the /v1/audio/transcriptions endpoint."""
    logger.info("Testing transcription endpoint...")

    # Generate test audio
    audio_data = generate_test_audio(duration_seconds=2)
    logger.info(f"Generated test audio: {len(audio_data)} bytes")

    # Prepare the request
    files = {
        'file': ('test_audio.wav', audio_data, 'audio/wav')
    }
    data = {
        'model': 'parakeet-tdt-v3',
        'response_format': 'json'
    }

    # Send the request
    url = f"{base_url}/v1/audio/transcriptions"
    logger.info(f"Sending POST request to {url}")

    try:
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = time.time() - start_time

        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response time: {elapsed:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Transcription result: {result}")
            return True
        else:
            logger.error(f"Error response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to server. Make sure heylookitsanllm is running.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def test_stt_models_endpoint(base_url="http://localhost:8080"):
    """Test the /v1/stt/models endpoint."""
    logger.info("Testing STT models endpoint...")

    url = f"{base_url}/v1/stt/models"

    try:
        response = requests.get(url)
        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            models = response.json()
            logger.info(f"Available STT models: {models}")
            return True
        else:
            logger.error(f"Error response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to server.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def test_direct_provider():
    """Test the MLX STT provider directly (without server)."""
    logger.info("Testing MLX STT provider directly...")

    try:
        from heylook_llm.providers.mlx_stt_provider import MLXSTTProvider

        config = {
            "model_path": "mlx-community/parakeet-tdt-0.6b-v3",
        }

        # Create provider
        provider = MLXSTTProvider("parakeet-tdt-v3", config, verbose=True)

        # Load model
        logger.info("Loading model...")
        try:
            provider.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Generate test audio
        audio_data = generate_test_audio(duration_seconds=2)

        # Test transcription
        logger.info("Transcribing...")
        result = provider.transcribe(audio_data)
        logger.info(f"Direct transcription result: {result}")

        # Cleanup
        provider.unload()
        return True

    except ImportError as e:
        logger.error(f"Could not import provider: {e}")
        logger.info("Make sure to install: uv sync --extra stt")
        return False
    except Exception as e:
        logger.error(f"Direct provider test failed: {e}")
        return False


def main():
    """Run all STT integration tests."""
    logger.info("Starting STT integration tests...")

    # Test 1: Direct provider test (no server needed)
    logger.info("\n=== Test 1: Direct Provider ===")
    direct_success = test_direct_provider()

    # Test 2: STT models endpoint
    logger.info("\n=== Test 2: STT Models Endpoint ===")
    models_success = test_stt_models_endpoint()

    # Test 3: Transcription endpoint
    logger.info("\n=== Test 3: Transcription Endpoint ===")
    transcription_success = test_transcription_endpoint()

    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Direct provider test: {'✓ PASSED' if direct_success else '✗ FAILED'}")
    logger.info(f"STT models endpoint: {'✓ PASSED' if models_success else '✗ FAILED'}")
    logger.info(f"Transcription endpoint: {'✓ PASSED' if transcription_success else '✗ FAILED'}")

    all_passed = direct_success and models_success and transcription_success
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.info("\n✗ Some tests failed. Check the logs above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())