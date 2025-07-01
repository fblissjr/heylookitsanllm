# Unified LLM Server (MLX + Llama.cpp)

A lightweight, unified OpenAI-compatible API server that can run both Apple's MLX models and GGUF models via `llama-cpp-python` in a single, unified interface for on-device inference. Models can be hot swapped between MLX and GGUF models.

## 1. Installation

### 1.1. Clone the Repository
```bash
git clone https://github.com/fblissjr/edge-llm-server
cd edge-llm-server
```

### 1.2. Install mlx-lm and mlx-vlm
```
pip install mlx-lm mlx-vlm
```

This project vendors the necessary components from `mlx-vlm` to ensure stability, minimize excessive dependencies, and avoid dependency conflicts (particularly for python 3.13+)

### 1.3. Install llama.cpp
**For Apple Silicon:**
You MUST install llama-cpp-python with Metal support.
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -r requirements.txt --force-reinstall --no-cache-dir
```

**For NVIDIA / CUDA:**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -r requirements.txt --force-reinstall --no-cache-dir
```

**For CPU Only:**
```bash
pip install -r requirements.txt
```

## 2. Configuration

All models are defined in the **`models.yaml`** file. You **must** edit this file to point to your local models.

-   `id`: A unique alias for the model. This is what you'll use in your API calls.
-   `provider`: Must be either `mlx` or `llama_cpp`.
-   `config`: Provider-specific settings (e.g., `model_path`, `mmproj_path`)

For speculative decoding, you can define a draft model by adding `draft_model: true` to the model configuration.

See the provided `models.yaml` for examples.

## 3. Running the Server

Once configured, start the server from the root `edge-llm-server` directory:

```bash
# For standard logging and default host and port
edge-llm

# for verbose logging and performance metrics
edge-llm --log-level DEBUG

# to change the the host and port:
edge-llm --host 0.0.0.0 --port 9999
```

The server will be available at `http://127.0.0.1:8080`.
```

## 4. Running Tests

This project uses `pytest` and `pytest-mock` for unit testing. The tests are located in the `tests/` directory and use mocks to validate the server's logic without needing to load actual models.

```bash
# Apple Silicon (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -e .[test]

# NVIDIA / CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -e .[test]

# CPU Only
pip install -e .[test]
```

```bash
python -m pytest
```
