# Hey Look, It's an LLM (!)
OpenAI API *and* ollama API compatible LLM and Vision LLM (VLM) / multimodal server for mlx + llama.cpp

a lightweight (and lighthearted, but still aiming for quality), OpenAI-compatible API server that runs both vision and text Apple MLX models (text via `mlx-lm`, and vision via `mlx-vlm`, with some `mlx` stitching) and GGUF models (via `llama-cpp-python`) behind one endpoint, with live on-the-fly model swapping via API calls. trying to take the best of what's out locally and put it under one roof in a smart, performant way. allows for running in openai api mode, ollama mode, or both together.

i'll aim to make a better install guide, given the dependencies, but hopefully this is fairly smooth sailing for now.

*note*: llama-cpp-python will by default install the cpu binary, but you can compile it yourself by following the instructions in the llama.cpp repo.

## 0.1 Quick Usage and Updates
You can now run mlx, mlx-vlm, and gguf/llama.cpp models, all hot swappable as before, in openai mode (default), ollama compatibility mode (middleware to translate requetss from ollama api format to openai format) which runs on the ollama default port, or both together.

```bash

# openai mode
heylookitsanllm --api "openai" --log-level DEBUG

# ollama mode, defaults to 11434 port
heylookitsanllm --api "ollama" --log-level DEBUG

# both together, but you can only run one port at a time here. not ideal, but future todo.
heylookitsanllm --api "both" --api "ollama" --log-level DEBUG --port 8080
```
---
## 1  Installation
tldr:
1. clone repo
2. uv or pip install -e . (this will install llama.cpp cpu but you can compile metal or cuda next)
5. decide if you want to compile llama.cpp with metal or cuda (you're probably using metal if you're here for mlx - if so, compile metal)


### 1.1  Clone & bootstrap
```bash
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm

# basic install (includes all needed dependencies now)
uv pip install -e .

# or, if you want the performance goodies (see [macOS Performance Guide](MACOS_PERFORMANCE.md) for details)
uv pip install -e .[performance]
```

the `[performance]` option gives you:
- **orjson**: 3-10x faster JSON operations (matters when you're sending images)
- **xxhash**: 50x faster image hashing for deduplication
- **turbojpeg**: 4-10x faster JPEG encoding/decoding
- **uvloop**: faster async event loop (linux/macos only)

these are optional but recommended if you're doing anything serious with vision models or high throughput.

### 1.2 Decide what flavor of llama.cpp you want
Install a pre-built llama.cpp binary or compile (my tip: compile if you're on macos/metal or CUDA).

#### 1.2.1 Install a pre-built CPU-only llama.cpp binary

Here's how to install the cpu binary:

```bash
# macOS / Linux
brew install llama.cpp

# Windows
winget install llama.cpp
```

Installing the binary first avoids a 10-15 min local C++ build. The official repo recommends these routes.

#### 1.2.2  Compiling llama-cpp-python for macOS (metal) or CUDA with the right flags

# macOS / Metal
`CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python`

# CPU only
`uv pip install --force-reinstall --no-cache-dir llama-cpp-python`

# NVIDIA CUDA (12.x shown)
`CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python`

If you skipped 1.2, the first pip install already built a CPU wheel; the commands above simply re-compile it in place with Metal or GPU support. See upstream docs for the full flag matrix.

### 1.3  trust (but verify)

# You should see a Metal or CUDA line if the compile succeeded (and you compiled it for that).
`llama.cpp`: python -c "import llama_cpp; print('llama.cpp version:', llama_cpp.llama_cpp_version())"

# mlx-lm
`mlx-lm`: python -c "from mlx_lm import __version__; print('mlx_lm version:', __version__)"

# mlx-vlm
`mlx-vlm`: python -c "from mlx_vlm import __version__; print('mlx_vlm version:', __version__)"

## 2. Configuration

All models are defined in the **`models.yaml`** file. You **must** edit this file to point to your local models. Here's some of the key fields - though since we're in active development, these are likely to be incomplete and/or changing. Will do my best not to make it madness.

- `id`: a unique alias for the model - or rather, the short name you’ll hit in API calls
- `provider`: must be either `mlx` or `llama_cpp`.
- `config`: provider and model specific settings (e.g., `model_path`, `mmproj_path`)
- `draft_model`: true – marks a fast “draft” model for speculative decoding

My setup is to use the included `modelzoo` directory and set up a symbolic link to it. On Mac and Linux and WSL2, you can do this: `ln -s /my_models/live/here/* .` and you'll have them all there without duplication. At some point I'll get hugging face cache going.

See the provided `models.yaml` for examples of different model setups with mlx, llama.cpp / gguf, including vision, text, both, etc.

## 3. Running the Server

Once configured, start the server from the root `heylookitsanllm` directory. Note that if you want to access the server from another machine on your local network, set the host to `0.0.0.0`. Otherwise, it defaults to `127.0.0.1`, which means it can only be accessed from the local machine it's running on.

```bash
# defaults to 127.0.0.1:8080
heylookllm

# verbose logging + perf metrics
heylookllm --log-level DEBUG

# custom host / port
heylookllm --host 0.0.0.0 --port 4242
```

### 3.1 Model Import (new!)

tired of manually editing models.yaml? we got you. the import command scans directories and generates model configs with smart defaults:

```bash
# scan a directory (follows symlinks)
heylookllm import --folder ~/modelzoo

# scan huggingface cache
heylookllm import --hf-cache

# use profiles for different use cases
heylookllm import --folder ~/models --profile fast      # speed optimized
heylookllm import --folder ~/models --profile quality   # quality optimized
heylookllm import --folder ~/models --profile memory    # low memory usage

# fine-tune with overrides
heylookllm import --folder ~/models --profile fast --override temperature=0.5
```

profiles:
- `fast`: aggressive sampling, quantized cache for speed
- `balanced`: default, good middle ground
- `quality`: conservative sampling, standard cache
- `memory`: maximum memory savings
- `interactive`: optimized for chat use

the importer detects:
- model size from filenames or file sizes
- vision support (mmproj files, config flags)
- quantization (4bit, 8bit, etc)
- model family (llama, qwen, gemma, mistral)

all imported models start disabled, so you can review before enabling.

## 4. Running Tests

For now, lots of debug tests in the `tests` directory that you can run manually.

## 5. Coming Soon

working on some cool stuff:

### 5.1 DuckDB Integration
- **request/response logging**: full conversation history in a queryable database
- **eval tracking**: compare model outputs, track quality over time
- **analytics**: token usage, response times, model performance metrics
- **sql your llm**: `SELECT response FROM chats WHERE model='qwen2.5' AND tokens/second > 5`

### 5.2 Example Client Apps
- **comfyui nodes**: already have [shrug-prompter](https://github.com/heylookitsanllm/shrug-prompter), but now even better with multipart support
- **native macOS app**: swift UI with model switcher, image drag-n-drop, conversation history
- **web dashboard**: react app with real-time streaming, model comparison, batch processing


### 5.3 Other Goodies
- **model recommendations**: based on your hardware and use case
- **automatic model downloads**: point at huggingface, get the right format
- **performance profiler**: see exactly where time is spent in your requests
- **fine-tuning integration**: use your conversations to improve models
