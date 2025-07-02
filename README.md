# Unified LLM Server (MLX + Llama.cpp)

A lightweight, OpenAI-compatible API server that runs Apple MLX models and GGUF models (via `llama-cpp-python`) behind one endpoint.

---
## 1  Installation

### 1.1  Clone & bootstrap
```bash
git clone https://github.com/fblissjr/edge-llm-server
cd edge-llm-server

uv pip install -e .
uv pip install --no-deps mlx-vlm          # skip its mlx-audio chain and gradio
uv pip install -r requirements-min.txt    # installs minimal dependencies needed
```

### 1.2  (Recommended) install a pre-built llama.cpp binary

```bash
# macOS / Linux
brew install llama.cpp
# Windows
winget install llama.cpp
```

Installing the binary first avoids a 10-15 min local C++ build. The official repo recommends these routes.

### 1.3  Build llama-cpp-python with the right flags

# Apple Silicon + Metal
`CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir llama-cpp-python`

# NVIDIA CUDA (12.x shown)
`CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install --force-reinstall --no-cache-dir llama-cpp-python`

If you skipped 1.2, the first pip install already built a CPU wheel; the commands above simply re-compile it in place with GPU support. See upstream docs for the full flag matrix.

### 1.4  Verify

python -c "import llama_cpp; print('llama.cpp version:', llama_cpp.llama_cpp_version())"

You should see a Metal or CUDA line if the compile succeeded.

## 1  Installation

### 1.1  Clone & bootstrap

```bash
# 1  Grab the code
git clone https://github.com/fblissjr/edge-llm-server
cd edge-llm-server

# 2  Install the core server
uv pip install -e .

# 3  Pull mlx-vlm *without* its heavy optional deps
uv pip install --no-deps mlx-vlm

# 4  Add the few libs mlx-vlm actually needs
uv pip install -r requirements-min.txt

# 5 Install llama.cpp backend
uv pip install edge-llm[metal] # macos
uv pip install edge-llm[cuda] # nvidia
uv pip install edge-llm[cpu] # cpu only
```

- need another CUDA version? Change the suffix (cu121, cu122) in the command above.
- edge-llm[cpu] installs the vanilla llama.cpp wheel; the metal and cuda extras swap in pre-built GPU wheels via --extra-index-url, so you avoid a local CMake build. but if you want to do a cmake (i typically do) check out the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) repo.

## 2. Configuration

All models are defined in the **`models.yaml`** file. You **must** edit this file to point to your local models.

- `id`: a unique alias for the model - or rather, the short name you’ll hit in API calls
- `provider`: must be either `mlx` or `llama_cpp`.
- `config`: provider and model specific settings (e.g., `model_path`, `mmproj_path`)
- `draft_model`: true – marks a fast “draft” model for speculative decoding

See the provided `models.yaml` for examples.

## 3. Running the Server

Once configured, start the server from the root `edge-llm-server` directory:

```bash
# defaults to 127.0.0.1:8080
edge-llm

# verbose logging + perf metrics
edge-llm --log-level DEBUG

# custom host / port
edge-llm --host 0.0.0.0 --port 4242
```

The server will be available at `http://127.0.0.1:8080`.

## 4. Running Tests
```bash
# Apple Silicon (Metal)
uv pip install -e .[test,metal]

# NVIDIA / CUDA
uv pip install -e .[test,cuda]

# CPU-only
uv pip install -e .[test,cpu]

python -m pytest
```
