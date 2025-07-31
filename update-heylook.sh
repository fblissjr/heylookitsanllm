#!/bin/bash
# update-heylook.sh

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect pip command to use
if command_exists uv; then
    PIP_CMD="uv pip"
    echo "Using uv for package management"
else
    # Try to find python command
    if command_exists python3; then
        PIP_CMD="python3 -m pip"
    elif command_exists python; then
        PIP_CMD="python -m pip"
    else
        echo "ERROR: Python not found. Please install Python 3.9+ first."
        exit 1
    fi
    echo "Using pip for package management"
fi

echo "Updating MLX packages..."
$PIP_CMD install --upgrade mlx-lm mlx-vlm mlx

# Check OS and recompile llama-cpp-python accordingly
if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected. Recompiling llama-cpp-python for Metal..."
    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 $PIP_CMD install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
elif [[ "$(uname)" == "Linux" ]]; then
    # Check for NVIDIA GPU and CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "Linux with NVIDIA GPU detected. Recompiling llama-cpp-python for CUDA..."
        CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 $PIP_CMD install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    else
        echo "Linux detected. Recompiling llama-cpp-python for CPU..."
        $PIP_CMD install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    fi
else
    echo "Unsupported OS. Please update llama-cpp-python manually."
fi

echo "Done! Verify with: heylookllm --version"
