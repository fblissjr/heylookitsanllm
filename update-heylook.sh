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

echo "Updating heylookitsanllm and dependencies..."
# Update the package and its dependencies using pyproject.toml
$PIP_CMD install --upgrade -e .

# Check which optional dependencies are installed and update them
echo "Checking for optional dependencies to update..."

# Check if llama-cpp is installed and update with GPU support if needed
if $PIP_CMD list | grep -q "llama-cpp-python"; then
    echo "Updating llama-cpp backend with GPU support..."
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
        echo "Unsupported OS. Updating llama-cpp-python without GPU acceleration."
        $PIP_CMD install --upgrade llama-cpp-python
    fi
fi

# Update MLX packages if installed
if $PIP_CMD list | grep -q "mlx"; then
    echo "Updating MLX backend..."
    $PIP_CMD install --upgrade -e ".[mlx]"
fi

# Update performance packages if installed
if $PIP_CMD list | grep -q "orjson"; then
    echo "Updating performance packages..."
    $PIP_CMD install --upgrade -e ".[performance]"
fi

# Update analytics packages if installed
if $PIP_CMD list | grep -q "duckdb"; then
    echo "Updating analytics packages..."
    $PIP_CMD install --upgrade -e ".[analytics]"
fi

# Update profiling packages if installed
if $PIP_CMD list | grep -q "py-spy"; then
    echo "Updating profiling packages..."
    $PIP_CMD install --upgrade -e ".[profile]"
fi

echo "Done! Verify with: heylookllm --version"
