#!/bin/bash
# update-heylook.sh

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect package manager
USE_UV=false
if command_exists uv; then
    USE_UV=true
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

# With uv, use sync which properly resolves dependencies from lockfile
if [[ "$USE_UV" == "true" ]]; then
    echo "Updating lockfile to get latest mlx-lm and mlx-vlm..."
    uv lock --upgrade-package mlx-lm --upgrade-package mlx-vlm 2>/dev/null || uv lock --upgrade
else
    # Update the package and its dependencies using pyproject.toml
    $PIP_CMD install --upgrade -e .
fi

# Check which optional dependencies are installed and update them
echo "Checking for optional dependencies to update..."

# Build list of extras to sync
EXTRAS=""

# Check if llama-cpp is installed
if $PIP_CMD list 2>/dev/null | grep -q "llama-cpp-python"; then
    echo "Found llama-cpp backend..."
    EXTRAS="$EXTRAS llama-cpp"

    # GPU recompilation still needs pip for CMAKE_ARGS
    echo "Recompiling llama-cpp-python with GPU support..."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "macOS detected. Recompiling for Metal..."
        CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 $PIP_CMD install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    elif [[ "$(uname)" == "Linux" ]]; then
        if command -v nvidia-smi &> /dev/null; then
            echo "Linux with NVIDIA GPU detected. Recompiling for CUDA..."
            CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 $PIP_CMD install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
        else
            echo "Linux detected. Recompiling for CPU..."
            $PIP_CMD install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
        fi
    else
        echo "Updating llama-cpp-python without GPU acceleration."
        $PIP_CMD install --upgrade llama-cpp-python
    fi
fi

# Check if MLX is installed
if $PIP_CMD list 2>/dev/null | grep -q "^mlx "; then
    echo "Found MLX backend..."
    EXTRAS="$EXTRAS mlx"
fi

# Check if performance packages are installed
if $PIP_CMD list 2>/dev/null | grep -q "orjson"; then
    echo "Found performance packages..."
    EXTRAS="$EXTRAS performance"
fi

# Check if analytics packages are installed
if $PIP_CMD list 2>/dev/null | grep -q "duckdb"; then
    echo "Found analytics packages..."
    EXTRAS="$EXTRAS analytics"
fi

# Check if profiling packages are installed
if $PIP_CMD list 2>/dev/null | grep -q "py-spy"; then
    echo "Found profiling packages..."
    EXTRAS="$EXTRAS profile"
fi

# Check if STT packages are installed
if $PIP_CMD list 2>/dev/null | grep -q "coremltools"; then
    echo "Found STT packages..."
    EXTRAS="$EXTRAS stt"
fi

# Sync with uv or update with pip
if [[ "$USE_UV" == "true" ]]; then
    if [[ -n "$EXTRAS" ]]; then
        echo "Syncing packages with extras:$EXTRAS"
        EXTRA_ARGS=""
        for extra in $EXTRAS; do
            EXTRA_ARGS="$EXTRA_ARGS --extra $extra"
        done
        uv sync $EXTRA_ARGS
    else
        echo "Syncing base packages..."
        uv sync
    fi
else
    # Fallback to pip for each extra
    for extra in $EXTRAS; do
        echo "Updating $extra packages..."
        $PIP_CMD install --upgrade -e ".[$extra]"
    done
fi

echo "Done! Verify with: heylookllm --version"
