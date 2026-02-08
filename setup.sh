#!/bin/bash
# Automated setup script for heylookitsanllm

set -e  # Exit on error

echo "Hey Look It's an LLM - Setup Script"
echo "==================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"
echo ""

# Require uv
if ! command_exists uv; then
    echo "ERROR: uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "Using uv for package management"
echo ""

# Function to install with uv sync (uses lockfile, properly resolves dependencies)
# Usage: uv_sync_install "extra1" "extra2" ...
uv_sync_install() {
    echo "Updating lockfile to get latest versions..."
    uv lock --upgrade-package mlx-lm --upgrade-package mlx-vlm --upgrade-package parakeet-mlx 2>/dev/null || uv lock
    if [[ $# -eq 0 ]] || [[ -z "$1" ]]; then
        uv sync
    else
        local extra_args=""
        for extra in "$@"; do
            extra_args="$extra_args --extra $extra"
        done
        uv sync $extra_args
    fi
}

# Ask user what to install
echo "What would you like to install?"
echo "1) MLX backend only (macOS, for MLX models)"
echo "2) STT backend only (Parakeet MLX Speech-to-Text)"
echo "3) MLX + STT backends"
echo "4) Everything (MLX + STT + performance + analytics + profiling)"
echo ""
read -p "Enter your choice (1-4): " choice

# Base installation
echo ""
echo "Installing base package..."
uv_sync_install ""

case $choice in
    1)
        echo ""
        echo "Installing MLX backend..."
        echo "Note: This includes mlx-vlm which requires scipy. If you get build errors, run: brew install gcc"
        uv_sync_install "mlx"

        echo ""
        echo "MLX backend installed successfully!"
        ;;
    2)
        echo ""
        echo "Installing STT backend (Parakeet MLX)..."
        uv_sync_install "stt"

        echo ""
        echo "STT backend installed successfully!"
        echo "Configure your Parakeet model in models.toml with provider: 'mlx_stt'"
        ;;
    3)
        echo ""
        echo "Installing MLX and STT backends..."
        echo "Note: This includes mlx-vlm which requires scipy. If you get build errors, run: brew install gcc"
        uv_sync_install "mlx" "stt"

        echo ""
        echo "MLX and STT backends installed successfully!"
        echo "Configure your STT models in models.toml with provider: 'mlx_stt'"
        ;;
    4)
        echo ""
        echo "Installing everything..."
        echo "Note: This includes mlx-vlm which requires scipy. If you get build errors, run: brew install gcc"
        uv_sync_install "all"

        echo ""
        echo "Everything installed successfully!"
        echo "Configure your STT models in models.toml with provider: 'mlx_stt'"
        ;;
    *)
        echo "ERROR: Invalid choice"
        exit 1
        ;;
esac

# Check if models.toml exists
echo ""
if [[ ! -f "models.toml" ]]; then
    if [[ -f "models.toml.example" ]]; then
        echo "Creating models.toml from example..."
        cp models.toml.example models.toml
        echo "WARNING:  Please edit models.toml to point to your model files"
    else
        echo "WARNING:  No models.toml found. You'll need to create one or use 'heylookllm import'"
    fi
else
    echo "models.toml already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit models.toml to configure your models (or use 'heylookllm import')"
echo "2. Start the server: heylookllm"
echo ""
echo "For more options, run: heylookllm --help"
