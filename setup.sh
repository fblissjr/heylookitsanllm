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

# Check for Python
if ! command_exists python && ! command_exists python3; then
    echo "ERROR: Python not found. Please install Python 3.9+ first."
    exit 1
fi

PYTHON_CMD=$(command_exists python3 && echo "python3" || echo "python")
echo "Using Python: $($PYTHON_CMD --version)"
echo ""

# Check for uv (optional but recommended)
USE_UV=false
if command_exists uv; then
    PIP_CMD="uv pip"
    USE_UV=true
    echo "Using uv for faster installation"
else
    PIP_CMD="$PYTHON_CMD -m pip"
    echo "Using pip (install uv for faster installs: pip install uv)"
fi
echo ""

# Ask user what to install
echo "What would you like to install?"
echo "1) MLX backend only (macOS, for MLX models)"
echo "2) Llama.cpp backend only (for GGUF models)"
echo "3) Both backends"
echo "4) Everything (all backends + performance + analytics + profiling)"
echo ""
read -p "Enter your choice (1-4): " choice

# Base installation
echo ""
echo "Installing base package..."
$PIP_CMD install -e .

case $choice in
    1)
        echo ""
        echo "Installing MLX backend..."
        echo "Note: This includes mlx-vlm which requires scipy. If you get build errors, run: brew install gcc"
        $PIP_CMD install -e ".[mlx]"
        
        echo ""
        echo "MLX backend installed successfully!"
        ;;
    2)
        echo ""
        echo "Installing llama.cpp backend..."
        $PIP_CMD install -e ".[llama-cpp]"
        
        # Ask about GPU acceleration
        echo ""
        echo "Would you like to enable GPU acceleration for llama.cpp?"
        if [[ "$OS" == "macos" ]]; then
            echo "1) Yes, compile with Metal support"
            echo "2) No, use CPU only"
            read -p "Enter your choice (1-2): " gpu_choice
            
            if [[ "$gpu_choice" == "1" ]]; then
                echo ""
                echo "Recompiling llama-cpp-python with Metal support..."
                if [[ "$USE_UV" == "true" ]]; then
                    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                else
                    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 $PYTHON_CMD -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                fi
            fi
        elif [[ "$OS" == "linux" ]]; then
            echo "1) Yes, compile with CUDA support (NVIDIA)"
            echo "2) No, use CPU only"
            read -p "Enter your choice (1-2): " gpu_choice
            
            if [[ "$gpu_choice" == "1" ]]; then
                echo ""
                echo "Recompiling llama-cpp-python with CUDA support..."
                if [[ "$USE_UV" == "true" ]]; then
                    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                else
                    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 $PYTHON_CMD -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                fi
            fi
        fi
        
        echo ""
        echo "Llama.cpp backend installed successfully!"
        ;;
    3)
        echo ""
        echo "Installing both backends..."
        echo "Note: This includes mlx-vlm which requires scipy. If you get build errors, run: brew install gcc"
        $PIP_CMD install -e ".[mlx,llama-cpp]"
        
        
        # Ask about GPU acceleration for llama.cpp
        echo ""
        echo "Would you like to enable GPU acceleration for llama.cpp?"
        if [[ "$OS" == "macos" ]]; then
            echo "1) Yes, compile with Metal support"
            echo "2) No, use CPU only"
            read -p "Enter your choice (1-2): " gpu_choice
            
            if [[ "$gpu_choice" == "1" ]]; then
                echo ""
                echo "Recompiling llama-cpp-python with Metal support..."
                if [[ "$USE_UV" == "true" ]]; then
                    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                else
                    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 $PYTHON_CMD -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                fi
            fi
        elif [[ "$OS" == "linux" ]]; then
            echo "1) Yes, compile with CUDA support (NVIDIA)"
            echo "2) No, use CPU only"
            read -p "Enter your choice (1-2): " gpu_choice
            
            if [[ "$gpu_choice" == "1" ]]; then
                echo ""
                echo "Recompiling llama-cpp-python with CUDA support..."
                if [[ "$USE_UV" == "true" ]]; then
                    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                else
                    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 $PYTHON_CMD -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                fi
            fi
        fi
        
        echo ""
        echo "Both backends installed successfully!"
        ;;
    4)
        echo ""
        echo "Installing everything..."
        echo "Note: This includes mlx-vlm which requires scipy. If you get build errors, run: brew install gcc"
        $PIP_CMD install -e ".[all]"
        
        
        # Ask about GPU acceleration for llama.cpp
        echo ""
        echo "Would you like to enable GPU acceleration for llama.cpp?"
        if [[ "$OS" == "macos" ]]; then
            echo "1) Yes, compile with Metal support"
            echo "2) No, use CPU only"
            read -p "Enter your choice (1-2): " gpu_choice
            
            if [[ "$gpu_choice" == "1" ]]; then
                echo ""
                echo "Recompiling llama-cpp-python with Metal support..."
                if [[ "$USE_UV" == "true" ]]; then
                    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                else
                    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 $PYTHON_CMD -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                fi
            fi
        elif [[ "$OS" == "linux" ]]; then
            echo "1) Yes, compile with CUDA support (NVIDIA)"
            echo "2) No, use CPU only"
            read -p "Enter your choice (1-2): " gpu_choice
            
            if [[ "$gpu_choice" == "1" ]]; then
                echo ""
                echo "Recompiling llama-cpp-python with CUDA support..."
                if [[ "$USE_UV" == "true" ]]; then
                    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                else
                    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 $PYTHON_CMD -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                fi
            fi
        fi
        
        echo ""
        echo "Everything installed successfully!"
        ;;
    *)
        echo "ERROR: Invalid choice"
        exit 1
        ;;
esac

# Check if models.yaml exists
echo ""
if [[ ! -f "models.yaml" ]]; then
    if [[ -f "models.yaml.example" ]]; then
        echo "Creating models.yaml from example..."
        cp models.yaml.example models.yaml
        echo "WARNING:  Please edit models.yaml to point to your model files"
    else
        echo "WARNING:  No models.yaml found. You'll need to create one or use 'heylookllm import'"
    fi
else
    echo "models.yaml already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit models.yaml to configure your models (or use 'heylookllm import')"
echo "2. Start the server: heylookllm"
echo ""
echo "For more options, run: heylookllm --help"