# setup.ps1
# Windows Setup Script for heylookitsanllm
# Automated setup for llama.cpp backend with GPU acceleration options

param(
    [switch]$SkipGPU,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Windows Setup Script for heylookitsanllm

Usage:
    .\setup.ps1              Interactive setup
    .\setup.ps1 -SkipGPU     Skip GPU acceleration prompts
    .\setup.ps1 -Help        Show this help message

This script will:
  - Detect Python installation
  - Install base dependencies
  - Guide through GPU acceleration options (CUDA/Vulkan)
  - Create initial models.yaml configuration

"@
    exit 0
}

# Banner
Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Hey Look It's an LLM - Setup Script" -ForegroundColor Cyan
Write-Host "Windows Edition" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-CommandExists {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check for Python
Write-Host "Checking for Python..." -ForegroundColor Yellow
$python = $null
if (Test-CommandExists python) {
    $python = "python"
} elseif (Test-CommandExists python3) {
    $python = "python3"
} else {
    Write-Host "ERROR: Python not found. Please install Python 3.11+ from python.org" -ForegroundColor Red
    Write-Host "Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

# Check Python version
$pythonVersion = & $python --version 2>&1
Write-Host "Using: $pythonVersion" -ForegroundColor Green

# Check for uv (optional but recommended)
$pip_cmd = "$python -m pip"
$use_uv = $false
if (Test-CommandExists uv) {
    $pip_cmd = "uv pip"
    $use_uv = $true
    Write-Host "Using uv for faster installation" -ForegroundColor Green
} else {
    Write-Host "Using pip (install uv for faster installs: pip install uv)" -ForegroundColor Yellow
}

Write-Host ""

# Check for Visual Studio Build Tools (informational)
$vsBuildTools = Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if (-not $vsBuildTools) {
    $vsBuildTools = Test-Path "C:\Program Files\Microsoft Visual Studio\2022\Community"
}

if (-not $vsBuildTools) {
    Write-Host "WARNING: Visual Studio Build Tools not detected" -ForegroundColor Yellow
    Write-Host "You'll need this to compile llama-cpp-python with GPU support" -ForegroundColor Yellow
    Write-Host "Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "Select 'Desktop development with C++' during installation" -ForegroundColor Cyan
    Write-Host ""
}

# Ask user what to install
Write-Host "What would you like to install?" -ForegroundColor Cyan
Write-Host "1) Llama.cpp backend only (for GGUF models)" -ForegroundColor White
Write-Host "2) Everything (llama.cpp + performance + analytics + profiling)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter your choice (1-2)"

# Base installation
Write-Host ""
Write-Host "Installing base package..." -ForegroundColor Yellow
if ($use_uv) {
    & uv pip install -e .
} else {
    & $python -m pip install -e .
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Base installation failed" -ForegroundColor Red
    exit 1
}

switch ($choice) {
    1 {
        Write-Host ""
        Write-Host "Installing llama.cpp backend..." -ForegroundColor Yellow
        if ($use_uv) {
            & uv pip install -e ".[llama-cpp]"
        } else {
            & $python -m pip install -e ".[llama-cpp]"
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: llama.cpp installation failed" -ForegroundColor Red
            exit 1
        }

        # GPU acceleration menu
        if (-not $SkipGPU) {
            Write-Host ""
            Write-Host "Would you like to enable GPU acceleration for llama.cpp?" -ForegroundColor Cyan
            Write-Host "1) Yes, compile with CUDA support (NVIDIA GPUs)" -ForegroundColor White
            Write-Host "2) Yes, compile with Vulkan support (AMD/Intel GPUs)" -ForegroundColor White
            Write-Host "3) No, use CPU only" -ForegroundColor White
            Write-Host ""
            $gpu_choice = Read-Host "Enter your choice (1-3)"

            switch ($gpu_choice) {
                1 {
                    Write-Host ""
                    Write-Host "Compiling llama-cpp-python with CUDA support..." -ForegroundColor Yellow
                    Write-Host "NOTE: This requires CUDA Toolkit and Visual Studio Build Tools" -ForegroundColor Red
                    Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
                    Read-Host

                    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
                    $env:FORCE_CMAKE = "1"
                    if ($use_uv) {
                        & uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                    } else {
                        & $python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                    }

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "ERROR: CUDA compilation failed" -ForegroundColor Red
                        Write-Host "Make sure CUDA Toolkit and Visual Studio Build Tools are installed" -ForegroundColor Yellow
                    } else {
                        Write-Host "CUDA support compiled successfully!" -ForegroundColor Green
                    }
                }
                2 {
                    Write-Host ""
                    Write-Host "Compiling llama-cpp-python with Vulkan support..." -ForegroundColor Yellow
                    Write-Host "NOTE: This requires Vulkan SDK and Visual Studio Build Tools" -ForegroundColor Red
                    Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
                    Read-Host

                    $env:CMAKE_ARGS = "-DGGML_VULKAN=on"
                    $env:FORCE_CMAKE = "1"
                    if ($use_uv) {
                        & uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                    } else {
                        & $python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                    }

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "ERROR: Vulkan compilation failed" -ForegroundColor Red
                        Write-Host "Make sure Vulkan SDK and Visual Studio Build Tools are installed" -ForegroundColor Yellow
                    } else {
                        Write-Host "Vulkan support compiled successfully!" -ForegroundColor Green
                    }
                }
                3 {
                    Write-Host "Using CPU-only llama-cpp-python" -ForegroundColor Green
                }
            }
        }

        Write-Host ""
        Write-Host "Llama.cpp backend installed successfully!" -ForegroundColor Green
    }
    2 {
        Write-Host ""
        Write-Host "Installing everything..." -ForegroundColor Yellow
        if ($use_uv) {
            & uv pip install -e ".[llama-cpp,performance,analytics,profile]"
        } else {
            & $python -m pip install -e ".[llama-cpp,performance,analytics,profile]"
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Installation failed" -ForegroundColor Red
            exit 1
        }

        # GPU acceleration menu (same as above)
        if (-not $SkipGPU) {
            Write-Host ""
            Write-Host "Would you like to enable GPU acceleration for llama.cpp?" -ForegroundColor Cyan
            Write-Host "1) Yes, compile with CUDA support (NVIDIA GPUs)" -ForegroundColor White
            Write-Host "2) Yes, compile with Vulkan support (AMD/Intel GPUs)" -ForegroundColor White
            Write-Host "3) No, use CPU only" -ForegroundColor White
            Write-Host ""
            $gpu_choice = Read-Host "Enter your choice (1-3)"

            switch ($gpu_choice) {
                1 {
                    Write-Host ""
                    Write-Host "Compiling llama-cpp-python with CUDA support..." -ForegroundColor Yellow
                    Write-Host "NOTE: This requires CUDA Toolkit and Visual Studio Build Tools" -ForegroundColor Red
                    Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
                    Read-Host

                    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
                    $env:FORCE_CMAKE = "1"
                    if ($use_uv) {
                        & uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                    } else {
                        & $python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                    }

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "ERROR: CUDA compilation failed" -ForegroundColor Red
                        Write-Host "Make sure CUDA Toolkit and Visual Studio Build Tools are installed" -ForegroundColor Yellow
                    } else {
                        Write-Host "CUDA support compiled successfully!" -ForegroundColor Green
                    }
                }
                2 {
                    Write-Host ""
                    Write-Host "Compiling llama-cpp-python with Vulkan support..." -ForegroundColor Yellow
                    Write-Host "NOTE: This requires Vulkan SDK and Visual Studio Build Tools" -ForegroundColor Red
                    Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
                    Read-Host

                    $env:CMAKE_ARGS = "-DGGML_VULKAN=on"
                    $env:FORCE_CMAKE = "1"
                    if ($use_uv) {
                        & uv pip install --force-reinstall --no-cache-dir llama-cpp-python
                    } else {
                        & $python -m pip install --force-reinstall --no-cache-dir llama-cpp-python
                    }

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "ERROR: Vulkan compilation failed" -ForegroundColor Red
                        Write-Host "Make sure Vulkan SDK and Visual Studio Build Tools are installed" -ForegroundColor Yellow
                    } else {
                        Write-Host "Vulkan support compiled successfully!" -ForegroundColor Green
                    }
                }
                3 {
                    Write-Host "Using CPU-only llama-cpp-python" -ForegroundColor Green
                }
            }
        }

        Write-Host ""
        Write-Host "Everything installed successfully!" -ForegroundColor Green
    }
    default {
        Write-Host "ERROR: Invalid choice" -ForegroundColor Red
        exit 1
    }
}

# Check if models.yaml exists
Write-Host ""
if (-not (Test-Path "models.yaml")) {
    if (Test-Path "models.yaml.example") {
        Write-Host "Creating models.yaml from example..." -ForegroundColor Yellow
        Copy-Item "models.yaml.example" "models.yaml"
        Write-Host "WARNING: Please edit models.yaml to point to your model files" -ForegroundColor Red
    } else {
        Write-Host "WARNING: No models.yaml found. You'll need to create one or use 'heylookllm import'" -ForegroundColor Yellow
    }
} else {
    Write-Host "models.yaml already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit models.yaml to configure your models (or use 'heylookllm import')" -ForegroundColor White
Write-Host "2. Start the server: heylookllm" -ForegroundColor White
Write-Host ""
Write-Host "For more options, run: heylookllm --help" -ForegroundColor White
Write-Host ""
Write-Host "For Windows-specific documentation, see: docs/WINDOWS_INSTALL.md" -ForegroundColor Cyan
