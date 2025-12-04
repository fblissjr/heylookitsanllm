# update-heylook.ps1
# Update heylookitsanllm and dependencies on Windows

param(
    [switch]$SkipGPU,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Update Script for heylookitsanllm (Windows)

Usage:
    .\update-heylook.ps1              Interactive update
    .\update-heylook.ps1 -SkipGPU     Skip GPU recompilation prompts
    .\update-heylook.ps1 -Help        Show this help message

This script will:
  - Pull latest changes from git
  - Update Python dependencies
  - Optionally recompile llama-cpp-python with GPU support

"@
    exit 0
}

Write-Host ""
Write-Host "Updating heylookitsanllm..." -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-CommandExists {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Check for git
if (-not (Test-CommandExists git)) {
    Write-Host "ERROR: git not found. Please install git first." -ForegroundColor Red
    exit 1
}

# Check for Python
$python = $null
if (Test-CommandExists python) {
    $python = "python"
} elseif (Test-CommandExists python3) {
    $python = "python3"
} else {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Check for uv
$pip_cmd = "$python -m pip"
$use_uv = $false
if (Test-CommandExists uv) {
    $pip_cmd = "uv pip"
    $use_uv = $true
    Write-Host "Using uv for faster installation" -ForegroundColor Green
}

# Update from git
Write-Host "Pulling latest changes from git..." -ForegroundColor Yellow
git pull

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: git pull failed" -ForegroundColor Red
    Write-Host "Please resolve any conflicts and try again" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Updating packages..." -ForegroundColor Yellow
if ($use_uv) {
    # Use uv sync for proper dependency resolution
    Write-Host "Updating lockfile to get latest versions..." -ForegroundColor Yellow
    & uv lock --upgrade-package mlx-lm --upgrade-package mlx-vlm 2>$null
    if ($LASTEXITCODE -ne 0) {
        & uv lock --upgrade
    }
    Write-Host "Syncing with llama-cpp backend..." -ForegroundColor Yellow
    & uv sync --extra llama-cpp
} else {
    & $python -m pip install -e . --upgrade
    & $python -m pip install -e ".[llama-cpp]" --upgrade
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Package update failed" -ForegroundColor Red
    exit 1
}

# Ask about GPU recompilation
if (-not $SkipGPU) {
    Write-Host ""
    Write-Host "Would you like to recompile llama-cpp-python with GPU support?" -ForegroundColor Cyan
    Write-Host "1) Yes, with CUDA support (NVIDIA)" -ForegroundColor White
    Write-Host "2) Yes, with Vulkan support (AMD/Intel)" -ForegroundColor White
    Write-Host "3) No, keep current configuration" -ForegroundColor White
    Write-Host ""
    $choice = Read-Host "Enter your choice (1-3)"

    switch ($choice) {
        1 {
            Write-Host ""
            Write-Host "Recompiling with CUDA support..." -ForegroundColor Yellow
            Write-Host "NOTE: This requires CUDA Toolkit and Visual Studio Build Tools" -ForegroundColor Red
            Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
            Read-Host

            $env:CMAKE_ARGS = "-DGGML_CUDA=on"
            $env:FORCE_CMAKE = "1"
            if ($use_uv) {
                & uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
            } else {
                & $python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
            }

            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: CUDA compilation failed" -ForegroundColor Red
                Write-Host "The update was successful, but GPU support compilation failed" -ForegroundColor Yellow
            } else {
                Write-Host "CUDA support recompiled successfully!" -ForegroundColor Green
            }
        }
        2 {
            Write-Host ""
            Write-Host "Recompiling with Vulkan support..." -ForegroundColor Yellow
            Write-Host "NOTE: This requires Vulkan SDK and Visual Studio Build Tools" -ForegroundColor Red
            Write-Host "Press Enter to continue or Ctrl+C to cancel..." -ForegroundColor Yellow
            Read-Host

            $env:CMAKE_ARGS = "-DGGML_VULKAN=on"
            $env:FORCE_CMAKE = "1"
            if ($use_uv) {
                & uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
            } else {
                & $python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
            }

            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Vulkan compilation failed" -ForegroundColor Red
                Write-Host "The update was successful, but GPU support compilation failed" -ForegroundColor Yellow
            } else {
                Write-Host "Vulkan support recompiled successfully!" -ForegroundColor Green
            }
        }
        3 {
            Write-Host "Keeping current configuration" -ForegroundColor Green
        }
        default {
            Write-Host "Invalid choice, keeping current configuration" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Update complete!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now restart the server with: heylookllm" -ForegroundColor Cyan
Write-Host ""
