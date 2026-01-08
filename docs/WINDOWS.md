# Windows Guide

Complete guide for installing and running heylookitsanllm on Windows 10/11.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [GPU Acceleration](#gpu-acceleration)
4. [Model Configuration](#model-configuration)
5. [Server Commands](#server-commands)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tips](#performance-tips)
9. [Command Reference](#command-reference)

---

## Quick Start

```powershell
# 1. Clone and setup
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm
.\setup.ps1                    # Or: .\setup.bat

# 2. Configure models (edit models.yaml with your paths)

# 3. Start server
heylookllm --log-level DEBUG

# 4. Test
curl http://localhost:8080/v1/models
```

---

## Prerequisites

### Required

- **Windows 10/11** (64-bit)
- **Python 3.11+** ([download](https://www.python.org/downloads/))
  - Check "Add Python to PATH" during installation
  - Verify: `python --version`
- **Git for Windows** ([download](https://git-scm.com/download/win))

### For GPU Acceleration (Optional)

**NVIDIA GPUs:**
- **Visual Studio Build Tools** ([download](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022))
  - Select "Desktop development with C++" (~7GB)
- **CUDA Toolkit 11.8 or 12.x** ([download](https://developer.nvidia.com/cuda-downloads))
  - Verify: `nvcc --version`

**AMD/Intel GPUs:**
- **Visual Studio Build Tools** (same as above)
- **Vulkan SDK** ([download](https://vulkan.lunarg.com/sdk/home))
  - Verify: `vulkaninfo`

---

## GPU Acceleration

### Detect Your GPU

```powershell
.\detect-gpu.ps1
```

### NVIDIA CUDA

```powershell
# 1. Install Visual Studio Build Tools + CUDA Toolkit
# 2. Compile llama-cpp-python with CUDA
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python

# 3. Verify
nvcc --version
nvidia-smi
```

### AMD/Intel Vulkan

```powershell
# 1. Install Visual Studio Build Tools + Vulkan SDK
# 2. Compile llama-cpp-python with Vulkan
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
$env:FORCE_CMAKE = "1"
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python

# 3. Verify
vulkaninfo
```

---

## Model Configuration

### Example models.yaml

```yaml
models:
  - id: qwen-coder
    provider: llama_cpp
    enabled: true
    config:
      model_path: C:\Users\YourName\models\qwen.gguf
      n_gpu_layers: -1  # -1 = all layers on GPU
      n_ctx: 4096
      n_batch: 512
      use_mmap: true
      chat_format: qwen

  - id: llama-vision
    provider: llama_cpp
    enabled: true
    config:
      model_path: C:\models\llama-vision.gguf
      mmproj_path: C:\models\llama-vision-mmproj.gguf
      vision: true
      n_gpu_layers: -1
      n_ctx: 4096
```

**Note:** Use Windows paths (`C:\...`) or forward slashes (`C:/...`)

### Model Import

```powershell
# Scan directory
heylookllm import --folder C:\Users\YourName\models --output models.yaml

# Scan HuggingFace cache
heylookllm import --hf-cache --output models.yaml

# Use profile
heylookllm import --folder C:\models --profile fast
heylookllm import --folder C:\models --profile quality
```

### Common Paths

| Path | Location |
|------|----------|
| Models | `C:\Users\YourName\models\` |
| Config | `C:\Users\YourName\heylookitsanllm\models.yaml` |
| HF Cache | `C:\Users\YourName\.cache\huggingface\hub\` |

---

## Server Commands

```powershell
# Start server (port 8080)
heylookllm

# Custom host and port
heylookllm --host 0.0.0.0 --port 4242

# With debug logging
heylookllm --log-level DEBUG

# With file logging
heylookllm --log-level INFO --file-log-level DEBUG --log-dir logs

# Check version
heylookllm --version
```

---

## Testing

```powershell
# Test server is running
Invoke-WebRequest -Uri http://localhost:8080/v1/models

# Using curl
curl http://localhost:8080/v1/models

# Test chat completion
$body = @{
    model = "qwen-coder"
    messages = @(@{ role = "user"; content = "Hello!" })
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8080/v1/chat/completions `
  -Method POST -ContentType "application/json" -Body $body
```

---

## Troubleshooting

### Quick Fixes

| Issue | Solution |
|-------|----------|
| Python not found | Add Python to PATH or use `C:\Python311\python.exe` |
| Port in use | Use `--port 8081` or kill process with `Stop-Process` |
| Setup won't run | Use `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| CUDA not working | Verify `nvcc --version` and reinstall CUDA Toolkit |
| Vulkan not working | Verify `vulkaninfo` and update GPU drivers |
| Models not loading | Check paths use Windows-style `C:\` format |
| Slow performance | Enable GPU, use quantized models, add Defender exclusion |
| Compilation fails | Install Visual Studio Build Tools (Desktop development with C++) |

### Port Already in Use

```powershell
# Find process using port
Get-NetTCPConnection -LocalPort 8080

# Kill process (replace PID)
Stop-Process -Id <PID>
```

### CUDA Compilation Fails

1. **CUDA not in PATH:**
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
   ```

2. **Missing Visual Studio:**
   ```powershell
   where cl.exe  # Should return path
   ```

3. **CMake errors:**
   ```powershell
   pip install cmake
   ```

### Windows Defender Exclusion

```powershell
# Run as Administrator
Add-MpPreference -ExclusionPath "C:\Users\YourName\models"
Add-MpPreference -ExclusionPath "C:\Users\YourName\heylookitsanllm"
```

### Firewall Rule

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "heylookllm Server" `
  -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

---

## Performance Tips

### Hardware

- **SSD Storage:** Keep models on SSD for faster loading
- **RAM:** 8GB minimum, 16GB+ recommended
- **GPU VRAM:** More VRAM = larger models fit in GPU memory

### Model Selection

- Use quantized GGUF models (Q4_K_M, Q5_K_M)
- Lower `n_ctx` if running out of memory

### Config Optimization

```yaml
config:
  n_gpu_layers: -1      # All layers on GPU
  n_ctx: 4096           # Context size (lower if OOM)
  n_batch: 512          # Batch size (higher = faster)
  n_threads: 8          # CPU threads (if CPU-only)
  use_mmap: true        # Memory-mapped file loading
```

---

## Command Reference

### Installation

```powershell
git clone https://github.com/fblissjr/heylookitsanllm
cd heylookitsanllm
.\setup.ps1              # Interactive setup
.\detect-gpu.ps1         # Check GPU
```

### Server

```powershell
heylookllm                           # Start (port 8080)
heylookllm --port 8081               # Custom port
heylookllm --log-level DEBUG         # Debug logging
heylookllm --host 0.0.0.0            # LAN access
```

### Model Import

```powershell
heylookllm import --folder C:\models --output models.yaml
heylookllm import --hf-cache --profile fast
```

### Diagnostics

```powershell
python --version                     # Check Python
where python                         # Python location
nvcc --version                       # CUDA version
vulkaninfo                           # Vulkan info
nvidia-smi                           # GPU status
Get-NetTCPConnection -LocalPort 8080 # Port check
Test-Path "C:\path\to\model.gguf"    # File exists
```

### Updating

```powershell
.\update-heylook.ps1
# Or manually:
git pull && pip install -e . --upgrade
```

---

## What Doesn't Work on Windows

These features are macOS-only:

- **MLX Provider** - Apple Silicon only (use llama.cpp instead)
- **CoreML STT** - Apple frameworks only
- **MLX STT** - Apple Silicon only

Windows support focuses on the llama.cpp backend with GGUF models.

---

## Links

- **Python:** https://www.python.org/downloads/
- **Git:** https://git-scm.com/download/win
- **Visual Studio Build Tools:** https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **Vulkan SDK:** https://vulkan.lunarg.com/sdk/home
- **GitHub Issues:** https://github.com/fblissjr/heylookitsanllm/issues
