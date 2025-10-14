# detect-gpu.ps1
# Detect GPU capabilities on Windows
# Helps users determine which GPU acceleration option to use

Write-Host ""
Write-Host "GPU Detection Tool" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host ""

# Get all video controllers
$gpus = Get-WmiObject Win32_VideoController

if (-not $gpus) {
    Write-Host "ERROR: Could not detect any GPUs" -ForegroundColor Red
    exit 1
}

Write-Host "Detected GPUs:" -ForegroundColor Yellow
Write-Host ""

$hasNvidia = $false
$hasAMD = $false
$hasIntel = $false

foreach ($gpu in $gpus) {
    Write-Host "Name: $($gpu.Name)" -ForegroundColor White
    Write-Host "Driver Version: $($gpu.DriverVersion)" -ForegroundColor Gray
    Write-Host "Video Processor: $($gpu.VideoProcessor)" -ForegroundColor Gray

    # Calculate VRAM
    if ($gpu.AdapterRAM) {
        $vramGB = [math]::Round($gpu.AdapterRAM / 1GB, 2)
        Write-Host "Video RAM: $vramGB GB" -ForegroundColor Gray
    }

    Write-Host ""

    # Detect vendor
    if ($gpu.Name -match "NVIDIA") {
        $hasNvidia = $true
    }
    if ($gpu.Name -match "AMD|Radeon") {
        $hasAMD = $true
    }
    if ($gpu.Name -match "Intel") {
        $hasIntel = $true
    }
}

Write-Host "Recommendations:" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host ""

if ($hasNvidia) {
    Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
    Write-Host "Recommended: Install CUDA Toolkit and compile with CUDA support" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    Write-Host "2. Install Visual Studio Build Tools (if not already installed)" -ForegroundColor White
    Write-Host "3. Run setup.ps1 and choose CUDA option" -ForegroundColor White
    Write-Host ""
    Write-Host "Or manually compile:" -ForegroundColor Cyan
    Write-Host '  $env:CMAKE_ARGS = "-DGGML_CUDA=on"' -ForegroundColor Gray
    Write-Host '  $env:FORCE_CMAKE = "1"' -ForegroundColor Gray
    Write-Host '  pip install --force-reinstall --no-cache-dir llama-cpp-python' -ForegroundColor Gray
    Write-Host ""
}

if ($hasAMD) {
    Write-Host "AMD GPU detected!" -ForegroundColor Green
    Write-Host "Recommended: Install Vulkan SDK and compile with Vulkan support" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "1. Install Vulkan SDK: https://vulkan.lunarg.com/sdk/home" -ForegroundColor White
    Write-Host "2. Install Visual Studio Build Tools (if not already installed)" -ForegroundColor White
    Write-Host "3. Run setup.ps1 and choose Vulkan option" -ForegroundColor White
    Write-Host ""
    Write-Host "Or manually compile:" -ForegroundColor Cyan
    Write-Host '  $env:CMAKE_ARGS = "-DGGML_VULKAN=on"' -ForegroundColor Gray
    Write-Host '  $env:FORCE_CMAKE = "1"' -ForegroundColor Gray
    Write-Host '  pip install --force-reinstall --no-cache-dir llama-cpp-python' -ForegroundColor Gray
    Write-Host ""
}

if ($hasIntel -and -not $hasNvidia -and -not $hasAMD) {
    Write-Host "Intel GPU detected!" -ForegroundColor Yellow
    Write-Host "Note: Integrated Intel GPUs may not provide significant acceleration" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can try Vulkan support, but CPU-only might be better:" -ForegroundColor Cyan
    Write-Host "1. For Vulkan: Install Vulkan SDK and Visual Studio Build Tools" -ForegroundColor White
    Write-Host "2. For CPU-only: Use default llama-cpp-python installation" -ForegroundColor White
    Write-Host ""
}

if (-not $hasNvidia -and -not $hasAMD -and -not $hasIntel) {
    Write-Host "No discrete GPU detected" -ForegroundColor Yellow
    Write-Host "Recommended: Use CPU-only installation" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "CPU-only is perfectly usable for:" -ForegroundColor Cyan
    Write-Host "- Small models (< 7B parameters)" -ForegroundColor White
    Write-Host "- Quantized models (Q4_K_M, Q5_K_M)" -ForegroundColor White
    Write-Host "- Non-real-time use cases" -ForegroundColor White
    Write-Host ""
}

# Check for CUDA Toolkit
Write-Host "Checking for CUDA Toolkit..." -ForegroundColor Cyan
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
    $cudaVersions = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" | Where-Object { $_.PSIsContainer }
    if ($cudaVersions) {
        Write-Host "CUDA Toolkit installed:" -ForegroundColor Green
        foreach ($version in $cudaVersions) {
            Write-Host "  - $($version.Name)" -ForegroundColor White
        }
    }
} else {
    Write-Host "CUDA Toolkit not found" -ForegroundColor Yellow
    if ($hasNvidia) {
        Write-Host "Install from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
    }
}
Write-Host ""

# Check for Vulkan SDK
Write-Host "Checking for Vulkan SDK..." -ForegroundColor Cyan
if (Test-Path "C:\VulkanSDK") {
    $vulkanVersions = Get-ChildItem "C:\VulkanSDK" | Where-Object { $_.PSIsContainer }
    if ($vulkanVersions) {
        Write-Host "Vulkan SDK installed:" -ForegroundColor Green
        foreach ($version in $vulkanVersions) {
            Write-Host "  - $($version.Name)" -ForegroundColor White
        }
    }
} else {
    Write-Host "Vulkan SDK not found" -ForegroundColor Yellow
    if ($hasAMD -or $hasIntel) {
        Write-Host "Install from: https://vulkan.lunarg.com/sdk/home" -ForegroundColor Cyan
    }
}
Write-Host ""

# Check for Visual Studio Build Tools
Write-Host "Checking for Visual Studio Build Tools..." -ForegroundColor Cyan
$vsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
)

$vsFound = $false
foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        Write-Host "Visual Studio Build Tools found: $path" -ForegroundColor Green
        $vsFound = $true
        break
    }
}

if (-not $vsFound) {
    Write-Host "Visual Studio Build Tools not found" -ForegroundColor Yellow
    Write-Host "Required for compiling llama-cpp-python with GPU support" -ForegroundColor Yellow
    Write-Host "Install from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "Select 'Desktop development with C++' during installation" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "========" -ForegroundColor Cyan
if ($hasNvidia) {
    Write-Host "Best option: CUDA (NVIDIA GPU)" -ForegroundColor Green
} elseif ($hasAMD) {
    Write-Host "Best option: Vulkan (AMD GPU)" -ForegroundColor Green
} elseif ($hasIntel) {
    Write-Host "Best option: CPU-only or Vulkan (integrated GPU)" -ForegroundColor Yellow
} else {
    Write-Host "Best option: CPU-only" -ForegroundColor Yellow
}
Write-Host ""
