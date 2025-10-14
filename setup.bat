@echo off
REM setup.bat - Batch wrapper for setup.ps1
REM Bypasses PowerShell execution policy for convenience

echo Windows Setup for heylookitsanllm
echo ==================================
echo.

REM Check if setup.ps1 exists
if not exist setup.ps1 (
    echo ERROR: setup.ps1 not found in current directory
    echo Please run this script from the heylookitsanllm directory
    pause
    exit /b 1
)

REM Run PowerShell script with execution policy bypass
echo Running setup.ps1...
echo.
powershell -ExecutionPolicy Bypass -File setup.ps1 %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
pause
