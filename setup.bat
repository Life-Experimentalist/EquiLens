@echo off
REM EquiLens One-Click Setup Launcher
REM Compatible with cmd, PowerShell, and various Windows shells

echo.
echo ============================================
echo 🔍 EquiLens One-Click Setup
echo ============================================
echo.

REM Check if PowerShell is available
powershell -Command "Write-Host '✅ PowerShell detected'" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🚀 Starting setup with PowerShell...
    echo.
    powershell -ExecutionPolicy Bypass -Command "& { try { $ErrorActionPreference = 'Stop'; Invoke-RestMethod -Uri 'https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1' | Invoke-Expression } catch { Write-Host '❌ Setup failed:' $_.Exception.Message -ForegroundColor Red; exit 1 } }"
    goto :end
)

REM Fallback: try to use curl and PowerShell Core if available
curl --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🚀 Downloading and running setup script...
    echo.
    curl -fsSL https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1 | powershell -ExecutionPolicy Bypass -
    goto :end
)

REM Last resort: try bitsadmin to download and run
echo 🚀 Downloading setup script...
bitsadmin /transfer "EquiLensSetup" https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1 "%TEMP%\equilens_setup.ps1" >nul
if %ERRORLEVEL% EQU 0 (
    echo 🚀 Running setup script...
    powershell -ExecutionPolicy Bypass -File "%TEMP%\equilens_setup.ps1"
    del "%TEMP%\equilens_setup.ps1" 2>nul
    goto :end
)

echo ❌ No compatible download method found.
echo.
echo Please ensure you have:
echo - PowerShell installed (comes with Windows)
echo - Or curl (from Git for Windows, WSL, etc.)
echo.
echo Manual installation:
echo 1. Download: https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1
echo 2. Run with PowerShell: powershell -ExecutionPolicy Bypass -File setup.ps1
echo.
pause
exit /b 1

:end
echo.
echo ============================================
echo 🎉 Setup process completed!
echo ============================================
echo.
pause
