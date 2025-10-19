# EquiLens One-Click Setup Script (moved to scripts/install)
﻿# Compatible with PowerShell, cmd, and various Windows shells

param(
    [switch]$SkipDocker,
    [switch]$SkipModels,
    [string]$InstallPath = "$env:USERPROFILE\EquiLens",
    [string]$PythonVersion = "3.11"
)

# Configuration
$REPO_URL = "https://github.com/Life-Experimentalist/EquiLens.git"
$SCRIPT_VERSION = "2.0.0"
$REQUIRED_PYTHON = "3.11"
$REQUIRED_MEMORY_GB = 4
$REQUIRED_DISK_GB = 2

# Colors for output
$Colors = @{
    Green = "Green"
    Red = "Red"
    Yellow = "Yellow"
    Blue = "Blue"
    Cyan = "Cyan"
    Magenta = "Magenta"
    White = "White"
}

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host "`n$("=" * 60)" -ForegroundColor Cyan
    Write-Host "🔍 $Title" -ForegroundColor Cyan
    Write-Host "$("=" * 60)" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Step, [string]$Description = "")
    Write-Host "`n📋 $Step" -ForegroundColor Yellow
    if ($Description) {
        Write-Host "   $Description" -ForegroundColor Gray
    }
}

function Test-Command {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Get-AvailableSpaceGB {
    param([string]$Path = ".")
    try {
        $drive = Get-PSDrive -Name ($Path -replace ":", "")
        return [math]::Round($drive.Free / 1GB, 2)
    } catch {
        return 0
    }
}

function Test-PythonVersion {
    try {
        $pythonVersion = & python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $version = $pythonVersion -replace "Python ", ""
            $versionParts = $version -split "\."
            $major = [int]$versionParts[0]
            $minor = [int]$versionParts[1]

            if ($major -eq 3 -and $minor -ge 11) {
                return $true, $version
            }
        }
    } catch {}

    try {
        $python3Version = & python3 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $version = $python3Version -replace "Python ", ""
            $versionParts = $version -split "\."
            $major = [int]$versionParts[0]
            $minor = [int]$versionParts[1]

            if ($major -eq 3 -and $minor -ge 11) {
                return $true, $version
            }
        }
    } catch {}

    return $false, $null
}

function Install-UV {
    Write-Step "Installing UV" "Fast Python package manager"

    try {
        # Install UV using pip
        & python -m pip install --user uv

        # Add to PATH for current session
        $uvPath = "$env:USERPROFILE\AppData\Roaming\Python\Scripts"
        if (Test-Path $uvPath) {
            $env:PATH = "$uvPath;$env:PATH"
        }

        # Verify installation
        if (Test-Command "uv") {
            Write-ColorOutput "✅ UV installed successfully" $Colors.Green
            return $true
        } else {
            Write-ColorOutput "❌ UV installation failed" $Colors.Red
            return $false
        }
    } catch {
        Write-ColorOutput "❌ Failed to install UV: $($_.Exception.Message)" $Colors.Red
        return $false
    }
}

function Copy-Repository {
    param([string]$Path)

    Write-Step "Cloning EquiLens Repository" "Downloading source code from GitHub"

    try {
        if (Test-Path $Path) {
            Write-ColorOutput "📁 Directory exists, updating..." $Colors.Yellow
            Push-Location $Path
            & git pull
            Pop-Location
        } else {
            Write-ColorOutput "📥 Cloning repository..." $Colors.Blue
            & git clone $REPO_URL $Path
        }

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Repository ready at: $Path" $Colors.Green
            return $true
        } else {
            Write-ColorOutput "❌ Failed to clone/update repository" $Colors.Red
            return $false
        }
    } catch {
        Write-ColorOutput "❌ Repository setup failed: $($_.Exception.Message)" $Colors.Red
        return $false
    }
}

function Install-Dependencies {
    param([string]$ProjectPath)

    Write-Step "Installing Dependencies" "Setting up Python environment with UV"

    try {
        Push-Location $ProjectPath

        # Create virtual environment and install dependencies
        Write-ColorOutput "🔧 Setting up virtual environment..." $Colors.Blue
        & uv sync

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Dependencies installed successfully" $Colors.Green
            Pop-Location
            return $true
        } else {
            Write-ColorOutput "❌ Dependency installation failed" $Colors.Red
            Pop-Location
            return $false
        }
    } catch {
        Write-ColorOutput "❌ Dependency setup failed: $($_.Exception.Message)" $Colors.Red
        Pop-Location
        return $false
    }
}

function Start-DockerServices {
    param([string]$ProjectPath)

    if ($SkipDocker) {
        Write-ColorOutput "⏭️ Skipping Docker services (as requested)" $Colors.Yellow
        return $true
    }

    Write-Step "Starting Docker Services" "Launching Ollama and other services"

    try {
        Push-Location $ProjectPath

        # Check if Docker is running
        $null = & docker --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "⚠️ Docker not detected. Please start Docker Desktop manually." $Colors.Yellow
            Write-ColorOutput "   Then run: docker-compose up -d ollama" $Colors.Cyan
            Pop-Location
            return $false
        }

        Write-ColorOutput "🐳 Starting Ollama service..." $Colors.Blue
        & docker-compose up -d ollama

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Docker services started" $Colors.Green
            Write-ColorOutput "⏳ Waiting for Ollama to initialize..." $Colors.Yellow
            Start-Sleep -Seconds 10
            Pop-Location
            return $true
        } else {
            Write-ColorOutput "❌ Failed to start Docker services" $Colors.Red
            Pop-Location
            return $false
        }
    } catch {
        Write-ColorOutput "❌ Docker services failed: $($_.Exception.Message)" $Colors.Red
        Pop-Location
        return $false
    }
}

function Get-AIModels {
    if ($SkipModels) {
        Write-ColorOutput "⏭️ Skipping model downloads (as requested)" $Colors.Yellow
        return $true
    }

    Write-Step "Downloading AI Models" "Setting up Phi-3 model for testing"

    try {
        # Wait a bit more for Ollama to be ready
        Start-Sleep -Seconds 5

        Write-ColorOutput "🤖 Downloading Phi-3 model (2GB)..." $Colors.Blue
        & docker-compose exec ollama ollama pull phi3 2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Phi-3 model downloaded successfully" $Colors.Green
            return $true
        } else {
            Write-ColorOutput "⚠️ Model download failed, but setup continues" $Colors.Yellow
            Write-ColorOutput "   You can download models later with: docker-compose exec ollama ollama pull <model-name>" $Colors.Cyan
            return $true  # Don't fail the entire setup for this
        }
    } catch {
        Write-ColorOutput "⚠️ Model download failed: $($_.Exception.Message)" $Colors.Yellow
        return $true  # Continue setup
    }
}

function Invoke-Verification {
    param([string]$ProjectPath)

    Write-Step "Running Final Verification" "Ensuring everything works"

    try {
        Push-Location $ProjectPath

        Write-ColorOutput "🔍 Running setup verification..." $Colors.Blue
        & uv run scripts/setup/verify_setup.py

        $exitCode = $LASTEXITCODE
        Pop-Location

        if ($exitCode -eq 0) {
            Write-ColorOutput "✅ All checks passed!" $Colors.Green
            return $true
        } else {
            Write-ColorOutput "⚠️ Some checks failed, but setup completed" $Colors.Yellow
            return $true
        }
    } catch {
        Write-ColorOutput "⚠️ Verification failed: $($_.Exception.Message)" $Colors.Yellow
        Pop-Location
        return $true  # Don't fail setup for verification
    }
}

function Show-NextSteps {
    param([string]$ProjectPath)

    Write-Header "🎉 EquiLens Setup Complete!"

    Write-Host "`n🚀 Quick Start Commands:" -ForegroundColor Green
    Write-Host "   cd '$ProjectPath'" -ForegroundColor Cyan
    Write-Host "   uv run equilens --help" -ForegroundColor White
    Write-Host "   uv run python -m equilens.gradio_ui" -ForegroundColor White

    Write-Host "`n🌐 Web Interface:" -ForegroundColor Green
    Write-Host "   uv run equilens gui" -ForegroundColor White
    Write-Host "   Opens at: http://localhost:7860" -ForegroundColor Gray

    Write-Host "`n📊 Check Status:" -ForegroundColor Green
    Write-Host "   uv run equilens status" -ForegroundColor White

    Write-Host "`n📚 Documentation:" -ForegroundColor Green
    Write-Host "   SETUP_COMPLETE.md" -ForegroundColor White
    Write-Host "   docs/QUICKSTART.md" -ForegroundColor White

    Write-Host "`n🔧 Useful Commands:" -ForegroundColor Green
    Write-Host "   docker-compose up -d ollama" -ForegroundColor White
    Write-Host "   docker-compose exec ollama ollama list" -ForegroundColor White
    Write-Host "   uv run scripts/setup/verify_setup.py" -ForegroundColor White

    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "🎯 Happy Bias Hunting with EquiLens!" -ForegroundColor Magenta
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

# Main setup function
function Install-EquiLens {
    Write-Header "EquiLens One-Click Setup v$SCRIPT_VERSION"

    Write-Host "`n🔍 EquiLens - AI Bias Detection Platform" -ForegroundColor Cyan
    Write-Host "📦 Automated Windows Setup Script" -ForegroundColor Gray
    Write-Host "🏠 Install Path: $InstallPath" -ForegroundColor Gray

    # Prerequisites check
    Write-Step "Checking Prerequisites"

    # Check Windows version
    $osInfo = Get-ComputerInfo
    Write-ColorOutput "✅ Windows $($osInfo.WindowsProductName)" $Colors.Green

    # Check available disk space
    $freeSpace = Get-AvailableSpaceGB
    if ($freeSpace -lt $REQUIRED_DISK_GB) {
        Write-ColorOutput "❌ Insufficient disk space: ${freeSpace}GB free (need ${REQUIRED_DISK_GB}GB+)" $Colors.Red
        exit 1
    }
    Write-ColorOutput "✅ Disk space: ${freeSpace}GB available" $Colors.Green

    # Check memory
    $memoryGB = [math]::Round((Get-ComputerInfo).TotalPhysicalMemory / 1GB, 1)
    if ($memoryGB -lt $REQUIRED_MEMORY_GB) {
        Write-ColorOutput "❌ Insufficient memory: ${memoryGB}GB (need ${REQUIRED_MEMORY_GB}GB+)" $Colors.Red
        exit 1
    }
    Write-ColorOutput "✅ Memory: ${memoryGB}GB available" $Colors.Green

    # Check Python
    $pythonOk, $pythonVersion = Test-PythonVersion
    if (-not $pythonOk) {
        Write-ColorOutput "❌ Python $REQUIRED_PYTHON+ required. Please install Python from https://python.org" $Colors.Red
        exit 1
    }
    Write-ColorOutput "✅ Python $pythonVersion" $Colors.Green

    # Check Git
    if (-not (Test-Command "git")) {
        Write-ColorOutput "❌ Git required. Please install from https://git-scm.com" $Colors.Red
        exit 1
    }
    Write-ColorOutput "✅ Git available" $Colors.Green

    # Install UV if needed
    if (-not (Test-Command "uv")) {
        if (-not (Install-UV)) {
            exit 1
        }
    } else {
        Write-ColorOutput "✅ UV already installed" $Colors.Green
    }

    # Clone repository
    if (-not (Copy-Repository $InstallPath)) {
        exit 1
    }

    # Install dependencies
    if (-not (Install-Dependencies $InstallPath)) {
        exit 1
    }

    # Start Docker services
    Start-DockerServices $InstallPath

    # Download models
    Get-AIModels

    # Run verification
    Invoke-Verification $InstallPath

    # Show next steps
    Show-NextSteps $InstallPath
}

# Run the setup
try {
    Install-EquiLens
} catch {
    Write-ColorOutput "`n❌ Setup failed with error: $($_.Exception.Message)" $Colors.Red
    Write-ColorOutput "Please check the error messages above and try again." $Colors.Yellow
    exit 1
}
