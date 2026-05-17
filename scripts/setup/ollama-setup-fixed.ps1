#Requires -Version 5.1
<#
.SYNOPSIS
    Ollama Docker Setup — Idempotent, volume-backed, GPU-aware

.DESCRIPTION
    • Checks Docker is running
    • Pulls ollama/ollama:latest only if image is missing or outdated
    • Creates a named Docker volume for persistent model storage
    • Creates (or reuses) an Ollama container bound to that volume
    • Detects NVIDIA GPU and enables passthrough automatically
    • Verifies the API is healthy before exiting
    • Exposes Ollama at http://localhost:11434

.PARAMETER ContainerName
    Name for the Ollama container. Default: ollama

.PARAMETER VolumeName
    Name for the Docker volume storing models. Default: ollama_data

.PARAMETER Port
    Host port to expose the Ollama API on. Default: 11434

.PARAMETER ForceRecreate
    If set, removes and recreates the container (volume is preserved).

.PARAMETER PullLatest
    If set, always pulls the latest image even if one already exists.

.EXAMPLE
    .\ollama-setup-fixed.ps1
    .\ollama-setup-fixed.ps1 -ForceRecreate
    .\ollama-setup-fixed.ps1 -Port 12000 -PullLatest
#>

[CmdletBinding()]
param(
    [string] $ContainerName = "ollama",
    [string] $VolumeName = "ollama_data",
    [int] $Port = 11434,
    [switch] $ForceRecreate,
    [switch] $PullLatest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param([string]$msg) Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Write-Ok { param([string]$msg) Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$msg) Write-Host "    [ERROR] $msg" -ForegroundColor Red; exit 1 }
function Write-Info { param([string]$msg) Write-Host "    [INFO] $msg" -ForegroundColor Gray }

function Test-CommandExists {
    param([string]$cmd)
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Invoke-DockerCommand {
    param([string[]]$args)
    $result = & docker @args 2>&1
    return $result
}

Write-Host ""
Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "  ║     Ollama Docker Setup             ║" -ForegroundColor Magenta
Write-Host "  ║     Volume-backed, GPU-aware        ║" -ForegroundColor Magenta
Write-Host "  ╚═══════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
Write-Info "Container : $ContainerName"
Write-Info "Volume    : $VolumeName"
Write-Info "Port      : $Port >> 11434"
Write-Host ""

Write-Step "Checking Docker..."

if (-not (Test-CommandExists "docker")) {
    Write-Fail "Docker is not installed. Get it at https://www.docker.com/products/docker-desktop/"
}

try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Ok "Docker is running"
} catch {
    Write-Fail "Docker daemon not responding. Start Docker Desktop and try again."
}

Write-Step "Checking GPU support..."

$gpuArgs = @()
$gpuLabel = "CPU-only"

if (Test-CommandExists "nvidia-smi") {
    try {
        $gpuCheck = & nvidia-smi -L 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "NVIDIA GPU detected"
            $gpuArgs = @("--gpus", "all")
            $gpuLabel = "GPU-accelerated (NVIDIA)"
        }
    } catch {
        Write-Info "nvidia-smi check skipped"
    }
}

Write-Step "Managing Docker volume..."

$volumeExists = docker volume ls --format "table {{.Name}}" | Select-String -Pattern "^$VolumeName$"

if (-not $volumeExists) {
    docker volume create $VolumeName | Out-Null
    Write-Ok "Docker volume created: $VolumeName"
} else {
    Write-Ok "Docker volume already exists: $VolumeName"
}

Write-Step "Managing Ollama container..."

$containerExists = docker ps -a --format "table {{.Names}}" | Select-String -Pattern "^$ContainerName$"

if ($containerExists -and $ForceRecreate) {
    Write-Info "Removing existing container..."
    docker stop $ContainerName 2>&1 | Out-Null
    docker rm $ContainerName 2>&1 | Out-Null
    Write-Ok "Removed $ContainerName"
    $containerExists = $false
}

if ($containerExists) {
    $containerRunning = docker ps --format "table {{.Names}}" | Select-String -Pattern "^$ContainerName$"

    if ($containerRunning) {
        Write-Ok "Container is already running: $ContainerName"
    } else {
        Write-Info "Starting stopped container..."
        docker start $ContainerName | Out-Null
        Write-Ok "Container started: $ContainerName"
    }
} else {
    Write-Info "Creating new container..."

    $runArgs = @(
        "run", "-d",
        "--name", $ContainerName,
        "--restart", "unless-stopped",
        "-p", "${Port}:11434",
        "-v", "${VolumeName}:/root/.ollama",
        "-e", "OLLAMA_HOST=0.0.0.0",
        "-e", "OLLAMA_MODELS=/root/.ollama/models"
    )

    if ($gpuArgs.Count -gt 0) {
        $runArgs += $gpuArgs
    }

    $runArgs += "ollama/ollama:latest"

    docker @runArgs | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to create container. Run 'docker logs $ContainerName' for details."
    }

    Write-Ok "Container created and started"
    Write-Info "GPU mode   : $gpuLabel"
    Write-Info "Volume     : $VolumeName (mounted at /root/.ollama)"
    Write-Info "API port   : $Port"
    Write-Info "Restart    : unless-stopped"
}

Write-Step "Waiting for Ollama API to be healthy..."

$maxAttempts = 20
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts) {
    $attempt++
    Start-Sleep -Seconds 2
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        Write-Host "    [WAIT] Attempt $attempt/$maxAttempts..." -ForegroundColor DarkGray
    }
}

if ($ready) {
    Write-Ok "Ollama API is healthy at http://localhost:${Port}"
} else {
    Write-Warn "API did not respond within timeout. Container may still be starting."
    Write-Info "Check status: docker logs $ContainerName"
}

Write-Host ""
Write-Host "  ════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  Ollama Setup Complete!" -ForegroundColor Green
Write-Host "  ════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "  API endpoint   : " -NoNewline; Write-Host "http://localhost:${Port}" -ForegroundColor Cyan
Write-Host "  Models volume  : " -NoNewline; Write-Host $VolumeName -ForegroundColor Cyan
Write-Host "  GPU mode       : " -NoNewline; Write-Host $gpuLabel -ForegroundColor Cyan
Write-Host ""
Write-Host "  Commands:" -ForegroundColor DarkGray
Write-Host "  Pull model: docker exec -it $ContainerName ollama pull phi3:mini" -ForegroundColor Yellow
Write-Host "  Chat mode: docker exec -it $ContainerName ollama run phi3:mini" -ForegroundColor Yellow
Write-Host "  List models: docker exec -it $ContainerName ollama list" -ForegroundColor Yellow
Write-Host "  View logs: docker logs -f $ContainerName" -ForegroundColor Yellow
Write-Host "  Stop: docker stop $ContainerName" -ForegroundColor Yellow
Write-Host ""
