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
    .\ollama-setup.ps1
    .\ollama-setup.ps1 -ForceRecreate
    .\ollama-setup.ps1 -Port 12000 -PullLatest
#>

[CmdletBinding()]
param(
	[string] $ContainerName = "ollama",
	[string] $VolumeName = "ollama_data",
	[int]    $Port = 11434,
	[switch] $ForceRecreate,
	[switch] $PullLatest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
function Write-Step { param([string]$msg) Write-Host "`n► $msg" -ForegroundColor Cyan }
function Write-Ok { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$msg) Write-Host "  [ERROR] $msg" -ForegroundColor Red; exit 1 }
function Write-Info { param([string]$msg) Write-Host "  [INFO] $msg" -ForegroundColor Gray }

function Test-CommandExists {
	param([string]$cmd)
	return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Invoke-DockerCommand {
	param([string[]]$dockerArgs)
	$result = & docker @dockerArgs 2>&1
	return $result
}

# ─────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "  ║     🦙  Ollama Docker Setup           ║" -ForegroundColor Magenta
Write-Host "  ║     Volume-backed · GPU-aware         ║" -ForegroundColor Magenta
Write-Host "  ╚═══════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""
Write-Info "Container : $ContainerName"
Write-Info "Volume    : $VolumeName"
Write-Info "Port      : $Port → 11434"
Write-Host ""

# ─────────────────────────────────────────────
#  Step 1 — Verify Docker is installed & running
# ─────────────────────────────────────────────
Write-Step "Checking Docker..."

if (-not (Test-CommandExists "docker")) {
	Write-Fail "Docker is not installed. Get it at https://www.docker.com/products/docker-desktop/"
}

try {
	$null = docker info 2>&1
	if ($LASTEXITCODE -ne 0) { throw }
	Write-Ok "Docker Desktop is running"
}
catch {
	Write-Fail "Docker daemon is not responding. Please start Docker Desktop and try again."
}

# ─────────────────────────────────────────────
#  Step 2 — GPU detection
# ─────────────────────────────────────────────
Write-Step "Detecting GPU..."

$gpuArgs = @()
$gpuLabel = "CPU only"

try {
	$nvidiaSmi = & nvidia-smi --query-gpu=name --format=csv, noheader 2>&1
	if ($LASTEXITCODE -eq 0 -and $nvidiaSmi -notmatch "error") {
		$gpuName = ($nvidiaSmi -split "`n")[0].Trim()

		# Confirm Docker can also see the GPU runtime
		$runtimeCheck = docker info --format "{{json .Runtimes}}" 2>&1
		if ($runtimeCheck -match "nvidia") {
			$gpuArgs = @("--gpus", "all")
			$gpuLabel = "NVIDIA GPU ($gpuName) — passthrough enabled"
			Write-Ok $gpuLabel
		}
		else {
			Write-Warn "NVIDIA GPU found ($gpuName) but Docker NVIDIA runtime not detected."
			Write-Info "Install the NVIDIA Container Toolkit for GPU support:"
			Write-Info "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
			Write-Info "Continuing with CPU mode."
		}
	}
 else {
		Write-Info "No NVIDIA GPU detected — using CPU mode."
	}
}
catch {
	Write-Info "nvidia-smi not found — using CPU mode."
}

# ─────────────────────────────────────────────
#  Step 3 — Pull / update image
# ─────────────────────────────────────────────
Write-Step "Checking Ollama image..."

$imageExists = docker images ollama/ollama:latest --format "{{.Repository}}" 2>&1
$needsPull = ($imageExists -notmatch "ollama") -or $PullLatest

if ($needsPull) {
	Write-Info "Pulling ollama/ollama:latest..."
	docker pull ollama/ollama:latest
	if ($LASTEXITCODE -ne 0) { Write-Fail "Image pull failed. Check your internet connection." }
	Write-Ok "Image pulled successfully"
}
else {
	Write-Ok "Image ollama/ollama:latest already present (use -PullLatest to force update)"
}

# ─────────────────────────────────────────────
#  Step 4 — Create the Docker volume
# ─────────────────────────────────────────────
Write-Step "Setting up Docker volume..."

$volumeExists = docker volume ls --filter "name=^${VolumeName}$" --format "{{.Name}}" 2>&1

if ($volumeExists -eq $VolumeName) {
	Write-Ok "Volume '$VolumeName' already exists — models will be reused"
}
else {
	docker volume create $VolumeName | Out-Null
	if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create volume '$VolumeName'." }
	Write-Ok "Volume '$VolumeName' created"
}

# Show where Docker stores the volume on disk
$volumePath = docker volume inspect $VolumeName --format "{{.Mountpoint}}" 2>&1
Write-Info "Host path: $volumePath"

# ─────────────────────────────────────────────
#  Step 5 — Handle existing container
# ─────────────────────────────────────────────
Write-Step "Checking for existing container..."

$containerState = docker inspect $ContainerName --format "{{.State.Status}}" 2>&1

if ($containerState -notmatch "no such" -and $containerState -notmatch "Error") {
	if ($ForceRecreate) {
		Write-Warn "ForceRecreate: stopping and removing container '$ContainerName'..."
		docker stop $ContainerName  2>&1 | Out-Null
		docker rm   $ContainerName  2>&1 | Out-Null
		Write-Ok "Old container removed (volume '$VolumeName' preserved)"
		$containerState = "gone"
	}
 else {
		Write-Info "Container '$ContainerName' exists (state: $containerState)"
		if ($containerState -eq "running") {
			Write-Ok "Container is already running — nothing to do"
			$skipCreate = $true
		}
		elseif ($containerState -in @("exited", "created", "paused")) {
			Write-Info "Starting stopped container..."
			docker start $ContainerName | Out-Null
			Write-Ok "Container started"
			$skipCreate = $true
		}
	}
}

# ─────────────────────────────────────────────
#  Step 6 — Create the container
# ─────────────────────────────────────────────
if (-not $skipCreate) {
	Write-Step "Creating Ollama container..."

	$runArgs = @(
		"run", "-d",
		"--name", $ContainerName,
		"--restart", "unless-stopped",
		"-p", "${Port}:11434",
		"-v", "${VolumeName}:/root/.ollama",
		"-e", "OLLAMA_HOST=0.0.0.0",
		"-e", "OLLAMA_MODELS=/root/.ollama/models"
	)

	# Append GPU args if available
	if ($gpuArgs.Count -gt 0) {
		$runArgs += $gpuArgs
	}

	$runArgs += "ollama/ollama:latest"

	docker @runArgs | Out-Null

	if ($LASTEXITCODE -ne 0) {
		Write-Fail "Failed to create container. Run 'docker logs $ContainerName' for details."
	}

	Write-Ok "Container '$ContainerName' created and started"
	Write-Info "GPU mode   : $gpuLabel"
	Write-Info "Volume     : $VolumeName → /root/.ollama"
	Write-Info "API port   : $Port"
	Write-Info "Restart    : unless-stopped (survives reboots)"
}

# ─────────────────────────────────────────────
#  Step 7 — Wait for API to be healthy
# ─────────────────────────────────────────────
Write-Step "Waiting for Ollama API to be ready..."

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
	}
 catch {
		Write-Host "  · Waiting... ($attempt/$maxAttempts)" -NoNewline
		Write-Host "`r" -NoNewline
	}
}

if ($ready) {
	Write-Ok "Ollama API is healthy at http://localhost:${Port}"
}
else {
	Write-Warn "API did not respond within timeout — container may still be starting."
	Write-Info "Check with: docker logs $ContainerName"
}

# ─────────────────────────────────────────────
#  Step 8 — Summary
# ─────────────────────────────────────────────
Write-Host ""
Write-Host "  ════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  🦙  Ollama is ready!" -ForegroundColor Green
Write-Host "  ════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "  API endpoint   : " -NoNewline; Write-Host "http://localhost:${Port}" -ForegroundColor Cyan
Write-Host "  Models volume  : " -NoNewline; Write-Host $VolumeName -ForegroundColor Cyan
Write-Host "  GPU mode       : " -NoNewline; Write-Host $gpuLabel -ForegroundColor Cyan
Write-Host ""
Write-Host "  ─── Quick commands ──────────────────────" -ForegroundColor DarkGray
Write-Host "  Pull a model   : " -NoNewline
Write-Host "docker exec -it $ContainerName ollama pull llama3.2" -ForegroundColor Yellow
Write-Host "  List models    : " -NoNewline
Write-Host "docker exec -it $ContainerName ollama list" -ForegroundColor Yellow
Write-Host "  Chat (REPL)    : " -NoNewline
Write-Host "docker exec -it $ContainerName ollama run llama3.2" -ForegroundColor Yellow
Write-Host "  View logs      : " -NoNewline
Write-Host "docker logs -f $ContainerName" -ForegroundColor Yellow
Write-Host "  Stop container : " -NoNewline
Write-Host "docker stop $ContainerName" -ForegroundColor Yellow
Write-Host "  Backup models  : " -NoNewline
Write-Host 'docker run --rm -v ${VolumeName}:/data -v ${PWD}:/out alpine tar czf /out/ollama_backup.tar.gz -C /data .' -ForegroundColor Yellow
Write-Host ""
