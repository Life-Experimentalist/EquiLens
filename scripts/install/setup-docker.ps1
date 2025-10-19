# EquiLens Smart Docker Setup (moved to scripts/install)
# One command to handle everything: install, start, repair, update

# ============================================================================
# CONFIGURATION - Customize these settings
# ============================================================================
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"   # Docker Hub image to use
$CONTAINER_NAME = "equilens-app"                 # Container name
$VOLUME_NAME = "equilens-data"                   # Data volume name
$OLLAMA_URL = "http://localhost:11434"           # Ollama API URL
# ============================================================================

Write-Host "=== EquiLens Smart Setup ===" -ForegroundColor Cyan
Write-Host "Image: $EQUILENS_IMAGE" -ForegroundColor Gray
Write-Host ""

# Check Docker
Write-Host "[1/4] Checking Docker..." -ForegroundColor Yellow
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
	Write-Host "❌ Docker not installed!" -ForegroundColor Red
	Write-Host "   Install from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
	exit 1
}

try {
	docker ps | Out-Null
	Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
	Write-Host "❌ Docker is not running!" -ForegroundColor Red
	Write-Host "   Please start Docker Desktop" -ForegroundColor Yellow
	exit 1
}

# Check if container exists
Write-Host "[2/4] Checking existing container..." -ForegroundColor Yellow
$containerExists = docker ps -a --format "{{.Names}}" | Select-String "^$CONTAINER_NAME$"

if ($containerExists) {
	# Container exists - check if running
	$isRunning = docker ps --format "{{.Names}}" | Select-String "^$CONTAINER_NAME$"

	if ($isRunning) {
		Write-Host "✅ Container '$CONTAINER_NAME' is already running!" -ForegroundColor Green
		Write-Host ""
		Write-Host "EquiLens is available at:" -ForegroundColor Cyan
		Write-Host "  • Gradio UI:  http://localhost:7860" -ForegroundColor White
		Write-Host "  • Web API:    http://localhost:8000" -ForegroundColor White
		Write-Host ""
		Write-Host "Commands:" -ForegroundColor Cyan
		Write-Host "  docker logs -f $CONTAINER_NAME      # View logs" -ForegroundColor Gray
		Write-Host "  docker stop $CONTAINER_NAME         # Stop container" -ForegroundColor Gray
		Write-Host "  docker restart $CONTAINER_NAME      # Restart container" -ForegroundColor Gray
		exit 0
	} else {
		Write-Host "⚠️  Container exists but stopped" -ForegroundColor Yellow
		Write-Host ""
		Write-Host "Choose action:" -ForegroundColor Cyan
		Write-Host "  [1] Start existing container (default)" -ForegroundColor White
		Write-Host "  [2] Recreate container (fresh start)" -ForegroundColor White
		Write-Host ""
		$choice = Read-Host "Enter choice [1 or 2 - default 1]"

		if ($choice -eq "2") {
			Write-Host ""
			Write-Host "🗑️  Removing old container..." -ForegroundColor Yellow
			docker rm $CONTAINER_NAME | Out-Null
			$containerExists = $false
			Write-Host "✅ Old container removed" -ForegroundColor Green
		} else {
			Write-Host ""
			Write-Host "▶️  Starting existing container..." -ForegroundColor Yellow
			docker start $CONTAINER_NAME | Out-Null

			if ($LASTEXITCODE -eq 0) {
				Write-Host "✅ Container started successfully!" -ForegroundColor Green
				Write-Host ""
				Write-Host "EquiLens is available at:" -ForegroundColor Cyan
				Write-Host "  • Gradio UI:  http://localhost:7860" -ForegroundColor White
				Write-Host "  • Web API:    http://localhost:8000" -ForegroundColor White
				Write-Host ""
				Write-Host "Commands:" -ForegroundColor Cyan
				Write-Host "  docker logs -f $CONTAINER_NAME      # View logs" -ForegroundColor Gray
				Write-Host "  docker stop $CONTAINER_NAME         # Stop container" -ForegroundColor Gray
				exit 0
			} else {
				Write-Host "❌ Failed to start container" -ForegroundColor Red
				Write-Host "   Try recreating (run script again and choose option 2)" -ForegroundColor Yellow
				exit 1
			}
		}
	}
} else {
	Write-Host "ℹ️  No existing container found" -ForegroundColor Gray
}

# Check Ollama (optional)
Write-Host "[3/4] Checking Ollama..." -ForegroundColor Yellow
try {
	$response = Invoke-WebRequest -Uri "$OLLAMA_URL/api/tags" -Method GET -TimeoutSec 3 -ErrorAction Stop
	if ($response.StatusCode -eq 200) {
		$data = $response.Content | ConvertFrom-Json
		$modelCount = $data.models.Count
		if ($modelCount -gt 0) {
			Write-Host "✅ Ollama running with $modelCount models" -ForegroundColor Green
		} else {
			Write-Host "⚠️  Ollama running but no models installed" -ForegroundColor Yellow
			Write-Host "   Run: ollama pull llama3.2" -ForegroundColor Gray
		}
	}
} catch {
	Write-Host "⚠️  Ollama not detected at $OLLAMA_URL" -ForegroundColor Yellow
	Write-Host "   Install from: https://ollama.ai/download" -ForegroundColor Gray
	Write-Host "   EquiLens will work once Ollama is running" -ForegroundColor Gray
}

# Create volume if needed
$volumeExists = docker volume ls --format "{{.Name}}" | Select-String "^$VOLUME_NAME$"
if (!$volumeExists) {
	Write-Host ""
	Write-Host "📦 Creating data volume..." -ForegroundColor Yellow
	docker volume create $VOLUME_NAME | Out-Null
	Write-Host "✅ Volume created: $VOLUME_NAME" -ForegroundColor Green
}

# Pull image and create container
Write-Host "[4/4] Setting up EquiLens..." -ForegroundColor Yellow

$localImage = docker images --format "{{.Repository}}:{{.Tag}}" | Select-String "^$EQUILENS_IMAGE$"

if ($localImage) {
	Write-Host "Found local image: $EQUILENS_IMAGE" -ForegroundColor Green

	# Get local image creation date
	$localImageInfo = docker images --format "{{.Repository}}:{{.Tag}},{{.CreatedAt}}" | Select-String "^$EQUILENS_IMAGE"
	if ($localImageInfo) {
		$localDate = ($localImageInfo -split ',')[1]
		Write-Host "     Created: $localDate" -ForegroundColor Gray
	}

	# Check if newer version available online
	Write-Host "Checking for updates..." -ForegroundColor Gray

	# Try to get remote image digest without pulling
	docker manifest inspect $EQUILENS_IMAGE 2>&1 | Out-Null
	$remoteAvailable = ($LASTEXITCODE -eq 0)

	# Handle three scenarios
	if (-not $remoteAvailable) {
		# Scenario 1: Cannot check remote
		Write-Host "Cannot check remote version (offline or not on Docker Hub)" -ForegroundColor Gray
		Write-Host "Using local image" -ForegroundColor Green
	}
	elseif ($remoteAvailable) {
		# Get digests for comparison
		$localDigest = docker images --no-trunc --format "{{.ID}}" $EQUILENS_IMAGE
		$remoteManifest = docker manifest inspect $EQUILENS_IMAGE 2>&1 | ConvertFrom-Json
		$remoteDigest = $remoteManifest.config.digest

		if ($localDigest -eq $remoteDigest) {
			# Scenario 2: Already up to date
			Write-Host "Local image is up to date" -ForegroundColor Green
		}
		else {
			# Scenario 3: Update available
			Write-Host "Newer version available online!" -ForegroundColor Yellow
			Write-Host ""
			Write-Host "Choose action:" -ForegroundColor Cyan
			Write-Host "  [1] Use existing local image (default)" -ForegroundColor White
			Write-Host "  [2] Pull latest version" -ForegroundColor White
			Write-Host ""
			$imageChoice = Read-Host "Enter choice [default 1]"

			if ($imageChoice -eq "2") {
				Write-Host ""
				Write-Host "Pulling latest image: $EQUILENS_IMAGE" -ForegroundColor Yellow
				docker pull $EQUILENS_IMAGE

				if ($LASTEXITCODE -ne 0) {
					Write-Host "Failed to pull image" -ForegroundColor Red
					Write-Host "Continuing with local image..." -ForegroundColor Yellow
				} else {
					Write-Host "Latest image pulled successfully" -ForegroundColor Green
				}
			} else {
				Write-Host "Using existing local image" -ForegroundColor Green
			}
		}
	}
} else {
	# No local image - must pull
	Write-Host "Image not found locally" -ForegroundColor Gray
	Write-Host "Pulling image: $EQUILENS_IMAGE" -ForegroundColor Yellow
	docker pull $EQUILENS_IMAGE

	if ($LASTEXITCODE -ne 0) {
		Write-Host "Failed to pull image: $EQUILENS_IMAGE" -ForegroundColor Red
		exit 1
	}
	Write-Host "Image pulled successfully" -ForegroundColor Green
}
Write-Host "  🚀 Starting container..." -ForegroundColor Gray
docker run -d `
	--name $CONTAINER_NAME `
	--network host `
	-v ${VOLUME_NAME}:/workspace/data `
	-e OLLAMA_BASE_URL=$OLLAMA_URL `
	--restart unless-stopped `
	$EQUILENS_IMAGE

if ($LASTEXITCODE -eq 0) {
	Write-Host "✅ Container created and started!" -ForegroundColor Green
} else {
	Write-Host "❌ Failed to start container" -ForegroundColor Red
	exit 1
}

# Wait for service to be ready
Write-Host ""
Write-Host "⏳ Waiting for EquiLens to be ready..." -ForegroundColor Yellow
$maxWait = 30
$waited = 0
$ready = $false

while ($waited -lt $maxWait) {
	try {
		$response = Invoke-WebRequest -Uri "http://localhost:7860" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
		if ($response.StatusCode -eq 200) {
			$ready = $true
			break
		}
	} catch {
		Start-Sleep -Seconds 2
		$waited += 2
		Write-Host "." -NoNewline
	}
}

Write-Host ""
if ($ready) {
	Write-Host "✅ EquiLens is ready!" -ForegroundColor Green
} else {
	Write-Host "⚠️  Service starting (may take a few more seconds)" -ForegroundColor Yellow
}

# Success message
Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access EquiLens:" -ForegroundColor Cyan
Write-Host "  • Gradio UI:  http://localhost:7860" -ForegroundColor White
Write-Host "  • Web API:    http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "📊 Quick Commands:" -ForegroundColor Cyan
Write-Host "  docker logs -f $CONTAINER_NAME           # View logs" -ForegroundColor Gray
Write-Host "  docker stop $CONTAINER_NAME              # Stop EquiLens" -ForegroundColor Gray
Write-Host "  docker start $CONTAINER_NAME             # Start EquiLens" -ForegroundColor Gray
Write-Host "  docker restart $CONTAINER_NAME           # Restart EquiLens" -ForegroundColor Gray
Write-Host "  docker rm -f $CONTAINER_NAME             # Remove container" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 Run this script again anytime to:" -ForegroundColor Cyan
Write-Host "   • Start stopped container" -ForegroundColor Gray
Write-Host "   • Recreate if there are issues" -ForegroundColor Gray
Write-Host "   • Check status" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Enjoy using EquiLens!" -ForegroundColor Green
Write-Host ""
