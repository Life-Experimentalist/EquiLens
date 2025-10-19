# EquiLens LOCAL Docker Setup Script
# For developers working inside the EquiLens repository
# This script builds and runs EquiLens from your current code

Write-Host "=== EquiLens Local Docker Setup ===" -ForegroundColor Cyan
Write-Host "Working directory: $PWD" -ForegroundColor Gray
Write-Host ""

# Verify we're in the EquiLens directory
if (!(Test-Path "docker-compose.yml") -or !(Test-Path "Dockerfile")) {
    Write-Host "[ERROR] Not in EquiLens directory!" -ForegroundColor Red
    Write-Host "Please run this script from the EquiLens root directory." -ForegroundColor Yellow
    Write-Host "Expected files: docker-compose.yml, Dockerfile" -ForegroundColor Gray
    exit 1
}

# Check if Docker is installed
Write-Host "[1/6] Checking Docker installation..." -ForegroundColor Yellow
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker is not installed!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Docker is installed" -ForegroundColor Green

# Check if Docker is running
Write-Host "[2/6] Checking if Docker is running..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "[OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check if docker-compose is available
Write-Host "[3/6] Checking docker-compose..." -ForegroundColor Yellow
if (!(Get-Command docker-compose -ErrorAction SilentlyContinue) -and !(docker compose version -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] docker-compose is not available!" -ForegroundColor Red
    Write-Host "Please ensure Docker Compose is installed." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] docker-compose is available" -ForegroundColor Green

# Create persistent volumes
Write-Host "[4/6] Creating Docker volumes..." -ForegroundColor Yellow
$volumeName = "equilens-data"
$exists = docker volume ls --format "{{.Name}}" | Select-String "^$volumeName$"
if ($exists) {
    Write-Host "  Volume already exists: $volumeName" -ForegroundColor Cyan
} else {
    docker volume create $volumeName 2>$null | Out-Null
    Write-Host "  Created volume: $volumeName" -ForegroundColor Cyan
}
Write-Host "[OK] Volume setup complete" -ForegroundColor Green
Write-Host "  [INFO] All data (results, logs, corpus) will be stored in: $volumeName" -ForegroundColor Gray

# Check Ollama availability
Write-Host "[5/6] Checking Ollama..." -ForegroundColor Yellow
$OLLAMA_URL = "http://localhost:11434"
$ollamaAccessible = $false
$ollamaContainerName = ""
$useExistingOllama = $false

# Try to access Ollama on port 11434
try {
    $response = Invoke-WebRequest -Uri "$OLLAMA_URL/api/tags" -Method GET -TimeoutSec 3 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        $ollamaAccessible = $true
        Write-Host "[OK] Ollama is accessible at $OLLAMA_URL" -ForegroundColor Green

        # Check if it's running in Docker or as desktop app
        $ollamaContainer = docker ps --format "{{.Names}}" | Select-String "ollama"
        if ($ollamaContainer) {
            Write-Host "   Running as Docker container: $ollamaContainer" -ForegroundColor Gray
            $useExistingOllama = $true
            $ollamaContainerName = $ollamaContainer.ToString().Trim()
        } else {
            Write-Host "   Running as Ollama Desktop app" -ForegroundColor Gray
            $useExistingOllama = $true
        }
    }
} catch {
    Write-Host "[WARN] Ollama not accessible at port 11434" -ForegroundColor Yellow

    # Check if Ollama container exists but is stopped
    $stoppedOllamaContainer = docker ps -a --format "{{.Names}}" | Select-String "ollama"

    if ($stoppedOllamaContainer) {
    Write-Host "  Found stopped Ollama container: $stoppedOllamaContainer" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Choose action:" -ForegroundColor Cyan
        Write-Host "    [1] Start this container: $stoppedOllamaContainer (default)" -ForegroundColor White
        Write-Host "    [2] Skip - I'll start Ollama manually" -ForegroundColor White
        Write-Host ""
        $ollamaChoice = Read-Host "  Enter choice [default 1]"

        if ($ollamaChoice -eq "2") {
        Write-Host "  [INFO] Please start Ollama manually before using EquiLens" -ForegroundColor Yellow
            $useExistingOllama = $true
        } else {
            # Start existing stopped container
            $ollamaContainerName = $stoppedOllamaContainer.ToString().Trim()
            Write-Host "  [ACTION] Starting container: $ollamaContainerName..." -ForegroundColor Gray
            docker start $ollamaContainerName | Out-Null
            Start-Sleep -Seconds 3
            $useExistingOllama = $true
            Write-Host "[OK] Ollama container started" -ForegroundColor Green
        }
    } else {
    Write-Host "  [INFO] No Ollama container found" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Options:" -ForegroundColor Cyan
        Write-Host "    [1] Use existing Ollama Desktop app (default)" -ForegroundColor White
        Write-Host "    [2] I'll set up Ollama manually" -ForegroundColor White
        Write-Host ""
        $ollamaChoice = Read-Host "  Enter choice [default 1]"

    Write-Host "  [INFO] Make sure Ollama is running before using EquiLens" -ForegroundColor Yellow
    Write-Host "  Download from: https://ollama.ai/download" -ForegroundColor Gray
        $useExistingOllama = $true
    }
}

# Build and start EquiLens
Write-Host "[6/6] Building and starting EquiLens..." -ForegroundColor Yellow

# Check if image already exists
$imageExists = docker images equilens:latest --format "{{.Repository}}:{{.Tag}}" | Select-String "equilens:latest"

if ($imageExists) {
    Write-Host "  [INFO] Image 'equilens:latest' already exists" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Do you want to rebuild? (builds take ~5 minutes)" -ForegroundColor Yellow
    Write-Host "    [1] Skip build - Use existing image (default)" -ForegroundColor White
    Write-Host "    [2] Rebuild - Get latest code changes" -ForegroundColor White
    Write-Host ""
    $buildChoice = Read-Host "  Enter choice [default 1]"

    if ($buildChoice -eq "2") {
        Write-Host "  [BUILD] Building EquiLens image from current code..." -ForegroundColor Gray
        docker-compose build

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Build failed!" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  [SKIP] Using existing image" -ForegroundColor Cyan
    }
} else {
    Write-Host "  [BUILD] Building EquiLens image from current code..." -ForegroundColor Gray
    docker-compose build

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Build failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  [ACTION] Starting EquiLens container..." -ForegroundColor Gray
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to start container!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] EquiLens is starting" -ForegroundColor Green

# Wait for EquiLens to be ready
Write-Host ""
Write-Host "[WAIT] Waiting for EquiLens to be ready..." -ForegroundColor Yellow
$maxWait = 60
$waited = 0
while ($waited -lt $maxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:7860" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "[OK] EquiLens is ready!" -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 2
        $waited += 2
        Write-Host "." -NoNewline
    }
}
if ($waited -ge $maxWait) {
    Write-Host ""
    Write-Host "[WARN] Timeout - EquiLens may still be starting" -ForegroundColor Yellow
    Write-Host "  Check logs with: docker-compose logs -f" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Services are running at:" -ForegroundColor Cyan
Write-Host "   - Gradio UI:  http://localhost:7860" -ForegroundColor White
Write-Host "   - Web API:    http://localhost:8000" -ForegroundColor White
Write-Host "   - Ollama API: $OLLAMA_URL" -ForegroundColor White
Write-Host ""

if ($useExistingOllama -and $ollamaContainerName) {
    Write-Host "OLLAMA Configuration:" -ForegroundColor Cyan
    Write-Host "   Using existing Docker container: $ollamaContainerName" -ForegroundColor White
    Write-Host ""
}

Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f              # View logs" -ForegroundColor Gray
Write-Host "  docker-compose ps                   # Check status" -ForegroundColor Gray
Write-Host "  docker-compose down                 # Stop services" -ForegroundColor Gray
Write-Host "  docker-compose up -d                # Start services" -ForegroundColor Gray
Write-Host "  docker-compose restart              # Restart services" -ForegroundColor Gray
Write-Host "  docker exec -it equilens-app bash   # Access container shell" -ForegroundColor Gray
Write-Host ""

Write-Host "Development workflow:" -ForegroundColor Cyan
Write-Host "  1. Make changes to your code" -ForegroundColor Gray
Write-Host "  2. Run: docker-compose build        # Rebuild with changes" -ForegroundColor Gray
Write-Host "  3. Run: docker-compose restart      # Restart with new code" -ForegroundColor Gray
Write-Host ""

if ($ollamaContainerName) {
    Write-Host "Model Management:" -ForegroundColor Cyan
    Write-Host "  docker exec $ollamaContainerName ollama list              # List models" -ForegroundColor Gray
    Write-Host "  docker exec $ollamaContainerName ollama pull <model>      # Download model" -ForegroundColor Gray
    Write-Host "  docker exec $ollamaContainerName ollama rm <model>        # Remove model" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "Alternative: Run locally without Docker" -ForegroundColor Cyan
Write-Host "  uv sync                             # Install dependencies" -ForegroundColor Gray
Write-Host "  uv run equilens                     # Run EquiLens locally" -ForegroundColor Gray
Write-Host "  uv run equilens --help              # See CLI options" -ForegroundColor Gray
Write-Host ""

Write-Host "Enjoy using EquiLens!" -ForegroundColor Green
Write-Host ""
