# EquiLens Docker Setup Script
# One-command installation: irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup-docker.ps1 | iex

Write-Host "=== EquiLens Docker Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
Write-Host "[1/8] Checking Docker installation..." -ForegroundColor Yellow
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host "After installation, restart this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Docker is installed" -ForegroundColor Green

# Check if Docker is running
Write-Host "[2/8] Checking if Docker is running..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check if docker-compose is available
Write-Host "[3/8] Checking docker-compose..." -ForegroundColor Yellow
if (!(Get-Command docker-compose -ErrorAction SilentlyContinue) -and !(docker compose version -ErrorAction SilentlyContinue)) {
    Write-Host "❌ docker-compose is not available!" -ForegroundColor Red
    Write-Host "Please ensure Docker Compose is installed." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ docker-compose is available" -ForegroundColor Green

# Clone or download EquiLens repository
Write-Host "[4/8] Setting up EquiLens..." -ForegroundColor Yellow
$repoUrl = "https://github.com/LifeExperimentalist/EquiLens"
$targetDir = "EquiLens"

if (Test-Path $targetDir) {
    Write-Host "Directory already exists. Pulling latest changes..." -ForegroundColor Yellow
    Push-Location $targetDir
    git pull 2>$null
    Pop-Location
} else {
    Write-Host "Cloning EquiLens repository..." -ForegroundColor Yellow
    git clone $repoUrl $targetDir 2>$null
}

if (!(Test-Path $targetDir)) {
    Write-Host "Creating directory structure..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    Push-Location $targetDir

    # Download docker-compose.yml
    $composeUrl = "$repoUrl/raw/main/docker-compose.yml"
    Invoke-WebRequest -Uri $composeUrl -OutFile "docker-compose.yml" -ErrorAction SilentlyContinue

    # Download Dockerfile
    $dockerfileUrl = "$repoUrl/raw/main/Dockerfile"
    Invoke-WebRequest -Uri $dockerfileUrl -OutFile "Dockerfile" -ErrorAction SilentlyContinue
} else {
    Push-Location $targetDir
}

Write-Host "✅ EquiLens setup complete" -ForegroundColor Green

# Create persistent volumes
Write-Host "[5/8] Creating Docker volumes..." -ForegroundColor Yellow

# Only create the equilens-data volume (results and logs are subdirectories inside it)
$volumeName = "equilens-data"
$exists = docker volume ls --format "{{.Name}}" | Select-String "^$volumeName$"
if ($exists) {
    Write-Host "  Volume already exists: $volumeName" -ForegroundColor Cyan
} else {
    docker volume create $volumeName 2>$null | Out-Null
    Write-Host "  Created volume: $volumeName" -ForegroundColor Cyan
}
Write-Host "✅ Volume setup complete" -ForegroundColor Green
Write-Host "  ℹ️  All data (results, logs, corpus) will be stored in: $volumeName" -ForegroundColor Gray

# Check Ollama availability
Write-Host "[6/8] Checking Ollama..." -ForegroundColor Yellow
$OLLAMA_URL = "http://localhost:11434"
$ollamaAccessible = $false
$ollamaContainerName = ""
$useExistingOllama = $false

# Try to access Ollama on port 11434
try {
    $response = Invoke-WebRequest -Uri "$OLLAMA_URL/api/tags" -Method GET -TimeoutSec 3 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        $ollamaAccessible = $true
        Write-Host "  ✅ Ollama is accessible at $OLLAMA_URL" -ForegroundColor Green

        # Check if it's running in Docker or as desktop app
        $ollamaContainer = docker ps --format "{{.Names}}" | Select-String "ollama"
        if ($ollamaContainer) {
            Write-Host "     Running as Docker container: $ollamaContainer" -ForegroundColor Gray
            $useExistingOllama = $true
            $ollamaContainerName = $ollamaContainer.ToString().Trim()
        } else {
            Write-Host "     Running as Ollama Desktop app" -ForegroundColor Gray
            $useExistingOllama = $true
        }
    }
} catch {
    Write-Host "  ℹ️  Ollama not accessible at port 11434" -ForegroundColor Yellow

    # Check if Ollama container exists but is stopped
    $stoppedOllamaContainer = docker ps -a --format "{{.Names}}" | Select-String "ollama"

    if ($stoppedOllamaContainer) {
        Write-Host "  ⚠️  Found stopped Ollama container: $stoppedOllamaContainer" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Choose action:" -ForegroundColor Cyan
        Write-Host "    [1] Start this container: $stoppedOllamaContainer (default)" -ForegroundColor White
        Write-Host "    [2] Create new Ollama container" -ForegroundColor White
        Write-Host "    [3] Use different container (specify name/ID)" -ForegroundColor White
        Write-Host ""
        $ollamaChoice = Read-Host "  Enter choice [default 1]"

        if ($ollamaChoice -eq "2") {
            Write-Host "  Creating new Ollama container..." -ForegroundColor Gray
            $useExistingOllama = $false
        } elseif ($ollamaChoice -eq "3") {
            Write-Host ""
            $customContainer = Read-Host "  Enter container name or ID"
            if ($customContainer) {
                # Verify container exists
                $containerExists = docker ps -a --format "{{.Names}}" | Select-String "^$customContainer$"
                if ($containerExists) {
                    $ollamaContainerName = $customContainer
                    Write-Host "  ✅ Using container: $ollamaContainerName" -ForegroundColor Green

                    # Start it if stopped
                    $isRunning = docker ps --format "{{.Names}}" | Select-String "^$ollamaContainerName$"
                    if (-not $isRunning) {
                        Write-Host "  ▶️  Starting container..." -ForegroundColor Gray
                        docker start $ollamaContainerName | Out-Null
                        Start-Sleep -Seconds 3
                    }
                    $useExistingOllama = $true
                } else {
                    Write-Host "  ❌ Container not found: $customContainer" -ForegroundColor Red
                    Write-Host "  Creating new Ollama container..." -ForegroundColor Yellow
                    $useExistingOllama = $false
                }
            } else {
                Write-Host "  No container specified. Creating new..." -ForegroundColor Yellow
                $useExistingOllama = $false
            }
        } else {
            # Start existing stopped container
            $ollamaContainerName = $stoppedOllamaContainer.ToString().Trim()
            Write-Host "  ▶️  Starting container: $ollamaContainerName..." -ForegroundColor Gray
            docker start $ollamaContainerName | Out-Null
            Start-Sleep -Seconds 3
            $useExistingOllama = $true
        }
    } else {
        Write-Host "  ℹ️  No Ollama container found" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Options:" -ForegroundColor Cyan
        Write-Host "    [1] Create new Ollama container (default)" -ForegroundColor White
        Write-Host "    [2] Use existing Ollama Desktop app" -ForegroundColor White
        Write-Host "    [3] Use existing container by name/ID" -ForegroundColor White
        Write-Host ""
        $ollamaChoice = Read-Host "  Enter choice [default 1]"

        if ($ollamaChoice -eq "2") {
            Write-Host "  Please start Ollama Desktop app manually" -ForegroundColor Yellow
            Write-Host "  Download from: https://ollama.ai/download" -ForegroundColor Gray
            Write-Host "  EquiLens will connect to it once running" -ForegroundColor Gray
            $useExistingOllama = $true
        } elseif ($ollamaChoice -eq "3") {
            Write-Host ""
            $customContainer = Read-Host "  Enter container name or ID"
            if ($customContainer) {
                # Verify container exists
                $containerExists = docker ps -a --format "{{.Names}}" | Select-String "^$customContainer$"
                if ($containerExists) {
                    $ollamaContainerName = $customContainer
                    Write-Host "  ✅ Using container: $ollamaContainerName" -ForegroundColor Green

                    # Start it if stopped
                    $isRunning = docker ps --format "{{.Names}}" | Select-String "^$ollamaContainerName$"
                    if (-not $isRunning) {
                        Write-Host "  ▶️  Starting container..." -ForegroundColor Gray
                        docker start $ollamaContainerName | Out-Null
                        Start-Sleep -Seconds 3
                    }
                    $useExistingOllama = $true
                } else {
                    Write-Host "  ❌ Container not found: $customContainer" -ForegroundColor Red
                    Write-Host "  Creating new Ollama container..." -ForegroundColor Yellow
                    $useExistingOllama = $false
                }
            } else {
                Write-Host "  No container specified. Creating new..." -ForegroundColor Yellow
                $useExistingOllama = $false
            }
        } else {
            Write-Host "  Creating new Ollama container..." -ForegroundColor Gray
            $useExistingOllama = $false
        }
    }
}

# Update docker-compose.yml based on Ollama choice
if ($useExistingOllama) {
    Write-Host "  📝 Configuring to use existing Ollama..." -ForegroundColor Gray

    # Modify docker-compose.yml to remove Ollama service
    $composeContent = Get-Content "docker-compose.yml" -Raw

    # Create a simplified version without Ollama service
    $newCompose = $composeContent -replace '(?s)  ollama:.*?(?=\n\S|\z)', ''

    # Update environment variable to point to existing Ollama
    if ($ollamaContainerName) {
        # Use Docker network to connect to existing container
        $newCompose = $newCompose -replace 'OLLAMA_BASE_URL=.*', "OLLAMA_BASE_URL=http://$ollamaContainerName:11434"
    } else {
        # Use host network to connect to Desktop app
        $newCompose = $newCompose -replace 'OLLAMA_BASE_URL=.*', "OLLAMA_BASE_URL=$OLLAMA_URL"
    }

    Set-Content "docker-compose.yml" -Value $newCompose
    Write-Host "✅ Configured for external Ollama" -ForegroundColor Green
} else {
    Write-Host "  📥 Pulling Ollama Docker image..." -ForegroundColor Gray
    docker pull ollama/ollama:latest
    Write-Host "✅ Ollama image ready" -ForegroundColor Green
}

# Pull/Build EquiLens image
Write-Host "[7/8] Building EquiLens image..." -ForegroundColor Yellow
docker-compose build

# Start services
Write-Host "  🚀 Starting EquiLens services..." -ForegroundColor Gray
docker-compose up -d

# Wait for services to be healthy
Write-Host "[8/8] Waiting for services to be ready..." -ForegroundColor Yellow

# Wait for Ollama
if (-not $useExistingOllama) {
    Write-Host "  ⏳ Waiting for Ollama..." -ForegroundColor Gray
    $maxWait = 60
    $waited = 0
    while ($waited -lt $maxWait) {
        try {
            $response = Invoke-WebRequest -Uri "$OLLAMA_URL/api/tags" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host ""
                Write-Host "✅ Ollama is ready" -ForegroundColor Green
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
        Write-Host "⚠️  Ollama startup timeout - may need manual check" -ForegroundColor Yellow
    }
}

# Wait for EquiLens
Write-Host "  ⏳ Waiting for EquiLens..." -ForegroundColor Gray
$maxWait = 60
$waited = 0
while ($waited -lt $maxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:7860" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "✅ EquiLens is ready" -ForegroundColor Green
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
    Write-Host "⚠️  EquiLens startup timeout - may need manual check" -ForegroundColor Yellow
}
Write-Host ""

# Pull default Ollama model if we created new Ollama container
if (-not $useExistingOllama) {
    Write-Host "  📥 Downloading default model (llama3.2:latest)..." -ForegroundColor Yellow
    Write-Host "     (This may take a few minutes)" -ForegroundColor Gray
    docker exec equilens-ollama ollama pull llama3.2:latest
    Write-Host "✅ Model downloaded" -ForegroundColor Green
} elseif ($ollamaContainerName) {
    Write-Host "  ℹ️  To download models to existing container:" -ForegroundColor Cyan
    Write-Host "     docker exec $ollamaContainerName ollama pull llama3.2" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Services are running at:" -ForegroundColor Cyan
Write-Host "   • Gradio UI:  http://localhost:7860" -ForegroundColor White
Write-Host "   • Web API:    http://localhost:8000" -ForegroundColor White
Write-Host "   • Ollama API: $OLLAMA_URL" -ForegroundColor White
Write-Host ""

if ($useExistingOllama) {
    Write-Host "📡 Ollama Configuration:" -ForegroundColor Cyan
    if ($ollamaContainerName) {
        Write-Host "   Using existing Docker container: $ollamaContainerName" -ForegroundColor White
        Write-Host "   Both EquiLens and Ollama are connected via Docker network" -ForegroundColor Gray
    } else {
        Write-Host "   Using Ollama Desktop app or external instance" -ForegroundColor White
        Write-Host "   EquiLens connects via: $OLLAMA_URL" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "📊 Useful commands:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f              # View logs" -ForegroundColor Gray
Write-Host "  docker-compose ps                   # Check status" -ForegroundColor Gray
Write-Host "  docker-compose down                 # Stop services" -ForegroundColor Gray
Write-Host "  docker-compose up -d                # Start services" -ForegroundColor Gray
Write-Host "  docker exec -it equilens-app bash   # Access EquiLens container" -ForegroundColor Gray

if ($useExistingOllama -and $ollamaContainerName) {
    Write-Host "  docker exec -it $ollamaContainerName bash  # Access Ollama container" -ForegroundColor Gray
}
Write-Host ""

Write-Host "🤖 Model Management:" -ForegroundColor Cyan
if ($ollamaContainerName) {
    Write-Host "  docker exec $ollamaContainerName ollama list              # List models" -ForegroundColor Gray
    Write-Host "  docker exec $ollamaContainerName ollama pull <model>      # Download model" -ForegroundColor Gray
    Write-Host "  docker exec $ollamaContainerName ollama rm <model>        # Remove model" -ForegroundColor Gray
} elseif ($useExistingOllama) {
    Write-Host "  ollama list                         # List models (in Desktop app)" -ForegroundColor Gray
    Write-Host "  ollama pull <model>                 # Download model" -ForegroundColor Gray
    Write-Host "  ollama rm <model>                   # Remove model" -ForegroundColor Gray
} else {
    Write-Host "  docker exec equilens-ollama ollama list              # List models" -ForegroundColor Gray
    Write-Host "  docker exec equilens-ollama ollama pull <model>      # Download model" -ForegroundColor Gray
    Write-Host "  docker exec equilens-ollama ollama rm <model>        # Remove model" -ForegroundColor Gray
}
Write-Host ""

Write-Host "💡 Popular models to try:" -ForegroundColor Cyan
Write-Host "  • llama3.2        (3B - Fast, good for testing)" -ForegroundColor Gray
Write-Host "  • llama3.1        (8B - Balanced performance)" -ForegroundColor Gray
Write-Host "  • mistral         (7B - Excellent for coding)" -ForegroundColor Gray
Write-Host "  • codellama       (7B - Specialized for code)" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Enjoy using EquiLens!" -ForegroundColor Green
Write-Host ""

Pop-Location
