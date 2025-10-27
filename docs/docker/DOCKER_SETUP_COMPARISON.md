# Docker Setup Scripts Comparison

**Comparing `setup-docker.ps1` vs `setup-docker-simple.ps1`**

---

## Overview

EquiLens has **TWO** Docker setup scripts with different purposes:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **`setup-docker.ps1`** | Build from source | Developers, contributors, customization |
| **`setup-docker-simple.ps1`** | Pull pre-built image | End users, quick setup, production |

---

## Key Differences Table

| Feature | `setup-docker.ps1` | `setup-docker-simple.ps1` |
|---------|-------------------|---------------------------|
| **Image Source** | Builds locally from Dockerfile | Pulls pre-built from Docker Hub |
| **Git Required** | ✅ Yes - clones repository | ❌ No git needed |
| **Build Time** | 5-10 minutes (slow) | 1-2 minutes (fast) |
| **Disk Space** | ~3GB (source + build layers) | ~1.5GB (just image) |
| **Customization** | ✅ Full - edit source code | ❌ Limited - use as-is |
| **Internet Speed** | Less important (small clone) | More important (large image download) |
| **Dependencies** | Git, Docker, docker-compose | Just Docker |
| **Updates** | `git pull` + rebuild | `docker pull` new version |
| **Configuration** | Edit `docker-compose.yml` | Edit `$EQUILENS_IMAGE` variable |

---

## Detailed Comparison

### 1. setup-docker.ps1 (Build from Source)

**Purpose:** Full installation for developers and those who want to customize

**What it does:**
```powershell
[1/8] Check Docker installation
[2/8] Check Docker is running
[3/8] Check docker-compose availability
[4/8] Clone/download EquiLens repository from GitHub  ⬅️ CLONES REPO
[5/8] Create Docker volumes (ollama, equilens-data, results, logs)
[6/8] Pull base images (ollama/ollama:latest)
[7/8] Build EquiLens from Dockerfile using docker-compose  ⬅️ BUILDS IMAGE
[8/8] Download default Ollama model (llama3.2)
```

**Key Steps:**
- **Clones Git repository** (lines 36-68)
- **Creates 4 volumes**: ollama-models, equilens-data, equilens-results, equilens-logs
- **Uses `docker-compose up -d`** which builds from Dockerfile
- **Checks for existing Ollama volumes** and offers to reuse them
- **Downloads default model** (llama3.2) automatically

**Pros:**
- ✅ Full source code access
- ✅ Easy to modify and customize
- ✅ Can make local changes and rebuild
- ✅ Includes complete project structure
- ✅ Development-friendly

**Cons:**
- ❌ Requires git
- ❌ Longer setup time (5-10 minutes)
- ❌ Needs to rebuild on updates
- ❌ More disk space used
- ❌ More dependencies to install

**Use Cases:**
- Contributing to EquiLens development
- Modifying source code
- Testing local changes
- Understanding the codebase
- Building custom versions

---

### 2. setup-docker-simple.ps1 (Pull Pre-Built)

**Purpose:** Quick installation for end users who just want to run EquiLens

**What it does:**
```powershell
[1/5] Check Docker installation
[2/5] Check Docker is running
[3/5] Check Ollama availability (assumes already installed)
[4/5] Create EquiLens data volume (just 1 volume)
[5/6] Pull pre-built image from Docker Hub  ⬅️ PULLS IMAGE
[6/6] Run container with docker run command  ⬅️ DIRECT RUN
```

**Key Steps:**
- **NO git cloning** - downloads just the Docker image
- **Creates 1 volume**: equilens-data
- **Configurable image URL**: `$EQUILENS_IMAGE = "vkrishna04/equilens:latest"`
- **Uses `docker run`** directly (not docker-compose)
- **Assumes Ollama is already running** on host (localhost:11434)

**Pros:**
- ✅ Very fast setup (1-2 minutes)
- ✅ No git required
- ✅ Minimal dependencies
- ✅ Easy to update (just pull new image)
- ✅ Configurable image source
- ✅ Production-ready images
- ✅ Simpler command structure

**Cons:**
- ❌ Cannot modify source code
- ❌ Need to push to Docker Hub for updates
- ❌ Assumes Ollama installed separately
- ❌ Less flexibility

**Use Cases:**
- Just want to run EquiLens
- Production deployments
- Testing released versions
- No development needed
- Quick demos

---

## Side-by-Side Code Comparison

### Repository Setup

**setup-docker.ps1:**
```powershell
# Clones entire repository
Write-Host "[4/8] Setting up EquiLens..." -ForegroundColor Yellow
$repoUrl = "https://github.com/LifeExperimentalist/equiLens"
$targetDir = "equiLens"

if (Test-Path $targetDir) {
    Push-Location $targetDir
    git pull 2>$null
} else {
    git clone $repoUrl $targetDir 2>$null
}
```

**setup-docker-simple.ps1:**
```powershell
# No repository cloning - just configuration
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"
# User can change this to any Docker Hub URL
```

---

### Volume Creation

**setup-docker.ps1:**
```powershell
# Creates 4 volumes
$volumes = @(
    "ollama-models",      # For Ollama models
    "equilens-data",      # For data
    "equilens-results",   # For results
    "equilens-logs"       # For logs
)

# Also checks for existing Ollama volumes
$existingOllamaVolumes = docker volume ls --format "{{.Name}}" | Select-String "ollama"
```

**setup-docker-simple.ps1:**
```powershell
# Creates 1 volume only
Write-Host "[4/5] Creating EquiLens data volume..." -ForegroundColor Yellow
docker volume create equilens-data

# Assumes Ollama runs on host, not in Docker
```

---

### Image Acquisition

**setup-docker.ps1:**
```powershell
# Builds from source using docker-compose
Write-Host "[7/8] Starting EquiLens services..." -ForegroundColor Yellow
docker-compose up -d
# This triggers: docker compose build (reads Dockerfile)
```

**setup-docker-simple.ps1:**
```powershell
# Pulls pre-built image from Docker Hub
Write-Host "[5/6] Pulling EquiLens image..." -ForegroundColor Yellow
docker pull $EQUILENS_IMAGE
# This downloads: vkrishna04/equilens:latest
```

---

### Container Startup

**setup-docker.ps1:**
```powershell
# Uses docker-compose for orchestration
docker-compose up -d

# Starts multiple containers defined in docker-compose.yml:
# - equilens-app
# - equilens-ollama (if configured)
# - equilens-web (if configured)
```

**setup-docker-simple.ps1:**
```powershell
# Direct docker run command
docker run -d \
    --name equilens-app \
    --network host \
    -v equilens-data:/workspace/data \
    -e OLLAMA_BASE_URL=http://localhost:11434 \
    $EQUILENS_IMAGE

# Single container, assumes Ollama on host
```

---

### Ollama Setup

**setup-docker.ps1:**
```powershell
# Can manage Ollama in Docker (if configured)
Write-Host "[6/8] Pulling Docker images..." -ForegroundColor Yellow
docker pull ollama/ollama:latest

# Downloads default model automatically
Write-Host "[8/8] Downloading default model..." -ForegroundColor Yellow
docker exec equilens-ollama ollama pull llama3.2:latest
```

**setup-docker-simple.ps1:**
```powershell
# Assumes Ollama is already installed on host
Write-Host "[3/5] Checking Ollama availability..." -ForegroundColor Yellow
$response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags"

# If not found, tells user to install Ollama separately
Write-Host "📖 To install and run Ollama:"
Write-Host "   1. Download from: https://ollama.ai/download"
```

---

## Workflow Diagrams

### setup-docker.ps1 (Build from Source)
```
┌─────────────────────────────────────────────────┐
│ 1. Check Docker & docker-compose                │
├─────────────────────────────────────────────────┤
│ 2. Clone EquiLens repository (git clone)        │
│    └─ Downloads: source code, Dockerfile, etc   │
├─────────────────────────────────────────────────┤
│ 3. Create 4 Docker volumes                      │
│    └─ ollama-models, data, results, logs        │
├─────────────────────────────────────────────────┤
│ 4. Pull base images (ollama/ollama)             │
├─────────────────────────────────────────────────┤
│ 5. Build EquiLens from Dockerfile (5-10 min)    │
│    └─ docker-compose up -d (builds image)       │
├─────────────────────────────────────────────────┤
│ 6. Download Ollama model (llama3.2)             │
├─────────────────────────────────────────────────┤
│ 7. Start all services                           │
│    └─ EquiLens + Ollama in containers           │
└─────────────────────────────────────────────────┘

Total Time: 5-10 minutes
Disk Space: ~3GB
Dependencies: Docker, docker-compose, git
```

### setup-docker-simple.ps1 (Pull Pre-Built)
```
┌─────────────────────────────────────────────────┐
│ 1. Check Docker                                 │
├─────────────────────────────────────────────────┤
│ 2. Check if Ollama is running on host           │
│    └─ http://localhost:11434                    │
├─────────────────────────────────────────────────┤
│ 3. Create 1 Docker volume (equilens-data)       │
├─────────────────────────────────────────────────┤
│ 4. Pull pre-built image from Docker Hub (1-2m)  │
│    └─ docker pull vkrishna04/equilens:latest    │
├─────────────────────────────────────────────────┤
│ 5. Run EquiLens container                       │
│    └─ docker run with host network              │
│    └─ Connects to host Ollama                   │
└─────────────────────────────────────────────────┘

Total Time: 1-2 minutes
Disk Space: ~1.5GB
Dependencies: Docker only
```

---

## Use Case Decision Tree

```
Do you need to modify EquiLens source code?
│
├─ YES → Use setup-docker.ps1
│        ✅ Full source access
│        ✅ Can rebuild with changes
│        ✅ Development environment
│
└─ NO → Do you want the fastest setup?
        │
        ├─ YES → Use setup-docker-simple.ps1
        │        ✅ 5x faster (1-2 min vs 5-10 min)
        │        ✅ Pre-tested production images
        │        ✅ Easy updates
        │
        └─ NO → Still use setup-docker.ps1 if:
                 • You want complete project structure
                 • You need to understand internals
                 • You're contributing to the project
```

---

## Configuration Comparison

### setup-docker.ps1 Configuration

**File:** `docker-compose.yml`
```yaml
services:
  equilens:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "7860:7860"
      - "8000:8000"
    volumes:
      - equilens-data:/workspace/data
      - equilens-results:/workspace/results
      - equilens-logs:/workspace/logs
```

**To change image:** Edit Dockerfile, then rebuild
**To update:** `git pull && docker-compose up -d --build`

---

### setup-docker-simple.ps1 Configuration

**File:** `setup-docker-simple.ps1` (line 8)
```powershell
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"

# Change to any Docker Hub image:
# $EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"
# $EQUILENS_IMAGE = "ghcr.io/vkrishna04/equilens:latest"
# $EQUILENS_IMAGE = "mycompany/equilens:custom"
```

**To change image:** Edit one line
**To update:** Change version in `$EQUILENS_IMAGE`, re-run script

---

## Commands After Setup

### setup-docker.ps1 Commands

```powershell
# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop services
docker-compose down

# Restart services
docker-compose up -d

# Rebuild after changes
docker-compose up -d --build

# Access container
docker exec -it equilens-app bash

# Update code
git pull
docker-compose up -d --build
```

---

### setup-docker-simple.ps1 Commands

```powershell
# View logs
docker logs -f equilens-app

# Check status
docker ps

# Stop service
docker stop equilens-app

# Start service
docker start equilens-app

# Restart service
docker restart equilens-app

# Remove container
docker rm equilens-app

# Update to new version
docker stop equilens-app
docker rm equilens-app
docker pull vkrishna04/equilens:v2.0.0
# Edit $EQUILENS_IMAGE in script, then re-run
.\setup-docker-simple.ps1
```

---

## Migration Between Scripts

### From setup-docker.ps1 to setup-docker-simple.ps1

**Scenario:** You started with build-from-source, now want pre-built images

**Steps:**
```powershell
# 1. Stop current setup
cd equilens
docker-compose down

# 2. (Optional) Backup volumes
docker volume ls | Select-String equilens

# 3. Download simplified setup
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/VKrishna04/EquiLens/main/setup-docker-simple.ps1" -OutFile "setup-simple.ps1"

# 4. Run simplified setup
.\setup-simple.ps1
```

**Note:** Data in `equilens-data` volume is preserved automatically

---

### From setup-docker-simple.ps1 to setup-docker.ps1

**Scenario:** You want to contribute code, need source access

**Steps:**
```powershell
# 1. Stop current container
docker stop equilens-app
docker rm equilens-app

# 2. Download full setup
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/VKrishna04/EquiLens/main/setup-docker.ps1" -OutFile "setup-full.ps1"

# 3. Run full setup (will clone repo and build)
.\setup-full.ps1
```

---

## Performance Comparison

### First-Time Setup

| Metric | setup-docker.ps1 | setup-docker-simple.ps1|
|--------|------------------|-------------------------|
| **Download Size** | 50MB (git repo) + build | 1.2GB (Docker image) |
| **Build Time** | 5-10 minutes | 0 (no build) |
| **Pull Time** | 1-2 minutes | 1-2 minutes |
| **Total Time** | 6-12 minutes | 1-2 minutes |
| **Disk Space Used** | ~3GB (source + layers) | ~1.5GB (image only) |

### Update/Reinstall

| Metric | setup-docker.ps1 | setup-docker-simple.ps1 |
|--------|------------------|-------------------------|
| **Code Update** | `git pull` (1-5 MB) | N/A |
| **Rebuild Time** | 2-5 minutes | 0 (no build) |
| **Image Update** | N/A | `docker pull` (1-2 min) |
| **Total Time** | 2-5 minutes | 1-2 minutes |

---

## Which Script Should You Use?

### Use `setup-docker.ps1` if you:

- ✅ Want to contribute to EquiLens development
- ✅ Need to modify source code
- ✅ Want to understand the codebase
- ✅ Are building custom features
- ✅ Need the complete project structure
- ✅ Have git installed
- ✅ Don't mind waiting 5-10 minutes
- ✅ Want to manage Ollama in Docker too

### Use `setup-docker-simple.ps1` if you:

- ✅ Just want to run EquiLens quickly
- ✅ Don't need source code access
- ✅ Want production-ready images
- ✅ Prefer faster setup (1-2 minutes)
- ✅ Want easy version switching
- ✅ Don't have git installed
- ✅ Already have Ollama on host
- ✅ Want minimal dependencies

---

## Recommendation

**For most users:** Start with `setup-docker-simple.ps1`
- Faster, simpler, production-ready
- Can always switch to full setup later if needed

**For developers:** Use `setup-docker.ps1`
- Full control and customization
- Necessary for code contributions

---

## Summary

| Aspect | setup-docker.ps1 | setup-docker-simple.ps1 |
|--------|-----------------|-------------------------|
| **Primary Audience** | Developers, Contributors | End Users, Production |
| **Installation Time** | 6-12 minutes | 1-2 minutes |
| **Git Required** | Yes | No |
| **Customization** | Full | Limited |
| **Image Source** | Build from Dockerfile | Pull from Docker Hub |
| **Update Process** | git pull + rebuild | docker pull new version |
| **Ollama Management** | Can run in Docker | Assumes host installation |
| **Complexity** | High (8 steps) | Low (6 steps) |
| **Best For** | Development | Production |

---

## Related Documentation

- **[DOCKER_HUB_DEPLOYMENT.md](./DOCKER_HUB_DEPLOYMENT.md)** - How to deploy your own images
- **[DOCKER_DEPLOY_QUICKREF.md](./DOCKER_DEPLOY_QUICKREF.md)** - Quick deployment reference
- **[DOCKER_SIMPLIFIED_SETUP.md](./DOCKER_SIMPLIFIED_SETUP.md)** - Implementation details
- **[ONE_CLICK_SETUP.md](./ONE_CLICK_SETUP.md)** - Original setup documentation

---

**Questions?**
- For deployment: See `DOCKER_HUB_DEPLOYMENT.md`
- For quick start: Use `setup-docker-simple.ps1`
- For development: Use `setup-docker.ps1`
