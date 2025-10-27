# EquiLens Docker Deployment Guide

Complete guide for deploying EquiLens using Docker containers with integrated Ollama support.

## Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

### One-Command Installation

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.ps1 | iex
```

**Linux/macOS (Bash):**
```bash
curl -fsSL https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.sh | bash
```

After installation completes, access:
- **Gradio UI**: http://localhost:7860
- **Web API**: http://localhost:8000
- **Ollama API**: http://localhost:11434

---

## Architecture

### Container Stack

```
┌────────────────────────────────────────────┐
│          Docker Compose Stack              │
├────────────────────────────────────────────┤
│                                            │
│   ┌──────────────┐      ┌──────────────┐   │
│   │  EquiLens    │◄─────┤   Ollama     │   │
│   │  Application │      │   Server     │   │
│   │              │      │              │   │
│   │  Port: 7860  │      │  Port: 11434 │   │
│   │  Port: 8000  │      │              │   │
│   └──────┬───────┘      └──────┬───────┘   │
│          │                     │           │
│          ▼                     ▼           │
│   ┌──────────────────────────────────┐     │
│   │     Persistent Volumes           │     │
│   ├──────────────────────────────────┤     │
│   │ • equilens-ollama-models         │     │
│   │ • equilens-data                  │     │
│   │ • equilens-results               │     │
│   │ • equilens-logs                  │     │
│   └──────────────────────────────────┘     │
│                                            │
└────────────────────────────────────────────┘
```

### Components

1. **Ollama Service**
   - Image: `ollama/ollama:latest`
   - Port: `11434`
   - Purpose: AI model server for report generation
   - GPU Support: NVIDIA GPU (optional)
   - Volume: Models stored in `ollama_models`

2. **EquiLens Application**
   - Base: Python 3.13.3-slim
   - Ports: `7860` (Gradio), `8000` (API)
   - Purpose: Bias auditing and analysis
   - Volumes: Data, results, logs persistence

3. **Network**
   - Bridge network: `equilens-network`
   - Subnet: `172.28.0.0/16`
   - Internal DNS: Containers communicate by name

---

## Prerequisites

### System Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space (models + data)
- **CPU**: Multi-core processor recommended

### Software Requirements

1. **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux)
   - Version: 20.10.0 or higher
   - Download: https://docs.docker.com/get-docker/

2. **Docker Compose**
   - Version: 2.0.0 or higher
   - Included with Docker Desktop
   - Linux: Install separately if needed

3. **Optional: NVIDIA GPU Support**
   - NVIDIA GPU with CUDA support
   - NVIDIA Container Toolkit
   - Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Verify Installation

```powershell
# Check Docker
docker --version
docker ps

# Check Docker Compose
docker-compose --version
# or
docker compose version

# Check GPU support (optional)
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## Installation Methods

### Method 1: One-Command Setup (Recommended)

**Windows PowerShell:**
```powershell
# Run with execution policy bypass if needed
Set-ExecutionPolicy Bypass -Scope Process -Force
irm https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.ps1 | iex
```

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.sh | bash
```

**What it does:**
1. Checks Docker installation
2. Creates persistent volumes
3. Pulls required images
4. Starts services
5. Downloads default Ollama model (llama3.2)

### Method 2: Manual Setup

#### Step 1: Clone Repository
```powershell
git clone https://github.com/[USER]/EquiLens.git
cd EquiLens
```

#### Step 2: Create Volumes
```powershell
docker volume create equilens-ollama-models
docker volume create equilens-data
docker volume create equilens-results
docker volume create equilens-logs
```

#### Step 3: Start Services
```powershell
docker-compose up -d
```

#### Step 4: Pull Default Model
```powershell
docker exec equilens-ollama ollama pull llama3.2:latest
```

### Method 3: Using Existing Ollama Installation

If you already have Ollama models downloaded in a Docker volume, you can reuse them:

#### Step 1: Find Your Existing Ollama Volume
```powershell
# List all volumes
docker volume ls | Select-String "ollama"

# Inspect a volume to verify it contains models
docker run --rm -v YOUR_VOLUME_NAME:/data alpine ls -lh /data/models
```

#### Step 2: Update docker-compose.yml
Edit the `volumes` section to use your existing Ollama volume:

```yaml
volumes:
  ollama_models:
    external: true
    name: YOUR_EXISTING_OLLAMA_VOLUME_NAME

  # EquiLens volumes remain unchanged
  equilens_data:
    driver: local
    name: equilens-data
  # ... rest of volumes
```

#### Step 3: Start Services
```powershell
docker-compose up -d
```

Your existing Ollama models will be available immediately without re-downloading!

### Method 4: Custom Configuration

1. **Copy and modify docker-compose.yml:**
```powershell
cp docker-compose.yml docker-compose.custom.yml
# Edit docker-compose.custom.yml with your settings
docker-compose -f docker-compose.custom.yml up -d
```

2. **Build custom image:**
```powershell
docker build -t equilens:custom .
# Update docker-compose.yml to use equilens:custom
docker-compose up -d
```

---

## Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

#### Ollama Configuration
```yaml
environment:
  - OLLAMA_HOST=0.0.0.0:11434        # Bind address
  - OLLAMA_ORIGINS=*                  # CORS origins
  - OLLAMA_KEEP_ALIVE=24h             # Model memory retention
  - OLLAMA_NUM_PARALLEL=4             # Concurrent requests
  - OLLAMA_MAX_LOADED_MODELS=2        # Loaded models limit
```

#### EquiLens Configuration
```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama:11434          # Ollama service URL
  - GRADIO_SERVER_PORT=7860                       # Gradio port
  - EQUILENS_DATA_DIR=/data                       # Data directory
  - EQUILENS_RESULTS_DIR=/workspace/results       # Results directory
  - EQUILENS_LOGS_DIR=/workspace/logs             # Logs directory
```

### Port Mapping

Change exposed ports if needed:

```yaml
services:
  ollama:
    ports:
      - "11434:11434"  # Change first number for host port

  equilens:
    ports:
      - "7860:7860"    # Gradio UI
      - "8000:8000"    # Web API
```

### Volume Configuration

#### Volume Separation Strategy

**Ollama and EquiLens use completely separate volumes:**

```yaml
volumes:
  # Ollama volume - INDEPENDENT from EquiLens
  ollama_models:
    driver: local
    name: ollama-models

  # EquiLens-specific volumes - SEPARATE from Ollama
  equilens_data:
    driver: local
    name: equilens-data
  equilens_results:
    driver: local
    name: equilens-results
  equilens_logs:
    driver: local
    name: equilens-logs
```

**Benefits of separation:**
- 🔄 **Reuse existing Ollama models** from other projects
- 🗑️ **Delete EquiLens data** without losing Ollama models
- 📦 **Share Ollama volume** across multiple containers
- 🔒 **Independent backups** for models vs. application data

#### Using Named Volumes (Default)
```yaml
volumes:
  - equilens_data:/data
```

#### Using Existing Ollama Volume

If you already have Ollama models in a Docker volume:

```yaml
volumes:
  ollama_models:
    external: true  # Use existing volume
    name: my-existing-ollama-volume  # Your volume name
```

**Steps to find your existing volume:**
```powershell
# List all Ollama-related volumes
docker volume ls | Select-String "ollama"

# Inspect volume to verify it has models
docker volume inspect YOUR_VOLUME_NAME

# Check models in the volume
docker run --rm -v YOUR_VOLUME_NAME:/data alpine ls -lh /data/models/manifests/registry.ollama.ai
```
```yaml
volumes:
  - equilens_data:/data
```

#### Using Host Directories
```yaml
volumes:
  - ./local_data:/data                    # Data
  - ./local_results:/workspace/results     # Results
  - ./local_logs:/workspace/logs           # Logs
```

### GPU Configuration

#### Enable NVIDIA GPU Support
Uncomment in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

#### CPU-Only Mode
Remove the entire `deploy` section.

---

## Usage

### Starting Services

```powershell
# Start in detached mode
docker-compose up -d

# Start with logs
docker-compose up

# Start specific service
docker-compose up -d ollama
```

### Stopping Services

```powershell
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# Stop specific service
docker-compose stop equilens
```

### Managing Models

#### List Available Models
```powershell
docker exec equilens-ollama ollama list
```

#### Pull Additional Models
```powershell
# Recommended models for bias analysis
docker exec equilens-ollama ollama pull llama3.2:latest
docker exec equilens-ollama ollama pull mistral:latest
docker exec equilens-ollama ollama pull phi3:latest
docker exec equilens-ollama ollama pull gemma2:2b
```

#### Remove Models
```powershell
docker exec equilens-ollama ollama rm <model-name>
```

### Accessing Services

#### Gradio Web UI
```
http://localhost:7860
```
- Interactive bias auditing interface
- Real-time analysis
- Model selection
- Report generation

#### Web API
```
http://localhost:8000
```
- RESTful API endpoints
- Programmatic access
- Batch processing

#### Ollama API
```
http://localhost:11434
```
- Direct model interaction
- API documentation: http://localhost:11434/api

### CLI Access

#### Interactive Shell
```powershell
# Access EquiLens container
docker exec -it equilens-app bash

# Inside container
python -m equilens --help
python -m equilens audit --help
```

#### Run Commands Directly
```powershell
# Run audit
docker exec equilens-app python -m equilens audit \
  --model llama3.2 \
  --output /workspace/results/audit_$(date +%Y%m%d)

# Generate report
docker exec equilens-app python -m equilens analyze \
  --input /workspace/results/audit.csv \
  --format html
```

### Viewing Logs

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f equilens
docker-compose logs -f ollama

# Last 100 lines
docker-compose logs --tail=100 equilens
```

### Data Management

#### Backup Data
```powershell
# Backup volumes
docker run --rm \
  -v equilens-results:/data \
  -v ${PWD}/backup:/backup \
  alpine tar czf /backup/results-$(date +%Y%m%d).tar.gz /data
```

#### Restore Data
```powershell
# Restore from backup
docker run --rm \
  -v equilens-results:/data \
  -v ${PWD}/backup:/backup \
  alpine tar xzf /backup/results-20240115.tar.gz -C /
```

#### Export Results
```powershell
# Copy from container
docker cp equilens-app:/workspace/results ./local_results
```

---

## Troubleshooting

### Common Issues

#### 1. Services Won't Start

**Problem:** `docker-compose up` fails
```
Error: Cannot start service ollama: driver failed
```

**Solutions:**
```powershell
# Check Docker is running
docker ps

# Restart Docker Desktop
# Check available resources
docker system df

# Remove and recreate
docker-compose down
docker-compose up -d
```

#### 2. Ollama Not Accessible

**Problem:** EquiLens can't connect to Ollama
```
Connection refused: http://ollama:11434
```

**Solutions:**
```powershell
# Check Ollama health
docker exec equilens-ollama curl http://localhost:11434/api/tags

# Check network
docker network inspect equilens-network

# Restart Ollama
docker-compose restart ollama
docker-compose logs ollama
```

#### 3. Port Already in Use

**Problem:** Port conflict
```
Bind for 0.0.0.0:7860 failed: port is already allocated
```

**Solutions:**
```powershell
# Find process using port
netstat -ano | findstr :7860  # Windows
lsof -i :7860                  # Linux/macOS

# Change port in docker-compose.yml
ports:
  - "7861:7860"  # Use 7861 instead

# Stop conflicting service
docker ps
docker stop <container-id>
```

#### 4. Out of Disk Space

**Problem:** No space left
```
Error: No space left on device
```

**Solutions:**
```powershell
# Check disk usage
docker system df

# Clean unused images/containers
docker system prune -a

# Remove specific volumes
docker volume rm equilens-logs

# Remove old Ollama models
docker exec equilens-ollama ollama rm <old-model>
```

#### 5. GPU Not Detected

**Problem:** GPU not available to Ollama
```
CUDA not available
```

**Solutions:**
```powershell
# Check GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify docker-compose.yml has GPU config
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Performance Issues

#### Slow Model Loading
```powershell
# Increase memory limits
docker-compose.yml:
  deploy:
    resources:
      limits:
        memory: 8G

# Use smaller models
docker exec equilens-ollama ollama pull phi3:mini
```

#### High CPU Usage
```powershell
# Limit CPU usage
docker-compose.yml:
  deploy:
    resources:
      limits:
        cpus: '2.0'
```

### Debugging

#### Enable Debug Logging
```powershell
# Add to docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=DEBUG
  - GRADIO_DEBUG=1
```

#### Inspect Container
```powershell
# View container details
docker inspect equilens-app

# Check resource usage
docker stats

# View processes
docker exec equilens-app ps aux
```

#### Health Checks
```powershell
# Check service health
docker-compose ps

# Manual health check
curl http://localhost:7860/
curl http://localhost:11434/api/tags
```

---

## Advanced Topics

### Custom Docker Image

Build a custom image with pre-installed models:

```dockerfile
# Dockerfile.custom
FROM python:3.13.3-slim

# Copy your custom dependencies
COPY custom-requirements.txt .
RUN pip install -r custom-requirements.txt

# Pre-download models (if needed)
# Add custom configuration
```

Build and use:
```powershell
docker build -f Dockerfile.custom -t equilens:custom .
# Update docker-compose.yml to use equilens:custom
```

### Production Deployment

#### Using Docker Swarm
```powershell
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml equilens

# Scale services
docker service scale equilens_equilens=3
```

#### Using Kubernetes
```yaml
# Convert to Kubernetes with Kompose
kompose convert -f docker-compose.yml

# Apply to cluster
kubectl apply -f equilens-deployment.yaml
kubectl apply -f equilens-service.yaml
```

### Monitoring

#### Prometheus + Grafana
```yaml
# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

#### Container Logs to ELK Stack
```yaml
services:
  equilens:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Security Hardening

#### Non-Root User (Already Implemented)
```dockerfile
RUN useradd -m -u 1000 equilens
USER equilens
```

#### Read-Only Filesystem
```yaml
services:
  equilens:
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
```

#### Network Isolation
```yaml
networks:
  equilens-network:
    internal: true  # No external access
```

### Backup and Restore

#### Automated Backups
```powershell
# Create backup script
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
docker run --rm `
  -v equilens-results:/data `
  -v ${PWD}/backups:/backup `
  alpine tar czf /backup/equilens-$timestamp.tar.gz /data
```

#### Disaster Recovery
```powershell
# Stop services
docker-compose down

# Restore volumes
docker run --rm `
  -v equilens-results:/data `
  -v ${PWD}/backups:/backup `
  alpine tar xzf /backup/equilens-20240115.tar.gz -C /

# Restart services
docker-compose up -d
```

---

## Support

### Getting Help

1. **Documentation**: Check `docs/` directory
2. **Issues**: GitHub Issues page
3. **Logs**: Always check `docker-compose logs`
4. **Community**: Discussion forums

### Reporting Bugs

Include:
- Docker version: `docker --version`
- Docker Compose version: `docker-compose --version`
- OS and version
- Error logs: `docker-compose logs`
- Steps to reproduce

### Contributing

See `CONTRIBUTING.md` for contribution guidelines.

---

## License

EquiLens is open-source software. See `LICENSE.md` for details.
