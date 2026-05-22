# Docker Setup Complete! 🐳

## What Was Created

### Core Docker Files
1. **Dockerfile** - Python 3.13.3-slim container with UV package manager
   - Non-root user (equilens:1000)
   - Health checks enabled
   - Exposes ports 7860 (Gradio) and 8000 (API)

2. **docker-compose.yml** - Multi-service orchestration
   - Ollama service (port 11434)
   - EquiLens app (ports 7860, 8000)
   - 4 persistent volumes
   - Custom bridge network
   - Health checks and dependencies

### Setup Scripts
3. **setup-docker.ps1** - Windows PowerShell one-command installer
   - Checks Docker installation
   - Creates volumes
   - Pulls images
   - Starts services
   - Downloads default model (llama3.2)

4. **setup-docker.sh** - Linux/macOS Bash one-command installer
   - Cross-platform equivalent
   - Same functionality as PowerShell version
   - Works on Ubuntu, Debian, macOS, etc.

### Documentation
5. **docs/DOCKER_SETUP.md** - Complete Docker deployment guide
   - Architecture diagrams
   - Installation methods
   - Configuration options
   - Troubleshooting
   - Advanced topics

6. **DOCKER_README.md** - Quick reference guide
   - One-command installation
   - Common commands
   - Quick troubleshooting

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Docker Compose Stack                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  EquiLens    │◄─────┤   Ollama     │   │
│  │  Application │      │   Server     │   │
│  │              │      │              │   │
│  │  Port: 7860  │      │  Port: 11434 │   │
│  │  Port: 8000  │      │              │   │
│  └──────┬───────┘      └──────┬───────┘   │
│         │                     │           │
│         ▼                     ▼           │
│  ┌──────────────────────────────────┐    │
│  │     Persistent Volumes           │    │
│  ├──────────────────────────────────┤    │
│  │ • equilens-ollama-models (10GB+) │    │
│  │ • equilens-data                  │    │
│  │ • equilens-results               │    │
│  │ • equilens-logs                  │    │
│  └──────────────────────────────────┘    │
│                                           │
└───────────────────────────────────────────┘
```

## Usage

### One-Command Installation

**Windows:**
```powershell
irm https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.ps1 | iex
```

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.sh | bash
```

### Manual Setup

```powershell
# Clone repository
git clone https://github.com/[USER]/EquiLens.git
cd EquiLens

# Start services
docker-compose up -d

# Pull default model
docker exec equilens-ollama ollama pull llama3.2:latest

# View logs
docker-compose logs -f

# Access services
# Gradio UI:  http://localhost:7860
# Web API:    http://localhost:8000
# Ollama API: http://localhost:11434
```

### Common Commands

```powershell
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f equilens
docker-compose logs -f ollama

# Check status
docker-compose ps

# Access container
docker exec -it equilens-app bash

# Manage Ollama models
docker exec equilens-ollama ollama list
docker exec equilens-ollama ollama pull mistral:latest
docker exec equilens-ollama ollama rm <model-name>
```

## Features

### Ollama Service
- **Port**: 11434 (accessible from host and EquiLens container)
- **GPU Support**: NVIDIA GPU support enabled (optional)
- **Models**: Persistent storage in `ollama_models` volume
- **Configuration**: Optimized for concurrent requests and memory management

### EquiLens Application
- **Gradio UI**: Port 7860 - Interactive web interface
- **Web API**: Port 8000 - RESTful API endpoints
- **Persistent Data**: Results, logs, and data stored in volumes
- **Environment**: Python 3.13.3 with UV package manager
- **Security**: Non-root user, read-only source code mounts

### Persistent Volumes
1. **equilens-ollama-models** - Ollama models (10GB+)
2. **equilens-data** - Application data
3. **equilens-results** - Analysis results and reports
4. **equilens-logs** - Application logs

### Network
- **Name**: equilens-network
- **Type**: Bridge
- **Subnet**: 172.28.0.0/16
- **DNS**: Containers resolve by name (ollama, equilens)

## Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

#### Ollama
```yaml
environment:
  - OLLAMA_HOST=0.0.0.0:11434        # Bind address
  - OLLAMA_ORIGINS=*                  # CORS origins
  - OLLAMA_KEEP_ALIVE=24h             # Model memory retention
  - OLLAMA_NUM_PARALLEL=4             # Concurrent requests
  - OLLAMA_MAX_LOADED_MODELS=2        # Loaded models limit
```

#### EquiLens
```yaml
environment:
  - OLLAMA_BASE_URL=http://ollama:11434
  - GRADIO_SERVER_PORT=7860
  - EQUILENS_RESULTS_DIR=/workspace/results
```

### Port Changes

```yaml
ports:
  - "7861:7860"  # Change host port (left) if 7860 is busy
  - "8001:8000"  # Change host port (left) if 8000 is busy
```

### GPU Support

To enable/disable NVIDIA GPU:

```yaml
# Enable GPU
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]

# Disable GPU - remove entire deploy section
```

## Data Management

### Backup Volumes

```powershell
# Backup results
docker run --rm `
  -v equilens-results:/data `
  -v ${PWD}/backup:/backup `
  alpine tar czf /backup/results-$(Get-Date -Format "yyyyMMdd").tar.gz /data

# Backup Ollama models
docker run --rm `
  -v equilens-ollama-models:/data `
  -v ${PWD}/backup:/backup `
  alpine tar czf /backup/models-$(Get-Date -Format "yyyyMMdd").tar.gz /data
```

### Restore Volumes

```powershell
# Restore results
docker run --rm `
  -v equilens-results:/data `
  -v ${PWD}/backup:/backup `
  alpine tar xzf /backup/results-20240115.tar.gz -C /
```

### Export Results

```powershell
# Copy from container
docker cp equilens-app:/workspace/results ./local_results
```

## Troubleshooting

### Services Won't Start

```powershell
# Check Docker is running
docker ps

# View logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

### Ollama Not Accessible

```powershell
# Check Ollama health
docker exec equilens-ollama curl http://localhost:11434/api/tags

# Check from EquiLens container
docker exec equilens-app curl http://ollama:11434/api/tags

# Restart Ollama
docker-compose restart ollama
```

### Port Conflicts

```powershell
# Find process using port
netstat -ano | findstr :7860  # Windows
lsof -i :7860                  # Linux/macOS

# Change port in docker-compose.yml
ports:
  - "7861:7860"
```

### Out of Disk Space

```powershell
# Check disk usage
docker system df

# Clean up
docker system prune -a
docker volume prune

# Remove old models
docker exec equilens-ollama ollama list
docker exec equilens-ollama ollama rm <old-model>
```

## Requirements

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **CPU**: Multi-core processor recommended

### Software Requirements
- **Docker Desktop** 20.10+ (Windows/macOS)
- **Docker Engine** 20.10+ (Linux)
- **Docker Compose** 2.0+

### Optional
- **NVIDIA GPU** with CUDA support
- **NVIDIA Container Toolkit** for GPU acceleration

## Next Steps

### 1. Update GitHub URLs
Replace `[USER]` in the following files with your GitHub username:
- `setup-docker.ps1` (line 2 and line 34)
- `setup-docker.sh` (line 2 and line 44)
- `docs/DOCKER_SETUP.md` (multiple occurrences)
- `DOCKER_README.md` (line 5 and 10)

### 2. Test Local Setup
```powershell
# Test docker-compose
docker-compose config

# Start services
docker-compose up -d

# Check health
docker-compose ps

# Access Gradio UI
Start-Process "http://localhost:7860"
```

### 3. Commit to Repository
```powershell
git add Dockerfile docker-compose.yml setup-docker.* DOCKER_README.md docs/DOCKER_SETUP.md
git commit -m "Add Docker deployment with Ollama integration"
git push
```

### 4. Test One-Command Installation
After pushing to GitHub:
```powershell
# Windows
irm https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.ps1 | iex

# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.sh | bash
```

## Documentation

- **Complete Guide**: `docs/DOCKER_SETUP.md`
- **Quick Reference**: `DOCKER_README.md`
- **Local Setup**: `docs/QUICKSTART.md`
- **Configuration**: `docs/CONFIGURATION_GUIDE.md`

## Support

- **Issues**: GitHub Issues
- **Logs**: `docker-compose logs`
- **Community**: Discussion forums

## What's Different from Local Setup?

### Local Setup (setup.ps1)
- Installs Python dependencies locally
- Ollama runs as system service
- Uses local file system
- Direct Python execution

### Docker Setup (setup-docker.ps1)
- Containerized environment
- Ollama in container
- Persistent volumes
- Isolated execution
- One-command deployment
- Cross-platform consistency

## Benefits of Docker Setup

1. **Isolation**: No conflicts with system packages
2. **Portability**: Same environment everywhere
3. **Easy Cleanup**: Remove containers and volumes
4. **Version Control**: Pin specific versions
5. **Scalability**: Easy to scale services
6. **Security**: Non-root execution
7. **Consistency**: Same setup on all platforms

## License

EquiLens is open-source software. See `LICENSE.md` for details.

---

**Setup completed successfully!** 🎉

You now have:
- ✅ Dockerfile
- ✅ docker-compose.yml
- ✅ setup-docker.ps1 (Windows)
- ✅ setup-docker.sh (Linux/macOS)
- ✅ Complete documentation
- ✅ Quick reference guide

Ready for deployment! 🚀
