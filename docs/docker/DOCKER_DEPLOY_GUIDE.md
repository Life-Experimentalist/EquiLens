# 🐳 Docker Deployment Quick Reference

Quick commands for deploying EquiLens using Docker and GitHub Container Registry.

## 📦 Using Pre-built Images

### Pull from GitHub Container Registry

```powershell
# Pull latest version
docker pull ghcr.io/life-experimentalist/equilens:latest

# Pull specific version
docker pull ghcr.io/life-experimentalist/equilens:2.0.0
```

### Run with Docker

```powershell
# Simple run (development)
docker run -d \
  --name equilens \
  -p 7860:7860 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  ghcr.io/life-experimentalist/equilens:latest

# Production run (with volumes)
docker run -d \
  --name equilens-prod \
  --restart unless-stopped \
  -p 7860:7860 \
  -v equilens-data:/workspace/data \
  -v equilens-results:/workspace/data/results \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_PORT=11434 \
  ghcr.io/life-experimentalist/equilens:latest

# Access at http://localhost:7860
```

### Run with Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  equilens:
    image: ghcr.io/life-experimentalist/equilens:2.0.0
    container_name: equilens
    restart: unless-stopped
    ports:
      - "7860:7860"
    volumes:
      - equilens-data:/workspace/data
      - equilens-results:/workspace/data/results
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - OLLAMA_PORT=11434

volumes:
  equilens-data:
  equilens-results:
```

```powershell
# Start services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Stop services
docker compose -f docker-compose.prod.yml down
```

## 🏗️ Building from Source

### Build Image

```powershell
# Clone repository
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens

# Build image
docker build -t equilens:local .

# Run locally built image
docker run -d -p 7860:7860 --name equilens equilens:local
```

### Build Multi-platform

```powershell
# Create builder
docker buildx create --use --name multiplatform-builder

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/life-experimentalist/equilens:latest \
  --push .
```

## 🚀 Deployment Workflow

### Version Release Process

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "2.1.0"
   ```

2. **Tag and push**:
   ```powershell
   git add pyproject.toml
   git commit -m "chore: bump version to 2.1.0"
   git tag v2.1.0
   git push origin main
   git push origin v2.1.0
   ```

3. **GitHub Actions automatically**:
   - Builds image for linux/amd64 and linux/arm64
   - Tags as `2.1.0`, `2.1`, `2`, `latest`
   - Pushes to GitHub Container Registry
   - Generates security attestation

4. **Deploy new version**:
   ```powershell
   docker pull ghcr.io/life-experimentalist/equilens:2.1.0
   docker stop equilens
   docker rm equilens
   docker run -d -p 7860:7860 --name equilens ghcr.io/life-experimentalist/equilens:2.1.0
   ```

## 🔧 Environment Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434  # Auto-detected
OLLAMA_PORT=11434                                   # Default port
OLLAMA_HOST=localhost                               # Alternative

# EquiLens Configuration
EQUILENS_DATA_DIR=/workspace/data
EQUILENS_RESULTS_DIR=/workspace/data/results

# Gradio Web UI
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

## 📊 Monitoring

### Check Container Status

```powershell
# View running containers
docker ps | findstr equilens

# View logs
docker logs -f equilens

# View resource usage
docker stats equilens

# Execute commands in container
docker exec -it equilens python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"
```

### Health Checks

```powershell
# Check web UI
curl http://localhost:7860

# Check Ollama connectivity from container
docker exec -it equilens curl http://host.docker.internal:11434

# Check container health
docker inspect equilens | findstr Health
```

## 🔄 Updates

### Update to Latest

```powershell
# Pull latest image
docker pull ghcr.io/life-experimentalist/equilens:latest

# Stop and remove old container
docker stop equilens
docker rm equilens

# Run new version
docker run -d \
  --name equilens \
  --restart unless-stopped \
  -p 7860:7860 \
  -v equilens-data:/workspace/data \
  ghcr.io/life-experimentalist/equilens:latest
```

### Rollback to Previous Version

```powershell
# Pull specific version
docker pull ghcr.io/life-experimentalist/equilens:2.0.0

# Stop current
docker stop equilens
docker rm equilens

# Run older version
docker run -d -p 7860:7860 --name equilens ghcr.io/life-experimentalist/equilens:2.0.0
```

## 🧹 Cleanup

```powershell
# Remove container
docker stop equilens
docker rm equilens

# Remove volumes (WARNING: deletes data)
docker volume rm equilens-data equilens-results

# Remove images
docker rmi ghcr.io/life-experimentalist/equilens:latest

# Clean up unused images
docker image prune -a
```

## 🔐 GitHub Container Registry Access

### Public Access (No Auth Required)

```powershell
# Public images can be pulled without authentication
docker pull ghcr.io/life-experimentalist/equilens:latest
```

### Authenticated Access

```powershell
# Login with Personal Access Token
$env:CR_PAT = "your_github_token"
echo $env:CR_PAT | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Pull image
docker pull ghcr.io/life-experimentalist/equilens:latest
```

## 📚 Further Documentation

- **Full Deployment Guide**: [../DEPLOYMENT.md](../DEPLOYMENT.md)
- **Environment Variables**: [ENVIRONMENT_VARIABLE_LOGIC.md](ENVIRONMENT_VARIABLE_LOGIC.md)
- **Docker Setup**: [DOCKER_SETUP.md](DOCKER_SETUP.md)
- **Main README**: [../../README.md](../../README.md)

---

**Quick Links**:
- [GitHub Container Registry](https://github.com/Life-Experimentalist/EquiLens/pkgs/container/equilens)
- [GitHub Repository](https://github.com/Life-Experimentalist/EquiLens)
- [Issues](https://github.com/Life-Experimentalist/EquiLens/issues)
