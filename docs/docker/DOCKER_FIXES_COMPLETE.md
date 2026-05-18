# Docker Container Fix - Complete Summary

## Problem Resolution

### Initial Issues
1. **PowerShell parse errors**: Emoji characters in `setup-docker-local.ps1` causing script failures
2. **Unnecessary rebuilds**: 10+ minute builds every time even when image exists
3. **Container crash loop**: `ModuleNotFoundError: No module named 'equilens'`
4. **Network access**: Web UI not accessible from host

### Root Causes Identified

#### 1. Module Import Failure
- **Issue**: System Python couldn't find the `equilens` package
- **Root Cause**: Package installed with `uv sync` creates editable install via `.pth` file pointing to `/workspace/src/equilens`
- **Volume Mount Problem**: `./src:/workspace/src:ro` mount was **empty** on Windows (V: drive access issue)
- **Solution**: Removed `./src` volume mount - source code already baked into image

#### 2. CMD Configuration
- **Evolution of fixes**:
  1. Initial: `CMD ["uv", "run", "equilens"]` → Failed (uv command issues)
  2. Attempt 2: `CMD ["python", "-m", "equilens.cli"]` → Failed (system Python missing deps)
  3. Attempt 3: `CMD [".venv/bin/python", "-m", "equilens.cli"]` → Works in `docker run`, fails in `docker-compose`
  4. **Final Solution**: `CMD [".venv/bin/equilens", "gui"]` → ✅ SUCCESS

- **Why it works**:
  - UV installs executable scripts in `.venv/bin/`
  - The `equilens` script uses venv Python automatically
  - Starts the GUI service which keeps container running

#### 3. Network Mode
- **Issue**: `network_mode: "host"` doesn't work on Docker Desktop for Windows
- **Solution**: Use explicit port mapping (`7860:7860`, `8000:8000`)
- **Ollama Connection**: Changed from `localhost:11434` to `host.docker.internal:11434`

## Final Configuration

### Dockerfile
```dockerfile
FROM python:3.13.3-slim

# System setup
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get autoremove -y

# Install UV package manager
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir uv

# Create user and directories
RUN useradd -m -u 1000 -s /bin/bash equilens && \
    mkdir -p /workspace/data/results /workspace/data/logs /workspace/data/corpus && \
    chown -R equilens:equilens /workspace

USER equilens
WORKDIR /workspace

# Install dependencies
COPY --chown=equilens:equilens pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy application code
COPY --chown=equilens:equilens . .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/workspace/src:/workspace \
    OLLAMA_BASE_URL=http://localhost:11434 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0

# Start Gradio GUI
CMD [".venv/bin/equilens", "gui"]
```

### docker-compose.yml
```yaml
services:
  equilens:
    build:
      context: .
      dockerfile: Dockerfile
    image: equilens:latest
    container_name: equilens-app
    ports:
      - "7860:7860"
      - "8000:8000"
    volumes:
      - equilens_data:/workspace/data
      # DO NOT mount ./src - it's already in the image
      # Mounting causes Windows path issues
      - ./public:/workspace/public:ro
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - OLLAMA_API_BASE=http://host.docker.internal:11434/api
      - EQUILENS_DATA_DIR=/workspace/data
      - EQUILENS_RESULTS_DIR=/workspace/data/results
      - EQUILENS_LOGS_DIR=/workspace/data/logs
      - EQUILENS_CORPUS_DIR=/workspace/src/Phase1_CorpusGenerator/corpus
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=7860
      - GRADIO_ANALYTICS_ENABLED=false
      - GRADIO_THEME=default
    restart: unless-stopped
    stdin_open: true
    tty: true
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:7860/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  equilens_data:
    driver: local
    name: equilens-data
```

## Testing & Verification

### Successful Tests
```powershell
# 1. Container starts without errors
docker-compose up -d
# Status: Up (healthy)

# 2. Web UI is accessible
Invoke-WebRequest -Uri http://localhost:7860
# Response: 200 OK

# 3. Container logs show GUI started
docker logs equilens-app
# Output: "Running on local URL: http://0.0.0.0:7860"

# 4. Ports are properly mapped
docker ps --filter name=equilens-app
# Shows: 0.0.0.0:7860->7860/tcp, 0.0.0.0:8000->8000/tcp
```

## Key Learnings

### 1. UV Package Management
- `uv sync` creates editable installs via `.pth` files
- Installs executable scripts in `.venv/bin/`
- Scripts automatically use venv Python (no need to specify interpreter)

### 2. Docker on Windows
- `network_mode: host` is Linux-only, doesn't work on Docker Desktop for Windows
- Use `host.docker.internal` to access host services from container
- Volume mounts can fail with non-standard drive letters (like V:)
- **Best practice**: Bake code into image, don't mount unless dev mode

### 3. Container Command Best Practices
- Always run a **service** (not just CLI) to keep container alive
- Use health checks to monitor service status
- Use venv executables directly (`.venv/bin/equilens`) instead of `python -m`

## Usage

### Start Services
```powershell
docker-compose up -d
```

### Access Web UI
Open browser to: http://localhost:7860

### Check Status
```powershell
docker ps
docker logs equilens-app
```

### Stop Services
```powershell
docker-compose down
```

### Rebuild After Changes
```powershell
docker-compose build
docker-compose up -d
```

## Development vs Production

### Production (Current)
- Source code baked into image
- No volume mounts for code
- Fast startup, consistent environment

### Development Mode
To enable live code editing, uncomment in docker-compose.yml:
```yaml
volumes:
  - ./src:/workspace/src:ro  # Enable for dev mode
```
**Note**: Only works if Docker has access to the drive!

## Next Steps

1. ✅ Container starts successfully
2. ✅ Web UI accessible
3. ✅ Health checks passing
4. ⏳ Test Ollama integration (requires Ollama running on host)
5. ⏳ Test full audit workflow
6. ⏳ Update `setup-docker-local.ps1` with smart rebuild detection
7. ⏳ Update deployment scripts with new configuration

## Related Documents
- `DOCKER_CONFIG_GUIDE.md` - Comprehensive configuration reference
- `SETUP_DOCKER_LOCAL_CHANGES.md` - Local setup script improvements
- `DOCKER_DEPLOY_QUICKREF.md` - Quick deployment reference
