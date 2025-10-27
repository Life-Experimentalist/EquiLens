# Smart Ollama Configuration System

## Overview

EquiLens now features an **intelligent Ollama configuration system** that automatically detects your deployment environment and configures the correct Ollama endpoint URL. No more manual configuration or guessing which URL to use!

## Deployment Scenarios

The system handles **4 different deployment scenarios** automatically:

### 1. **Container → Container** 🐳→🐳
- **EquiLens**: Running in Docker container
- **Ollama**: Running in Docker container
- **URL Used**: `http://ollama:11434` (Docker Compose service name)
- **How it works**: Containers on the same Docker network can communicate using service names

### 2. **Container → Host** 🐳→💻
- **EquiLens**: Running in Docker container
- **Ollama**: Running on host machine (Docker Desktop app or native install)
- **URL Used**: `http://host.docker.internal:11434`
- **How it works**: Special DNS name that resolves to host from inside containers

### 3. **Local → Container** 💻→🐳
- **EquiLens**: Running locally (not in container)
- **Ollama**: Running in Docker container with exposed port
- **URL Used**: `http://localhost:11434`
- **How it works**: Container exposes port 11434 to host, accessible via localhost

### 4. **Local → Local** 💻→💻
- **EquiLens**: Running locally (not in container)
- **Ollama**: Running locally (native install or Docker Desktop)
- **URL Used**: `http://localhost:11434`
- **How it works**: Standard localhost communication

## How Detection Works

### Container Detection

The system checks if EquiLens is running inside a Docker container by:

1. **Method 1**: Checking for `/.dockerenv` file (Docker creates this)
2. **Method 2**: Inspecting `/proc/1/cgroup` for "docker" or "containerd"
3. **Method 3**: Environment variable `EQUILENS_IN_CONTAINER=true`

### Ollama Detection

The system checks if Ollama is containerized by:

1. Running `docker ps --filter ancestor=ollama/ollama` to find Ollama containers
2. Testing connections to various Ollama endpoints
3. Using environment variable overrides if provided

### URL Selection Priority

1. **Explicit Override**: `OLLAMA_BASE_URL` or `OLLAMA_HOST` environment variable
2. **Smart Detection**: Auto-detects environment and tests candidate URLs
3. **Fallback Chain**: Tries multiple URLs in order until one works
4. **Caching**: Caches the working URL for performance

## Usage

### Automatic (Recommended)

The system works automatically - just run EquiLens and it will figure out the correct configuration:

```bash
# In Docker
docker compose up

# Locally
uv run equilens gui
```

### Environment Variable Override

If you need to explicitly specify the Ollama URL:

```bash
# Docker Compose (in docker-compose.yml)
environment:
  - OLLAMA_BASE_URL=http://my-custom-ollama:11434

# Local execution
export OLLAMA_BASE_URL=http://192.168.1.100:11434
uv run equilens gui
```

### Force Container Detection

If auto-detection fails:

```bash
export EQUILENS_IN_CONTAINER=true
```

## API Reference

### Python API

```python
from equilens.core.ollama_config import get_ollama_url, get_environment_info, is_running_in_container

# Get the correct Ollama URL
url = get_ollama_url()
print(f"Ollama URL: {url}")

# Get detailed environment information
env_info = get_environment_info()
print(f"Scenario: {env_info['scenario']}")
print(f"Description: {env_info['description']}")
print(f"EquiLens in container: {env_info['equilens_in_container']}")
print(f"Ollama in container: {env_info['ollama_in_container']}")

# Check if running in container
in_container = is_running_in_container()
print(f"Running in container: {in_container}")

# Force re-detection (bypass cache)
url = get_ollama_url(force_refresh=True)
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Override Ollama URL (highest priority) | `http://ollama:11434` |
| `OLLAMA_HOST` | Alternative override | `host.docker.internal:11434` |
| `EQUILENS_IN_CONTAINER` | Force container detection | `true`, `1`, `yes` |

## Architecture

### Smart Configuration Module

Location: `src/equilens/core/ollama_config.py`

Key components:

1. **OllamaConfig class**: Core detection and configuration logic
2. **get_ollama_url()**: Convenience function for getting URL
3. **get_environment_info()**: Get detailed environment information
4. **is_running_in_container()**: Check container status

### Integration Points

The smart configuration is integrated into:

- ✅ `src/Phase2_ModelAuditor/audit_model.py`
- ✅ `src/Phase2_ModelAuditor/enhanced_audit_model.py`
- ✅ `src/Phase3_Analysis/analytics.py`
- ✅ `src/equilens/core/manager.py`
- ✅ `src/equilens/core/docker.py`
- ✅ `src/equilens/web_ui.py`
- ✅ `src/equilens/gradio_ui.py`
- ✅ `src/equilens/cli.py`

## Testing

### Test Connectivity

```bash
# From inside container
docker exec -it equilens-app python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"

# From local
python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"
```

### Test Script

Use the provided test script:

```bash
# From inside container
docker exec -it equilens-app python scripts/tools/test_docker_networking.py

# From local
python scripts/tools/test_docker_networking.py
```

## Troubleshooting

### Issue: Wrong URL detected

**Solution**: Use explicit environment variable override:
```bash
export OLLAMA_BASE_URL=http://correct-url:11434
```

### Issue: Connection fails to all URLs

**Checks**:
1. Is Ollama running? `docker ps | grep ollama` or check Docker Desktop
2. Is port 11434 exposed? Check docker-compose.yml or container settings
3. Firewall blocking? Check Windows Firewall or antivirus

### Issue: Container detection fails

**Solution**: Explicitly set environment variable:
```bash
export EQUILENS_IN_CONTAINER=true
```

### Issue: Need to force re-detection

**Solution**: Use force_refresh parameter:
```python
from equilens.core.ollama_config import get_ollama_url
url = get_ollama_url(force_refresh=True)
```

## Migration from Previous Setup

If you were using hardcoded URLs before:

### Before (Hardcoded)
```python
ollama_url = "http://localhost:11434"
response = requests.get(f"{ollama_url}/api/tags")
```

### After (Smart Detection)
```python
from equilens.core.ollama_config import get_ollama_url

ollama_url = get_ollama_url()
response = requests.get(f"{ollama_url}/api/tags")
```

## Performance

- **First call**: ~100-300ms (detection + connection tests)
- **Subsequent calls**: <1ms (cached URL)
- **Cache invalidation**: Use `force_refresh=True` parameter

## Future Enhancements

Potential improvements for future versions:

- [ ] Auto-retry with different URLs on connection failure
- [ ] Health check endpoint for monitoring
- [ ] Support for multiple Ollama instances (load balancing)
- [ ] Kubernetes service discovery
- [ ] Configuration UI in web interface

## Example: docker-compose.yml

```yaml
services:
  equilens:
    image: equilens:latest
    container_name: equilens-app
    environment:
      # Optional: Override if auto-detection doesn't work
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      # Optional: Force container detection
      - EQUILENS_IN_CONTAINER=true
    ports:
      - "7860:7860"
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs: `docker compose logs equilens` or `logs/` directory
3. Test with verification script: `scripts/tools/test_docker_networking.py`
4. Enable debug logging: Set `LOG_LEVEL=DEBUG` environment variable
