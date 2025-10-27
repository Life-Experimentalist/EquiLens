# Port Management Guide

## Overview

EquiLens implements flexible port management to allow multiple instances to run simultaneously without conflicts. The system automatically detects available ports and provides environment variable overrides for custom configurations.

## Architecture

### Port Management Module

Location: `src/equilens/core/ports.py`

The centralized port management module provides utilities for all EquiLens services:

- **Backend API**: Default port 8000
- **Gradio Frontend**: Default port 7860

### Key Features

1. **Automatic Port Detection**: Checks if default ports are available
2. **Fallback Mechanism**: Automatically finds next available port if default is taken
3. **Environment Variable Support**: Customize ports via environment variables
4. **Docker-aware**: Detects Docker environment and adjusts URLs accordingly
5. **User-friendly Feedback**: Shows actual ports and URLs on startup

## Port Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_PORT` | 8000 | Backend API port |
| `FRONTEND_PORT` | 7860 | Gradio frontend port |
| `GRADIO_PORT` | 7860 | Alternative frontend port variable |

### Setting Ports in PowerShell

```powershell
# Set custom backend port
$env:BACKEND_PORT = 8001
uv run equilens backend

# Set custom frontend port
$env:FRONTEND_PORT = 8080
uv run equilens serve

# Multiple instances with different ports
$env:FRONTEND_PORT = 7861; uv run equilens serve
$env:FRONTEND_PORT = 7862; uv run equilens web
```

### Setting Ports in Docker

```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - BACKEND_PORT=8001
    ports:
      - "8001:8001"

  frontend:
    environment:
      - FRONTEND_PORT=7861
    ports:
      - "7861:7861"
```

## Usage Scenarios

### Scenario 1: Default Ports

Simply run the service without any configuration:

```powershell
# Backend starts on 8000, frontend on 7860
uv run equilens start
```

**What happens:**
- Backend checks port 8000 → available → uses 8000
- Frontend checks port 7860 → available → uses 7860
- Services start successfully on default ports

### Scenario 2: Port Conflict

Another Gradio app is already running on port 7860:

```powershell
# Old Gradio running on 7860
# Start new EquiLens instance
uv run equilens serve
```

**What happens:**
- Frontend checks port 7860 → occupied → tries 7861
- Frontend checks port 7861 → available → uses 7861
- Service starts on next available port automatically

**Output:**
```
🎯 EquiLens Gradio Frontend Starting...
🔗 Backend URL: http://localhost:8000
🌐 Frontend Port: 7861 (port 7860 was unavailable)
✅ Backend connection: OK
```

### Scenario 3: Multiple EquiLens Instances

Running multiple EquiLens instances for different projects:

```powershell
# Terminal 1 - Project A (defaults)
uv run equilens serve

# Terminal 2 - Project B (custom ports)
$env:BACKEND_PORT = 8001; $env:FRONTEND_PORT = 7861; uv run equilens start

# Terminal 3 - Project C (custom ports)
$env:BACKEND_PORT = 8002; $env:FRONTEND_PORT = 7862; uv run equilens serve
```

**Result:**
- Project A: Backend 8000, Frontend 7860
- Project B: Backend 8001, Frontend 7861
- Project C: Backend 8002, Frontend 7862

### Scenario 4: Custom Ports

Explicitly set ports for your environment:

```powershell
# Use ports that fit your infrastructure
$env:BACKEND_PORT = 9000
$env:FRONTEND_PORT = 9001
uv run equilens start
```

**Result:**
- Backend runs on port 9000
- Frontend runs on port 9001
- No automatic fallback (uses specified ports or fails if taken)

## Service Launchers

### Individual Services

#### Backend Only
```powershell
uv run equilens backend

# Output:
# 🚀 EquiLens Backend API Starting...
# 📊 API URL: http://localhost:8000/api
# 🏥 Health Check: http://localhost:8000/api/health
# 📚 API Docs: http://localhost:8000/docs
```

#### Frontend Only (with Backend)
```powershell
uv run equilens serve

# Output:
# 🎯 EquiLens Gradio Frontend Starting...
# 🔗 Backend URL: http://localhost:8000
# 🌐 Frontend Port: 7860
# ✅ Backend connection: OK
```

#### Legacy Frontend (Standalone)
```powershell
uv run equilens web

# Output:
# 🎯 EquiLens Gradio Frontend (Legacy Standalone) Starting...
# 🌐 Frontend Port: 7860
# 💡 Set FRONTEND_PORT or GRADIO_PORT environment variable for custom port
```

### Combined Services

Start both backend and frontend together:

```powershell
uv run equilens start

# Output:
# ════════════════════════════════════════════════════════
#   EquiLens Services Starting
# ════════════════════════════════════════════════════════
# 🚀 Backend API:   http://localhost:8000/api
# 🎯 Frontend UI:   http://localhost:7860
# 📚 API Docs:      http://localhost:8000/docs
# ════════════════════════════════════════════════════════
```

## Port Availability Checking

### How It Works

The port management system uses socket binding to check availability:

```python
import socket

def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            return True
    except OSError:
        return False
```

### Fallback Algorithm

1. Check preferred port (default or env var)
2. If available → use it
3. If taken → try next port (port + 1)
4. Repeat up to 10 attempts
5. If all attempts fail → raise error

## Docker Environment

### Automatic Detection

The system automatically detects Docker environment:

```python
def get_backend_url(port: int = None) -> str:
    """Get backend URL with Docker detection."""
    in_docker = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"

    if in_docker:
        return f"http://backend:{port or 8000}"
    else:
        return f"http://localhost:{port or 8000}"
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  backend:
    build: .
    container_name: equilens-backend
    environment:
      - DOCKER_ENV=true
      - BACKEND_PORT=8000
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    networks:
      - equilens-network

  frontend:
    build: .
    container_name: equilens-frontend
    environment:
      - DOCKER_ENV=true
      - FRONTEND_PORT=7860
      - BACKEND_URL=http://backend:8000
    ports:
      - "7860:7860"
    depends_on:
      - backend
    networks:
      - equilens-network

networks:
  equilens-network:
    driver: bridge
```

## Troubleshooting

### Problem: "Address already in use"

**Symptoms:**
```
OSError: [Errno 98] Address already in use
```

**Solutions:**

1. **Let EquiLens auto-detect next port:**
   ```powershell
   # Just run - it will find an available port
   uv run equilens serve
   ```

2. **Use custom port explicitly:**
   ```powershell
   $env:FRONTEND_PORT = 8080
   uv run equilens serve
   ```

3. **Kill process using the port:**
   ```powershell
   # Find process on Windows
   netstat -ano | findstr :7860

   # Kill by PID
   taskkill /PID <pid> /F
   ```

### Problem: "Connection refused" (Frontend → Backend)

**Symptoms:**
```
❌ Backend connection: Failed
Connection refused at http://localhost:8000
```

**Solutions:**

1. **Start backend first:**
   ```powershell
   # Terminal 1
   uv run equilens backend

   # Terminal 2
   uv run equilens serve
   ```

2. **Use combined launcher:**
   ```powershell
   uv run equilens start
   ```

3. **Check backend URL:**
   ```powershell
   # In Docker, ensure BACKEND_URL is set
   $env:BACKEND_URL = "http://backend:8000"
   ```

### Problem: Multiple Instances on Same Port

**Symptoms:**
Second instance doesn't start or crashes

**Solution:**
The system automatically handles this, but you can force ports:

```powershell
# Instance 1 (Terminal 1)
uv run equilens start

# Instance 2 (Terminal 2) - force different ports
$env:BACKEND_PORT = 8001
$env:FRONTEND_PORT = 7861
uv run equilens start
```

### Problem: Docker Port Mapping Issues

**Symptoms:**
Service accessible inside container but not from host

**Solution:**
Ensure Docker Compose port mapping matches internal port:

```yaml
services:
  frontend:
    environment:
      - FRONTEND_PORT=7860  # Internal port
    ports:
      - "7860:7860"  # Host:Container mapping must match
```

## Best Practices

### Development

1. **Use defaults**: Let EquiLens auto-detect ports during development
2. **Custom ports for conflicts**: Only set env vars when you have conflicts
3. **Document custom configs**: Keep track of custom ports in your project README

### Production

1. **Explicit port configuration**: Always set ports explicitly via env vars
2. **Port mapping documentation**: Document port mappings in deployment docs
3. **Health checks**: Monitor service health endpoints regularly
4. **Load balancer configuration**: Ensure load balancer knows correct ports

### CI/CD

1. **Dynamic port allocation**: Use ephemeral port ranges (e.g., 49152-65535)
2. **Environment-based config**: Different ports per environment (dev/staging/prod)
3. **Container orchestration**: Let Kubernetes/Docker Swarm manage port allocation

## API Reference

### Functions

#### `find_available_port(start_port: int, max_attempts: int = 10) -> int`
Find an available port starting from `start_port`.

**Parameters:**
- `start_port`: Initial port to check
- `max_attempts`: Maximum number of ports to try (default: 10)

**Returns:**
- Available port number

**Raises:**
- `RuntimeError`: If no available port found within attempts

**Example:**
```python
from equilens.core.ports import find_available_port

port = find_available_port(8000)
print(f"Using port: {port}")  # e.g., 8000 or 8001 if 8000 is taken
```

#### `is_port_available(port: int) -> bool`
Check if a specific port is available.

**Parameters:**
- `port`: Port number to check

**Returns:**
- `True` if available, `False` otherwise

**Example:**
```python
from equilens.core.ports import is_port_available

if is_port_available(8000):
    print("Port 8000 is available")
else:
    print("Port 8000 is in use")
```

#### `get_backend_port() -> int`
Get available backend port with environment variable support.

**Environment Variables:**
- `BACKEND_PORT`: Custom backend port

**Returns:**
- Available backend port (default starts at 8000)

**Example:**
```python
from equilens.core.ports import get_backend_port

port = get_backend_port()
# Uses $env:BACKEND_PORT if set, otherwise finds available port starting at 8000
```

#### `get_frontend_port() -> int`
Get available frontend port with environment variable support.

**Environment Variables:**
- `FRONTEND_PORT` or `GRADIO_PORT`: Custom frontend port

**Returns:**
- Available frontend port (default starts at 7860)

**Example:**
```python
from equilens.core.ports import get_frontend_port

port = get_frontend_port()
# Uses $env:FRONTEND_PORT if set, otherwise finds available port starting at 7860
```

#### `get_backend_url(port: int = None) -> str`
Get backend URL with Docker environment detection.

**Parameters:**
- `port`: Optional port override (uses `get_backend_port()` if not specified)

**Returns:**
- Backend URL string (http://backend:{port} in Docker, http://localhost:{port} locally)

**Example:**
```python
from equilens.core.ports import get_backend_url

url = get_backend_url()
# In Docker: http://backend:8000
# Locally:   http://localhost:8000
```

#### `get_service_ports() -> tuple[int, int]`
Get both backend and frontend ports.

**Returns:**
- Tuple of (backend_port, frontend_port)

**Example:**
```python
from equilens.core.ports import get_service_ports

backend_port, frontend_port = get_service_ports()
print(f"Backend: {backend_port}, Frontend: {frontend_port}")
```

#### `print_service_info(backend_port: int, frontend_port: int)`
Print formatted service information banner.

**Parameters:**
- `backend_port`: Backend API port
- `frontend_port`: Frontend UI port

**Example:**
```python
from equilens.core.ports import print_service_info

print_service_info(8000, 7860)
# Displays formatted banner with service URLs
```

## Integration Examples

### Custom Service Launcher

```python
from equilens.core.ports import get_backend_port, get_frontend_port
import uvicorn

def launch_custom_service():
    """Launch custom service with flexible ports."""
    backend_port = get_backend_port()
    frontend_port = get_frontend_port()

    print(f"🚀 Custom Service Starting...")
    print(f"📊 Backend: http://localhost:{backend_port}")
    print(f"🎯 Frontend: http://localhost:{frontend_port}")

    # Your service logic here
    uvicorn.run("app:app", host="0.0.0.0", port=backend_port)
```

### Health Check Script

```python
from equilens.core.ports import get_backend_url
import requests

def check_service_health():
    """Check if EquiLens backend is healthy."""
    backend_url = get_backend_url()

    try:
        response = requests.get(f"{backend_url}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Backend is healthy at {backend_url}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend unreachable: {e}")
        return False
```

## See Also

- [Configuration Guide](../CONFIGURATION_GUIDE.md) - General configuration options
- [Docker Setup](../docker/DOCKER_SETUP.md) - Docker-specific configuration
- [Interface Architecture](../architecture/INTERFACE_ARCHITECTURE.md) - System architecture overview
- [Deployment Guide](../DEPLOYMENT.md) - Production deployment recommendations
