# EquiLens Environment Variable Logic

## Overview

EquiLens uses smart environment variable detection to determine whether it's running in a Docker container or locally, and how to connect to Ollama.

## Key Principle

**If `OLLAMA_BASE_URL` environment variable doesn't exist → EquiLens is running locally**

## Why This Works

When you run EquiLens locally (e.g., `uv run equilens audit`):
- The environment variables from `docker-compose.yml` are NOT set
- No `OLLAMA_BASE_URL` env var exists
- System uses localhost to connect to Ollama

When you run EquiLens in Docker container:
- Docker Compose automatically sets all environment variables from `docker-compose.yml`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434` is present
- System uses host.docker.internal to connect to Ollama

## Environment Variables

### Core Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | (none) | Primary Ollama URL. If set and working, used directly. If not set, indicates local install. |
| `OLLAMA_HOST` | (none) | Alternative to OLLAMA_BASE_URL |
| `OLLAMA_PORT` | `11434` | Configurable Ollama port |
| `EQUILENS_IN_CONTAINER` | (auto-detect) | Force container detection ("true"/"1"/"yes") |

### Detection Logic Flow

```
1. Check OLLAMA_BASE_URL env var
   ├─ Not set? → Local install, use localhost
   ├─ Set AND works? → Use it directly
   └─ Set but doesn't work? → In container, try container URLs

2. Auto-detect container environment
   ├─ Check for .dockerenv file
   ├─ Check /proc/1/cgroup for docker/containerd
   └─ Check EQUILENS_IN_CONTAINER override

3. Build URL candidates based on environment
   ├─ In container? → host.docker.internal:PORT (primary)
   └─ Local? → localhost:PORT (primary)

4. Test each URL and cache first working one
```

## URL Selection Rules

### When EquiLens is in Docker Container
```
Primary:   http://host.docker.internal:11434
Fallback:  http://localhost:11434
```

**Why host.docker.internal?**
- Docker Desktop special DNS name
- Routes to host machine's network
- Works for BOTH containerized Ollama (exposed port) AND native Ollama on host
- No custom networks needed

### When EquiLens is Local
```
Primary:   http://localhost:11434
Fallback:  http://127.0.0.1:11434
```

**Why localhost?**
- Standard local connection
- Works for BOTH containerized Ollama (exposed port) AND native Ollama on host
- Containerized Ollama exposes port 11434 to host network

## Configuring Custom Ports

If your Ollama runs on a non-standard port:

### Docker Container
```yaml
# docker-compose.yml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:12345
  - OLLAMA_PORT=12345
```

### Local Installation
```powershell
# PowerShell
$env:OLLAMA_PORT = "12345"
uv run equilens audit --model llama2

# Or specify URL directly
$env:OLLAMA_BASE_URL = "http://localhost:12345"
uv run equilens audit --model llama2
```

## Troubleshooting

### EquiLens can't find Ollama

1. **Check which mode you're running in:**
   ```powershell
   # From Python
   from equilens.core.ollama_config import get_environment_info
   print(get_environment_info())
   ```

2. **Verify Ollama is accessible:**
   ```powershell
   # Test from host terminal - executes curl inside the equilens-app container
   docker exec -it equilens-app curl http://host.docker.internal:11434/api/version

   # From host directly
   curl http://localhost:11434/api/version
   ```

3. **Check environment variables:**
   ```powershell
   # Check env vars inside container (command run from host terminal)
   docker exec -it equilens-app env | findstr OLLAMA

   # On host
   Get-ChildItem Env: | Where-Object Name -like "*OLLAMA*"
   ```

### Override Detection

Force container mode (if auto-detection fails):
```powershell
$env:EQUILENS_IN_CONTAINER = "true"
```

Force specific Ollama URL:
```powershell
$env:OLLAMA_BASE_URL = "http://your-custom-url:11434"
```

## Examples

### Example 1: Standard Local Setup
```powershell
# Ollama running locally (native or Docker with exposed port)
# No env vars needed
PS> uv run equilens audit --model llama2

# Auto-detects:
# - OLLAMA_BASE_URL not set → Local install
# - Uses http://localhost:11434
```

### Example 2: Docker Container Setup
```powershell
# Start EquiLens container
PS> docker compose up -d

# Inside container, OLLAMA_BASE_URL is set
# Auto-detects:
# - OLLAMA_BASE_URL=http://host.docker.internal:11434
# - Uses that URL directly
```

### Example 3: Custom Port
```powershell
# Local with custom port
PS> $env:OLLAMA_PORT = "12345"
PS> uv run equilens audit --model llama2

# Auto-detects:
# - Uses http://localhost:12345
```

### Example 4: External Ollama Server
```powershell
# Connect to remote Ollama
PS> $env:OLLAMA_BASE_URL = "http://192.168.1.100:11434"
PS> uv run equilens audit --model llama2

# Uses specified URL directly
```

## Best Practices

1. **Let auto-detection work**: Don't set environment variables unless needed
2. **Use OLLAMA_PORT for port changes**: Easier than full URL
3. **Validate with curl**: Always test Ollama connectivity before running audits
4. **Check environment info**: Use `get_environment_info()` to verify detection
5. **Container-to-host**: Always use `host.docker.internal` in Docker Compose files

## Related Documentation

- [Simplified Ollama Config](./SIMPLIFIED_OLLAMA_CONFIG.md) - Implementation details
- [Docker Setup Guide](./DOCKER_SETUP.md) - Container configuration
