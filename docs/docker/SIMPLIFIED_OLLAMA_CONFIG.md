# Simplified Ollama Configuration

## Simple Two-Rule System

EquiLens now uses a **dead-simple** configuration that works universally without Docker networks:

### Rule 1: EquiLens in Container → Use `host.docker.internal:11434`
When EquiLens runs in a Docker container, it uses `host.docker.internal:11434` which works for **BOTH**:
- ✅ Ollama running in a separate container (like `ollama-gpu`)
- ✅ Ollama running on the host machine (Docker Desktop or native)

### Rule 2: EquiLens Local → Use `localhost:11434`
When EquiLens runs locally, it uses `localhost:11434` which works for **BOTH**:
- ✅ Ollama running in a container with exposed port 11434
- ✅ Ollama running locally (native or Docker Desktop)

## Why This Works

### Container → host.docker.internal
Docker Desktop (Windows/Mac) provides the special DNS name `host.docker.internal` that always resolves to the host machine's IP address from inside any container. This works because:
- **Ollama in container**: The container exposes port 11434 to the host, so `host.docker.internal:11434` reaches it
- **Ollama on host**: Direct communication via the host network interface

### Local → localhost
When running locally, `localhost:11434` works because:
- **Ollama in container**: Docker exposes the container's port 11434 to host's localhost
- **Ollama on host**: Direct local communication

## No Docker Networks Needed! 🎉

Unlike complex Docker networking setups, this approach:
- ✅ **No custom networks** - uses default Docker bridge + host routing
- ✅ **No service discovery** - simple DNS resolution
- ✅ **No network configuration** - works out of the box
- ✅ **Universal** - same pattern for all scenarios

## Configuration

### docker-compose.yml
```yaml
services:
  equilens:
    container_name: equilens-app
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Local Execution
```bash
# Automatically uses localhost:11434
uv run equilens gui
```

## Verified Scenarios

| EquiLens Location | Ollama Location | URL Used | Status |
|-------------------|-----------------|----------|---------|
| 🐳 Container | 🐳 Container (ollama-gpu) | `host.docker.internal:11434` | ✅ Verified |
| 🐳 Container | 💻 Host (Desktop) | `host.docker.internal:11434` | ✅ Verified |
| 💻 Local | 🐳 Container | `localhost:11434` | ✅ Verified |
| 💻 Local | 💻 Host | `localhost:11434` | ✅ Verified |

## Testing

```bash
# From inside EquiLens container
docker exec -it equilens-app curl http://host.docker.internal:11434
# Should return: "Ollama is running"

# From local machine
curl http://localhost:11434
# Should return: "Ollama is running"
```

## Override (if needed)

```bash
# Force a specific URL
export OLLAMA_BASE_URL=http://custom-server:11434
docker compose up -d
```

## How Smart Config Implements This

The `src/equilens/core/ollama_config.py` module:

```python
if in_container:
    # Use host.docker.internal - works for ANY Ollama setup
    candidate_urls = ["http://host.docker.internal:11434"]
else:
    # Use localhost - works for ANY Ollama setup
    candidate_urls = ["http://localhost:11434"]
```

That's it! No complex logic, no network detection, no service discovery. Just two simple rules that cover all cases.
