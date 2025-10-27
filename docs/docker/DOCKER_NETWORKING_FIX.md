# Docker Networking Fix - Ollama Container Communication

## Problem
The `equilens-app` container could not access Ollama service (either as a desktop app or in container form) using `localhost:11434` because containers have isolated networking from the host machine.

## Solution
Updated all Ollama connection code to use `host.docker.internal:11434` which is a special DNS name that resolves to the host machine from within Docker containers.

## Changes Made

### 1. Environment Variable Configuration
- **docker-compose.yml** already had the correct configuration:
  ```yaml
  environment:
    - OLLAMA_BASE_URL=http://host.docker.internal:11434
    - OLLAMA_API_BASE=http://host.docker.internal:11434/api
  ```

### 2. Python Source Code Updates
All Python files now respect the `OLLAMA_BASE_URL` environment variable with a sensible fallback:

#### Core Auditing Files
- **src/Phase2_ModelAuditor/audit_model.py**
  - Updated `ollama_hosts` list to prioritize `OLLAMA_BASE_URL` env var
  - Default: `http://host.docker.internal:11434`
  - Fallbacks: `ollama:11434`, `localhost:11434`, `127.0.0.1:11434`

- **src/Phase2_ModelAuditor/enhanced_audit_model.py**
  - Same prioritization as above
  - Added environment variable support to `check_ollama_service()`

#### Analytics Module
- **src/Phase3_Analysis/analytics.py**
  - Constructor now accepts `ollama_url` parameter (optional)
  - Defaults to `OLLAMA_BASE_URL` environment variable
  - Final fallback: `http://host.docker.internal:11434`

#### Core Management
- **src/equilens/core/manager.py**
  - All Ollama API calls now use `os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")`
  - Functions updated: `_display_models_status()`, `list_models()`, `pull_model()`

- **src/equilens/core/docker.py**
  - Added `self.ollama_url` instance variable with env var support
  - Updated all connection tests to use configurable URL
  - Functions updated: `_check_existing_ollama()`, `_wait_for_services()`, `get_service_status()`

#### User Interfaces
- **src/equilens/web_ui.py**
  - Updated `get_system_info()` to use env var for Ollama URL display
  - Updated `list_models()` to use configurable URL

- **src/equilens/gradio_ui.py**
  - Same updates as web_ui.py for consistency

- **src/equilens/cli.py**
  - Updated `get_available_models()` to use env var

### 3. Connection Priority Order
All files now try connections in this order:
1. **Environment Variable**: `OLLAMA_BASE_URL` (set in docker-compose.yml)
2. **Primary Default**: `http://host.docker.internal:11434` (for containers)
3. **Fallback Options**:
   - `http://ollama:11434` (Docker Compose service name)
   - `http://localhost:11434` (for non-containerized runs)
   - `http://127.0.0.1:11434` (loopback)

## Testing Instructions

### Build the Image
```powershell
# Navigate to project directory
cd v:\Code\ProjectCode\EquiLens

# Build the Docker image
docker build -t equilens:latest .
```

### Run with Docker Compose
```powershell
# Start all services (with environment variables from docker-compose.yml)
docker compose up -d

# Check logs
docker compose logs -f equilens

# Test Ollama connectivity from inside container
docker exec -it equilens-app python -c "import os, requests; url = os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434'); print(f'Testing {url}...'); r = requests.get(f'{url}/api/tags', timeout=5); print(f'Status: {r.status_code}'); print(f'Models: {len(r.json().get(\"models\", []))}')"
```

### Verify Environment Variables
```powershell
# Check environment variables inside container
docker exec equilens-app env | Select-String "OLLAMA"

# Should show:
# OLLAMA_BASE_URL=http://host.docker.internal:11434
# OLLAMA_API_BASE=http://host.docker.internal:11434/api
```

## Docker Desktop for Windows Notes
- `host.docker.internal` is automatically provided by Docker Desktop on Windows
- This DNS name resolves to the Windows host machine IP from within containers
- Works for both Docker Desktop with WSL2 and Hyper-V backends
- Ollama running on host at `localhost:11434` becomes accessible at `host.docker.internal:11434` from containers

## Verification Checklist
- [x] Updated all Python source files to use `OLLAMA_BASE_URL` environment variable
- [x] Set sensible default fallback: `http://host.docker.internal:11434`
- [x] Maintained backward compatibility with localhost for non-containerized runs
- [x] docker-compose.yml already configured correctly
- [ ] Build Docker image successfully
- [ ] Test container-to-host Ollama communication
- [ ] Verify model listing works from inside container
- [ ] Run full audit workflow in containerized environment

## Next Steps
1. Build the Docker image: `docker build -t equilens:latest .`
2. Test with Docker Compose: `docker compose up -d`
3. Verify Ollama connectivity from container
4. Run a complete bias detection audit to ensure end-to-end functionality

## Technical References
- Docker networking docs: https://docs.docker.com/network/
- host.docker.internal: https://docs.docker.com/desktop/networking/#i-want-to-connect-from-a-container-to-a-service-on-the-host
- Docker Compose environment variables: https://docs.docker.com/compose/environment-variables/
