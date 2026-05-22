# Smart Ollama Configuration - Complete Implementation

## Summary

✅ **All changes have been implemented according to your requirements:**

1. **Environment variable detection** - Uses `OLLAMA_BASE_URL` presence to detect if running locally or in container
2. **Configurable port** - `OLLAMA_PORT` env var (default: 11434)
3. **Smart URL selection** - Container tries host.docker.internal first, local tries localhost first
4. **Proper fallbacks** - If env var URL doesn't work, falls back to auto-detection
5. **No Docker networks** - Uses simple host.docker.internal routing

## Key Changes

### 1. Enhanced `ollama_config.py`

**Container Detection Logic:**
```python
# If OLLAMA_BASE_URL is NOT set → Local install
# If OLLAMA_BASE_URL is set → Likely in container (validated with .dockerenv, cgroup)
```

**URL Selection Logic:**
```python
# In Container:
#   1st: http://host.docker.internal:PORT  # Primary - works for all cases
#   2nd: http://localhost:PORT              # Fallback

# Local:
#   1st: http://localhost:PORT              # Primary - works for all cases
#   2nd: http://127.0.0.1:PORT              # Fallback
```

**Configurable Port:**
```python
ollama_port = os.getenv("OLLAMA_PORT", "11434")  # Default 11434, customizable
```

**Priority Order:**
1. `OLLAMA_BASE_URL` env var (if it works)
2. Auto-detect environment (check for OLLAMA_BASE_URL existence)
3. Try candidate URLs based on environment
4. Cache first working URL

### 2. Updated Integration Files

All 8 files now use smart configuration:
- ✅ `src/Phase2_ModelAuditor/audit_model.py`
- ✅ `src/Phase2_ModelAuditor/enhanced_audit_model.py`
- ✅ `src/equilens/core/docker.py`
- ✅ `src/equilens/core/manager.py`
- ✅ `src/Phase3_Analysis/analytics.py` (from previous changes)
- ✅ `src/equilens/web_ui.py` (from previous changes)
- ✅ `src/equilens/gradio_ui.py` (from previous changes)
- ✅ `src/equilens/cli.py` (from previous changes)

**Import pattern:**
```python
from equilens.core.ollama_config import get_ollama_url

# Usage
ollama_url = get_ollama_url()  # Automatically detects correct URL
```

### 3. Docker Compose Configuration

**Simple environment variable setup:**
```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_API_BASE=http://host.docker.internal:11434/api
  - OLLAMA_PORT=11434  # NEW: Configurable port
```

**No Docker networks needed** - Uses default bridge + Docker Desktop's host.docker.internal

### 4. Documentation

Created comprehensive documentation:
- ✅ `docs/docker/ENVIRONMENT_VARIABLE_LOGIC.md` - Complete env var logic
- ✅ `docs/docker/SIMPLIFIED_OLLAMA_CONFIG.md` - Simple two-rule system

## How It Works

### Scenario 1: Local EquiLens + Any Ollama

```powershell
# Run EquiLens locally
PS> uv run equilens audit --model llama2

# Detection:
# - OLLAMA_BASE_URL not set → Local install
# - Uses: http://localhost:11434
# - Works for: Containerized Ollama OR Native Ollama
```

### Scenario 2: Container EquiLens + Any Ollama

```powershell
# Start container
PS> docker compose up -d

# Detection:
# - OLLAMA_BASE_URL=http://host.docker.internal:11434 (set by docker-compose)
# - Uses: http://host.docker.internal:11434
# - Works for: Your ollama-gpu container OR Native Ollama on host
```

### Scenario 3: Custom Port

```powershell
# Local with custom port
PS> $env:OLLAMA_PORT = "12345"
PS> uv run equilens audit --model llama2

# Or in docker-compose.yml:
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:12345
  - OLLAMA_PORT=12345
```

## Testing

### Test Local Detection

```powershell
# Local Python test
PS> python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"

# Expected output:
{
  "equilens_in_container": false,
  "ollama_in_container": true,  # or false
  "ollama_port": "11434",
  "ollama_url": "http://localhost:11434",
  "scenario": "Local → localhost",
  "description": "EquiLens running locally, using localhost:11434 ..."
}
```

### Test Container Detection

```powershell
# Build and start container (from host PowerShell terminal)
PS> docker compose build
PS> docker compose up -d

# Test from inside container (run this command from HOST terminal, not from inside container)
# This executes Python code INSIDE the container, but you run the command FROM OUTSIDE
PS> docker exec -it equilens-app python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"

# Expected output:
{
  "equilens_in_container": true,
  "ollama_in_container": true,  # or false
  "ollama_port": "11434",
  "ollama_url": "http://host.docker.internal:11434",
  "scenario": "Container → host.docker.internal",
  "description": "EquiLens in Docker container, using host.docker.internal:11434 ..."
}
```

### Test Ollama Connectivity

```powershell
# From host (local)
PS> curl http://localhost:11434/api/version

# From host terminal - test container's connectivity to Ollama
# (This command runs curl INSIDE the container, executed FROM host terminal)
PS> docker exec -it equilens-app curl http://host.docker.internal:11434/api/version

# Both should return Ollama version info
```

## Environment Variables Reference

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OLLAMA_BASE_URL` | No | (auto) | Explicit Ollama URL. If not set → local install. If set → container mode. |
| `OLLAMA_HOST` | No | (auto) | Alternative to OLLAMA_BASE_URL |
| `OLLAMA_PORT` | No | `11434` | Configurable Ollama port |
| `EQUILENS_IN_CONTAINER` | No | (auto) | Force container detection ("true"/"1"/"yes") |

## Advantages of This Approach

1. **Zero Configuration for Standard Setup**
   - Local install: No env vars needed
   - Docker: Uses docker-compose.yml defaults

2. **Flexible Port Configuration**
   - Change port without changing full URL
   - Single env var: `OLLAMA_PORT`

3. **Universal Connectivity**
   - `host.docker.internal` works for any Ollama (container or host)
   - `localhost` works for any Ollama (container with exposed port or host)

4. **Smart Fallbacks**
   - If env var URL fails → Auto-detect and try alternatives
   - If primary URL fails → Try fallback URLs

5. **No Docker Networks**
   - Simpler configuration
   - Works with external containers (like your `ollama-gpu`)
   - No network creation/management

6. **Clear Detection Logic**
   - Absence of `OLLAMA_BASE_URL` → Local
   - Presence of `OLLAMA_BASE_URL` → Container
   - Validated with multiple checks

## Next Steps

1. **Rebuild Docker image:**
   ```powershell
   docker compose build
   ```

2. **Test local mode:**
   ```powershell
   uv run python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"
   ```

3. **Test container mode (from host PowerShell terminal):**
   ```powershell
   docker compose up -d
   docker exec -it equilens-app python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"
   ```

4. **Run audit to verify:**
   ```powershell
   # Local
   uv run equilens audit --model llama2 --corpus data.csv

   # Container (command executed from host terminal)
   docker exec -it equilens-app python src/Phase2_ModelAuditor/audit_model.py --model llama2 --corpus data.csv
   ```

## Files Modified

### Core Implementation
- ✅ `src/equilens/core/ollama_config.py` - Smart configuration module
- ✅ `docker-compose.yml` - Added OLLAMA_PORT env var

### Integration Files (8 total)
- ✅ `src/Phase2_ModelAuditor/audit_model.py`
- ✅ `src/Phase2_ModelAuditor/enhanced_audit_model.py`
- ✅ `src/equilens/core/docker.py`
- ✅ `src/equilens/core/manager.py`
- ✅ `src/Phase3_Analysis/analytics.py`
- ✅ `src/equilens/web_ui.py`
- ✅ `src/equilens/gradio_ui.py`
- ✅ `src/equilens/cli.py`

### Documentation
- ✅ `docs/docker/ENVIRONMENT_VARIABLE_LOGIC.md` - New comprehensive guide
- ✅ `docs/docker/SIMPLIFIED_OLLAMA_CONFIG.md` - Existing, still relevant

## Troubleshooting

### Issue: "Can't connect to Ollama"

**Check 1: Environment detection**
```powershell
python -c "from equilens.core.ollama_config import get_environment_info; import json; print(json.dumps(get_environment_info(), indent=2))"
```

**Check 2: Ollama accessibility**
```powershell
# Local
curl http://localhost:11434/api/version

# Container
docker exec -it equilens-app curl http://host.docker.internal:11434/api/version
```

**Check 3: Environment variables**
```powershell
# On host
Get-ChildItem Env: | Where-Object Name -like "*OLLAMA*"

# In container
docker exec -it equilens-app env | findstr OLLAMA
```

### Issue: "Wrong URL being used"

**Force specific URL:**
```powershell
$env:OLLAMA_BASE_URL = "http://your-url:11434"
```

**Force container detection:**
```powershell
$env:EQUILENS_IN_CONTAINER = "true"
```

**Clear cache and re-detect:**
```python
from equilens.core.ollama_config import _ollama_config
_ollama_config.clear_cache()
url = _ollama_config.get_ollama_url(force_refresh=True)
```

## Summary

All changes are complete and follow your exact requirements:
- ✅ Environment variable detection (OLLAMA_BASE_URL presence)
- ✅ Configurable port (OLLAMA_PORT)
- ✅ Container-first tries host.docker.internal
- ✅ Local-first tries localhost
- ✅ Smart fallbacks when env vars don't work
- ✅ No Docker networks needed
- ✅ Works with external ollama-gpu container
- ✅ Comprehensive documentation

Ready to test! 🚀
