# Flexible Ollama Setup Guide

## Overview

EquiLens supports multiple Ollama configurations, allowing you to choose the best setup for your needs. The setup script automatically detects your environment and provides options.

## Setup Scenarios

### 1. 🎯 Ollama Already Running (Port 11434 Accessible)

**What happens:**
- Script detects Ollama at `http://localhost:11434`
- Automatically identifies if it's:
  - Docker container (shows container name)
  - Ollama Desktop app
- Uses existing Ollama instance
- No additional setup needed

**Advantages:**
- Fastest setup
- Keeps existing models
- No duplication

```powershell
.\setup-docker-dev.ps1
# ✅ Ollama is accessible at http://localhost:11434
#    Running as Docker container: ollama-container
```

---

### 2. 🔌 Ollama Container Stopped

**What happens:**
- Script finds stopped Ollama container
- Presents 3 options:
  1. **Start existing container (default)** - Resumes stopped container
  2. **Create new container** - Fresh Ollama instance
  3. **Use existing by name/ID** - Specify custom container

**When to use each:**
- **Option 1:** Normal case, container just stopped
- **Option 2:** Want fresh Ollama, separate from existing
- **Option 3:** Have multiple Ollama containers, want specific one

```powershell
.\setup-docker-dev.ps1
# ⚠️  Found stopped Ollama container: my-ollama
# Choose action:
#   [1] Start existing Ollama container (default)
#   [2] Create new Ollama container
#   [3] Use existing container by name/ID
```

---

### 3. 🆕 No Ollama Detected

**What happens:**
- Script detects port 11434 not accessible
- Presents 3 options:
  1. **Create new Ollama container (default)** - Full Docker setup
  2. **Use Ollama Desktop app** - Manual desktop install
  3. **Use existing by name/ID** - Connect to specific container

**When to use each:**
- **Option 1:** First-time setup, want everything in Docker
- **Option 2:** Prefer desktop app, want GUI management
- **Option 3:** Have existing Ollama not auto-detected

```powershell
.\setup-docker-dev.ps1
# ℹ️  No Ollama container found
# Options:
#   [1] Create new Ollama container (default)
#   [2] Use existing Ollama Desktop app
#   [3] Use existing container by name/ID
```

---

## Connection Methods

### Docker Network Connection

**When:** EquiLens connects to Ollama container

```yaml
# Configured automatically in docker-compose.yml
OLLAMA_BASE_URL=http://ollama-container:11434
```

**Advantages:**
- Internal Docker network
- Faster communication
- Isolated from host

**Example:**
```powershell
# Enter choice: 1 (or 3 with container name)
# Result: Both containers on same network
docker network inspect equilens_default
```

---

### Host Network Connection

**When:** EquiLens connects to Ollama Desktop app

```yaml
# Configured automatically in docker-compose.yml
OLLAMA_BASE_URL=http://localhost:11434
```

**Advantages:**
- Use Ollama GUI
- Manage models easily
- Share with other apps

**Example:**
```powershell
# Enter choice: 2 (Use Desktop app)
# Result: EquiLens connects via localhost
```

---

## Use Cases

### Scenario A: Developer with Existing Ollama

**Goal:** Keep current Ollama setup, add EquiLens

**Setup:**
```powershell
# 1. Ensure Ollama running (container or desktop)
docker ps  # or check Ollama app

# 2. Run setup
.\setup-docker-dev.ps1

# 3. Script auto-detects, no choices needed
# ✅ Ollama is accessible at http://localhost:11434
#    Using existing Ollama Desktop app
```

**Result:** EquiLens uses your existing Ollama, all models available

---

### Scenario B: Fresh Installation

**Goal:** Install both from scratch

**Setup:**
```powershell
# 1. Run setup
.\setup-docker-dev.ps1

# 2. Choose option 1 (default)
# Options:
#   [1] Create new Ollama container (default)
# Enter choice: [Press Enter]

# 3. Wait for model download
# 📥 Downloading default model (llama3.2:latest)...
```

**Result:** Complete isolated setup in Docker

---

### Scenario C: Multiple Ollama Instances

**Goal:** Use specific Ollama from several containers

**Setup:**
```powershell
# 1. List existing containers
docker ps -a --filter "name=ollama"
# my-ollama-prod
# my-ollama-dev
# my-ollama-test

# 2. Run setup
.\setup-docker-dev.ps1

# 3. Choose option 3
# Options:
#   [3] Use existing container by name/ID
# Enter choice: 3
# Enter container name or ID: my-ollama-dev

# ✅ Using container: my-ollama-dev
```

**Result:** EquiLens connects to specific Ollama instance

---

### Scenario D: Prefer Desktop App

**Goal:** Use Ollama GUI for model management

**Setup:**
```powershell
# 1. Install Ollama Desktop (if not installed)
# Download from: https://ollama.ai/download

# 2. Start Ollama app

# 3. Run setup
.\setup-docker-dev.ps1

# 4. If not auto-detected, choose option 2
# Options:
#   [2] Use existing Ollama Desktop app
# Enter choice: 2
```

**Result:** EquiLens uses desktop Ollama, manage via GUI

---

## Verification

### Check Ollama Connection

```powershell
# Test Ollama API
Invoke-WebRequest -Uri "http://localhost:11434/api/tags" | ConvertFrom-Json

# List available models
docker exec <container-name> ollama list
# or
ollama list  # if using desktop app
```

### Check EquiLens Connection

```powershell
# View EquiLens logs
docker-compose logs -f equilens

# Should see:
# Connected to Ollama at http://localhost:11434
# Available models: llama3.2, ...
```

---

## Model Management

### Using Docker Container

```powershell
# List models
docker exec <ollama-container> ollama list

# Download model
docker exec <ollama-container> ollama pull llama3.2

# Remove model
docker exec <ollama-container> ollama rm llama3.2

# Run model test
docker exec <ollama-container> ollama run llama3.2 "Hello"
```

### Using Desktop App

```powershell
# List models
ollama list

# Download model
ollama pull llama3.2

# Remove model
ollama rm llama3.2

# Run model test
ollama run llama3.2 "Hello"
```

---

## Troubleshooting

### Port 11434 Already in Use

**Problem:** Another service using port 11434

**Solution:**
```powershell
# Find what's using the port
netstat -ano | findstr :11434

# Stop the service or change Ollama port
# For desktop app: Settings > Advanced > Port
# For container: docker run -p 11435:11434 ...
```

### Container Not Accessible

**Problem:** EquiLens can't reach Ollama container

**Solution:**
```powershell
# Check both on same network
docker network inspect equilens_default

# Verify Ollama container name
docker ps --filter "name=ollama"

# Test connection from EquiLens container
docker exec equilens-app curl http://ollama-container:11434/api/tags
```

### Desktop App Not Detected

**Problem:** Ollama desktop running but script says not found

**Solution:**
```powershell
# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Check if port exposed correctly
# Desktop app: Settings > General > Allow connections from network

# Re-run setup, choose option 2 manually
.\setup-docker-dev.ps1
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Setup Detection                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Check port 11434 accessible?                   │
│     ├─ Yes → Detect if container or desktop        │
│     │         Use existing (auto-configure)        │
│     │                                               │
│     └─ No  → Check stopped containers              │
│              ├─ Found → Start/Create/Custom        │
│              └─ None  → Create/Desktop/Custom      │
│                                                     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Connection Configuration               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Container → Container:                             │
│  ┌──────────┐           ┌──────────┐              │
│  │ EquiLens │ ─network─▶│  Ollama  │              │
│  └──────────┘           └──────────┘              │
│       │                       │                     │
│  http://ollama-container:11434                     │
│                                                     │
│  Container → Desktop:                               │
│  ┌──────────┐                                      │
│  │ EquiLens │ ─localhost─▶ Ollama Desktop         │
│  └──────────┘                                      │
│       │                                             │
│  http://localhost:11434                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Advanced Configuration

### Custom Ollama URL

Edit `docker-compose.yml`:

```yaml
environment:
  - OLLAMA_BASE_URL=http://custom-host:11434
```

### Multiple EquiLens, One Ollama

```powershell
# Start first EquiLens
cd equilens-project-1
.\setup-docker-dev.ps1
# Choose existing Ollama

# Start second EquiLens (different directory)
cd equilens-project-2
.\setup-docker-dev.ps1
# Choose same Ollama
```

**Result:** Both EquiLens instances share models, save disk space

### Persistent Models Across Recreations

```yaml
# In docker-compose.yml
volumes:
  ollama-models:
    external: true  # Use existing volume
    name: my-persistent-ollama-models
```

---

## Summary

| Scenario | Detection | User Action | Result |
|----------|-----------|-------------|--------|
| Ollama running (container) | Auto | None | Use existing container |
| Ollama running (desktop) | Auto | None | Use desktop app |
| Ollama stopped | Prompt | Choose 1/2/3 | Start/Create/Custom |
| No Ollama | Prompt | Choose 1/2/3 | Create/Desktop/Custom |
| Custom container | Manual | Enter name | Use specified |

**Key Benefits:**
- ✅ Automatic detection
- ✅ Flexible configuration
- ✅ Separation of concerns
- ✅ Reuse existing setups
- ✅ Easy management
- ✅ Multiple configurations

**Next Steps:**
1. Run `.\setup-docker-dev.ps1`
2. Follow prompts for your scenario
3. Access EquiLens at `http://localhost:7860`
4. Manage models as needed

For more help, see:
- [QUICKSTART.md](./QUICKSTART.md) - Basic setup
- [DOCKER_SETUP_COMPARISON.md](./DOCKER_SETUP_COMPARISON.md) - Setup vs Dev comparison
- [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) - Ollama installation guide
