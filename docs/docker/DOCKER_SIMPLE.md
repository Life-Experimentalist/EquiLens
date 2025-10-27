# EquiLens Docker Setup - Simple Guide

## Overview

**Simple approach:** EquiLens runs in Docker, Ollama runs on your host machine.

```
┌─────────────────────────────────────┐
│  Your Computer (Host)               │
├─────────────────────────────────────┤
│                                     │
│  📦 Ollama (localhost:11434)        │
│     • User manages models           │
│     • Runs natively on host         │
│                                     │
│  🐳 Docker Container                │
│     ├─ EquiLens App                 │
│     │  • Gradio UI (7860)           │
│     │  • Web API (8000)             │
│     │  • Connects to host Ollama    │
│     │                               │
│     └─ Volume: equilens-data        │
│        ├─ /data/results             │
│        └─ /data/logs                │
│                                     │
└─────────────────────────────────────┘
```

## Prerequisites

### 1. Install Ollama

**Download:** https://ollama.ai/download

**After installation:**
```powershell
# Pull a model
ollama pull llama3.2

# Verify it's running
curl http://localhost:11434/api/tags
```

### 2. Install Docker

**Download:** https://www.docker.com/products/docker-desktop

## Quick Start

### Option 1: Automatic Setup

```powershell
# Windows
.\setup-docker-simple.ps1
```

### Option 2: Manual Setup

```powershell
# 1. Create volume
docker volume create equilens-data

# 2. Start EquiLens
docker-compose up -d

# 3. Access Gradio UI
Start-Process "http://localhost:7860"
```

## Access Points

- **Gradio UI**: http://localhost:7860
- **Web API**: http://localhost:8000
- **Ollama** (on host): http://localhost:11434

## Data Storage

All EquiLens data is in **one volume**: `equilens-data`

```
equilens-data/
├── results/          # Analysis results (Gradio accesses these)
├── logs/             # Application logs
└── corpus/           # Generated corpus data
```

### View Data

```powershell
# Browse the data volume
docker run --rm -it -v equilens-data:/data alpine sh
cd /data
ls -la
```

### Backup Data

```powershell
# Backup all EquiLens data
docker run --rm `
  -v equilens-data:/data `
  -v ${PWD}/backups:/backup `
  alpine tar czf /backup/equilens-data-$(Get-Date -Format "yyyyMMdd").tar.gz /data
```

### Restore Data

```powershell
# Restore from backup
docker run --rm `
  -v equilens-data:/data `
  -v ${PWD}/backups:/backup `
  alpine tar xzf /backup/equilens-data-20241018.tar.gz -C /
```

## Managing EquiLens

```powershell
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Access container shell
docker exec -it equilens-app bash
```

## Managing Ollama (on Host)

```powershell
# List models
ollama list

# Pull new model
ollama pull mistral

# Remove model
ollama rm llama3.2

# Run model directly
ollama run llama3.2 "Hello!"

# Check if Ollama is accessible
python scripts/check_ollama.py
```

## Troubleshooting

### EquiLens can't connect to Ollama

**Symptom:** Error connecting to Ollama

**Check:**
```powershell
# 1. Is Ollama running?
curl http://localhost:11434/api/tags

# 2. Check from inside container
docker exec equilens-app curl http://localhost:11434/api/tags
```

**Solution:** Make sure Ollama is running on the host.

### Gradio can't see results

**Symptom:** No results showing in Gradio UI

**Check:**
```powershell
# View data volume contents
docker exec equilens-app ls -la /workspace/data/results
```

**Solution:** All results should be in `/workspace/data/results` which is mounted from the `equilens-data` volume.

### Port already in use

**Error:** `port is already allocated`

**Solution:**
```powershell
# Check what's using the port
netstat -ano | findstr :7860

# Stop the conflicting service or change port in docker-compose.yml
```

## Configuration

### Change Ollama URL

If Ollama is running on a different host/port:

Edit `docker-compose.yml`:
```yaml
environment:
  - OLLAMA_BASE_URL=http://YOUR_HOST:YOUR_PORT
```

### Change EquiLens Ports

Edit `docker-compose.yml`:
```yaml
ports:
  - "7861:7860"  # Change first number (host port)
  - "8001:8000"
```

## CLI Usage

The CLI is mature and works from anywhere, both inside and outside the container.

### From Host (Recommended)

```powershell
# Run CLI commands directly without entering container
docker exec equilens-app equilens --help
docker exec equilens-app equilens audit --model llama3.2
docker exec equilens-app equilens analyze --advanced

# Works from any directory!
docker exec equilens-app equilens audit --model mistral --output /workspace/data/results
```

### From Inside Container

```powershell
# Access container
docker exec -it equilens-app bash

# Run CLI from anywhere (no need to be in /workspace)
cd /tmp
equilens --help
equilens audit --model llama3.2
equilens analyze --advanced

# CLI automatically finds project files
```

### Key Features

✅ **Works from any directory** - PYTHONPATH and entry point configured
✅ **No need to use `python -m`** - Direct `equilens` command
✅ **Finds results automatically** - Scans `/workspace/data/results`
✅ **Unified analytics** - Advanced and standard modes use same module

## Gradio/API Focus

The Docker setup is optimized for Gradio and Web API usage:

- ✅ Single data volume for easy access
- ✅ Results in `/workspace/data/results` (Gradio can read easily)
- ✅ Logs in `/workspace/data/logs`
- ✅ Ports exposed: 7860 (Gradio), 8000 (API)
- ✅ Ollama assumed available at localhost:11434

## Why This Approach?

1. **Simple**: User manages Ollama separately (their models, their control)
2. **Fast**: No need to containerize Ollama
3. **Flexible**: Use existing Ollama installation
4. **Clean**: One EquiLens volume with all data together
5. **Gradio-friendly**: Results in one place, easy to access

## Helper Script

Check Ollama connectivity:

```powershell
python scripts/check_ollama.py
```

Output:
```
Checking Ollama availability...

✅ Ollama is running with 3 model(s): llama3.2, mistral, phi3
```

## Next Steps

1. ✅ Make sure Ollama is running: `ollama list`
2. ✅ Start EquiLens: `docker-compose up -d`
3. ✅ Open Gradio: http://localhost:7860
4. 🚀 Start working on Gradio improvements!

---

**Focus:** Gradio & API development
**Don't touch:** CLI (it's mature)
**Ollama:** User manages it, we just use it at localhost:11434
