# Docker Setup - Simplified ✅

## What Changed

**OLD approach:** EquiLens + Ollama both in Docker containers
**NEW approach:** EquiLens in Docker, Ollama on host (user manages it)

## Architecture

```
┌─────────────────────────────────────┐
│  Host Machine                       │
├─────────────────────────────────────┤
│                                     │
│  📦 Ollama (localhost:11434)        │
│     └─ User manages entirely        │
│                                     │
│  🐳 EquiLens Container              │
│     ├─ Gradio UI: 7860              │
│     ├─ Web API: 8000                │
│     └─ Connects to host Ollama      │
│                                     │
│  💾 Single Volume: equilens-data    │
│     ├─ /data/results (Gradio ✅)    │
│     ├─ /data/logs                   │
│     └─ /data/corpus                 │
│                                     │
└─────────────────────────────────────┘
```

## Key Simplifications

### 1. No Ollama in Docker ✅
- **Before:** Ollama container managed by docker-compose
- **After:** User runs Ollama on host at localhost:11434
- **Why:** User manages their own models, simpler setup

### 2. Single Data Volume ✅
- **Before:** 3 separate volumes (data, results, logs)
- **After:** 1 volume with all data in `/workspace/data/`
- **Why:** Gradio can easily access results, cleaner structure

### 3. Network Mode: Host ✅
- **Before:** Custom bridge network
- **After:** Host network (container uses localhost)
- **Why:** Direct access to host's Ollama at localhost:11434

### 4. Simplified Environment ✅
- Ollama URL: `http://localhost:11434` (not `http://ollama:11434`)
- Data paths: All under `/workspace/data/`
- No dependencies on other containers

## Files Modified

### docker-compose.yml
```yaml
services:
  equilens:
    # Removed: ollama service entirely
    network_mode: "host"  # Direct access to host
    volumes:
      - equilens_data:/workspace/data  # Single volume

volumes:
  equilens_data:  # Only one volume now
```

### Dockerfile
```dockerfile
# Data structure simplified
mkdir -p /workspace/data/results /workspace/data/logs /workspace/data/corpus

# Ollama URL points to host
OLLAMA_BASE_URL=http://localhost:11434
```

### New Files Created

1. **scripts/check_ollama.py** - Helper to verify Ollama is available
   ```python
   # Usage: python scripts/check_ollama.py
   # Checks localhost:11434 and lists models
   ```

2. **setup-docker-simple.ps1** - Simplified setup script
   ```powershell
   # 1. Checks Docker
   # 2. Checks Ollama at localhost:11434
   # 3. Creates volume
   # 4. Starts EquiLens
   ```

3. **DOCKER_SIMPLE.md** - Complete simple setup guide
   - Prerequisites (Ollama + Docker)
   - Quick start
   - Data management
   - Troubleshooting
   - Focus on Gradio/API

## Usage

### Setup (One Time)

```powershell
# 1. Install Ollama (on host)
# Download from: https://ollama.ai/download
ollama pull llama3.2

# 2. Run setup script
.\setup-docker-simple.ps1

# 3. Access Gradio
# http://localhost:7860
```

### Daily Usage

```powershell
# Start EquiLens
docker-compose up -d

# Stop EquiLens
docker-compose down

# View logs
docker-compose logs -f

# Check Ollama connection
python scripts/check_ollama.py
```

### Managing Ollama (Separately)

```powershell
# User manages Ollama on host
ollama list
ollama pull mistral
ollama rm old-model
```

## Data Access

### Everything in One Volume

```
equilens-data/
├── results/          # Analysis results (Gradio reads from here)
│   ├── audit_20241018.csv
│   └── report.html
├── logs/             # Application logs
│   └── equilens.log
└── corpus/           # Generated corpus
    └── prompts.csv
```

### Gradio Access

Gradio can easily access results because everything is in one place:

```python
# In Gradio code
RESULTS_DIR = "/workspace/data/results"
results = load_results(f"{RESULTS_DIR}/audit.csv")
```

## Benefits

### For You (Developer)

✅ **Focus on Gradio/API**: One data volume, easy to work with
✅ **Don't touch CLI**: It's mature, runs as-is
✅ **Simple debugging**: One container, one volume
✅ **Fast iteration**: Just rebuild EquiLens container

### For Users

✅ **Simple setup**: Install Ollama, run one script
✅ **Control models**: They manage Ollama themselves
✅ **No complexity**: No networks, no container dependencies
✅ **Easy updates**: Update Ollama independently

## What to Expect from Ollama

EquiLens **assumes Ollama is available at localhost:11434**.

That's it! We don't:
- ❌ Manage Ollama installation
- ❌ Pull models automatically
- ❌ Run Ollama in Docker
- ❌ Configure Ollama settings

We just:
- ✅ Provide helper script to check it's running
- ✅ Connect to it at localhost:11434
- ✅ Use whatever models user has installed

## Helper Script

```powershell
python scripts/check_ollama.py
```

**Output if working:**
```
Checking Ollama availability...

✅ Ollama is running with 3 model(s): llama3.2, mistral, phi3
```

**Output if not working:**
```
Checking Ollama availability...

❌ Cannot connect to Ollama at localhost:11434. Is Ollama running?

📖 To install and run Ollama:
   • Download: https://ollama.ai/download
   • Install and run the application
   • Pull a model: ollama pull llama3.2
```

## Validation

```powershell
# docker-compose.yml is valid ✅
docker-compose config --quiet
# Exit code: 0 (success)

# Configuration shows:
# - network_mode: host ✅
# - Single volume: equilens-data ✅
# - Ollama URL: localhost:11434 ✅
# - Ports: 7860, 8000 ✅
```

## Next Steps

### For Gradio Development

```python
# All data is in one place
RESULTS_DIR = "/workspace/data/results"

# Easy to:
# 1. List all results
results = os.listdir(RESULTS_DIR)

# 2. Load specific result
df = pd.read_csv(f"{RESULTS_DIR}/audit.csv")

# 3. Generate reports
save_report(f"{RESULTS_DIR}/report.html")
```

### For API Development

```python
# FastAPI can serve results directly
@app.get("/results")
def list_results():
    return os.listdir("/workspace/data/results")

@app.get("/results/{filename}")
def get_result(filename: str):
    return FileResponse(f"/workspace/data/results/{filename}")
```

## Summary

✅ **Simplified Docker setup**
✅ **Ollama on host (user manages)**
✅ **Single data volume (Gradio-friendly)**
✅ **Host networking (direct access)**
✅ **Helper script (check Ollama)**
✅ **Focus on Gradio/API work**
✅ **CLI untouched (mature)**

**Ready to work on Gradio!** 🚀

---

## Quick Reference

| What | Where | How |
|------|-------|-----|
| Ollama | Host machine | `ollama list` |
| EquiLens | Docker container | `docker-compose up -d` |
| Data | Volume: equilens-data | `/workspace/data/` |
| Results | In data volume | `/workspace/data/results/` |
| Logs | In data volume | `/workspace/data/logs/` |
| Gradio | http://localhost:7860 | Browser |
| API | http://localhost:8000 | HTTP requests |

**Documentation:**
- Quick guide: `DOCKER_SIMPLE.md`
- Helper script: `scripts/check_ollama.py`
- Setup script: `setup-docker-simple.ps1`
