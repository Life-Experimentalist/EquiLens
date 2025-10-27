# EquiLens Interface Architecture

## Overview

EquiLens now provides **two complementary interfaces** that work together seamlessly:

1. **Command Line Interface (CLI)** - Full-featured, scriptable, traditional terminal interface
2. **Web Interface (Gradio)** - Modern browser-based GUI with job management

**Both interfaces share the same job database** - you can start work in one and continue in the other!

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interaction Layer                      │
├────────────────────────────────┬────────────────────────────────┤
│         CLI Interface          │      Web Interface (Gradio)    │
│    uv run equilens <cmd>       │   uv run equilens web/serve    │
│                                │                                │
│  • Interactive commands        │  • Browser-based GUI           │
│  • Real-time progress          │  • Job submission & monitoring │
│  • Scriptable automation       │  • Results visualization       │
│  • Resume interrupted audits   │  • Export packages             │
└────────────────┬───────────────┴──────────────┬─────────────────┘
                 │                              │
                 │  ┌───────────────────────┐   │
                 └──┤  Shared Job Database  ├───┘
                    │ data/jobs/            │
                    │ equilens_jobs.db      │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┴───────────────┐
                 │     Backend API Server       │
                 │  FastAPI on port 8000        │
                 │                              │
                 │  • Job queue management      │
                 │  • Background task execution │
                 │  • Progress tracking         │
                 │  • Results packaging         │
                 └──────────────┬───────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    ┌─────▼─────┐         ┌─────▼─────┐       ┌──────▼──────┐
    │  Phase 1  │         │  Phase 2  │       │   Phase 3   │
    │  Corpus   │────────▶│   Audit   │──────▶│  Analysis   │
    │ Generator │         │ (Ollama)  │       │  Analytics  │
    └───────────┘         └───────────┘       └─────────────┘
```

## Interface Comparison

| Feature | CLI | Web UI |
|---------|-----|--------|
| **Accessibility** | Terminal required | Browser-based |
| **Job Submission** | ✅ Interactive | ✅ Form-based |
| **Progress Monitoring** | ✅ Real-time | ✅ Live updates |
| **Job History** | ❌ Session only | ✅ Persistent view |
| **Resume Audits** | ✅ Automatic | ✅ View & continue |
| **Results Export** | ❌ Manual | ✅ One-click .zip |
| **Multi-day Jobs** | ⚠️ Requires tmux | ✅ Native support |
| **Scriptable** | ✅ Full automation | ❌ Browser only |
| **Remote Access** | ✅ SSH | ✅ HTTP |

## Shared Components

### Job Database (`data/jobs/equilens_jobs.db`)

**Schema:**
```sql
-- Jobs table
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    job_type TEXT NOT NULL,  -- 'corpus', 'audit', 'analysis'
    status TEXT NOT NULL,     -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    progress INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    config TEXT,              -- JSON configuration
    result_path TEXT,
    error_message TEXT,
    pid INTEGER              -- Process ID for cancellation
);

-- Job logs table
CREATE TABLE job_logs (
    id INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,     -- 'INFO', 'WARNING', 'ERROR'
    message TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);
```

**Key Points:**
- **Location**: `data/jobs/equilens_jobs.db` (created automatically)
- **Thread-safe**: Uses thread-local connections
- **Persistent**: Survives restarts
- **Shared**: Both CLI and Web UI access the same file

### Environment Detection

**Backend (`src/equilens/backend/api.py`):**
```python
# Auto-detects if running in Docker
docker_detected = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"

# Determines Ollama URL based on environment
if docker_detected:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
else:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

**Frontend (`src/equilens/gradio_app.py`):**
```python
# Auto-detects backend URL
in_docker = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"

if in_docker:
    backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
else:
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
```

## Deployment Scenarios

### Scenario 1: Local Development (Both Local)

```powershell
# Terminal 1: Start backend
uv run equilens backend

# Terminal 2: Start web UI
uv run equilens web

# Or start both at once:
uv run equilens serve
```

**Configuration:**
- Backend: `http://localhost:8000`
- Web UI: `http://localhost:7860`
- Ollama: `http://localhost:11434`
- Database: `data/jobs/equilens_jobs.db`

### Scenario 2: Full Docker Stack

```powershell
docker-compose -f docker-compose.full-stack.yml up
```

**docker-compose.full-stack.yml:**
```yaml
services:
  backend:
    image: ghcr.io/life-experimentalist/equilens:latest
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./data:/workspace/data

  frontend:
    image: ghcr.io/life-experimentalist/equilens:latest
    environment:
      - BACKEND_URL=http://backend:8000
    ports:
      - "7860:7860"

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
```

**Configuration:**
- Backend: `http://backend:8000` (internal)
- Web UI: `http://localhost:7860` (exposed)
- Ollama: `http://ollama:11434` (internal)
- Database: `/workspace/data/jobs/equilens_jobs.db` (mounted)

### Scenario 3: Hybrid (Backend in Docker, UI Local)

```powershell
# Start backend in Docker
docker run -p 8000:8000 -v ./data:/workspace/data equilens:latest equilens backend

# Start web UI locally
uv run equilens web
```

**Configuration:**
- Backend: `http://localhost:8000` (Docker port forward)
- Web UI: `http://localhost:7860` (local)
- Ollama: `http://localhost:11434` or `http://ollama:11434`
- Database: Shared via volume mount

### Scenario 4: Remote Backend

```powershell
# On server
uv run equilens backend

# On local machine
$env:BACKEND_URL = "http://server-ip:8000"
uv run equilens web
```

## CLI-Web Integration

### Example Workflow: Start in CLI, Monitor in Web

```powershell
# 1. Start audit via CLI
uv run equilens audit --model llama3.2

# Output: Job submitted: job_audit_20251020_143022
# Backend URL: http://localhost:8000

# 2. Open web UI in browser
# Navigate to http://localhost:7860

# 3. Go to "Monitor Jobs" tab
# See your audit running with live progress

# 4. Click "View Results" when complete
# Export as .zip package
```

### Example Workflow: Submit via Web, Resume via CLI

```powershell
# 1. Submit job via Web UI
# Go to http://localhost:7860
# Click "Submit Jobs" → Configure audit → Submit
# Note the job_id: job_audit_20251020_150000

# 2. Close browser (job continues in backend)

# 3. Check status via CLI
uv run equilens backend --status job_audit_20251020_150000

# 4. Job gets interrupted (Ctrl+C in backend)

# 5. Resume via CLI
uv run equilens audit --resume job_audit_20251020_150000
```

## File Structure

```
EquiLens/
├── src/equilens/
│   ├── cli.py                    # CLI commands
│   ├── web_ui.py                 # Legacy standalone Gradio
│   ├── gradio_app.py             # New Gradio frontend
│   ├── backend_server.py         # Backend launcher
│   ├── start_all.py              # Multi-process launcher
│   └── backend/
│       ├── api.py                # FastAPI endpoints
│       ├── database.py           # SQLite interface
│       ├── jobs.py               # Job execution
│       └── export.py             # Results packaging
├── data/
│   └── jobs/
│       └── equilens_jobs.db      # Shared job database
├── docker-compose.full-stack.yml # Full stack deployment
└── docs/
    ├── setup/
    │   └── GRADIO_QUICKSTART.md  # Web UI quick start
    └── architecture/
        └── BACKEND_ARCHITECTURE.md # API reference
```

## API Endpoints

The backend provides these key endpoints:

### Job Management
- `POST /api/jobs` - Create new job
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job status
- `POST /api/jobs/{job_id}/cancel` - Cancel job
- `DELETE /api/jobs/{job_id}` - Delete job

### Results
- `GET /api/results` - List available results
- `GET /api/results/{name}/export` - Download .zip package

### System
- `GET /api/status` - System status (Docker/Ollama detection)
- `GET /api/health` - Health check

### Models
- `GET /api/models` - List Ollama models
- `POST /api/models/pull` - Pull new model

See `docs/architecture/BACKEND_ARCHITECTURE.md` for full API reference.

## Environment Variables

### Backend
```powershell
# Ollama API endpoint
$env:OLLAMA_BASE_URL = "http://localhost:11434"  # or http://ollama:11434

# Mark as Docker environment
$env:DOCKER_ENV = "true"

# Database location (optional, default: data/jobs/)
$env:DATABASE_PATH = "./custom/path"
```

### Frontend
```powershell
# Backend API endpoint
$env:BACKEND_URL = "http://localhost:8000"  # or http://backend:8000

# Mark as Docker environment
$env:DOCKER_ENV = "true"
```

## Benefits of Dual Interface

1. **Flexibility**: Choose the right tool for the job
   - CLI for scripting and automation
   - Web for visual monitoring and exploration

2. **Continuity**: Start work in one interface, continue in another
   - Submit via CLI, monitor via Web
   - Submit via Web, resume via CLI

3. **Remote Access**:
   - SSH for CLI access
   - Browser for Web access
   - Both work over network

4. **Multi-day Jobs**:
   - Backend keeps running
   - Web UI reconnects seamlessly
   - CLI can check status anytime

5. **Team Collaboration**:
   - Share backend URL with team
   - Everyone monitors same jobs
   - Coordinate long-running audits

## Troubleshooting

### Web UI can't connect to backend

```powershell
# Check backend is running
curl http://localhost:8000/api/health

# If not running, start it
uv run equilens backend

# Check environment variable
echo $env:BACKEND_URL
```

### CLI and Web show different jobs

```powershell
# Verify they're using the same database
Get-Content data/jobs/equilens_jobs.db | Select-String "job_"

# Check database location
uv run python -c "from equilens.backend.database import get_db_path; print(get_db_path())"
```

### Backend can't find Ollama

```powershell
# Check Ollama is running
curl http://localhost:11434/api/tags

# Set correct URL
$env:OLLAMA_BASE_URL = "http://localhost:11434"

# Restart backend
uv run equilens backend
```

## Future Enhancements

Potential improvements to the dual interface:

1. **WebSocket Support**: Real-time push notifications instead of polling
2. **Multi-user Support**: User authentication and job ownership
3. **Distributed Backend**: Multiple backend workers for parallel jobs
4. **Mobile App**: Native mobile interface using the same backend
5. **VS Code Extension**: Direct integration with editor
6. **Slack/Discord Bot**: Chat-based interface to backend API

---

**Related Documentation:**
- [Gradio Quick Start](../setup/GRADIO_QUICKSTART.md)
- [Backend Architecture](BACKEND_ARCHITECTURE.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Docker Deployment](../docker/DEPLOYMENT.md)
