# CLI Commands Compatibility with Backend Architecture

## ✅ YES - All CLI Commands Work Independently!

The EquiLens CLI commands (`audit`, `generate`, `analyze`) are **fully compatible** and work **completely independently** of the backend/Gradio architecture.

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Two Independent Paths                        │
├──────────────────────────┬──────────────────────────────────┤
│     CLI Path             │      Web Path                    │
│  (Direct Execution)      │   (Backend API)                  │
├──────────────────────────┼──────────────────────────────────┤
│                          │                                  │
│  uv run equilens audit   │   Backend receives job request   │
│         ↓                │           ↓                      │
│  Runs Python scripts     │   Creates job in database        │
│  directly:               │           ↓                      │
│  - Phase1 Generator      │   Executes subprocess:           │
│  - Phase2 Auditor        │   python audit_model.py          │
│  - Phase3 Analytics      │           ↓                      │
│         ↓                │   Updates job progress           │
│  Outputs to results/     │           ↓                      │
│         ↓                │   Stores in database             │
│  Returns to terminal     │           ↓                      │
│                          │   Gradio displays results        │
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

## CLI Commands - How They Work

### 1. `uv run equilens audit`

**What it does:**
```powershell
uv run equilens audit --model llama3.2 --corpus corpus.csv
```

**Execution Flow:**
1. CLI parses arguments
2. Validates model and corpus file
3. Runs **directly** via subprocess:
   ```python
   subprocess.run([
       "python",
       "src/Phase2_ModelAuditor/enhanced_audit_model.py",
       "--model", "llama3.2",
       "--corpus", "corpus.csv",
       "--output-dir", "results"
   ])
   ```
4. Progress shown in terminal with Rich progress bars
5. Results saved to `results/` directory
6. **No backend required!**

**Backend Integration (Optional):**
- If backend is running, it can track the job in the database
- But the audit runs independently regardless

### 2. `uv run equilens generate`

**What it does:**
```powershell
uv run equilens generate
# Or with config:
uv run equilens generate --config config.json
```

**Execution Flow:**
1. Interactive mode (no config):
   ```python
   subprocess.run([
       "python",
       "src/Phase1_CorpusGenerator/generate_corpus.py"
   ])
   ```
2. Config mode (with config):
   - Uses `EquiLensManager.generate_corpus()`
   - Calls Phase1 generator directly

3. Outputs to `src/Phase1_CorpusGenerator/corpus/`
4. **No backend required!**

### 3. `uv run equilens analyze`

**What it does:**
```powershell
uv run equilens analyze --results results/audit.csv
# Or advanced:
uv run equilens analyze --results results/audit.csv --advanced
```

**Execution Flow:**
1. Standard analytics:
   ```python
   subprocess.run([
       "python",
       "src/Phase3_Analysis/analytics.py",
       results_file
   ])
   ```

2. Advanced analytics:
   ```python
   subprocess.run([
       "python",
       "src/Phase3_Analysis/analytics.py",
       results_file,
       "--n-category-support"
   ])
   ```

3. Generates HTML/Markdown reports + PNGs
4. Outputs to `results/` directory
5. **No backend required!**

## Comparison: CLI vs Backend

| Feature | CLI (`uv run equilens`) | Backend (`uv run equilens serve`) |
|---------|-------------------------|-----------------------------------|
| **Execution** | Direct Python subprocess | Background task via FastAPI |
| **Progress** | Terminal output (Rich bars) | Web UI live updates |
| **State** | Session-based (resume via file) | Database-persisted |
| **Results** | `results/` directory | `results/` + database record |
| **Dependencies** | Python, Ollama | Python, FastAPI, SQLite, Gradio, Ollama |
| **Use Case** | Quick audits, scripting | Long-running jobs, team access |

## Compatibility Matrix

| Command | Works Without Backend? | Works With Backend? | Shared State? |
|---------|------------------------|---------------------|---------------|
| `audit` | ✅ Yes | ✅ Yes | ⚠️ File-based resume |
| `generate` | ✅ Yes | ✅ Yes | ❌ No shared state |
| `analyze` | ✅ Yes | ✅ Yes | ❌ No shared state |
| `web` | ❌ Requires backend | ✅ Yes | ✅ Database |
| `backend` | ✅ Standalone | N/A | ✅ Database |
| `serve` | ✅ Starts both | N/A | ✅ Database |

## Integration Points

### CLI → Backend Integration

**How CLI jobs can appear in backend:**

The backend currently doesn't automatically detect CLI jobs, but you can integrate them:

**Option 1: Manual job creation**
```powershell
# Start audit via CLI
uv run equilens audit --model llama3.2

# Separately, track in backend (future enhancement)
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_type": "audit", "config": {...}}'
```

**Option 2: Backend watches results directory (future enhancement)**
- Backend could monitor `results/` for new files
- Auto-import completed CLI audits into database

### Backend → CLI Resume

**You can resume backend jobs via CLI:**

1. Backend creates progress file in `results/`
2. CLI detects and offers to resume:
   ```powershell
   uv run equilens audit --resume results/progress_job123.json
   ```

## Real-World Workflows

### Workflow 1: Pure CLI (No Backend)

```powershell
# 1. Generate corpus
uv run equilens generate

# 2. Run audit
uv run equilens audit --model llama3.2

# 3. Analyze results
uv run equilens analyze --results results/llama3.2_latest.csv --advanced

# All done! No backend needed.
```

**Benefits:**
- ✅ Fastest for quick audits
- ✅ No extra processes
- ✅ Scriptable for automation

### Workflow 2: Pure Web (With Backend)

```powershell
# Start everything
uv run equilens serve

# Then use browser at http://localhost:7860
# Submit jobs, monitor, download results

# All done via Web UI!
```

**Benefits:**
- ✅ Visual interface
- ✅ Multi-day job support
- ✅ Team collaboration

### Workflow 3: Hybrid (Best of Both)

```powershell
# Start backend in background
uv run equilens backend &

# Run quick audit via CLI
uv run equilens audit --model phi3:mini

# Monitor long audit via Web
# (Submit via browser for multi-day job)

# Analyze results via CLI
uv run equilens analyze --results results/latest.csv
```

**Benefits:**
- ✅ CLI speed for quick jobs
- ✅ Web persistence for long jobs
- ✅ Choose right tool for each task

## Technical Details

### Why CLI Works Independently

**CLI Implementation (`src/equilens/cli.py`):**
```python
@app.command()
def audit(...):
    # Direct subprocess execution
    subprocess.run([
        "python",
        "src/Phase2_ModelAuditor/enhanced_audit_model.py",
        "--model", model,
        "--corpus", corpus,
        "--output-dir", output_dir
    ])
```

**No Backend Dependencies:**
- Imports only `typer`, `rich`, `subprocess`
- No FastAPI imports
- No database imports
- No Gradio imports

### Backend Job Execution

**Backend Implementation (`src/equilens/backend/jobs.py`):**
```python
def run_audit_job(job_id, config):
    # Also uses subprocess!
    subprocess.Popen([
        "python",
        "src/Phase2_ModelAuditor/enhanced_audit_model.py",
        "--model", config["model"],
        "--corpus", config["corpus"],
        "--output-dir", config["output_dir"]
    ])
```

**Same Scripts, Different Wrappers:**
- Both CLI and backend call the same Python scripts
- CLI wraps with Rich progress bars
- Backend wraps with database tracking
- Scripts work identically either way

## Resume Functionality

### CLI Resume (File-Based)

```powershell
# Audit gets interrupted
uv run equilens audit --model llama3.2
# Ctrl+C

# Resume from progress file
uv run equilens audit --resume results/progress_llama3.2.json
```

**How it works:**
- Phase2 auditor saves progress to JSON file
- CLI detects progress files on startup
- Offers to resume automatically

### Backend Resume (Database-Based)

```powershell
# Backend job gets interrupted
# (Backend crash, container restart, etc.)

# Restart backend
uv run equilens backend

# Web UI shows job as "running" or "failed"
# Click "Resume" or "Retry" to continue
```

**How it works:**
- Job state persisted in SQLite
- PID tracked for cancellation
- Can resume from last checkpoint

### Cross-Resume (Future Enhancement)

**Currently NOT supported:**
- Starting in CLI, resuming in Web UI
- Starting in Web UI, resuming in CLI

**Why:**
- Different progress tracking mechanisms
- File-based vs database-based

**Could be added:**
- Backend monitors progress files
- CLI reads from database
- Unified progress format

## Summary

### ✅ All CLI Commands Work Independently

**No backend required for:**
- `uv run equilens audit`
- `uv run equilens generate`
- `uv run equilens analyze`
- `uv run equilens status`
- `uv run equilens start/stop`
- `uv run equilens models list/pull`

**Backend required for:**
- `uv run equilens web` (Gradio frontend)
- `uv run equilens backend` (API server)
- `uv run equilens serve` (both together)

**Backend optional for:**
- Everything else! Use backend for persistence and web monitoring, or don't use it at all.

### Key Takeaway

**The backend/Gradio architecture is an ADD-ON, not a replacement:**

```
Old: CLI → Direct execution → Results
New: CLI → Direct execution → Results  (still works!)
New: Web → Backend → Subprocess → Results (new option!)
```

You can use:
- **CLI only** - Just like before, fast and direct
- **Web only** - Submit and monitor via browser
- **Both together** - Mix and match as needed

All your existing scripts and workflows continue to work exactly as before! 🎯

---

**Related Documentation:**
- [CLI Reference](../setup/EXECUTION_GUIDE.md)
- [Gradio Quick Start](../setup/GRADIO_QUICKSTART.md)
- [Interface Architecture](INTERFACE_ARCHITECTURE.md)
- [Backend Architecture](BACKEND_ARCHITECTURE.md)
