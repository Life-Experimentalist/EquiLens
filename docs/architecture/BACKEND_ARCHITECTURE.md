# EquiLens Backend & Frontend Architecture

## Overview

EquiLens now features a modern architecture with a FastAPI backend and Gradio frontend, supporting persistent job tracking for long-running tasks.

## Architecture Components

### 1. Backend API (FastAPI)
- **Port**: 8000
- **Purpose**: Handles all operations via REST API
- **Features**:
  - Job submission and management
  - Persistent job tracking with SQLite
  - Background task execution
  - Results export as .zip files
  - Ollama model management

### 2. Frontend (Gradio)
- **Port**: 7860
- **Purpose**: Pure UI interface
- **Features**:
  - Job submission forms
  - Real-time job monitoring
  - Progress tracking and live logs
  - Results viewing and export
  - System status dashboard

### 3. Job Database (SQLite)
- **Location**: `data/jobs/equilens_jobs.db`
- **Purpose**: Persistent storage for jobs and logs
- **Features**:
  - Job state tracking (queued/running/completed/failed/cancelled)
  - Progress tracking
  - Log streaming
  - Job history

## Running the System

### Local Development

#### Option 1: Start All Services Together
```powershell
uv run equilens serve
```
This starts both backend (port 8000) and frontend (port 7860) in one command.

#### Option 2: Start Services Separately
```powershell
# Terminal 1: Start backend
uv run equilens backend

# Terminal 2: Start frontend
uv run equilens web
```

### Access Points

- **Gradio Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/api/health

## API Endpoints

### Health & Status
- `GET /api/health` - Health check
- `GET /api/status` - System status (Docker, Ollama availability)

### Job Management
- `POST /api/jobs` - Create new job
- `GET /api/jobs` - List all jobs (optional `?status=` filter)
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/jobs/{job_id}/cancel` - Cancel running job
- `DELETE /api/jobs/{job_id}` - Delete job
- `GET /api/jobs/{job_id}/logs` - Get job logs

### Results
- `GET /api/results` - List available results
- `GET /api/results/{name}/export` - Download results as .zip
- `GET /api/results/{name}/html` - View HTML report

### Models
- `GET /api/models` - List Ollama models
- `POST /api/models/pull?model_name=X` - Pull Ollama model

## Job Types

### 1. Corpus Generation
```json
{
  "job_type": "corpus_generation",
  "config": {
    "config_file": "path/to/config.json"
  }
}
```

### 2. Audit
```json
{
  "job_type": "audit",
  "config": {
    "model": "llama3.2:latest",
    "corpus": "path/to/corpus.csv",
    "output_dir": "results",
    "silent": false
  }
}
```

### 3. Analysis
```json
{
  "job_type": "analysis",
  "config": {
    "results_file": "results/session.csv",
    "no_ai": false,
    "advanced": true
  }
}
```

## Job States

- **queued**: Job created, waiting to run
- **running**: Job currently executing
- **completed**: Job finished successfully
- **failed**: Job encountered an error
- **cancelled**: Job was cancelled by user

## Features

### AI-Independent Reports

HTML and Markdown reports are always generated, even if AI analysis fails:
- Reports include statistical analysis and visualizations
- AI sections show placeholder text if unavailable
- Reports can be regenerated with AI later

### Chart Descriptions

Each visualization includes a description explaining what it shows:
- Violin Plot: Score distributions
- Heatmap: Color-coded bias patterns
- Effect Sizes: Cohen's d values
- Box Plots: Statistical comparisons
- Scatter: Correlations
- Time Series: Temporal patterns
- Dashboard: Multi-panel overview

### Results Export

Export packages include:
- All PNG visualization files
- HTML report (with embedded base64 images for offline viewing)
- Markdown report (with linked images)
- Raw CSV data
- README file explaining contents

## Environment Detection

The system automatically detects if running in Docker:
- **Docker**: Uses service names for inter-container communication
- **Local**: Uses localhost URLs
- **Ollama URL**: Automatically configured based on environment

## Job Cancellation

Long-running jobs (especially audits) can be cancelled:
- Frontend provides cancel button
- Backend terminates the process
- Job marked as cancelled in database
- Partial results preserved if available

## Database Schema

### Jobs Table
- `job_id`: Unique identifier
- `job_type`: Type of operation
- `status`: Current state
- `progress`/`total`: Progress tracking
- `created_at`, `started_at`, `completed_at`: Timestamps
- `config`: JSON configuration
- `result_path`: Path to results
- `error_message`: Error details if failed
- `pid`: Process ID for cancellation

### Job Logs Table
- `job_id`: Foreign key to jobs
- `timestamp`: Log entry time
- `level`: info/warning/error
- `message`: Log content

## CLI Commands

### New Commands
- `equilens serve` - Start both backend and frontend
- `equilens backend` - Start backend API only
- `equilens web` - Start Gradio frontend only

### Existing Commands (still work)
- `equilens gui` - Legacy Gradio UI (direct CLI wrapper)
- `equilens generate` - Generate corpus
- `equilens audit` - Run audit
- `equilens analyze` - Run analysis

## Migration from Legacy UI

The legacy `equilens gui` command still works but uses direct CLI calls.

The new architecture (`equilens serve` or `web`+`backend`) provides:
- ✅ Persistent job tracking
- ✅ Multi-day job support
- ✅ Progress monitoring
- ✅ Job cancellation
- ✅ Results export
- ✅ Separation of concerns

## Troubleshooting

### Backend won't start
- Check if port 8000 is available
- Ensure all dependencies installed: `uv sync`

### Frontend can't connect to backend
- Verify backend is running: `curl http://localhost:8000/api/health`
- Check `BACKEND_URL` environment variable

### Jobs not appearing
- Check database exists: `data/jobs/equilens_jobs.db`
- Verify backend logs for errors

### Ollama not detected
- Ensure Ollama is running
- Check `OLLAMA_BASE_URL` environment variable
- In Docker: Use service name (e.g., `http://ollama:11434`)
- Local: Use `http://localhost:11434`

## Best Practices

1. **Long-running jobs**: Always use the new backend/frontend architecture
2. **Job monitoring**: Check logs regularly for progress updates
3. **Results backup**: Export results as .zip for archival
4. **Database maintenance**: Periodically clean old jobs
5. **AI failures**: Reports generate regardless of AI availability

## Future Enhancements

- WebSocket support for real-time updates
- Job scheduling and recurring tasks
- Multi-user support with authentication
- Cloud deployment guides
- Performance metrics and analytics
