# EquiLens Gradio Interface - Quick Start Guide

## Overview

The new EquiLens web interface provides a complete dashboard for managing bias detection jobs, monitoring progress, and viewing results.

## Starting the Interface

### Option 1: All-in-One (Recommended)
```powershell
uv run equilens serve
```

This starts both the backend API and Gradio frontend automatically.

### Option 2: Separate Services
```powershell
# Terminal 1: Backend
uv run equilens backend

# Terminal 2: Frontend
uv run equilens web
```

### Option 3: Docker Compose
```powershell
docker-compose -f docker-compose.full-stack.yml up
```

## Accessing the Interface

- **Web Interface**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Interface Tabs

### 1. 🏠 System Status
- View backend status
- Check Docker detection
- Verify Ollama availability
- See configured URLs

**Actions**:
- Click "Refresh Status" to update

### 2. 📦 Models
- View installed Ollama models
- Pull/download new models

**Actions**:
- Click "Refresh Models" to see current models
- Enter model name (e.g., `llama3.2`, `phi3:mini`) and click "Pull Model"

### 3. 🚀 Submit Jobs
Submit three types of jobs:

#### 📝 Generate Corpus
- **Config File**: Optional path to JSON config
- **Action**: Click "Generate Corpus"

#### 🔍 Run Audit
- **Model Name**: Required (e.g., `llama3.2:latest`)
- **Corpus File**: Optional path to CSV
- **Output Directory**: Optional (defaults to `results`)
- **Silent Mode**: Checkbox to suppress verbose output
- **Action**: Click "Run Audit"

#### 📊 Run Analysis
- **Results File**: Required path to audit CSV
- **Skip AI Insights**: Checkbox (reports still generate)
- **Advanced Analytics**: Checkbox for comprehensive analysis
- **Action**: Click "Run Analysis"

**Output**: Each submission returns a Job ID for tracking

### 4. 📊 Monitor Jobs

#### Job Status
- **Job ID**: Enter the job ID from submission
- **Actions**:
  - "Check Status" - View current status and progress
  - "View Logs" - See real-time logs

#### All Jobs
- **Filter by Status**: Select All/queued/running/completed/failed
- **Action**: Click "List Jobs" to view all matching jobs

#### Cancel Job
- **Job ID to Cancel**: Enter job ID
- **Action**: Click "Cancel Job" (works for running jobs only)

### 5. 📈 Results
- View all available result directories
- See creation dates and file info
- Export results

**Actions**:
- Click "List Available Results"
- To download as .zip: Visit `http://localhost:8000/api/results/{result_name}/export`

## Understanding Job Status

### Status Icons
- ⏳ **queued**: Waiting to start
- 🔄 **running**: Currently executing
- ✅ **completed**: Finished successfully
- ❌ **failed**: Encountered an error
- 🚫 **cancelled**: Stopped by user

### Progress Tracking
Jobs show progress as `current/total` (e.g., `45/100`)
- Corpus generation: Based on generation phases
- Audit: Based on number of tests completed
- Analysis: Based on analysis stages

## Workflow Example

### Complete Bias Audit

1. **Check System** (System Status tab)
   - Verify Ollama is available
   - Ensure backend is running

2. **Get a Model** (Models tab)
   - Check if you have a model
   - If not, pull one: `llama3.2`

3. **Generate Corpus** (Submit Jobs tab)
   - Use default config or provide custom
   - Note the Job ID
   - Monitor in Monitor Jobs tab

4. **Run Audit** (Submit Jobs tab)
   - Enter model name: `llama3.2:latest`
   - Optionally specify corpus from step 3
   - Note the Job ID
   - **This can run for hours/days!**

5. **Monitor Progress** (Monitor Jobs tab)
   - Enter Job ID
   - Check status regularly
   - View logs for details
   - Cancel if needed

6. **Analyze Results** (Submit Jobs tab)
   - When audit completes, use results file path
   - Enable "Advanced Analytics"
   - Keep "Skip AI Insights" unchecked for full reports

7. **View & Export** (Results tab)
   - List all results
   - Export as .zip for sharing/archival

## Tips & Best Practices

### For Long-Running Jobs
- ✅ Use the new architecture (serve/web+backend)
- ✅ Jobs persist even if you close the browser
- ✅ Backend keeps running in terminal
- ✅ Can check status anytime by reopening interface

### For Quick Tasks
- Can use legacy CLI commands directly
- `uv run equilens analyze results/file.csv`

### Monitoring
- Check logs frequently for errors
- Progress updates show in status panel
- Backend terminal shows detailed output

### Results Management
- Export important results as .zip
- .zip includes HTML (with embedded images), Markdown, PNGs, CSV
- HTML report works offline (images embedded as base64)

### AI Insights
- Reports generate even if AI fails
- Placeholder text appears if AI unavailable
- All statistics and visualizations still created
- Can regenerate with AI later

## Troubleshooting

### "Cannot connect to backend"
- Verify backend running: `curl http://localhost:8000/api/health`
- Check backend terminal for errors
- Ensure port 8000 not in use

### "Ollama not available"
- Start Ollama service
- Verify URL: http://localhost:11434
- In Docker: Check service name

### Job stuck in "queued"
- Backend may be busy
- Check backend terminal for errors
- Try restarting backend

### Progress not updating
- Click refresh buttons manually
- Check job logs for activity
- Verify job hasn't failed

### Results not appearing
- Ensure job completed successfully
- Check output directory (default: `results/`)
- Verify file permissions

## Advanced Usage

### Custom Backend URL
```powershell
$env:BACKEND_URL="http://custom-host:8000"
uv run equilens web
```

### Custom Ollama URL
```powershell
$env:OLLAMA_BASE_URL="http://custom-ollama:11434"
uv run equilens backend
```

### Docker Deployment
```powershell
docker-compose -f docker-compose.full-stack.yml up -d
```

Access at same URLs (localhost:7860 and localhost:8000)

## Next Steps

- Read [BACKEND_ARCHITECTURE.md](./BACKEND_ARCHITECTURE.md) for technical details
- Check [API documentation](http://localhost:8000/docs) for direct API usage
- See main [README.md](../README.md) for overall project info

## Support

- **Issues**: https://github.com/Life-Experimentalists/EquiLens/issues
- **Documentation**: https://github.com/Life-Experimentalists/EquiLens/tree/main/docs
- **Repository**: https://github.com/Life-Experimentalists/EquiLens
