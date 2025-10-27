# EquiLens Web Interface - New Features 🎉

## What's New

EquiLens now features a **modern, production-ready architecture** with:

- 🚀 **FastAPI Backend** - Robust API for all operations
- 🌐 **Gradio Frontend** - Beautiful, intuitive web interface
- 💾 **Persistent Jobs** - Track jobs that run for days
- 📊 **Real-Time Monitoring** - Watch progress and logs live
- 📦 **Complete Exports** - Download everything as .zip
- 🤖 **AI-Independent Reports** - Always get results, with or without AI

## Quick Start

### Start Everything
```powershell
uv run equilens serve
```

Then open http://localhost:7860 in your browser!

### What You Can Do

1. **📦 Manage Models** - View and download Ollama models
2. **🚀 Submit Jobs** - Create corpus, run audits, analyze results
3. **📊 Monitor Progress** - Track jobs with real-time updates
4. **📝 View Logs** - See detailed execution logs
5. **📈 Export Results** - Download complete result packages

## Architecture

```
┌─────────────────┐      REST API      ┌──────────────────┐
│ Gradio Frontend │ ◄─────────────────► │ FastAPI Backend  │
│  (Port 7860)    │                     │   (Port 8000)    │
└─────────────────┘                     └──────────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  SQLite DB   │
                                        │ (Job Tracking)│
                                        └──────────────┘
```

### Benefits

✅ **Long-Running Jobs**: Close your browser, jobs keep running
✅ **Progress Tracking**: See exactly where your audit is
✅ **Job Cancellation**: Stop long audits anytime
✅ **Persistent History**: Jobs survive restarts
✅ **Complete Exports**: Get everything in one .zip file
✅ **AI Fallback**: Reports generate even if AI fails

## New CLI Commands

```powershell
# Start both backend and frontend together
uv run equilens serve

# Start just the backend API
uv run equilens backend

# Start just the Gradio frontend
uv run equilens web

# Legacy UI (still works)
uv run equilens gui
```

## API Access

The backend provides a full REST API:

- **API Base**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

Example API call:
```powershell
# Create an audit job
curl -X POST http://localhost:8000/api/jobs `
  -H "Content-Type: application/json" `
  -d '{"job_type":"audit","config":{"model":"llama3.2:latest"}}'
```

## Docker Deployment

```powershell
docker-compose -f docker-compose.full-stack.yml up
```

This starts:
- Backend API (port 8000)
- Gradio Frontend (port 7860)
- Ollama Service (port 11434)

All with persistent data and automatic restarts.

## Features in Detail

### Job Management
- **Create**: Submit jobs via web interface or API
- **Monitor**: Real-time progress and status updates
- **Track**: Complete history of all jobs
- **Cancel**: Stop running jobs safely
- **Logs**: Stream detailed execution logs

### Results Export
Every export includes:
- 📊 All visualization PNG files
- 📄 HTML report (offline-ready with embedded images)
- 📝 Markdown report
- 📁 Raw CSV data
- 📖 README explaining contents

### AI Integration
- Reports **always generate**, even if AI fails
- Placeholder text shown when AI unavailable
- Can regenerate with AI later
- Statistics and charts independent of AI

### Environment Detection
Automatically configures for:
- 💻 Local development (localhost URLs)
- 🐳 Docker containers (service names)
- ☁️ Cloud deployments (environment variables)

## Documentation

- **Quick Start**: [docs/GRADIO_QUICKSTART.md](docs/GRADIO_QUICKSTART.md)
- **Architecture**: [docs/BACKEND_ARCHITECTURE.md](docs/BACKEND_ARCHITECTURE.md)
- **Implementation**: [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)

## Migration from Legacy UI

The old `equilens gui` command still works, but the new architecture provides:

| Feature | Legacy UI | New Architecture |
|---------|-----------|------------------|
| Job Persistence | ❌ | ✅ |
| Multi-Day Jobs | ❌ | ✅ |
| Progress Tracking | ⚠️ Basic | ✅ Detailed |
| Job Cancellation | ❌ | ✅ |
| Result Export | ⚠️ Manual | ✅ Automated |
| API Access | ❌ | ✅ |

## Requirements

All dependencies included in base installation:
- FastAPI & Uvicorn (backend)
- Gradio (frontend)
- Requests (API communication)
- SQLite (built-in with Python)

No additional installation needed!

## Troubleshooting

### Backend won't start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /F /PID <process_id>
```

### Frontend can't connect
```powershell
# Verify backend is running
curl http://localhost:8000/api/health

# Check backend terminal for errors
```

### Jobs not appearing
- Ensure backend is running
- Check database at `data/jobs/equilens_jobs.db`
- Verify backend terminal logs

## Support

- **📖 Documentation**: [docs/](docs/)
- **🐛 Issues**: [GitHub Issues](https://github.com/Life-Experimentalists/EquiLens/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Life-Experimentalists/EquiLens/discussions)

---

**Ready to get started?**

```powershell
uv run equilens serve
```

Then visit http://localhost:7860 and start detecting bias! 🔍
