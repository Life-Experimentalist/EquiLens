# EquiLens Setup Quick Reference

## 🎯 Which Script to Use?

```
┌──────────────────────────────────────────────────────────┐
│          I want to...                    Use...          │
├──────────────────────────────────────────────────────────┤
│  Try EquiLens quickly               setup-docker.ps1     │
│  Develop/modify EquiLens            setup-docker-dev.ps1 │
│  Deploy to Docker Hub               deploy-docker.ps1    │
│  Develop with Python locally        setup.ps1            │
└──────────────────────────────────────────────────────────┘
```

---

## ⚡ One-Command Setups

### Users (Fastest)
```powershell
.\setup-docker.ps1
# Press Enter for all prompts → Done in 2-5 min
```

### Developers (Full Control)
```powershell
.\setup-docker-dev.ps1
# Choose your Ollama option → Done in 10-15 min
```

---

## 🤖 Ollama Options (setup-docker-dev.ps1)

### If Ollama Already Running
```
✅ Auto-detected → Just press Enter → Uses existing
```

### If Ollama Stopped
```
Choose:
[1] Start existing    ← Recommended
[2] Create new
[3] Use custom by name
```

### If No Ollama
```
Choose:
[1] Create container  ← Recommended for Docker setup
[2] Use desktop app   ← Recommended if prefer GUI
[3] Use custom by name
```

---

## 🔍 Quick Diagnostics

### Check Ollama
```powershell
# Test if accessible
curl http://localhost:11434/api/tags

# List running containers
docker ps | findstr ollama

# List all containers
docker ps -a | findstr ollama
```

### Check EquiLens
```powershell
# Test if running
curl http://localhost:7860

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

---

## 🛠️ Common Commands

### Manage EquiLens
```powershell
docker-compose up -d         # Start
docker-compose down          # Stop
docker-compose logs -f       # View logs
docker-compose ps            # Status
docker-compose restart       # Restart
```

### Manage Models
```powershell
# If using Docker Ollama
docker exec <ollama-container> ollama list
docker exec <ollama-container> ollama pull llama3.2
docker exec <ollama-container> ollama rm llama3.2

# If using Desktop Ollama
ollama list
ollama pull llama3.2
ollama rm llama3.2
```

---

## 📊 Decision Tree

```
START
  │
  ├─ Have Docker? ──NO──> Install Docker Desktop
  │       │
  │      YES
  │       │
  ├─ Want to modify code?
  │       │
  │      NO ─────> Use: setup-docker.ps1
  │       │             (Pull pre-built image)
  │       │
  │      YES ────> Use: setup-docker-dev.ps1
  │                     (Build from source)
  │                           │
  │                           │
  │                    Have Ollama?
  │                           │
  │                    ┌──────┴──────┐
  │                   YES            NO
  │                    │              │
  │              Press Enter    Choose option:
  │              (Auto-uses)    [1] Create new
  │                             [2] Use desktop
  │                             [3] Custom
  │
  └─> Access: http://localhost:7860
```

---

## 🚨 Troubleshooting Fast Track

### Error: Docker not running
```powershell
# Solution: Start Docker Desktop, wait 30s, retry
```

### Error: Port already in use
```powershell
# Find process
netstat -ano | findstr :7860

# Kill and retry
```

### Error: Image not found
```powershell
# Solution: Use setup-docker-dev.ps1 instead
# (Image not yet deployed to Docker Hub)
```

### Error: Cannot connect to Ollama
```powershell
# Check if running
curl http://localhost:11434/api/tags

# If not, start Ollama:
# Option 1: Start Docker container
docker start <ollama-container>

# Option 2: Start Desktop app
# Launch Ollama from Start menu
```

---

## 📚 Documentation Index

| Topic | File |
|-------|------|
| **Overview** | SETUP_SCRIPTS_GUIDE.md |
| **Ollama Details** | docs/OLLAMA_FLEXIBLE_SETUP.md |
| **Quick Start** | docs/QUICKSTART.md |
| **Comparison** | docs/DOCKER_SETUP_COMPARISON.md |
| **Deployment** | docs/DOCKER_HUB_DEPLOYMENT.md |

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Install Docker Desktop | 5-10 min |
| setup-docker.ps1 (first time) | 2-5 min |
| setup-docker-dev.ps1 (first time) | 10-15 min |
| setup-docker-dev.ps1 (subsequent) | 2-3 min |
| deploy-docker.ps1 | 5-10 min |

---

## 🎯 Success Checklist

After running setup:
- [ ] Script completed without errors
- [ ] Can access http://localhost:7860
- [ ] Can access http://localhost:8000
- [ ] Ollama accessible at http://localhost:11434
- [ ] Can see "Setup Complete!" message

---

## 💡 Pro Tips

1. **Reuse Ollama** - Multiple EquiLens instances can share one Ollama
2. **Keep Docker Running** - Faster subsequent starts
3. **Use Defaults** - Just press Enter for quickest setup
4. **Check Logs** - `docker-compose logs -f` shows everything
5. **Update Easily** - Rerun setup script to pull latest

---

## 🆘 Quick Help

```powershell
# View available commands
.\setup-docker.ps1 -?

# Check Docker version
docker --version

# Check docker-compose version
docker-compose --version

# List all containers
docker ps -a

# List all volumes
docker volume ls

# Check disk space
docker system df
```

---

## 🎉 Next Steps

After setup completes:

1. **Open browser** → http://localhost:7860
2. **Upload your data** → CSV with bias categories
3. **Select model** → Choose from available Ollama models
4. **Run audit** → Generate bias analysis
5. **View results** → Interactive visualizations + reports

---

## 📞 Support

- **Issues**: https://github.com/Life-Experimentalists/EquiLens/issues
- **Docs**: `docs/` directory
- **Quick Ref**: This file!

---

**Last Updated**: October 19, 2025
**EquiLens Version**: Latest
