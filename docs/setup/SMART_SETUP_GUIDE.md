# EquiLens Smart Setup - Quick Guide

## ✨ One Command Does Everything!

**Just run:**
```powershell
.\setup-docker.ps1
```

## 🎯 What It Does

The script is **intelligent** and handles all scenarios:

### 📊 Scenario 1: Container Already Running
```
✅ Container 'equilens-app' is already running!

EquiLens is available at:
  • Gradio UI:  http://localhost:7860
  • Web API:    http://localhost:8000

→ Script exits (nothing to do)
```

### 🔄 Scenario 2: Container Exists But Stopped
```
⚠️  Container exists but stopped

Choose action:
  [1] Start existing container (default)
  [2] Recreate container (fresh start)

Enter choice (1-2, default=1): _

→ Press Enter or 1: Starts existing container
→ Press 2: Removes old container and creates fresh one
```

### 🆕 Scenario 3: No Container Exists
```
[1/4] Checking Docker... ✅
[2/4] Checking existing container... ℹ️  No existing container found
[3/4] Checking Ollama... ✅
[4/4] Setting up EquiLens...
  📥 Pulling image (first time: 1-2 minutes)
  🚀 Starting container...
  ✅ Container created and started!

→ Creates and starts new container
```

---

## 🚀 Quick Start

### First Time Setup
```powershell
# 1. Make sure Docker Desktop is running

# 2. Run the script
.\setup-docker.ps1

# 3. Wait 1-2 minutes for image download

# 4. Open browser to http://localhost:7860
```

### Daily Use
```powershell
# Just run the script again!
.\setup-docker.ps1

# If container is stopped: Press Enter to start it
# If container is running: Script shows you it's ready
```

---

## ⚙️ Configuration

Edit the top of `setup-docker.ps1`:

```powershell
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"  # Change Docker Hub image
$CONTAINER_NAME = "equilens-app"                # Change container name
$VOLUME_NAME = "equilens-data"                  # Change volume name
$OLLAMA_URL = "http://localhost:11434"          # Change Ollama URL
```

---

## 📋 Common Tasks

### Start EquiLens
```powershell
.\setup-docker.ps1
# Press Enter if asked (starts existing container)
```

### Stop EquiLens
```powershell
docker stop equilens-app
```

### Restart EquiLens
```powershell
docker restart equilens-app
# Or run: .\setup-docker.ps1 and choose option 1
```

### Recreate EquiLens (Fix Issues)
```powershell
.\setup-docker.ps1
# Choose option 2 when asked
```

### View Logs
```powershell
docker logs -f equilens-app
```

### Update to New Version
```powershell
# 1. Stop and remove container
docker stop equilens-app
docker rm equilens-app

# 2. Pull new image
docker pull vkrishna04/equilens:latest

# 3. Run setup script
.\setup-docker.ps1
```

---

## 🎯 Key Features

✅ **Smart Detection**: Automatically detects existing containers
✅ **Default Action**: Just press Enter to start (no typing needed)
✅ **Error Handling**: Clear error messages with solutions
✅ **Health Check**: Waits for EquiLens to be ready
✅ **Ollama Check**: Warns if Ollama not running (optional)
✅ **Auto-restart**: Container restarts automatically on Docker restart
✅ **Data Persistence**: Your data survives container recreation

---

## 🆘 Troubleshooting

### "Docker not installed"
```powershell
# Install Docker Desktop
# Windows: https://www.docker.com/products/docker-desktop
```

### "Docker is not running"
```powershell
# Start Docker Desktop from Start Menu
# Wait for it to fully start (whale icon in system tray)
```

### "Container exists but won't start"
```powershell
# Run script and choose option 2 (Recreate)
.\setup-docker.ps1
# Choose: 2
```

### "Failed to pull image"
```powershell
# Check internet connection
# Or verify image exists: docker search vkrishna04/equilens
```

### "Port already in use"
```powershell
# Check what's using port 7860
netstat -ano | findstr :7860

# Stop the conflicting process or change EquiLens port
```

---

## 🔄 Workflow Comparison

### Old Way (Multiple Commands)
```powershell
# Check if container exists
docker ps -a | Select-String "equilens"

# If exists and stopped
docker start equilens-app

# If doesn't exist
docker pull vkrishna04/equilens:latest
docker run -d --name equilens-app ...

# Check if running
docker ps | Select-String "equilens"
```

### New Way (One Command)
```powershell
.\setup-docker.ps1
# Done! ✨
```

---

## 📂 File Locations

- **Script**: `setup-docker.ps1` (main setup script)
- **Dev Script**: `setup-docker-dev.ps1` (build from source)
- **Data Volume**: Docker volume `equilens-data`
  - Results: `/workspace/data/results`
  - Logs: `/workspace/data/logs`
- **Container**: `equilens-app`

---

## 🎓 Understanding the Script

### What happens when you run it:

```
Step 1: Check Docker installed and running
  ↓
Step 2: Check if container exists
  ├─ Running? → Show info and exit
  ├─ Stopped? → Ask: [1] Start or [2] Recreate
  └─ Not exists? → Continue to Step 3
  ↓
Step 3: Check Ollama (optional warning)
  ↓
Step 4: Setup EquiLens
  ├─ Pull image (if not local)
  ├─ Create container
  ├─ Start container
  └─ Wait for ready
  ↓
Done! Show access URLs
```

---

## 💡 Pro Tips

1. **Quick Access**: Create a desktop shortcut to the script
2. **Always Running**: Container auto-restarts with Docker Desktop
3. **Data Safety**: Your data persists even if you recreate container
4. **Update Check**: Run script weekly to check for updates
5. **Default Choice**: Just press Enter for most common action

---

## 🎯 When to Choose Each Option

### Choose Option 1 (Start - Default)
- ✅ Container stopped normally
- ✅ Docker Desktop restarted
- ✅ No issues, just need to start it

### Choose Option 2 (Recreate)
- ✅ Container not starting properly
- ✅ Want to pull updated image
- ✅ Something seems broken
- ✅ Want fresh start

---

## 📊 Quick Reference

| Command | What It Does |
|---------|-------------|
| `.\setup-docker.ps1` | Smart setup - handles everything |
| `docker logs -f equilens-app` | View live logs |
| `docker stop equilens-app` | Stop EquiLens |
| `docker start equilens-app` | Start EquiLens |
| `docker restart equilens-app` | Restart EquiLens |
| `docker ps` | Check if running |
| `docker rm -f equilens-app` | Force remove container |

---

## 🚀 You're Ready!

**To get started right now:**

```powershell
# 1. Open PowerShell in EquiLens directory
cd v:\Code\ProjectCode\EquiLens

# 2. Run the script
.\setup-docker.ps1

# 3. Open browser
start http://localhost:7860
```

**That's it! 🎉**

---

## 📚 Related Documentation

- **DOCKER_HUB_DEPLOYMENT.md** - How to deploy your own images
- **DOCKER_SETUP_COMPARISON.md** - Comparing setup vs setup-dev
- **README.md** - Full project documentation

---

**Questions?** The script has helpful error messages and will guide you! 💪
