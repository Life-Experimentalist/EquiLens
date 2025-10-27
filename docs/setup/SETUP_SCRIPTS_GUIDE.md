# EquiLens Setup Scripts Guide

This directory contains multiple setup scripts for different use cases. Choose the one that fits your needs.

## 📋 Available Scripts

### 1. `setup-docker.ps1` (Recommended for Users)
**Purpose:** Quick setup using pre-built Docker image from Docker Hub

**Best for:**
- End users
- Quick testing
- Production deployments
- No build tools needed

**Features:**
- ✅ Pulls pre-built image from Docker Hub
- ✅ Smart container detection (running/stopped/not exists)
- ✅ Automatic version checking (when image exists)
- ✅ User-friendly prompts with defaults
- ✅ Configurable image URL
- ✅ Ollama health checking
- ⚡ Fast setup (no compilation)

**Usage:**
```powershell
.\setup-docker.ps1
```

**Requirements:**
- Docker Desktop
- Internet connection
- 2GB free disk space

**Time:** 2-5 minutes (depending on download speed)

---

### 2. `setup-docker-dev.ps1` (For Developers)
**Purpose:** Build from source with flexible Ollama configuration

**Best for:**
- Developers
- Custom modifications
- Testing changes
- Flexible Ollama setups

**Features:**
- ✅ Builds image from source
- ✅ Clones repository if needed
- ✅ Smart Ollama detection & management
- ✅ Multiple Ollama options:
  - Use existing container
  - Use existing desktop app
  - Create new container
  - Specify custom container by name/ID
- ✅ Automatic network configuration
- ✅ Persistent volumes
- 🔧 Full control over build process

**Usage:**
```powershell
.\setup-docker-dev.ps1
```

**Requirements:**
- Docker Desktop
- Git (optional, will download if not available)
- 5GB free disk space
- Build tools (handled by Docker)

**Time:** 10-15 minutes (first build)

**Ollama Scenarios:**
1. **Port 11434 accessible** → Auto-detects and uses
2. **Container stopped** → Start existing / Create new / Use custom
3. **No Ollama found** → Create new / Use desktop / Use custom

---

### 3. `deploy-docker.ps1` (For Maintainers)
**Purpose:** Deploy EquiLens to Docker Hub

**Best for:**
- Project maintainers
- Release managers
- CI/CD pipelines

**Features:**
- ✅ Automated build process
- ✅ Multi-tag deployment (version, major.minor, major, latest)
- ✅ Version validation
- ✅ Docker Hub login verification
- ✅ Build timing metrics
- ✅ Auto-updates setup scripts

**Usage:**
```powershell
.\deploy-docker.ps1 -Version "v1.0.0"
.\deploy-docker.ps1 -Version "v1.0.0" -Force  # Skip confirmation
.\deploy-docker.ps1 -Version "v1.2.0" -Username "myuser"
```

**Requirements:**
- Docker Desktop
- Docker Hub account & login
- Push permissions to repository

**Time:** 5-10 minutes (build + push)

---

### 4. `setup.ps1` (For Python Developers)
**Purpose:** Local Python development setup without Docker

**Best for:**
- Python developers
- Direct code editing
- IDE integration
- Debugging

**Features:**
- ✅ UV package manager setup
- ✅ Virtual environment creation
- ✅ Dependency installation
- ✅ Development tools
- 🐍 Native Python environment

**Usage:**
```powershell
.\setup.ps1
```

**Requirements:**
- Python 3.13+
- UV package manager (auto-installed)
- 1GB free disk space

**Time:** 3-5 minutes

---

## 🎯 Quick Decision Guide

**I want to USE EquiLens:**
→ `setup-docker.ps1` (fastest, easiest)

**I want to DEVELOP EquiLens:**
→ `setup-docker-dev.ps1` (source code, flexible)

**I want to DEPLOY EquiLens:**
→ `deploy-docker.ps1` (release to Docker Hub)

**I want PYTHON DEVELOPMENT:**
→ `setup.ps1` (local Python env)

---

## 📊 Comparison Matrix

| Feature | setup-docker.ps1 | setup-docker-dev.ps1 | deploy-docker.ps1 | setup.ps1 |
|---------|------------------|----------------------|-------------------|-----------|
| **Speed** | ⚡⚡⚡ Fast | ⚡ Slow (first time) | ⚡⚡ Medium | ⚡⚡ Medium |
| **Ease** | ✅ Easy | ⚠️ Intermediate | ⚠️ Advanced | ⚠️ Intermediate |
| **Source Code** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Custom Builds** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Ollama Flexibility** | ⚠️ Basic | ✅ Advanced | N/A | ⚠️ Manual |
| **Updates** | ✅ Pull latest | 🔄 Rebuild | 📤 Push | 🔄 Git pull |
| **Use Case** | End users | Developers | Maintainers | Python devs |

---

## 🚀 Step-by-Step Walkthroughs

### First-Time User Setup

1. Install Docker Desktop: https://www.docker.com/products/docker-desktop
2. Start Docker Desktop
3. Open PowerShell in EquiLens directory
4. Run: `.\setup-docker.ps1`
5. Wait for download
6. Access: http://localhost:7860

**Total time:** ~5 minutes

---

### Developer Setup

1. Install Docker Desktop
2. Install Git (optional)
3. Open PowerShell in EquiLens directory
4. Run: `.\setup-docker-dev.ps1`
5. Choose Ollama option:
   - **Have Ollama already?** → Press Enter (uses existing)
   - **New to Ollama?** → Press Enter (creates new container)
   - **Custom setup?** → Choose option 3, enter container name
6. Wait for build
7. Access: http://localhost:7860

**Total time:** ~15 minutes (first time), ~2 minutes (subsequent)

---

### Release Process

1. Ensure clean build: `docker-compose build`
2. Test locally: `docker-compose up`
3. Login to Docker Hub: `docker login`
4. Run: `.\deploy-docker.ps1 -Version "v1.0.0"`
5. Confirm deployment
6. Wait for push
7. Update GitHub: `git tag v1.0.0 && git push --tags`

**Total time:** ~10 minutes

---

## 🔧 Advanced Configuration

### Custom Image URL (setup-docker.ps1)

Edit script, line 8:
```powershell
$EQUILENS_IMAGE = "your-registry/equilens:latest"
```

### Custom Ollama Port (setup-docker-dev.ps1)

Edit script, line 147:
```powershell
$OLLAMA_URL = "http://localhost:12345"
```

### Skip Model Download (setup-docker-dev.ps1)

Comment out lines 250-254:
```powershell
# Write-Host "  📥 Downloading default model..." -ForegroundColor Yellow
# docker exec equilens-ollama ollama pull llama3.2:latest
# Write-Host "  ✅ Model downloaded" -ForegroundColor Green
```

---

## 🐛 Troubleshooting

### Docker Not Running
```powershell
# Error: Docker is not running!
# Solution: Start Docker Desktop and retry
```

### Port Already in Use
```powershell
# Error: Port 7860 or 8000 already in use
# Solution: Stop conflicting service
netstat -ano | findstr :7860
# Kill process or change port in docker-compose.yml
```

### Image Not Found (setup-docker.ps1)
```powershell
# Error: vkrishna04/equilens:latest not found
# Solution 1: Wait for first release
# Solution 2: Use setup-docker-dev.ps1 to build from source
```

### Build Failed (setup-docker-dev.ps1)
```powershell
# Error: Build failed
# Solution: Check Docker Desktop has enough resources
# Settings > Resources > Advanced
# Recommended: 4GB RAM, 2 CPUs
```

### Ollama Connection Issues
```powershell
# Error: Cannot connect to Ollama
# Solution 1: Check Ollama is running
docker ps | findstr ollama
# or check Ollama Desktop app

# Solution 2: Verify port 11434 accessible
curl http://localhost:11434/api/tags

# Solution 3: Check firewall/antivirus
```

---

## 📚 Related Documentation

- **[OLLAMA_FLEXIBLE_SETUP.md](./OLLAMA_FLEXIBLE_SETUP.md)** - Detailed Ollama scenarios
- **[DOCKER_SETUP_COMPARISON.md](./DOCKER_SETUP_COMPARISON.md)** - Setup vs Dev comparison
- **[QUICKSTART.md](./QUICKSTART.md)** - Quick start guide
- **[DOCKER_HUB_DEPLOYMENT.md](./DOCKER_HUB_DEPLOYMENT.md)** - Deployment guide

---

## 🎯 Best Practices

### For Users
1. Use `setup-docker.ps1` for quickest setup
2. Let script handle everything (press Enter for defaults)
3. Keep Docker Desktop running
4. Update by running script again

### For Developers
1. Use `setup-docker-dev.ps1` for source access
2. Choose Ollama option based on your setup
3. Reuse existing Ollama to save disk space
4. Commit changes to your fork

### For Maintainers
1. Test builds locally first
2. Use semantic versioning (v1.2.3)
3. Update CHANGELOG before deployment
4. Tag releases in Git

---

## 📝 Script Locations

```
EquiLens/
├── setup-docker.ps1         → User setup (pull image)
├── setup-docker-dev.ps1     → Dev setup (build source)
├── deploy-docker.ps1        → Deploy to Docker Hub
├── setup.ps1                → Python local setup
├── setup.bat                → Python local setup (Windows)
└── setup-docker.sh          → User setup (Linux/Mac)
```

---

## ❓ FAQ

**Q: Which script should I use?**
A: For trying EquiLens: `setup-docker.ps1`. For development: `setup-docker-dev.ps1`

**Q: Can I use existing Ollama Desktop app?**
A: Yes! `setup-docker-dev.ps1` detects and uses it automatically

**Q: Do I need to install Ollama separately?**
A: No, scripts handle it. But you can use existing installation

**Q: How do I update EquiLens?**
A: Run the setup script again. It checks for updates

**Q: Can I run multiple EquiLens instances?**
A: Yes! Use different directories and share same Ollama

**Q: What if I have multiple Ollama containers?**
A: `setup-docker-dev.ps1` option 3 lets you specify which one

**Q: How much disk space needed?**
A: User setup: ~2GB, Dev setup: ~5GB (includes build cache)

**Q: Can I customize the setup?**
A: Yes! Scripts have configuration sections at the top

---

## 🆘 Getting Help

1. Check relevant documentation in `docs/`
2. Review error messages carefully
3. Verify prerequisites (Docker, ports, disk space)
4. Check GitHub issues: https://github.com/Life-Experimentalists/EquiLens/issues
5. Join community discussions

---

## 🎉 Success Indicators

After running any setup script, you should see:

```
=== Setup Complete! ===

Services are running at:
  • Gradio UI:  http://localhost:7860
  • Web API:    http://localhost:8000
  • Ollama API: http://localhost:11434
```

Visit http://localhost:7860 to start using EquiLens! 🚀
