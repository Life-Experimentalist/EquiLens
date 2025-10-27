# EquiLens Setup Scripts Guide

## 🎯 Which Script Should You Use?

### For New Users / Installation

#### `setup.ps1` - **RECOMMENDED for most users**
```powershell
# One-click installation - clones repo and sets up everything
irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1 | iex
```
- ✅ Clones the repository
- ✅ Installs UV and dependencies
- ✅ Sets up Python environment
- ✅ Verifies installation
- ❌ Does NOT use Docker

**Use this if:** You want to run EquiLens **locally** (not in Docker)

---

#### `setup-docker.ps1` - For Docker deployment
```powershell
# Downloads and runs EquiLens in Docker
irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup-docker.ps1 | iex
```
- ✅ Clones the repository into a subdirectory
- ✅ Sets up Docker containers
- ✅ Configures Ollama
- ✅ Pulls required images

**Use this if:** You're a **NEW USER** who wants Docker deployment

---

### For Developers (Already Have the Repo)

#### `setup-docker-local.ps1` - **NEW! For local development**
```powershell
# Run from INSIDE the EquiLens directory
cd V:\Code\ProjectCode\EquiLens
.\setup-docker-local.ps1
```
- ✅ Builds EquiLens from your current code
- ✅ Uses existing Ollama container
- ✅ Hot-reload on code changes (with rebuild)
- ✅ Proper dev workflow

**Use this if:** You're a **DEVELOPER** with the repo already cloned

---

#### Local Development (No Docker)
```powershell
# Install dependencies
uv sync

# Run EquiLens
uv run equilens

# Run with CLI options
uv run equilens --help
```

**Use this if:** You want to develop **without Docker**

---

## 🔍 What Each Script Does

| Script | Target Users | Clones Repo? | Uses Docker? | Best For |
|--------|-------------|--------------|--------------|----------|
| `setup.ps1` | New users | ✅ Yes | ❌ No | Quick local setup |
| `setup-docker.ps1` | New users | ✅ Yes | ✅ Yes | Production Docker |
| `setup-docker-local.ps1` | Developers | ❌ No | ✅ Yes | Docker dev |
| `uv run equilens` | Developers | ❌ No | ❌ No | Local dev |

---

## 🐳 Docker vs Local Development

### Docker Mode
```powershell
# Build and run in container
.\setup-docker-local.ps1

# Or manually
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Pros:**
- ✅ Isolated environment
- ✅ Consistent across systems
- ✅ Easy Ollama integration
- ✅ Production-ready

**Cons:**
- ❌ Slower iteration (need rebuild)
- ❌ More complex debugging
- ❌ Extra resource usage

---

### Local Mode (UV)
```powershell
# Install and run
uv sync
uv run equilens

# Run specific commands
uv run equilens audit --model llama3.2
uv run python scripts/setup/verify_setup.py
```

**Pros:**
- ✅ Fast iteration
- ✅ Easy debugging
- ✅ Direct file access
- ✅ IDE integration

**Cons:**
- ❌ Need Python setup
- ❌ Platform differences
- ❌ Manual Ollama setup

---

## 🔧 Common Commands

### Docker Workflow
```powershell
# Start services
docker-compose up -d

# View logs
docker-compose logs -f equilens

# Rebuild after code changes
docker-compose build
docker-compose restart

# Stop everything
docker-compose down

# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### UV/Local Workflow
```powershell
# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run EquiLens
uv run equilens

# Run tests
uv run pytest

# Update dependencies
uv sync

# Add new package
uv add <package-name>
```

---

## 🛠️ Troubleshooting

### "Empty compose file" Error
**Problem:** Running `setup-docker-dev.ps1` from inside EquiLens directory

**Solution:** Use `setup-docker-local.ps1` instead:
```powershell
.\setup-docker-local.ps1
```

### "Cannot find docker-compose.yml"
**Problem:** Wrong working directory

**Solution:** Make sure you're in the EquiLens root:
```powershell
cd V:\Code\ProjectCode\EquiLens
ls  # Should see docker-compose.yml, Dockerfile, etc.
```

### Ollama Connection Issues
**Problem:** EquiLens can't connect to Ollama

**Solutions:**
1. Check Ollama is running:
   ```powershell
   curl http://localhost:11434/api/tags
   ```

2. Start Ollama container:
   ```powershell
   docker start ollama-gpu
   ```

3. Or run Ollama Desktop app from https://ollama.ai/download

---

## 📝 Quick Reference

### I want to... → Use this:

- **Install EquiLens for the first time**
  ```powershell
  irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1 | iex
  ```

- **Run in Docker (already have repo)**
  ```powershell
  .\setup-docker-local.ps1
  ```

- **Develop locally without Docker**
  ```powershell
  uv sync
  uv run equilens
  ```

- **Test my changes in Docker**
  ```powershell
  docker-compose build
  docker-compose restart
  ```

- **See EquiLens logs**
  ```powershell
  # Docker
  docker-compose logs -f equilens

  # Local
  cat .\logs\equilens.log
  ```
