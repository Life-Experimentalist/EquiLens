# What Happened & How to Fix It

## 🔍 The Problem You Experienced

### What You Saw:
```
Cannot find path 'V:\Code\ProjectCode\EquiLens\EquiLens\docker-compose.yml'
empty compose file
```

### What Went Wrong:

1. **Wrong Script for Your Situation**
   - You ran: `setup-docker-dev.ps1`
   - This script is designed for **NEW USERS** who don't have EquiLens yet
   - It tries to **clone the repo** into a subdirectory called "EquiLens"
   - But you're **ALREADY** inside the EquiLens directory!

2. **Path Confusion**
   ```
   Your location:  V:\Code\ProjectCode\EquiLens  ← You're here
   Script tried:   V:\Code\ProjectCode\EquiLens\EquiLens  ← Wrong!
   ```

3. **The Script Logic**
   ```powershell
   $targetDir = "EquiLens"  # Creates subdirectory
   Push-Location $targetDir # Goes INTO subdirectory
   # Now looking for docker-compose.yml in wrong place!
   ```

---

## ✅ The Solution

### For Docker Development (What You Want):

Use the **NEW** script I just created:

```powershell
cd V:\Code\ProjectCode\EquiLens
.\setup-docker-local.ps1
```

This script:
- ✅ Works from INSIDE the EquiLens directory
- ✅ Builds from your current code
- ✅ Uses your existing Ollama container
- ✅ Proper developer workflow

---

### For Local Development (Without Docker):

If you don't need Docker, just use UV:

```powershell
# Make sure you're in the EquiLens directory
cd V:\Code\ProjectCode\EquiLens

# Sync dependencies
uv sync

# Run EquiLens
uv run equilens
```

---

## 🎯 Understanding the Different Setups

### Scenario 1: Docker Development (You)
**What you want:** Run EquiLens in Docker, use existing Ollama

```powershell
# Option A: Use the new script
.\setup-docker-local.ps1

# Option B: Manual Docker commands
docker-compose build
docker-compose up -d
docker-compose logs -f
```

**Commands to use EquiLens:**
- Access web UI: http://localhost:7860
- View logs: `docker-compose logs -f`
- Run commands: `docker exec -it equilens-app bash`
- Then inside: `python -m equilens.cli`

---

### Scenario 2: Local Development (No Docker)
**What you want:** Run directly on your machine

```powershell
# Setup (once)
uv sync

# Run EquiLens
uv run equilens

# Or activate venv and run directly
.\.venv\Scripts\Activate.ps1
python -m equilens.cli
```

**Commands to use EquiLens:**
```powershell
uv run equilens --help
uv run equilens audit --model llama3.2
uv run equilens analyze results/latest
```

---

### Scenario 3: New User Installation
**NOT YOUR CASE** - but for reference

```powershell
# From any directory, downloads and sets up everything
irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup.ps1 | iex
```

---

## 🤔 Why "uv run equilens" vs Docker?

### UV Commands (Local):
```powershell
uv run equilens
```
- Runs **directly** on your machine
- Uses your local Python
- Fast iteration
- No container overhead
- Great for development

### Docker Commands:
```powershell
docker-compose up -d
```
- Runs **inside** a container
- Isolated environment
- Consistent across machines
- Need to rebuild after changes
- Great for deployment

---

## 📋 Quick Decision Guide

**Choose Docker if:**
- ✅ You want isolated environment
- ✅ You're deploying to production
- ✅ You want everything containerized
- ✅ You're testing deployment setup

**Choose Local (UV) if:**
- ✅ You're actively developing
- ✅ You want fast iteration
- ✅ You want easy debugging
- ✅ You're testing features

**You can use BOTH!** Develop locally, test in Docker.

---

## 🔧 Fixing Your Current Situation

### Step 1: Clean Up (if needed)
```powershell
# Stop any running containers
docker-compose down

# Remove that problematic subdirectory (if it exists)
if (Test-Path "EquiLens") {
    Remove-Item -Path "EquiLens" -Recurse -Force
}
```

### Step 2: Choose Your Path

#### Path A: Docker Development
```powershell
# Use the new script
.\setup-docker-local.ps1

# Or manually
docker-compose build
docker-compose up -d

# Access at http://localhost:7860
```

#### Path B: Local Development
```powershell
# Install dependencies
uv sync

# Run EquiLens
uv run equilens

# Or with venv
.\.venv\Scripts\Activate.ps1
python -m equilens.cli
```

---

## 🎓 Understanding the Commands

### The `uv run` Prefix
```powershell
uv run equilens           # Runs equilens via UV
uv run python script.py   # Runs Python script with UV-managed env
uv run pytest             # Runs tests
```

UV automatically:
- Creates/uses virtual environment
- Installs dependencies
- Runs the command

### The Docker Approach
```powershell
docker-compose up -d              # Start containers
docker exec equilens-app <cmd>    # Run command in container
docker-compose logs -f            # View logs
```

Docker:
- Runs everything in containers
- Needs explicit exec to run commands
- Changes need rebuild

---

## 🚀 Recommended Workflow for You

Since you're developing, I recommend:

### For Daily Development:
```powershell
# Use local mode for fast iteration
uv run equilens
```

### For Testing Deployment:
```powershell
# Use Docker to test production setup
.\setup-docker-local.ps1
```

### For Code Changes:
```powershell
# Make your changes
# Test locally
uv run equilens

# Then test in Docker
docker-compose build
docker-compose restart
```

---

## 📝 Summary

| What You Did | What Happened | What to Do Instead |
|-------------|---------------|-------------------|
| Ran `setup-docker-dev.ps1` | Tried to clone repo into subdirectory | Run `setup-docker-local.ps1` |
| Expected Docker setup | Got path errors | Use script for developers |
| Wanted `uv run` commands | Script is for Docker | Choose: Docker OR local |

**Bottom line:**
- **Docker development:** Use `setup-docker-local.ps1`
- **Local development:** Use `uv run equilens`
- **Pick one based on your workflow!**
