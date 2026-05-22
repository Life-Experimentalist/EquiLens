# EquiLens Setup Scripts Reorganization Plan

## 🎯 Current Issues
1. Too many setup scripts in root directory (confusing for users)
2. `setup-docker-dev.ps1` is redundant
3. One-command remote installation scripts should be separate
4. `deploy-docker.ps1` needs fixes

---

## 📂 Proposed Structure

### Root Directory (Keep)
```
deploy-docker.ps1         # Production deployment to Docker Hub
setup-docker-local.ps1    # Local Docker testing before shipping
docker-compose.yml        # Docker compose configuration
Dockerfile                # Container build file
```

### scripts/install/ (Move Here)
```
setup.ps1                 # One-command PowerShell install
setup.bat                 # One-command batch install
setup-docker.ps1          # One-command Docker install (for end users)
setup-docker.sh           # One-command Docker install (Linux/Mac)
```

---

## 🔧 Script Purposes (Clarified)

### 1. **deploy-docker.ps1** (Root)
**Purpose:** Deploy EquiLens to Docker Hub for distribution

**Who uses:** Maintainers/developers shipping releases

**What it does:**
- Builds production Docker image
- Tags with version numbers
- Pushes to Docker Hub
- Creates latest, major, major.minor tags

**Usage:**
```powershell
.\deploy-docker.ps1 -Version v2.0.0 -Username vkrishna04
```

**Issues to fix:**
- ✅ Version validation
- ⚠️  Missing post-deployment verification
- ⚠️  No rollback mechanism

---

### 2. **setup-docker-local.ps1** (Root)
**Purpose:** Test Docker setup locally before shipping

**Who uses:** Developers testing containerized version

**What it does:**
- Builds EquiLens image from current code
- Connects to existing Ollama (container or desktop)
- Starts services locally
- Verifies everything works

**Usage:**
```powershell
.\setup-docker-local.ps1
```

**Already created** ✅

---

### 3. **setup.ps1** (Move to scripts/install/)
**Purpose:** One-command installation for new users (local, no Docker)

**Who uses:** End users wanting local installation

**What it does:**
- Clones repository
- Installs UV
- Sets up Python environment
- Downloads models (optional)
- Runs verification

**Usage:**
```powershell
irm https://raw.githubusercontent.com/.../setup.ps1 | iex
```

**Move to:** `scripts/install/setup.ps1`

---

### 4. **setup.bat** (Move to scripts/install/)
**Purpose:** One-command installation launcher for CMD users

**Who uses:** Windows users without PowerShell knowledge

**What it does:**
- Detects PowerShell
- Downloads and runs setup.ps1
- Fallback methods for older systems

**Usage:**
```cmd
curl -fsSL https://raw.githubusercontent.com/.../setup.bat | cmd
```

**Move to:** `scripts/install/setup.bat`

---

### 5. **setup-docker.ps1** (Move to scripts/install/)
**Purpose:** One-command Docker installation for end users

**Who uses:** New users wanting Docker deployment

**What it does:**
- Clones repository into subdirectory
- Sets up Docker containers
- Configures Ollama
- Pulls images

**Usage:**
```powershell
irm https://raw.githubusercontent.com/.../setup-docker.ps1 | iex
```

**Move to:** `scripts/install/setup-docker.ps1`

---

### 6. **setup-docker.sh** (Move to scripts/install/)
**Purpose:** One-command Docker installation for Linux/Mac

**Who uses:** Linux/Mac users wanting Docker deployment

**Move to:** `scripts/install/setup-docker.sh`

---

### 7. **setup-docker-dev.ps1** (REMOVE)
**Purpose:** Originally for first-time development setup

**Why remove:**
- Redundant with `setup.ps1` + `setup-docker-local.ps1`
- Causes confusion (downloads repo into subdirectory when already in repo)
- Not needed - developers use one of two workflows:
  - Local: `uv sync && uv run equilens`
  - Docker: `.\setup-docker-local.ps1`

**Action:** DELETE THIS FILE ❌

---

## 🎬 Implementation Steps

### Step 1: Create scripts/install/ directory
```powershell
New-Item -ItemType Directory -Path "scripts/install" -Force
```

### Step 2: Move one-command scripts
```powershell
Move-Item setup.ps1 scripts/install/
Move-Item setup.bat scripts/install/
Move-Item setup-docker.ps1 scripts/install/
Move-Item setup-docker.sh scripts/install/
```

### Step 3: Delete redundant script
```powershell
Remove-Item setup-docker-dev.ps1
```

### Step 4: Update documentation
- Update README.md with new script locations
- Update all docs referencing old paths
- Add clear explanation of each script's purpose

### Step 5: Fix deploy-docker.ps1
- Add image verification after push
- Add rollback option
- Improve error handling

---

## 📚 Documentation Updates Needed

### README.md
```markdown
## Installation

### Quick Install (Recommended)
```powershell
# Local installation (no Docker)
irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/scripts/install/setup.ps1 | iex

# Docker installation
irm https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/scripts/install/setup-docker.ps1 | iex
```

### For Developers

**Local Development:**
```powershell
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens
uv sync
uv run equilens
```

**Docker Testing:**
```powershell
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens
.\setup-docker-local.ps1
```

**Production Deployment:**
```powershell
.\deploy-docker.ps1 -Version v2.0.0
```
```

---

## 🔍 Before & After Comparison

### Before (Root Directory)
```
setup.ps1                 # 🤔 What's this for?
setup.bat                 # 🤔 Same as above?
setup-docker.ps1          # 🤔 Docker setup?
setup-docker-dev.ps1      # 🤔 Development?
setup-docker-local.ps1    # 🤔 Local Docker?
deploy-docker.ps1         # 🤔 Deployment?
```
**Problem:** 6 setup scripts, unclear purposes

### After (Organized)
```
Root:
  deploy-docker.ps1         # 👔 Deploy to Docker Hub
  setup-docker-local.ps1    # 🧪 Test Docker locally

scripts/install/:
  setup.ps1                 # 🚀 One-command local install
  setup.bat                 # 🚀 One-command CMD install
  setup-docker.ps1          # 🐳 One-command Docker install
  setup-docker.sh           # 🐧 One-command Linux install
```
**Solution:** Clear separation of concerns

---

## 🎯 Final Structure Summary

| Script | Location | Purpose | Target Users |
|--------|----------|---------|--------------|
| `deploy-docker.ps1` | Root | Deploy to Docker Hub | Maintainers |
| `setup-docker-local.ps1` | Root | Test Docker locally | Developers |
| `setup.ps1` | scripts/install/ | Remote local install | End users |
| `setup.bat` | scripts/install/ | Remote CMD install | End users |
| `setup-docker.ps1` | scripts/install/ | Remote Docker install | End users |
| `setup-docker.sh` | scripts/install/ | Remote Linux install | End users |

**Deleted:** `setup-docker-dev.ps1` (redundant)

---

## ✅ Benefits

1. **Clarity:** Clear distinction between developer and end-user scripts
2. **Simplicity:** Root directory only has what developers need
3. **Organization:** Installation scripts grouped together
4. **Discoverability:** Easy to find the right script for your use case
5. **Maintainability:** Less confusion, better documentation
