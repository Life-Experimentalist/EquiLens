# Setup Docker Local - Recent Changes

## ✅ Fixed Issues

### 1. **PowerShell Parse Errors** (FIXED)
**Problem:** Emoji/unicode characters caused parse errors
```
Missing argument in parameter list.
Unexpected token '}' in expression or statement.
```

**Solution:** Replaced all emoji with ASCII labels:
- ❌ → `[ERROR]`
- ✅ → `[OK]`
- ⚠️ → `[WARN]`
- ℹ️ → `[INFO]`
- 📦 → `[BUILD]`
- 🚀 → `[ACTION]`
- ⏳ → `[WAIT]`

**Result:** Script now parses correctly in all PowerShell versions

---

### 2. **Smart Build Detection** (NEW FEATURE)
**Problem:** Script always rebuilt image (~5 minutes) even if it already exists

**Solution:** Added check for existing `equilens:latest` image with user prompt:
```
Do you want to rebuild? (builds take ~5 minutes)
  [1] Skip build - Use existing image (default)
  [2] Rebuild - Get latest code changes
```

**Result:**
- Saves 5 minutes when image already exists
- User control over when to rebuild
- Default is to skip (option 1)

---

## ⚠️ Current Container Issue

### **Problem:** Container Fails to Start
The EquiLens container is crashing with:
```
ModuleNotFoundError: No module named 'equilens'
```

### **Root Cause:**
The package structure has `equilens` under `src/equilens/`, but the Docker container can't import it.

### **Fix Applied:**
Updated `Dockerfile` to add `/workspace/src` to `PYTHONPATH`:
```dockerfile
PYTHONPATH=/workspace/src:/workspace
```

### **Why It's Not Working Yet:**
The existing `equilens:latest` image was built BEFORE this fix. The image needs to be rebuilt with the updated Dockerfile.

---

## 🔧 How to Fix

### Option 1: Rebuild Image (Recommended)
```powershell
# Run the setup script and choose option 2 to rebuild
.\setup-docker-local.ps1
# When prompted: Enter 2 to rebuild

# OR rebuild manually
docker-compose build --no-cache equilens
docker-compose up -d
```

### Option 2: Use Local Development (Faster)
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run EquiLens locally (no Docker)
uv run equilens
```

---

## 📊 Script Features

### Current Checks (In Order)
1. **[1/6]()** Check Docker installation
2. **[2/6]()** Check Docker is running
3. **[3/6]()** Check docker-compose available
4. **[4/6]()** Create persistent volumes
5. **[5/6]()** Check Ollama availability
6. **[6/6]()** Build/Start EquiLens (with smart detection)

### Smart Build Logic
```
IF image exists:
  → Ask user: Skip or Rebuild?
  → Default: Skip (saves time)
ELSE:
  → Always build (first time)
```

---

## 🎯 Next Steps

1. **Rebuild the image** to include the PYTHONPATH fix
2. **Test container startup** to verify it works
3. **Update docker-compose.yml** if needed (add `external: true` for volume)

---

## 📝 Summary of Changes

| File | Change | Impact |
|------|--------|--------|
| `setup-docker-local.ps1` | Removed emoji characters | ✅ No more parse errors |
| `setup-docker-local.ps1` | Added smart build check | ✅ Saves ~5 minutes |
| `Dockerfile` | Added `/workspace/src` to PYTHONPATH | 🔄 Needs rebuild |

**Status:** Script runs correctly, but container needs rebuild with updated Dockerfile to work.
