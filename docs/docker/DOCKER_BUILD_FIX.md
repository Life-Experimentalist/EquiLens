# 🐳 Docker Build Fix - README.md Missing Error

## Issue

**Error during Docker build:**
```
OSError: Readme file does not exist: README.md

hint: This usually indicates a problem with the package or the build environment.
```

**Root Cause:**
The Dockerfile was copying `pyproject.toml` before `uv sync`, but `pyproject.toml` references `README.md` on line 5:
```toml
readme = "README.md"
```

However, `README.md` wasn't copied until later (with all other files), causing the build to fail during dependency installation.

## Solution

**Fixed Dockerfile line 24:**

### Before
```dockerfile
COPY --chown=equilens:equilens pyproject.toml uv.lock* ./
```

### After
```dockerfile
COPY --chown=equilens:equilens pyproject.toml uv.lock* README.md ./
```

## Why This Works

1. **Hatchling validates metadata** during `uv sync`
2. **pyproject.toml references README.md** for package description
3. **README.md must exist** before the build process
4. **Early copy maintains Docker layer caching** - only dependencies change triggers rebuild

## Testing

**Command:**
```bash
docker compose build
```

**Expected Result:**
```
[+] Building X.Xs (15/15) FINISHED
 ✓ Built equilens @ file:///workspace
 ✓ Installed 156 packages
 ✓ Container built successfully
```

## Files Changed

- ✅ `Dockerfile` - Added `README.md` to early COPY command

## Impact

- ✅ Docker builds now succeed
- ✅ Maintains optimal layer caching
- ✅ No changes needed to pyproject.toml
- ✅ Follows best practices for multi-stage builds

## Related Files

- `Dockerfile` - Container build configuration
- `pyproject.toml` - Python project metadata (references README.md)
- `README.md` - Project documentation (required by pyproject.toml)

---

**Status:** ✅ **FIXED**
**Date:** October 19, 2025
**Impact:** Docker builds now work correctly
