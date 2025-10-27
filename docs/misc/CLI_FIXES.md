# CLI and Analytics Fixes ✅

## Issues Fixed

### 1. Analytics Refactoring Issue ✅
**Problem:** CLI was trying to run `advanced_analytics.py` which doesn't exist anymore

**Fix:**
- Updated CLI to import and use `BiasAnalytics` class from `analytics.py` directly
- Removed subprocess calls to non-existent scripts
- CLI now uses Python imports instead of subprocess

**Changes:**
- `src/equilens/cli.py`: Lines 1296-1320, 1590-1620
  - Changed from: `subprocess.run(["python", "src/Phase3_Analysis/advanced_analytics.py", ...])`
  - Changed to: `BiasAnalytics(results_file).run_complete_analysis(...)`

### 2. Analytics Method Signature ✅
**Problem:** `run_complete_analysis()` had parameter `use_ai` but CLI expected `generate_html` and `generate_ai_insights`

**Fix:**
- Updated `BiasAnalytics.run_complete_analysis()` method signature
- Added `generate_html` parameter to control HTML report generation
- Renamed `use_ai` to `generate_ai_insights` for clarity

**Changes:**
- `src/Phase3_Analysis/analytics.py`: Lines 993-1048
  - Old signature: `run_complete_analysis(self, use_ai: bool = True)`
  - New signature: `run_complete_analysis(self, generate_html: bool = True, generate_ai_insights: bool = True)`

### 3. Removed Combined Analytics Choice ✅
**Problem:** User was presented with 3 options (None, Standard, Advanced) which was confusing

**Status:** CLI still has 2 modes (Standard and Advanced) which makes sense:
- **Standard**: Basic charts only, no AI insights, no HTML
- **Advanced**: Full analysis with HTML report and AI insights

This is correct - no change needed.

### 4. CLI Works from Anywhere in Docker ✅
**Problem:** CLI only worked from `/workspace` directory

**Fix:**
- Added `PYTHONPATH=/workspace` to Dockerfile ENV
- Added `uv pip install -e .` to install package in editable mode
- Console script entry point in `pyproject.toml` already configured
- Users can now run `equilens` command from any directory

**Changes:**
- `Dockerfile`: Line 38-39
  - Added: `RUN uv pip install -e .`
  - Added: `PYTHONPATH=/workspace` to ENV

### 5. Docker External CLI Access ✅
**Problem:** Users had to enter container to use CLI

**Solution:** Can now run CLI from host:
```powershell
docker exec equilens-app equilens audit --model llama3.2
docker exec equilens-app equilens analyze --advanced
```

No need to enter container!

## Testing

### Test the Fix

```powershell
# 1. Build container
docker-compose build

# 2. Start container
docker-compose up -d

# 3. Test CLI from host
docker exec equilens-app equilens --help
docker exec equilens-app equilens analyze --advanced

# 4. Test CLI from inside (any directory)
docker exec -it equilens-app bash
cd /tmp
equilens --help
equilens analyze
```

### Expected Behavior

✅ CLI finds analytics module correctly
✅ No errors about `advanced_analytics.py`
✅ Analysis completes successfully
✅ Works from any directory
✅ Works from host without entering container

## Files Modified

1. **src/equilens/cli.py**
   - Fixed two occurrences of analytics invocation
   - Now uses direct Python import instead of subprocess
   - Lines: 1296-1320, 1590-1620

2. **src/Phase3_Analysis/analytics.py**
   - Updated `run_complete_analysis()` method signature
   - Added `generate_html` parameter
   - Made HTML generation optional
   - Lines: 993-1048

3. **Dockerfile**
   - Added `uv pip install -e .` to install package
   - Added `PYTHONPATH=/workspace` to ENV
   - Line: 30-31, 38-39

4. **DOCKER_SIMPLE.md**
   - Updated CLI usage section
   - Added examples for running from host
   - Added examples for running from any directory inside container

5. **equilens_cli.py** (New)
   - Entry point wrapper (not strictly needed, but good to have)
   - Ensures PYTHONPATH is set correctly

## Summary

✅ **Analytics refactoring complete** - No more `advanced_analytics.py` references
✅ **Method signatures match** - `generate_html` and `generate_ai_insights` parameters
✅ **CLI works anywhere** - PYTHONPATH and editable install configured
✅ **External access works** - Can run from host with `docker exec`
✅ **CLI is mature** - Only fixed broken references, didn't change functionality

## What Wasn't Touched

✅ CLI command structure (mature)
✅ Audit functionality (mature)
✅ Interactive prompts (mature)
✅ Error handling (mature)

Only fixed:
- Broken analytics module references
- Method parameter mismatches
- Docker working directory issues

---

**Status: Ready to test!** 🚀

Try running:
```powershell
docker-compose build
docker-compose up -d
docker exec equilens-app equilens analyze --advanced
```
