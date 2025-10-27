# ⚡ Docker Build Optimization - Quick Reference

## 🎯 Performance Results

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| **First build** | ~13 min | ~13 min | Same (expected) |
| **Source change** | ~13 min | **~1 min** | ⚡ **12.4x faster** |
| **Dependency change** | ~13 min | ~2-3 min | ⚡ **4-6x faster** |
| **No changes** | ~30 sec | ~10 sec | ⚡ **3x faster** |

## 🚀 Quick Start

### Enable BuildKit (One-time Setup)

**PowerShell:**
```powershell
# Set permanently
[Environment]::SetEnvironmentVariable("DOCKER_BUILDKIT", "1", "User")

# Restart PowerShell or run:
$env:DOCKER_BUILDKIT=1
```

**Verify:**
```powershell
$env:DOCKER_BUILDKIT
# Output: 1 ✅
```

### Build Commands

**Standard build (fast with cache):**
```powershell
docker build -t equilens:latest .
```

**Docker Compose (automatic):**
```powershell
docker compose build
```

**See detailed progress:**
```powershell
docker build --progress=plain -t equilens:latest .
```

**Force clean build (slow, but sometimes needed):**
```powershell
docker build --no-cache -t equilens:latest .
```

## 🔍 How It Works

### The Magic Line

```dockerfile
RUN --mount=type=cache,target=/home/equilens/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev
```

**What it does:**
- Creates a persistent cache at `/home/equilens/.cache/uv`
- Cache survives between builds (not in final image)
- UV stores downloaded packages here
- Next build reuses cached packages instead of re-downloading

### Layer Caching Strategy

```dockerfile
# 1. Base image + system deps (rarely changes)
FROM python:3.13.3-slim
RUN apt-get install ...

# 2. UV installation (rarely changes)
RUN pip install uv

# 3. Dependency files ONLY (changes when deps change)
COPY pyproject.toml uv.lock ./

# 4. Install deps with cache mount (uses cache!)
RUN --mount=type=cache,... uv sync

# 5. Source code (changes frequently)
COPY . .
```

**Result**: Source changes only rebuild steps 5+ ⚡

## 📊 Build Analysis

### First Build
```
[8/10] RUN uv sync ... 490.2s  ← Downloads all packages
Total: ~13 minutes
```

### Subsequent Build (Source Change)
```
[8/10] CACHED RUN uv sync ... 0.0s  ← Uses cache! ⚡
Total: ~1 minute
```

### Cache Hit Indicators

Look for these in build output:
```
=> CACHED [8/10] RUN --mount=type=cache ...  ← Good! ✅
=> [internal] load build context ... 1.5s    ← Fast! ✅
=> exporting layers ... 37.8s                 ← Expected
```

## 🧹 Cache Management

### View cache size
```powershell
docker builder du
```

### Clear cache (if needed)
```powershell
# Clear all build cache
docker builder prune

# Clear everything
docker system prune -a
```

### Monitor cache usage
```powershell
docker build --progress=plain -t equilens:latest . 2>&1 | Select-String "cache"
```

## 💡 Pro Tips

### Development Workflow
```powershell
# 1. Build once
docker build -t equilens:dev .

# 2. Make code changes

# 3. Rebuild (fast!)
docker build -t equilens:dev .

# 4. Test
docker run -d -p 7860:7860 equilens:dev
```

### Dependency Updates
```powershell
# Update dependencies in pyproject.toml
uv sync

# Rebuild (downloads only new packages)
docker build -t equilens:latest .
```

### Multi-platform Builds
```powershell
# Setup (one-time)
docker buildx create --use --name multiplatform

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t equilens:latest \
  .
```

## 🐛 Troubleshooting

### Build still slow?

**1. Check BuildKit is enabled:**
```powershell
docker info | Select-String "BuildKit"
# Should show: BuilderVersion: 2
```

**2. Verify cache is used:**
```powershell
docker build --progress=plain -t equilens:latest . 2>&1 | Select-String "CACHED"
# Should see multiple "CACHED" lines
```

**3. Check Docker version:**
```powershell
docker --version
# Need: Docker 20.10+ for full BuildKit support
```

### Cache not working?

**Clear and rebuild:**
```powershell
docker builder prune -af
docker build --no-cache -t equilens:latest .
```

**Ensure BuildKit enabled:**
```powershell
# Set environment variable
$env:DOCKER_BUILDKIT=1

# Or add to Docker Desktop settings:
# Settings → Docker Engine → "features": {"buildkit": true}
```

## 📈 Expected Build Times

| Scenario | Time | Notes |
|----------|------|-------|
| First build | 10-15 min | Full download of all packages |
| Source change | 1-2 min | Cache reused ⚡ |
| Dependency change | 2-4 min | Only new packages downloaded |
| No changes | 10-30 sec | All layers cached |
| Clean rebuild | 10-15 min | `--no-cache` flag used |

## 🎓 Key Takeaways

✅ **BuildKit cache mounts** = Persistent UV cache between builds
✅ **Separate dependency layers** = Source changes don't trigger re-download
✅ **Smart layer ordering** = Maximize cache hits
✅ **12x faster rebuilds** = Happy developers 🎉

---

**Full documentation**: [DOCKER_BUILD_OPTIMIZATION.md](DOCKER_BUILD_OPTIMIZATION.md)

**Result**: From 13-minute rebuilds to 1-minute rebuilds! 🚀
