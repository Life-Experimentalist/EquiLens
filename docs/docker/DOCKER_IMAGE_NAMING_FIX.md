# 🐳 Docker Image Naming Fix

## Issue

Docker Compose was creating an image with a duplicate name:
```
=> => naming to docker.io/library/equilens-equilens:latest
```

This happened because Docker Compose automatically generates image names by combining:
- **Project directory name** (EquiLens → `equilens`)
- **Service name** (`equilens`)
- Result: `equilens-equilens:latest` ❌

## Solution

Added explicit `image` tag in `docker-compose.yml`:

```yaml
services:
  equilens:
    build:
      context: .
      dockerfile: Dockerfile
    image: equilens:latest          # ← Added this line
    container_name: equilens-app
    ports:
      - "7860:7860"
      ...
```

## Result

Now the image is named cleanly:
```
=> => naming to docker.io/library/equilens:latest  ✅
```

## Benefits

1. ✅ **Cleaner naming** - Just `equilens:latest` instead of `equilens-equilens:latest`
2. ✅ **Explicit control** - Image name is now explicitly defined
3. ✅ **Better for CI/CD** - Consistent naming across environments
4. ✅ **Docker Hub ready** - Can easily tag for registry push

## Commands

**Build:**
```bash
docker compose build
```

**Run:**
```bash
docker compose up -d
```

**Check image:**
```bash
docker images | grep equilens
# Output: equilens  latest  ...
```

## Files Changed

- ✅ `docker-compose.yml` - Added `image: equilens:latest`

---

**Status:** ✅ **FIXED**
**Impact:** Cleaner Docker image naming
