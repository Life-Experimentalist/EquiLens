# Docker Hub Deployment Guide

Complete guide for building and deploying EquiLens to Docker Hub or other container registries.

---

## Quick Reference

```powershell
# Build locally
docker compose build

# Tag for Docker Hub
docker tag equilens:latest vkrishna04/equilens:latest
docker tag equilens:latest vkrishna04/equilens:v1.0.0

# Push to Docker Hub
docker push vkrishna04/equilens:latest
docker push vkrishna04/equilens:v1.0.0
```

---

## Table of Contents

1. [Docker Hub Setup](#1-docker-hub-setup)
2. [Build & Tag Image](#2-build--tag-image)
3. [Push to Docker Hub](#3-push-to-docker-hub)
4. [GitHub Container Registry](#4-github-container-registry)
5. [Version Management](#5-version-management)
6. [Multi-Architecture Builds](#6-multi-architecture-builds)
7. [Automated CI/CD](#7-automated-cicd)

---

## 1. Docker Hub Setup

### Create Docker Hub Account

1. Visit [Docker Hub](https://hub.docker.com/)
2. Click **Sign Up** and create an account
3. Verify your email address

### Login from Command Line

```powershell
# Login to Docker Hub
docker login

# You'll be prompted for:
# Username: vkrishna04
# Password: ********
```

**Tip:** Use an access token instead of your password for better security:

1. Go to **Account Settings** → **Security** → **Access Tokens**
2. Click **New Access Token**
3. Name: `EquiLens Deployment`
4. Permissions: **Read, Write, Delete**
5. Copy the token and use it as the password

---

## 2. Build & Tag Image

### Build the Image Locally

```powershell
# From the EquiLens project root
cd v:\Code\ProjectCode\EquiLens

# Build using docker-compose (recommended)
docker compose build

# Or build directly with Dockerfile
docker build -t equilens:latest .
```

**Build output:**
```
[+] Building 550.0s
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 1.50kB
 => naming to docker.io/library/equilens:latest
```

### Tag for Docker Hub

```powershell
# Format: docker tag <local-image> <username>/<repository>:<tag>

# Tag as latest
docker tag equilens:latest vkrishna04/equilens:latest

# Tag specific version
docker tag equilens:latest vkrishna04/equilens:v1.0.0

# Tag with date
docker tag equilens:latest vkrishna04/equilens:2025-01-15
```

### Verify Tags

```powershell
# List all images with 'equilens' in the name
docker images | Select-String "equilens"
```

**Expected output:**
```
vkrishna04/equilens   latest    abc123def456   5 minutes ago   1.2GB
vkrishna04/equilens   v1.0.0    abc123def456   5 minutes ago   1.2GB
equilens              latest    abc123def456   5 minutes ago   1.2GB
```

---

## 3. Push to Docker Hub

### Push Tagged Images

```powershell
# Push latest tag
docker push vkrishna04/equilens:latest

# Push version tag
docker push vkrishna04/equilens:v1.0.0

# Push all tags at once
docker push vkrishna04/equilens --all-tags
```

**Push output:**
```
The push refers to repository [docker.io/vkrishna04/equilens]
layer1: Pushed
layer2: Pushed
layer3: Pushed
latest: digest: sha256:abc123... size: 4321
```

### Verify on Docker Hub

1. Go to [Docker Hub](https://hub.docker.com/)
2. Navigate to **Repositories** → **vkrishna04/equilens**
3. Check **Tags** tab for your pushed tags
4. Verify **Last Pushed** timestamp

### Update Repository Description

1. Click on your repository: `vkrishna04/equilens`
2. Edit the **Overview** section
3. Add description from `README.md`
4. Add usage instructions for `setup-docker-simple.ps1`

---

## 4. GitHub Container Registry

Alternative to Docker Hub - GitHub Container Registry (GHCR).

### Login to GHCR

```powershell
# Create Personal Access Token (PAT)
# Go to GitHub → Settings → Developer settings → Personal access tokens
# Scopes: write:packages, read:packages, delete:packages

# Login using PAT
$env:CR_PAT = "ghp_YourPersonalAccessToken"
echo $env:CR_PAT | docker login ghcr.io -u vkrishna04 --password-stdin
```

### Tag & Push to GHCR

```powershell
# Tag for GHCR
docker tag equilens:latest ghcr.io/vkrishna04/equilens:latest
docker tag equilens:latest ghcr.io/vkrishna04/equilens:v1.0.0

# Push to GHCR
docker push ghcr.io/vkrishna04/equilens:latest
docker push ghcr.io/vkrishna04/equilens:v1.0.0
```

### Make Package Public

1. Go to GitHub → Your Profile → Packages
2. Click on `equilens` package
3. **Package settings** → **Change visibility** → **Public**

---

## 5. Version Management

### Semantic Versioning

Follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

```powershell
# Major release (breaking changes)
docker tag equilens:latest vkrishna04/equilens:v2.0.0
docker tag equilens:latest vkrishna04/equilens:v2

# Minor release (new features)
docker tag equilens:latest vkrishna04/equilens:v1.1.0
docker tag equilens:latest vkrishna04/equilens:v1.1

# Patch release (bug fixes)
docker tag equilens:latest vkrishna04/equilens:v1.0.1
```

### Recommended Tagging Strategy

Always push **THREE** tags for each release:

```powershell
# Example: Releasing v1.2.3

# 1. Exact version
docker tag equilens:latest vkrishna04/equilens:v1.2.3

# 2. Minor version (auto-update patches)
docker tag equilens:latest vkrishna04/equilens:v1.2

# 3. Major version (auto-update minor/patches)
docker tag equilens:latest vkrishna04/equilens:v1

# 4. Latest (always update)
docker tag equilens:latest vkrishna04/equilens:latest

# Push all
docker push vkrishna04/equilens:v1.2.3
docker push vkrishna04/equilens:v1.2
docker push vkrishna04/equilens:v1
docker push vkrishna04/equilens:latest
```

**Benefits:**
- Users can pin exact versions: `vkrishna04/equilens:v1.2.3`
- Users can auto-update patches: `vkrishna04/equilens:v1.2`
- Users can auto-update minor versions: `vkrishna04/equilens:v1`
- Users can always get latest: `vkrishna04/equilens:latest`

---

## 6. Multi-Architecture Builds

Build images for multiple CPU architectures (AMD64, ARM64).

### Setup Buildx

```powershell
# Create a new builder
docker buildx create --name equilens-builder --use

# Verify builder
docker buildx inspect --bootstrap
```

### Build Multi-Arch Image

```powershell
# Build for AMD64 and ARM64
docker buildx build --platform linux/amd64,linux/arm64 `
    -t vkrishna04/equilens:latest `
    -t vkrishna04/equilens:v1.0.0 `
    --push `
    .
```

**Supported platforms:**
- `linux/amd64` - Intel/AMD x86_64 (most common)
- `linux/arm64` - Apple M1/M2, Raspberry Pi 4
- `linux/arm/v7` - Raspberry Pi 3

### Why Multi-Arch?

- **Intel/AMD laptops/servers**: Need `linux/amd64`
- **Apple Silicon (M1/M2)**: Need `linux/arm64`
- **Cloud ARM instances**: AWS Graviton uses `linux/arm64`

---

## 7. Automated CI/CD

### GitHub Actions Workflow

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    tags: [ 'v*.*.*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: docker.io
  IMAGE_NAME: vkrishna04/equilens

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          platforms: linux/amd64,linux/arm64
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Add Secrets to GitHub

1. Go to GitHub → Repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add:
   - `DOCKERHUB_USERNAME` = `vkrishna04`
   - `DOCKERHUB_TOKEN` = Your Docker Hub access token

### Trigger Workflow

```powershell
# Push to main branch
git add .
git commit -m "Update EquiLens"
git push origin main

# Or create a release tag
git tag v1.0.0
git push origin v1.0.0
```

**Automated builds will:**
- Build on every push to `main`
- Build multi-arch images (AMD64 + ARM64)
- Push to Docker Hub automatically
- Tag with version from git tag

---

## Complete Workflow Example

### Developer Workflow (Local Build & Push)

```powershell
# 1. Make changes to code
cd v:\Code\ProjectCode\EquiLens

# 2. Test locally
docker compose build
docker compose up -d
# Test at http://localhost:7860

# 3. Tag version
$VERSION = "v1.0.0"
docker tag equilens:latest vkrishna04/equilens:$VERSION
docker tag equilens:latest vkrishna04/equilens:latest

# 4. Push to Docker Hub
docker push vkrishna04/equilens:$VERSION
docker push vkrishna04/equilens:latest

# 5. Update setup script
# Edit setup-docker-simple.ps1:
# $EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"

# 6. Test user workflow
.\setup-docker-simple.ps1
```

### User Workflow (Pull & Run)

```powershell
# 1. Download setup script
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/VKrishna04/EquiLens/main/setup-docker-simple.ps1" -OutFile "setup-docker-simple.ps1"

# 2. Run setup
.\setup-docker-simple.ps1

# 3. Access EquiLens
# Open http://localhost:7860
```

---

## Troubleshooting

### "denied: requested access to the resource is denied"

**Solution:** Login again
```powershell
docker logout
docker login
```

### "manifest unknown"

**Solution:** Make sure you pushed the tag
```powershell
docker push vkrishna04/equilens:latest
```

### Image size too large

**Solution:** Multi-stage builds already implemented in Dockerfile
```powershell
# Check image size
docker images vkrishna04/equilens

# Optimize further:
# - Remove unnecessary dependencies in pyproject.toml
# - Use .dockerignore to exclude files
# - Use alpine-based Python images (not recommended for UV)
```

### Push timeout

**Solution:** Increase timeout or push layers separately
```powershell
# Check network connection
Test-Connection -ComputerName hub.docker.com

# Push with retry
for ($i = 1; $i -le 3; $i++) {
    docker push vkrishna04/equilens:latest
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "Retry $i of 3..."
    Start-Sleep -Seconds 10
}
```

---

## Best Practices

### Security

1. ✅ **Use access tokens**, not passwords
2. ✅ **Rotate tokens** every 90 days
3. ✅ **Scan images** for vulnerabilities: `docker scan vkrishna04/equilens:latest`
4. ✅ **Don't commit secrets** to git
5. ✅ **Use minimal base images** (already using Python 3.13-slim)

### Performance

1. ✅ **Multi-stage builds** (already implemented)
2. ✅ **Layer caching** - Order Dockerfile commands by change frequency
3. ✅ **Build cache** - Use GitHub Actions cache
4. ✅ **.dockerignore** - Exclude unnecessary files

### Maintenance

1. ✅ **Tag every release** with semantic versioning
2. ✅ **Keep `latest` tag** updated
3. ✅ **Document changes** in `CHANGELOG.md`
4. ✅ **Test before pushing** to Docker Hub
5. ✅ **Monitor image size** and download counts

---

## Resources

- [Docker Hub](https://hub.docker.com/)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Semantic Versioning](https://semver.org/)
- [Docker Buildx](https://docs.docker.com/buildx/working-with-buildx/)

---

## Next Steps

1. **Test local build**: `docker compose build`
2. **Create Docker Hub account**: [hub.docker.com](https://hub.docker.com/)
3. **Login**: `docker login`
4. **Tag & push**: See [Quick Reference](#quick-reference)
5. **Update setup script**: Change `$EQUILENS_IMAGE` to your Docker Hub image
6. **Share with users**: Distribute `setup-docker-simple.ps1`

---

**Ready to deploy? Follow the [Quick Reference](#quick-reference) at the top of this guide!**
