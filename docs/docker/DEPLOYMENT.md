# 🚀 EquiLens Deployment Guide

Complete guide for deploying EquiLens to production using Docker and GitHub Container Registry.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [GitHub Container Registry Setup](#github-container-registry-setup)
- [Automated Deployment](#automated-deployment)
- [Manual Deployment](#manual-deployment)
- [Version Management](#version-management)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
- **Git** - Version control
- **Docker** - Container runtime
- **GitHub Account** - For container registry

### Optional Tools
- **Docker Compose** - Multi-container orchestration
- **GitHub CLI** - Easier GitHub Actions management

## GitHub Container Registry Setup

We use **GitHub Container Registry (ghcr.io)** instead of Docker Hub because:
- ✅ **Free** for public repositories
- ✅ **Integrated** with GitHub Actions
- ✅ **Better security** with fine-grained permissions
- ✅ **Automatic** image attestation
- ✅ **No rate limits** for pulls

### 1. Enable GitHub Packages

1. Go to your repository settings
2. Navigate to **Actions** → **General**
3. Under **Workflow permissions**, select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
4. Save changes

### 2. Generate Personal Access Token (Optional)

For local builds/pushes:

1. Go to **GitHub Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **Generate new token** → **Generate new token (classic)**
3. Select scopes:
   - ✅ `write:packages`
   - ✅ `read:packages`
   - ✅ `delete:packages`
4. Generate and **save the token securely**

### 3. Login to GitHub Container Registry

```powershell
# Login using Personal Access Token
$env:CR_PAT = "your_personal_access_token"
echo $env:CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Or using GitHub CLI (recommended)
gh auth login
echo $env:GH_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

## Automated Deployment

EquiLens uses **GitHub Actions** for automated Docker builds and publishing.

### Workflow Triggers

The Docker build workflow (`.github/workflows/docker-publish.yml`) runs on:

1. **Push to main branch** - Builds and tags as `latest`
2. **Version tag push** (`v*.*.*`) - Builds and tags with semantic versioning
3. **Pull requests** - Builds but doesn't push (validation only)
4. **Manual dispatch** - Trigger from GitHub Actions tab

### Version Update Workflow

**Single Source of Truth: `pyproject.toml`**

```toml
[project]
name = "equilens"
version = "2.0.0"  # ← Change this version number
```

**Automated deployment steps:**

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "2.1.0"  # Increment version
   ```

2. **Commit and tag**:
   ```powershell
   # Commit version change
   git add pyproject.toml
   git commit -m "chore: bump version to 2.1.0"

   # Create version tag
   git tag v2.1.0

   # Push commit and tag
   git push origin main
   git push origin v2.1.0
   ```

3. **Automatic build** - GitHub Actions will:
   - ✅ Extract version from `pyproject.toml`
   - ✅ Build Docker image for multiple platforms (amd64, arm64)
   - ✅ Tag image with:
     - `ghcr.io/life-experimentalist/equilens:2.1.0`
     - `ghcr.io/life-experimentalist/equilens:2.1`
     - `ghcr.io/life-experimentalist/equilens:2`
     - `ghcr.io/life-experimentalist/equilens:latest`
   - ✅ Push to GitHub Container Registry
   - ✅ Generate attestation for security

4. **Verify deployment**:
   ```powershell
   # Check GitHub Packages
   # Go to: https://github.com/Life-Experimentalist/EquiLens/pkgs/container/equilens

   # Pull and test new version
   docker pull ghcr.io/life-experimentalist/equilens:2.1.0
   docker run -d -p 7860:7860 ghcr.io/life-experimentalist/equilens:2.1.0
   ```

### Monitoring Builds

1. Go to **GitHub** → **Actions** tab
2. Click on the **Docker Build and Publish** workflow
3. View real-time build logs
4. Check for any errors

## Manual Deployment

For manual builds and pushes:

### 1. Build Image Locally

```powershell
# Extract version from pyproject.toml
$VERSION = python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Build for current platform
docker build -t ghcr.io/life-experimentalist/equilens:$VERSION .
docker build -t ghcr.io/life-experimentalist/equilens:latest .

# Build for multiple platforms (requires buildx)
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/life-experimentalist/equilens:$VERSION \
  -t ghcr.io/life-experimentalist/equilens:latest \
  --push .
```

### 2. Test Image Locally

```powershell
# Run image
docker run -d \
  --name equilens-test \
  -p 7860:7860 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  ghcr.io/life-experimentalist/equilens:$VERSION

# Test web UI
# Open http://localhost:7860

# Check logs
docker logs -f equilens-test

# Stop and remove
docker stop equilens-test
docker rm equilens-test
```

### 3. Push to Registry

```powershell
# Push specific version
docker push ghcr.io/life-experimentalist/equilens:$VERSION

# Push latest tag
docker push ghcr.io/life-experimentalist/equilens:latest

# Verify on GitHub Packages
# https://github.com/Life-Experimentalist/EquiLens/pkgs/container/equilens
```

## Version Management

### Semantic Versioning

EquiLens follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH (e.g., 2.1.3)
```

- **MAJOR** - Breaking changes (e.g., 2.0.0 → 3.0.0)
- **MINOR** - New features, backward compatible (e.g., 2.0.0 → 2.1.0)
- **PATCH** - Bug fixes, backward compatible (e.g., 2.1.0 → 2.1.1)

### Version Tagging Strategy

```powershell
# Patch release (bug fixes)
git tag v2.0.1
git push origin v2.0.1

# Minor release (new features)
git tag v2.1.0
git push origin v2.1.0

# Major release (breaking changes)
git tag v3.0.0
git push origin v3.0.0
```

### Changelog Management

Update `CHANGELOG.md` before each release:

```markdown
## [2.1.0] - 2025-01-15

### Added
- Smart Ollama configuration with auto-detection
- Configurable port support via OLLAMA_PORT
- Comprehensive deployment documentation

### Changed
- Improved Docker environment detection
- Enhanced error messages in web UI

### Fixed
- Container-to-host communication issues
- Documentation clarity for docker exec commands
```

## Production Deployment

### Using Docker Compose (Recommended)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  equilens:
    image: ghcr.io/life-experimentalist/equilens:2.0.0
    container_name: equilens-prod
    restart: unless-stopped
    ports:
      - "7860:7860"
    volumes:
      - equilens-data:/workspace/data
      - equilens-results:/workspace/data/results
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - OLLAMA_PORT=11434
      - EQUILENS_DATA_DIR=/workspace/data
      - EQUILENS_RESULTS_DIR=/workspace/data/results
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=7860
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:7860/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  equilens-data:
    name: equilens-prod-data
  equilens-results:
    name: equilens-prod-results
```

Deploy:

```powershell
# Pull latest image
docker compose -f docker-compose.prod.yml pull

# Start services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Update to new version
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --force-recreate
```

### Using Plain Docker

```powershell
# Pull specific version
docker pull ghcr.io/life-experimentalist/equilens:2.0.0

# Run in production
docker run -d \
  --name equilens-prod \
  --restart unless-stopped \
  -p 7860:7860 \
  -v equilens-prod-data:/workspace/data \
  -v equilens-prod-results:/workspace/data/results \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  ghcr.io/life-experimentalist/equilens:2.0.0

# Check status
docker ps | findstr equilens

# View logs
docker logs -f equilens-prod

# Update to new version
docker stop equilens-prod
docker rm equilens-prod
docker pull ghcr.io/life-experimentalist/equilens:latest
# Re-run docker run command above
```

### Production Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests: `uv run pytest`
- [ ] Commit changes: `git commit -m "chore: release v2.1.0"`
- [ ] Create tag: `git tag v2.1.0`
- [ ] Push tag: `git push origin v2.1.0`
- [ ] Wait for GitHub Actions to complete
- [ ] Verify image on GitHub Packages
- [ ] Test pulled image locally
- [ ] Deploy to production
- [ ] Create GitHub Release with notes

## Troubleshooting

### Build Failures

**Issue**: GitHub Actions build fails

```bash
# Check workflow logs
# Go to GitHub → Actions → Failed workflow → View logs

# Common fixes:
1. Ensure pyproject.toml has valid version
2. Check Dockerfile syntax
3. Verify all files in .dockerignore are correct
4. Re-run workflow from GitHub Actions UI
```

### Permission Errors

**Issue**: Cannot push to ghcr.io

```powershell
# Verify login
docker login ghcr.io

# Check token permissions
# Token needs write:packages scope

# Re-generate token if needed
# GitHub Settings → Developer settings → PAT
```

### Image Pull Errors

**Issue**: Cannot pull from ghcr.io

```powershell
# Make package public
# Go to Package settings → Change visibility → Public

# Or authenticate pull
echo $env:CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
docker pull ghcr.io/life-experimentalist/equilens:latest
```

### Version Mismatch

**Issue**: Image version doesn't match pyproject.toml

```powershell
# Verify version extraction
python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Check GitHub Actions logs for version
# Actions → Docker Build → View logs → Extract version step

# Ensure tag matches version
git tag -l
```

## Security Best Practices

1. **Use specific version tags** in production (not `latest`)
2. **Enable image scanning** in GitHub Security tab
3. **Review dependencies** regularly with `uv sync --upgrade`
4. **Enable attestation** (already in workflow)
5. **Use secrets** for sensitive data (not hardcoded in compose files)

## Support

- **GitHub Issues**: [Report bugs](https://github.com/Life-Experimentalist/EquiLens/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Life-Experimentalist/EquiLens/discussions)
- **Email**: krishnagsvv@gmail.com

---

**Next Steps**: See [README.md](README.md) for usage guide or [docs/QUICKSTART.md](docs/QUICKSTART.md) for getting started.
