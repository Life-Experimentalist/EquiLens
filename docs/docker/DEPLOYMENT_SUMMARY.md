# 🚀 EquiLens Deployment - Summary

Complete deployment infrastructure has been set up for EquiLens with automated Docker publishing to GitHub Container Registry.

## ✅ What's Been Created

### 1. **GitHub Actions Workflow** (`.github/workflows/docker-publish.yml`)
- Automatically builds and publishes Docker images
- Triggers on:
  - Push to main → tags as `latest`
  - Version tags (`v*.*.*`) → semantic version tags
  - Pull requests → build validation only
- Multi-platform support: `linux/amd64`, `linux/arm64`
- **Single source of truth**: Extracts version from `pyproject.toml`

### 2. **Documentation**
- **DEPLOYMENT.md** - Comprehensive deployment guide
- **CHANGELOG.md** - Version history and upgrade guide
- **CONTRIBUTING.md** - Contribution guidelines
- **docs/docker/DOCKER_DEPLOY_GUIDE.md** - Quick reference for Docker deployment

### 3. **Build Optimization**
- **.dockerignore** - Excludes unnecessary files from Docker build
- **scripts/extract_version.py** - Utility to extract version from pyproject.toml

## 🎯 Key Features

### Automated Deployment
1. **Update version** in `pyproject.toml`:
   ```toml
   [project]
   version = "2.1.0"
   ```

2. **Create tag and push**:
   ```powershell
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to 2.1.0"
   git tag v2.1.0
   git push origin main
   git push origin v2.1.0
   ```

3. **GitHub Actions automatically**:
   - Extracts version from `pyproject.toml`
   - Builds Docker image for multiple platforms
   - Tags image with semantic versions:
     - `ghcr.io/life-experimentalist/equilens:2.1.0`
     - `ghcr.io/life-experimentalist/equilens:2.1`
     - `ghcr.io/life-experimentalist/equilens:2`
     - `ghcr.io/life-experimentalist/equilens:latest`
   - Pushes to GitHub Container Registry
   - Generates security attestation

### Single Source of Truth
- **Version number**: Defined only in `pyproject.toml`
- **Automatic extraction**: GitHub Actions reads version during build
- **No manual updates**: Tag version automatically propagates to Docker image

## 📦 Using the Deployment

### For End Users

```powershell
# Pull and run latest version
docker pull ghcr.io/life-experimentalist/equilens:latest
docker run -d -p 7860:7860 --name equilens ghcr.io/life-experimentalist/equilens:latest

# Or pull specific version
docker pull ghcr.io/life-experimentalist/equilens:2.0.0
docker run -d -p 7860:7860 --name equilens ghcr.io/life-experimentalist/equilens:2.0.0
```

### For Developers

```powershell
# Build locally
docker build -t equilens:dev .

# Test locally
docker run -d -p 7860:7860 --name equilens-dev equilens:dev

# Extract version
python scripts/extract_version.py
# Output: 2.0.0
```

## 🔧 Configuration

### GitHub Repository Settings

**Required for automated deployment:**

1. **Enable GitHub Packages**:
   - Repository Settings → Actions → General
   - Workflow permissions: **Read and write permissions**
   - Allow GitHub Actions to create and approve pull requests: ✅

2. **Make Package Public** (after first build):
   - Go to package: https://github.com/Life-Experimentalist/EquiLens/pkgs/container/equilens
   - Package settings → Change visibility → Public

### Environment Variables (Docker Runtime)

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434  # Auto-detected
OLLAMA_PORT=11434                                   # Default port

# EquiLens Configuration
EQUILENS_DATA_DIR=/workspace/data
EQUILENS_RESULTS_DIR=/workspace/data/results

# Gradio Web UI
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

## 🎉 Benefits

### Why GitHub Container Registry?

✅ **Free** for public repositories
✅ **Integrated** with GitHub Actions
✅ **No rate limits** on pulls
✅ **Better security** with fine-grained permissions
✅ **Automatic attestation** for supply chain security
✅ **Multi-platform** support (amd64, arm64)

### Why Not Docker Hub?

❌ **Rate limits** on anonymous pulls (100 pulls/6h)
❌ **Paid plans** for private repos with teams
❌ **Separate credentials** from GitHub
❌ **No native GitHub integration**

## 📊 Workflow Visualization

```
Developer                    GitHub Actions                 GHCR
    |                              |                          |
    | 1. Update pyproject.toml     |                          |
    |    version = "2.1.0"         |                          |
    |                              |                          |
    | 2. git tag v2.1.0           |                          |
    |    git push                  |                          |
    |----------------------------->|                          |
    |                              |                          |
    |                              | 3. Extract version       |
    |                              |    from pyproject.toml   |
    |                              |                          |
    |                              | 4. Build Docker image    |
    |                              |    (multi-platform)      |
    |                              |                          |
    |                              | 5. Tag image with        |
    |                              |    semantic versions     |
    |                              |                          |
    |                              | 6. Push to GHCR          |
    |                              |------------------------->|
    |                              |                          |
    |                              | 7. Generate attestation  |
    |                              |                          |
    |                              | ✅ Complete              |
    |                              |                          |
End User                          |                          |
    |                              |                          |
    | 8. docker pull ghcr.io/life-experimentalist/equilens:2.1.0
    |<---------------------------------------------------------|
    |                              |                          |
    | 9. docker run ...            |                          |
    |                              |                          |
```

## 🔍 Monitoring Builds

1. **Go to GitHub Actions**:
   - https://github.com/Life-Experimentalist/EquiLens/actions

2. **Select workflow**: "Docker Build and Publish"

3. **View logs**: Real-time build progress

4. **Check packages**:
   - https://github.com/Life-Experimentalist/EquiLens/pkgs/container/equilens

## ✨ Next Steps

### Before First Release

- [ ] Update `CHANGELOG.md` with release notes
- [ ] Test Docker image locally:
  ```powershell
  docker build -t equilens:test .
  docker run -d -p 7860:7860 equilens:test
  ```
- [ ] Verify version extraction:
  ```powershell
  python scripts/extract_version.py
  ```
- [ ] Update main README with deployment badges
- [ ] Push to GitHub and create first tag

### After First Build

- [ ] Make package public in GitHub Packages
- [ ] Test pulling public image:
  ```powershell
  docker pull ghcr.io/life-experimentalist/equilens:latest
  ```
- [ ] Create GitHub Release with notes
- [ ] Announce deployment to users

## 📚 Documentation Links

- **Main README**: [../../README.md](../../README.md)
- **Full Deployment Guide**: [../../DEPLOYMENT.md](../DEPLOYMENT.md)
- **Docker Quick Reference**: [DOCKER_DEPLOY_GUIDE.md](DOCKER_DEPLOY_GUIDE.md)
- **Changelog**: [../../CHANGELOG.md](../../CHANGELOG.md)
- **Contributing**: [../../CONTRIBUTING.md](../../CONTRIBUTING.md)

## 🆘 Troubleshooting

### Build Fails

**Check**:
- `pyproject.toml` has valid version string
- All dependencies listed in Dockerfile
- `.dockerignore` not excluding required files

**Solution**:
- View GitHub Actions logs
- Test build locally: `docker build .`

### Cannot Pull Image

**Check**:
- Package is public (not private)
- Image tag exists: https://github.com/Life-Experimentalist/EquiLens/pkgs/container/equilens

**Solution**:
- Make package public in settings
- Use correct image name with `ghcr.io` prefix

### Version Mismatch

**Check**:
- `pyproject.toml` version matches git tag
- GitHub Actions extracted correct version (view logs)

**Solution**:
- Ensure tag format is `v*.*.*` (e.g., `v2.0.0`)
- Re-run workflow if needed

---

**🎉 Deployment infrastructure is complete and ready to use!**

Simply update the version in `pyproject.toml`, create a git tag, push, and GitHub Actions handles the rest.
