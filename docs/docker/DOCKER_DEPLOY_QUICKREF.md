# Docker Hub Quick Deploy

**One-page reference for deploying EquiLens to Docker Hub**

---

## Prerequisites

```powershell
# Login to Docker Hub (one-time)
docker login
# Username: vkrishna04
# Password: (use access token from hub.docker.com)
```

---

## Deploy Workflow

### 1. Build Locally

```powershell
cd v:\Code\ProjectCode\EquiLens
docker compose build
```

### 2. Tag for Docker Hub

```powershell
# Replace vkrishna04 with your Docker Hub username
$USERNAME = "vkrishna04"
$VERSION = "v1.0.0"  # Change this for each release

# Tag with version
docker tag equilens:latest ${USERNAME}/equilens:${VERSION}

# Tag as latest
docker tag equilens:latest ${USERNAME}/equilens:latest
```

### 3. Push to Docker Hub

```powershell
# Push both tags
docker push ${USERNAME}/equilens:${VERSION}
docker push ${USERNAME}/equilens:latest
```

### 4. Update Setup Script

Edit `setup-docker-simple.ps1` line 8:

```powershell
$EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"  # Update version
```

### 5. Commit & Tag Release

```powershell
git add setup-docker-simple.ps1
git commit -m "Release v1.0.0"
git tag v1.0.0
git push origin main --tags
```

---

## User Installation

Users can now install with ONE command:

```powershell
# Download and run setup script
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/VKrishna04/EquiLens/main/setup-docker-simple.ps1" -OutFile "setup.ps1"; .\setup.ps1
```

---

## Quick Commands

```powershell
# Check local images
docker images | Select-String "equilens"

# Test locally before pushing
docker compose up -d
# Visit http://localhost:7860

# Stop containers
docker compose down

# Clean up old images
docker image prune -f
```

---

## Version Strategy

**Always tag THREE versions:**

```powershell
# Example: v1.2.3 release
docker tag equilens:latest vkrishna04/equilens:v1.2.3   # Exact
docker tag equilens:latest vkrishna04/equilens:v1.2     # Minor
docker tag equilens:latest vkrishna04/equilens:v1       # Major
docker tag equilens:latest vkrishna04/equilens:latest   # Latest

# Push all
docker push vkrishna04/equilens --all-tags
```

---

## Troubleshooting

### "denied: requested access to the resource is denied"

```powershell
docker logout
docker login
```

### "manifest unknown: manifest unknown"

```powershell
# Make sure you pushed the image
docker push vkrishna04/equilens:latest
```

### Image not updating

```powershell
# Clear cache and rebuild
docker compose build --no-cache
docker tag equilens:latest vkrishna04/equilens:latest
docker push vkrishna04/equilens:latest
```

---

## Complete Script

Save this as `deploy.ps1` for automated deployment:

```powershell
# deploy.ps1 - Automated Docker Hub deployment
param(
    [Parameter(Mandatory=$true)]
    [string]$Version,

    [string]$Username = "vkrishna04"
)

Write-Host "=== EquiLens Docker Deployment ===" -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Yellow
Write-Host ""

# Build
Write-Host "[1/4] Building image..." -ForegroundColor Yellow
docker compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build complete" -ForegroundColor Green

# Tag
Write-Host "[2/4] Tagging image..." -ForegroundColor Yellow
docker tag equilens:latest ${Username}/equilens:$Version
docker tag equilens:latest ${Username}/equilens:latest
Write-Host "✅ Tagged as $Version and latest" -ForegroundColor Green

# Push
Write-Host "[3/4] Pushing to Docker Hub..." -ForegroundColor Yellow
docker push ${Username}/equilens:$Version
docker push ${Username}/equilens:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Pushed to Docker Hub" -ForegroundColor Green

# Update setup script
Write-Host "[4/4] Updating setup script..." -ForegroundColor Yellow
(Get-Content setup-docker-simple.ps1) -replace 'EQUILENS_IMAGE = ".*"', "EQUILENS_IMAGE = `"${Username}/equilens:$Version`"" | Set-Content setup-docker-simple.ps1
Write-Host "✅ Updated setup-docker-simple.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "=== Deployment Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. git add setup-docker-simple.ps1" -ForegroundColor White
Write-Host "  2. git commit -m 'Release $Version'" -ForegroundColor White
Write-Host "  3. git tag $Version" -ForegroundColor White
Write-Host "  4. git push origin main --tags" -ForegroundColor White
Write-Host ""
Write-Host "Image URL: ${Username}/equilens:$Version" -ForegroundColor Cyan
Write-Host "Docker Hub: https://hub.docker.com/r/${Username}/equilens" -ForegroundColor Cyan
Write-Host ""
```

**Usage:**

```powershell
.\deploy.ps1 -Version "v1.0.0"
```

---

## Resources

- **Full Guide**: [DOCKER_HUB_DEPLOYMENT.md](./DOCKER_HUB_DEPLOYMENT.md)
- **Docker Hub**: https://hub.docker.com/
- **Create Access Token**: https://hub.docker.com/settings/security

---

**Ready to deploy?** Run the commands in [Deploy Workflow](#deploy-workflow) above!
