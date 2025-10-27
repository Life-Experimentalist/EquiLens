# Smart Image Management - Feature Documentation

## ✨ New Feature: Intelligent Image Checking

The `setup-docker.ps1` script now intelligently manages Docker images with version detection and user choice.

---

## 🎯 How It Works

### Scenario 1: Image Exists Locally & Is Up-to-Date
```powershell
PS> .\setup-docker.ps1

[4/4] Setting up EquiLens...
  ✅ Found local image: vkrishna04/equilens:latest
     Created: 2025-10-15 10:30:00
  🔍 Checking for updates...
  ✅ Local image is up to date
  🚀 Starting container...
```

**Result:** Uses existing local image (fast!)

---

### Scenario 2: Image Exists But Older Version Available
```powershell
PS> .\setup-docker.ps1

[4/4] Setting up EquiLens...
  ✅ Found local image: vkrishna04/equilens:latest
     Created: 2025-10-10 08:00:00
  🔍 Checking for updates...
  ⚠️  Newer version available online!

  Choose action:
    [1] Use existing local image (default)
    [2] Pull latest version

  Enter choice (1-2, default=1): _
```

**User presses Enter or 1:**
```
  ✅ Using existing local image
  🚀 Starting container...
```

**User types 2:**
```
  📥 Pulling latest image: vkrishna04/equilens:latest
     (This may take 1-2 minutes)
  ✅ Latest image pulled successfully
  🚀 Starting container...
```

**Result:** User controls whether to update!

---

### Scenario 3: Image Not Found Locally
```powershell
PS> .\setup-docker.ps1

[4/4] Setting up EquiLens...
  ℹ️  Image not found locally
  📥 Pulling image: vkrishna04/equilens:latest
     (This may take 1-2 minutes on first run)
  ✅ Image pulled successfully
  🚀 Starting container...
```

**Result:** Automatically pulls image (required for first run)

---

### Scenario 4: Offline / Can't Check Remote
```powershell
PS> .\setup-docker.ps1

[4/4] Setting up EquiLens...
  ✅ Found local image: vkrishna04/equilens:latest
     Created: 2025-10-15 10:30:00
  🔍 Checking for updates...
  ℹ️  Cannot check remote version (offline or not on Docker Hub)
  ✅ Using local image
  🚀 Starting container...
```

**Result:** Uses local image when offline (graceful degradation)

---

## ⚙️ Configuration

### Change Docker Image

Edit `setup-docker.ps1` line 8:

```powershell
# Use latest version (recommended)
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"

# Use specific version (stable)
$EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"

# Use GitHub Container Registry
$EQUILENS_IMAGE = "ghcr.io/vkrishna04/equilens:latest"

# Use your own custom build
$EQUILENS_IMAGE = "mycompany/equilens:custom"

# Use local development build
$EQUILENS_IMAGE = "equilens:dev"
```

---

## 🎓 Version Detection Logic

### How It Checks Versions

```powershell
# 1. Check if local image exists
docker images --format "{{.Repository}}:{{.Tag}}" | Select-String "^$EQUILENS_IMAGE$"

# 2. Get local image digest (unique ID)
docker images --no-trunc --format "{{.ID}}" $EQUILENS_IMAGE

# 3. Get remote image digest (without downloading)
docker manifest inspect $EQUILENS_IMAGE

# 4. Compare digests
if ($localDigest -ne $remoteDigest) {
    # Newer version available - ask user
}
```

**Benefits:**
- ✅ No unnecessary downloads
- ✅ Accurate version detection (uses digest, not date)
- ✅ Works offline (graceful fallback)
- ✅ User controls updates

---

## 💡 Best Practices

### For Development
```powershell
# Pin to specific version for stability
$EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"

# No prompts - always uses exact version
```

### For Testing
```powershell
# Use latest for newest features
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"

# Always choose option 2 to pull updates
```

### For Production
```powershell
# Pin to tested version
$EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"

# Or use major version for auto-updates
$EQUILENS_IMAGE = "vkrishna04/equilens:v1"
```

---

## 🔄 Update Workflow

### Check for Updates
```powershell
# Run setup script
.\setup-docker.ps1

# If update available:
#   Choose option 2 to update
#   Or press Enter to skip

# Container will restart with new/old image
```

### Force Update
```powershell
# Remove local image
docker rmi vkrishna04/equilens:latest

# Run setup
.\setup-docker.ps1

# Will pull latest version
```

### Manual Update
```powershell
# Pull manually
docker pull vkrishna04/equilens:latest

# Run setup
.\setup-docker.ps1

# Will detect update and offer choice
```

---

## 📊 Scenarios Comparison

| Scenario | Local Image | Remote Check | Action | User Prompt |
|----------|-------------|--------------|--------|-------------|
| First run | ❌ No | ✅ Available | Pull | ❌ No |
| Up-to-date | ✅ Yes | ✅ Same version | Use local | ❌ No |
| Older version | ✅ Yes | ✅ Newer available | Ask user | ✅ Yes |
| Offline | ✅ Yes | ❌ Can't check | Use local | ❌ No |
| Custom build | ✅ Yes | ❌ Not on Hub | Use local | ❌ No |

---

## 🎯 Key Benefits

### 1. Smart Default
- ✅ Press Enter to use existing image (fast)
- ✅ No unnecessary downloads
- ✅ Saves bandwidth and time

### 2. User Control
- ✅ Always informed about updates
- ✅ Choose when to update
- ✅ No forced updates

### 3. Offline Support
- ✅ Works without internet
- ✅ Graceful fallback to local
- ✅ No errors when offline

### 4. Configurable
- ✅ Change image URL easily
- ✅ Pin versions for stability
- ✅ Use custom registries

### 5. Safe
- ✅ No automatic overwrites
- ✅ Always asks before pulling
- ✅ Preserves local images

---

## 🔍 Technical Details

### Image Digest vs Tag

**Tag (can change):**
```
vkrishna04/equilens:latest
```
- Same tag can point to different images
- Not reliable for version checking

**Digest (unique):**
```
sha256:abc123def456...
```
- Unique identifier for each image
- Reliable for version detection

**Script uses digests for accurate detection!** ✨

---

### Manifest Inspection

```powershell
# Check remote without downloading
docker manifest inspect vkrishna04/equilens:latest

# Returns JSON with digest:
{
  "schemaVersion": 2,
  "config": {
    "digest": "sha256:abc123..."
  }
}
```

**Benefits:**
- Fast (just metadata, no download)
- Accurate (uses digest)
- Efficient (minimal bandwidth)

---

## 🆘 Troubleshooting

### "Cannot check remote version"

**Causes:**
- Offline / no internet
- Image not on Docker Hub
- Docker Hub rate limit
- Private registry without auth

**Solution:**
- Script continues with local image
- Check internet connection if online expected
- For private registries: `docker login` first

---

### Update Not Detected

**Cause:**
- Docker cache issue
- Same digest (no actual update)

**Solution:**
```powershell
# Clear manifest cache
docker system prune -a

# Pull manually
docker pull vkrishna04/equilens:latest

# Run setup
.\setup-docker.ps1
```

---

### Always Asks for Update

**Cause:**
- Using `:latest` tag which changes frequently

**Solution:**
```powershell
# Pin to specific version
$EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"

# No more update prompts
```

---

## 📋 Quick Reference

### Configuration Examples
```powershell
# Latest version (auto-update available)
$EQUILENS_IMAGE = "vkrishna04/equilens:latest"

# Specific version (no updates)
$EQUILENS_IMAGE = "vkrishna04/equilens:v1.0.0"

# Major version (minor/patch updates)
$EQUILENS_IMAGE = "vkrishna04/equilens:v1"

# GitHub Container Registry
$EQUILENS_IMAGE = "ghcr.io/vkrishna04/equilens:latest"

# Local custom build
$EQUILENS_IMAGE = "equilens:dev"
```

### Update Commands
```powershell
# Check and optionally update
.\setup-docker.ps1
# Choose option 2 if update available

# Force pull latest
docker pull vkrishna04/equilens:latest
.\setup-docker.ps1

# Check what's running
docker inspect equilens-app | Select-String "Image"
```

---

## 🎉 Summary

The smart image management feature provides:

✅ **Intelligent detection** - Knows if updates available
✅ **User control** - You decide when to update
✅ **Offline support** - Works without internet
✅ **Configurable** - Easy to change image source
✅ **Safe defaults** - Existing image by default
✅ **Fast** - No unnecessary downloads

**One command, smart decisions!** 🚀

---

## 📚 Related Documentation

- **SMART_SETUP_GUIDE.md** - Complete setup guide
- **DOCKER_SETUP_COMPARISON.md** - Setup vs Dev comparison
- **DOCKER_HUB_DEPLOYMENT.md** - Deploying your own images

---

**Ready to use!** Just run `.\setup-docker.ps1` 🎊
