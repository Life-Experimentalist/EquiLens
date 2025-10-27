# Docker Volume Management Guide

## Volume Architecture

EquiLens uses **completely separate volumes** for Ollama and application data:

```
┌─────────────────────────────────────────┐
│        Volume Separation                │
├─────────────────────────────────────────┤
│                                         │
│  📦 Ollama Volume (Independent)         │
│  └─ ollama-models                       │
│     └─ AI models (10GB+)                │
│     └─ Can be shared/reused             │
│                                         │
│  📦 EquiLens Volumes (Separate)         │
│  ├─ equilens-data                       │
│  │  └─ Application data                 │
│  ├─ equilens-results                    │
│  │  └─ Analysis results & reports       │
│  └─ equilens-logs                       │
│     └─ Application logs                 │
│                                         │
└─────────────────────────────────────────┘
```

## Benefits of Volume Separation

✅ **Reuse Existing Models** - Bring your own Ollama models from other projects
✅ **Independent Lifecycle** - Delete EquiLens data without losing models
✅ **Share Across Projects** - Use same Ollama volume in multiple containers
✅ **Separate Backups** - Backup models and application data independently
✅ **Easy Migration** - Move EquiLens to new system while keeping models

---

## Using Existing Ollama Volume

### Step 1: Find Your Existing Volume

**PowerShell:**
```powershell
# List all volumes containing "ollama"
docker volume ls | Select-String "ollama"

# Inspect a specific volume
docker volume inspect YOUR_VOLUME_NAME

# Verify it contains models
docker run --rm -v YOUR_VOLUME_NAME:/data alpine ls -lh /data/models/manifests/registry.ollama.ai
```

**Bash:**
```bash
# List all volumes containing "ollama"
docker volume ls | grep ollama

# Inspect a specific volume
docker volume inspect YOUR_VOLUME_NAME

# Verify it contains models
docker run --rm -v YOUR_VOLUME_NAME:/data alpine ls -lh /data/models/manifests/registry.ollama.ai
```

### Step 2: Update docker-compose.yml

Edit the `volumes` section at the bottom of `docker-compose.yml`:

**Before (creates new volume):**
```yaml
volumes:
  ollama_models:
    driver: local
    name: ollama-models
    # external: false (default)
```

**After (uses existing volume):**
```yaml
volumes:
  ollama_models:
    external: true  # ← Add this line
    name: YOUR_EXISTING_OLLAMA_VOLUME_NAME  # ← Change this name
```

### Step 3: Start Services

```powershell
docker-compose up -d
```

Your existing models are now available immediately! 🎉

---

## Volume Operations

### List All EquiLens Volumes

**PowerShell:**
```powershell
docker volume ls | Select-String "ollama|equilens"
```

**Bash:**
```bash
docker volume ls | grep -E "ollama|equilens"
```

### Inspect Volume Details

```powershell
# Check volume location and size
docker volume inspect ollama-models

# See what's inside a volume
docker run --rm -v ollama-models:/data alpine ls -lh /data
```

### Backup Individual Volumes

**Backup Ollama models (separate from EquiLens data):**
```powershell
# PowerShell
docker run --rm `
  -v ollama-models:/data `
  -v ${PWD}/backups:/backup `
  alpine tar czf /backup/ollama-models-$(Get-Date -Format "yyyyMMdd").tar.gz /data

# Bash
docker run --rm \
  -v ollama-models:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/ollama-models-$(date +%Y%m%d).tar.gz /data
```

**Backup EquiLens results (separate from models):**
```powershell
# PowerShell
docker run --rm `
  -v equilens-results:/data `
  -v ${PWD}/backups:/backup `
  alpine tar czf /backup/equilens-results-$(Get-Date -Format "yyyyMMdd").tar.gz /data

# Bash
docker run --rm \
  -v equilens-results:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/equilens-results-$(date +%Y%m%d).tar.gz /data
```

### Restore Volumes

```powershell
# PowerShell
docker run --rm `
  -v ollama-models:/data `
  -v ${PWD}/backups:/backup `
  alpine tar xzf /backup/ollama-models-20241018.tar.gz -C /

# Bash
docker run --rm \
  -v ollama-models:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/ollama-models-20241018.tar.gz -C /
```

### Copy Models Between Volumes

```powershell
# Create a new volume and copy models from existing one
docker volume create new-ollama-models

docker run --rm `
  -v old-ollama-volume:/source:ro `
  -v new-ollama-models:/dest `
  alpine sh -c "cp -av /source/* /dest/"
```

### Clean Up EquiLens Data (Keep Ollama Models)

```powershell
# Remove ONLY EquiLens data volumes (models remain safe)
docker volume rm equilens-data
docker volume rm equilens-results
docker volume rm equilens-logs

# Ollama models are untouched!
docker volume inspect ollama-models  # Still there
```

---

## Common Scenarios

### Scenario 1: Fresh Install with New Volumes

**docker-compose.yml:**
```yaml
volumes:
  ollama_models:
    driver: local
    name: ollama-models
  equilens_data:
    driver: local
    name: equilens-data
  # ... other volumes
```

**Action:** Run `docker-compose up -d` - all volumes created automatically.

### Scenario 2: Use Existing Ollama, New EquiLens Data

**docker-compose.yml:**
```yaml
volumes:
  ollama_models:
    external: true
    name: my-existing-ollama-volume  # Your existing volume
  equilens_data:
    driver: local
    name: equilens-data  # New volume created
  # ... other volumes
```

**Benefits:**
- Reuse existing Ollama models (save bandwidth & time)
- Fresh EquiLens data (clean start)

### Scenario 3: Share Ollama Volume Across Multiple Projects

**Project 1 (EquiLens):**
```yaml
volumes:
  ollama_models:
    external: true
    name: shared-ollama-models
```

**Project 2 (Other App):**
```yaml
volumes:
  ollama_models:
    external: true
    name: shared-ollama-models  # Same volume!
```

**Benefits:**
- One set of models for all projects
- Saves disk space
- Consistent model versions

### Scenario 4: Migrate to New System

**On old system:**
```powershell
# Backup only what you need
docker run --rm -v ollama-models:/data -v ${PWD}:/backup alpine tar czf /backup/ollama.tar.gz /data
docker run --rm -v equilens-results:/data -v ${PWD}:/backup alpine tar czf /backup/results.tar.gz /data
```

**On new system:**
```powershell
# Restore volumes
docker volume create ollama-models
docker volume create equilens-results

docker run --rm -v ollama-models:/data -v ${PWD}:/backup alpine tar xzf /backup/ollama.tar.gz -C /
docker run --rm -v equilens-results:/data -v ${PWD}:/backup alpine tar xzf /backup/results.tar.gz -C /

# Start EquiLens
docker-compose up -d
```

---

## Volume Naming Convention

| Volume Name | Purpose | Typical Size | Shareable |
|------------|---------|--------------|-----------|
| `ollama-models` | Ollama AI models | 10GB+ | ✅ Yes |
| `equilens-data` | Application data | <1GB | ❌ No |
| `equilens-results` | Analysis results | Variable | ⚠️ Maybe |
| `equilens-logs` | Application logs | <500MB | ❌ No |

---

## Troubleshooting

### Volume Already Exists Error

**Error:**
```
Error response from daemon: create ollama-models: volume already exists
```

**Solution:**
Either use the existing volume or remove it first:

```powershell
# Option 1: Use existing volume (edit docker-compose.yml)
volumes:
  ollama_models:
    external: true
    name: ollama-models

# Option 2: Remove and recreate (⚠️ DELETES DATA)
docker-compose down -v
docker volume rm ollama-models
docker-compose up -d
```

### Volume Not Found Error

**Error:**
```
Error response from daemon: volume ollama-models not found
```

**Solution:**
Create the volume manually:

```powershell
docker volume create ollama-models
docker-compose up -d
```

### Check Volume Disk Usage

```powershell
# See all volume sizes
docker system df -v

# Inspect specific volume
docker volume inspect ollama-models | Select-String "Mountpoint"

# Check size at mount point (PowerShell - run as admin)
$mountpoint = (docker volume inspect ollama-models | ConvertFrom-Json).Mountpoint
Get-ChildItem -Path $mountpoint -Recurse | Measure-Object -Property Length -Sum
```

---

## Best Practices

1. **Separate Concerns**: Keep Ollama models separate from application data
2. **Regular Backups**: Backup volumes independently based on importance
3. **Name Clearly**: Use descriptive volume names (`ollama-models` not `vol1`)
4. **Document External Volumes**: If using `external: true`, document which volume
5. **Test Restores**: Verify backup/restore procedures work before you need them
6. **Monitor Disk Space**: Large models can fill up disk quickly
7. **Version Control**: Keep docker-compose.yml in git to track volume config changes

---

## Quick Reference Card

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect VOLUME_NAME

# Create volume
docker volume create VOLUME_NAME

# Remove volume (⚠️ deletes data)
docker volume rm VOLUME_NAME

# Remove all unused volumes (⚠️ careful!)
docker volume prune

# Backup volume
docker run --rm -v VOLUME_NAME:/data -v $(pwd):/backup alpine tar czf /backup/NAME.tar.gz /data

# Restore volume
docker run --rm -v VOLUME_NAME:/data -v $(pwd):/backup alpine tar xzf /backup/NAME.tar.gz -C /

# Browse volume contents
docker run --rm -it -v VOLUME_NAME:/data alpine sh

# Copy between volumes
docker run --rm -v SOURCE:/src:ro -v DEST:/dst alpine cp -av /src/* /dst/
```

---

## Support

For more information:
- Full Docker setup guide: `docs/DOCKER_SETUP.md`
- Docker checklist: `DOCKER_CHECKLIST.md`
- Main documentation: `README.md`

**Questions?** Open an issue on GitHub or check the documentation.
