# Quick Guide: Using Your Existing Ollama Models

If you already have Ollama models downloaded, you can reuse them with EquiLens instead of downloading them again!

## 🔍 Step 1: Find Your Existing Ollama Volume

**PowerShell:**
```powershell
docker volume ls | Select-String "ollama"
```

**Bash/Linux/macOS:**
```bash
docker volume ls | grep ollama
```

**Example output:**
```
local     ollama
local     my-ollama-models
local     project1-ollama
```

## ✅ Step 2: Verify It Has Models

Check if the volume contains models:

**PowerShell:**
```powershell
docker run --rm -v ollama:/data alpine ls -lh /data/models/manifests/registry.ollama.ai/library
```

**Bash:**
```bash
docker run --rm -v ollama:/data alpine ls -lh /data/models/manifests/registry.ollama.ai/library
```

If you see folders like `llama3.2`, `mistral`, etc., you have models! 🎉

## ✏️ Step 3: Edit docker-compose.yml

Open `docker-compose.yml` and find the `volumes:` section at the bottom (around line 80).

**Change FROM:**
```yaml
volumes:
  ollama_models:
    driver: local
    name: ollama-models
    # To use existing Ollama volume, uncomment below and set name above:
    # external: true
```

**Change TO:**
```yaml
volumes:
  ollama_models:
    external: true           # ← UNCOMMENT this line
    name: ollama             # ← CHANGE to your volume name (from Step 1)
```

**Example if your volume is called `my-ollama-models`:**
```yaml
volumes:
  ollama_models:
    external: true
    name: my-ollama-models
```

## 🚀 Step 4: Start EquiLens

```powershell
docker-compose up -d
```

That's it! EquiLens will now use your existing Ollama models.

## ✨ Verify It Worked

**Check available models:**
```powershell
docker exec equilens-ollama ollama list
```

You should see your existing models listed!

**Test a model:**
```powershell
docker exec equilens-ollama ollama run llama3.2 "Hello!"
```

---

## 💡 Benefits

- ✅ **No re-download** - Save bandwidth and time
- ✅ **Shared models** - Use same models across multiple projects
- ✅ **Disk space** - No duplicate model storage
- ✅ **Instant setup** - Models available immediately

---

## 🔧 Troubleshooting

### "Volume not found" error

**Error:**
```
Error response from daemon: volume ollama not found
```

**Solution:** Double-check the volume name:
```powershell
docker volume ls | Select-String "ollama"
```

Make sure the `name:` in docker-compose.yml matches exactly!

### Models not showing up

**Check volume is mounted:**
```powershell
docker exec equilens-ollama ls -la /root/.ollama/models
```

If empty, the wrong volume might be mounted. Verify the volume name in docker-compose.yml.

### Want to switch back to new volume

**Edit docker-compose.yml:**
```yaml
volumes:
  ollama_models:
    driver: local        # ← Change external: true to this
    name: ollama-models  # ← New volume name
    # external: true     # ← Comment this out
```

Then:
```powershell
docker-compose down
docker-compose up -d
```

---

## 📚 More Information

- **Complete volume guide:** `VOLUME_MANAGEMENT.md`
- **Docker setup guide:** `docs/DOCKER_SETUP.md`
- **Full documentation:** `README.md`

**Questions?** Open an issue on GitHub!
