# 🦙 Ollama Docker — Quick Reference Cheatsheet

## One-Time Setup

```powershell
# Windows (PowerShell) — run once
.\ollama-setup.ps1

# With options
.\ollama-setup.ps1 -PullLatest          # force image update
.\ollama-setup.ps1 -ForceRecreate       # rebuild container (keeps volume)
.\ollama-setup.ps1 -Port 12000          # different port
```

```bash
# WSL2 / Linux / macOS — run once
chmod +x ollama-setup.sh && ./ollama-setup.sh

# With options
./ollama-setup.sh --pull-latest
./ollama-setup.sh --force-recreate
./ollama-setup.sh --port 12000
```

---

## Container Lifecycle

```powershell
docker start ollama       # start stopped container
docker stop  ollama       # stop (models stay in volume)
docker restart ollama     # restart

docker logs -f ollama     # live logs
docker stats ollama       # CPU / RAM / GPU usage
docker inspect ollama     # full container info
```

---

## Model Management

```powershell
# Pull models (stored in ollama_data volume)
docker exec -it ollama ollama pull llama3.2        # ~2 GB
docker exec -it ollama ollama pull phi3:mini       # ~2.3 GB
docker exec -it ollama ollama pull mistral         # ~4 GB
docker exec -it ollama ollama pull codellama       # ~4 GB
docker exec -it ollama ollama pull gemma2:9b       # ~5.5 GB
docker exec -it ollama ollama pull llava           # ~4.5 GB (vision)

# List downloaded models
docker exec -it ollama ollama list

# Remove a model
docker exec -it ollama ollama rm mistral

# Show model info / parameters
docker exec -it ollama ollama show llama3.2

# Interactive chat REPL
docker exec -it ollama ollama run llama3.2
```

---

## API Usage

```powershell
# Health check
curl http://localhost:11434

# List models via API
curl http://localhost:11434/api/tags

# Generate (non-streaming)
curl http://localhost:11434/api/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"llama3.2","prompt":"Why is the sky blue?","stream":false}'

# Chat completions (OpenAI-compatible)
curl http://localhost:11434/v1/chat/completions `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "model": "llama3.2",
    "messages": [{"role":"user","content":"Hello!"}]
  }'

# Pull a model via API
curl http://localhost:11434/api/pull `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"name":"phi3:mini"}'
```

---

## Volume Management

```powershell
# Inspect the volume
docker volume inspect ollama_data

# List all volumes
docker volume ls

# Disk usage
docker system df -v

# Backup models to a .tar.gz
docker run --rm `
  -v ollama_data:/data `
  -v ${PWD}:/backup `
  alpine tar czf /backup/ollama_backup.tar.gz -C /data .

# Restore from backup
docker run --rm `
  -v ollama_data:/data `
  -v ${PWD}:/backup `
  alpine tar xzf /backup/ollama_backup.tar.gz -C /data

# Delete volume (DESTROYS ALL DOWNLOADED MODELS)
docker volume rm ollama_data
```

---

## GPU Verification

```powershell
# Check host GPU
nvidia-smi

# Confirm GPU is visible inside the container
docker exec -it ollama nvidia-smi

# Watch GPU utilisation while running a model
nvidia-smi dmon -s u
```

---

## Cleanup

```powershell
# Remove container only (volume + models preserved)
docker stop ollama && docker rm ollama

# Remove everything including models (destructive!)
docker stop ollama
docker rm ollama
docker volume rm ollama_data

# Remove image
docker rmi ollama/ollama:latest
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `127.0.0.1` | API listen address (set to `0.0.0.0` for LAN) |
| `OLLAMA_MODELS` | `/root/.ollama/models` | Where models are stored inside container |
| `OLLAMA_NUM_PARALLEL` | `1` | Concurrent request handling |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Models kept hot in VRAM |
| `OLLAMA_FLASH_ATTENTION` | `0` | Enable flash attention (`1` = on) |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long model stays loaded after last request |

Set via `docker run -e VARIABLE=value ...` or in `docker-compose.yml`.

---

## Useful Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/api/tags` | List local models |
| POST | `/api/generate` | Text generation |
| POST | `/api/chat` | Chat completion |
| POST | `/api/pull` | Pull a model |
| DELETE | `/api/delete` | Delete a model |
| GET | `/api/ps` | Running models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
