# EquiLens Docker Quick Reference

## One-Command Installation

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.ps1 | iex
```

### Linux/macOS (Bash)
```bash
curl -fsSL https://raw.githubusercontent.com/[USER]/EquiLens/main/setup-docker.sh | bash
```

## Access Points

After installation:
- **Gradio UI**: http://localhost:7860
- **Web API**: http://localhost:8000  
- **Ollama API**: http://localhost:11434

## Common Commands

```powershell
# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop services
docker-compose down

# Start services
docker-compose up -d

# Access container
docker exec -it equilens-app bash

# Pull Ollama models
docker exec equilens-ollama ollama pull llama3.2:latest
docker exec equilens-ollama ollama pull mistral:latest
```

## Persistent Data

All data is stored in Docker volumes:
- `equilens-ollama-models` - Ollama models
- `equilens-data` - Application data
- `equilens-results` - Analysis results
- `equilens-logs` - Application logs

## Documentation

See `docs/DOCKER_SETUP.md` for complete documentation.

## Troubleshooting

### Services won't start
```powershell
docker-compose down
docker-compose up -d
docker-compose logs
```

### Port conflicts
Edit `docker-compose.yml` and change host ports:
```yaml
ports:
  - "7861:7860"  # Change first number
```

### Check Ollama connectivity
```powershell
docker exec equilens-ollama curl http://localhost:11434/api/tags
```

## Requirements

- Docker Desktop 20.10+ or Docker Engine
- Docker Compose 2.0+
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

## Support

- Full docs: `docs/DOCKER_SETUP.md`
- Local setup: `docs/QUICKSTART.md`
- Issues: GitHub Issues page
