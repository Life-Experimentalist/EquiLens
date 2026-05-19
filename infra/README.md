# infra/

Infrastructure files for running EquiLens in Docker.

## Usage

```bash
# From project root:
docker compose -f infra/docker-compose.yml up

# Full stack with GPU:
docker compose -f infra/docker-compose.full-stack.yml up
```

## Files

- `Dockerfile` — EquiLens container image (port 8000, dashboard + API)
- `docker-compose.yml` — Standard stack (Ollama + EquiLens)
- `docker-compose.full-stack.yml` — Extended stack
- `.dockerignore` — Files excluded from build context
- `equilens_cli.py` — Docker entry point shim (adds src/ to PYTHONPATH for container use)
