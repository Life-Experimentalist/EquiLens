# infra/

Infrastructure files for running EquiLens in Docker.

## Usage

```bash
# From project root:
docker compose -f infra/docker-compose.yml up

# Full stack with GPU:
docker compose -f infra/docker-compose.full-stack.yml up
```

## Prerequisites

Before running the Docker stack, create the Ollama data volume if it doesn't exist:
```bash
docker volume create equilens-ollama-data
```

## GPU Requirements (full-stack only)

`docker-compose.full-stack.yml` requires:
- NVIDIA GPU on the host
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Run `nvidia-smi` to verify GPU is detected

For CPU-only setups, use `docker-compose.yml` instead.

## Notes

- `equilens_cli.py` is a legacy Python path shim. The container uses `/workspace/.venv/bin/equilens` directly.
- Corpus CSV files should be placed in the `data/corpus/` directory on the host, which maps to `/workspace/data/corpus` in the container.

## Files

- `Dockerfile` — EquiLens container image (port 8000, dashboard + API)
- `docker-compose.yml` — Standard stack (Ollama + EquiLens)
- `docker-compose.full-stack.yml` — Extended stack
- `.dockerignore` — Files excluded from build context
- `equilens_cli.py` — Docker entry point shim (adds src/ to PYTHONPATH for container use)
