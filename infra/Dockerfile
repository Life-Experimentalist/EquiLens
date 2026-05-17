FROM python:3.13.3-slim-bullseye

LABEL author="VKrishna04"
LABEL org.opencontainers.image.source="https://github.com/Life-Experimentalist/EquiLens"
LABEL org.opencontainers.image.description="EquiLens AI Bias Detection Platform"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="2.2.0"

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir uv

RUN useradd -m -u 1000 -s /bin/bash equilens && \
    mkdir -p /workspace/data/results /workspace/data/logs /workspace/data/corpus /workspace/backups && \
    chown -R equilens:equilens /workspace

USER equilens
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/workspace/src:/workspace \
    PATH="/workspace/.venv/bin:$PATH" \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    EQUILENS_BACKUP_INTERVAL_MINUTES=30 \
    EQUILENS_BACKUP_RETENTION=10

COPY --chown=equilens:equilens pyproject.toml README.md ./
COPY --chown=equilens:equilens uv.lock* ./

RUN --mount=type=cache,target=/home/equilens/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

COPY --chown=equilens:equilens . .

RUN mkdir -p data/results data/logs data/corpus src/Phase1_CorpusGenerator/corpus public backups && \
    chmod -R 755 data backups

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

CMD ["/workspace/.venv/bin/equilens", "web"]
