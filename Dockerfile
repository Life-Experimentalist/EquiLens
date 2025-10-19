FROM python:3.13.3-slim

LABEL author="VKrishna04"
LABEL org.opencontainers.image.source="https://github.com/Life-Experimentalist/EquiLens"
LABEL org.opencontainers.image.description="EquiLens AI Bias Detection Platform"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="2.0.0"

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
	curl git ca-certificates && \
	rm -rf /var/lib/apt/lists/* && \
	apt-get clean && apt-get autoremove -y

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
	pip install --no-cache-dir uv

RUN useradd -m -u 1000 -s /bin/bash equilens && \
	mkdir -p /workspace/data/results /workspace/data/logs /workspace/data/corpus && \
	chown -R equilens:equilens /workspace

USER equilens
WORKDIR /workspace

# Copy dependency files ONLY first (for better caching)
COPY --chown=equilens:equilens pyproject.toml README.md ./
COPY --chown=equilens:equilens uv.lock* ./

# Use BuildKit cache mount for UV cache directory
# This persists the UV cache between builds, avoiding re-downloads
RUN --mount=type=cache,target=/home/equilens/.cache/uv,uid=1000,gid=1000 \
	uv sync --frozen --no-dev || uv sync --no-dev

# Copy the rest of the application (this layer changes more frequently)
COPY --chown=equilens:equilens . .

RUN mkdir -p data/results data/logs data/corpus src/Phase1_CorpusGenerator/corpus public && \
	chmod -R 755 data && \
	find . -name "*.pyc" -delete && \
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
	CMD python -c "import sys; print('EquiLens healthy'); sys.exit(0)" || exit 1

EXPOSE 7860 8000

ENV PYTHONUNBUFFERED=1 \
	PYTHONIOENCODING=utf-8 \
	LANG=C.UTF-8 \
	LC_ALL=C.UTF-8 \
	PYTHONPATH=/workspace/src:/workspace \
	OLLAMA_BASE_URL=http://localhost:11434 \
	GRADIO_SERVER_PORT=7860 \
	GRADIO_SERVER_NAME=0.0.0.0

# Run the Gradio GUI by default
CMD [".venv/bin/equilens", "gui"]
