# EquiLens Application Dockerfile - ULTRA-OPTIMIZED & SECURE
FROM python:3.13.3-slim

# Metadata for security and traceability
LABEL author="VKrishna04 <https://github.com/VKrishna04>"
LABEL org.opencontainers.image.source="https://github.com/Life-Experimentalists/EquiLens"
LABEL org.opencontainers.image.description="EquiLens AI Bias Detection Platform"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.created="2025-08-08"

# Security: Update base packages and install only essential tools
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
	curl \
	git \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean \
	&& apt-get autoremove -y

# Install UV for fast Python package management
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
	pip install --no-cache-dir uv

# Security: Create non-root user with explicit shell
RUN useradd -m -u 1000 -s /bin/bash equilens && \
	mkdir -p /workspace && \
	chown equilens:equilens /workspace

# Switch to non-root user for security
USER equilens
WORKDIR /workspace

# Copy UV configuration files first for better layer caching
COPY --chown=equilens:equilens pyproject.toml uv.lock ./

# Install Python dependencies with UV (OPTIMIZED)
RUN uv sync --frozen --no-dev

# Copy application files (this layer changes most often)
COPY --chown=equilens:equilens . .

# Create necessary directories with proper permissions
RUN mkdir -p logs results src/Phase1_CorpusGenerator/corpus && \
	chmod 755 logs results

# Security: Clean up Python cache and temporary files
RUN find . -name "*.pyc" -delete && \
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
	CMD python -c "import sys; print('🚀 EquiLens healthy'); sys.exit(0)" || exit 1

# Expose port for future web interface
EXPOSE 8000

# Set environment variables for better Unicode support
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Default command - ready for interactive use
CMD ["uv", "run", "equilens", "--help"]
