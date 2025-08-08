# EquiLens Application Dockerfile - ULTRA-OPTIMIZED for speed
FROM python:3.12-slim
LABEL author="VKrishna04 <https://github.com/VKrishna04>"

# Install ONLY essential system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	git \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Create working directory and user
WORKDIR /workspace
RUN useradd -m -u 1000 equilens

# Copy ONLY requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with uv (OPTIMIZED - no heavy packages)
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Skip spaCy model download during build - do it at runtime instead
# This saves ~500MB and 5+ minutes during build
# RUN python -m spacy download en_core_web_sm --quiet

# Copy application files (this layer changes most often, so keep it last)
COPY . .

# Set permissions and create directories
RUN chown -R equilens:equilens /workspace && \
	mkdir -p logs results corpus

# Switch to non-root user
USER equilens

# Expose port for future web interface
EXPOSE 8000

# Default command - lightweight and fast
CMD ["python", "-c", "print('EquiLens container ready - optimized build!'); import time; time.sleep(999999)"]
