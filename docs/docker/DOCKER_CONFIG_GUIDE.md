# EquiLens Docker Configuration Guide

## 🎛️ All Customizable Parameters

### 1. **Container Configuration**

#### Ports
```yaml
ports:
  - "7860:7860"  # Gradio Web UI
  - "8000:8000"  # FastAPI/Web API
```

**Customization:**
```yaml
# Change host port (left side) to avoid conflicts
ports:
  - "8080:7860"  # Access UI at http://localhost:8080
  - "9000:8000"  # Access API at http://localhost:9000
```

---

#### Container Name
```yaml
container_name: equilens-app
```

**Customization:**
```yaml
# Useful for running multiple instances
container_name: equilens-production
container_name: equilens-dev
container_name: equilens-testing
```

---

#### Image Name and Tag
```yaml
image: equilens:latest
```

**Customization:**
```yaml
image: equilens:v2.0.0           # Specific version
image: vkrishna04/equilens:latest  # Docker Hub image
image: equilens:dev              # Development build
```

---

### 2. **Volume Configuration**

#### Single Data Volume (Current Setup - RECOMMENDED)
```yaml
volumes:
  - equilens_data:/workspace/data           # Persistent data
  - ./src:/workspace/src:ro                 # Source code (read-only)
  - ./public:/workspace/public:ro           # Public assets (read-only)
```

**Volume Locations Inside Container:**
- `/workspace/data` - All persistent data
- `/workspace/data/results` - Audit results
- `/workspace/data/logs` - Application logs
- `/workspace/data/corpus` - Generated corpus files

**Customization Options:**

**Option A: Change Volume Name**
```yaml
volumes:
  equilens_data:
    driver: local
    name: my-custom-equilens-data  # Custom name
```

**Option B: Use Host Directory (Development)**
```yaml
volumes:
  - ./data:/workspace/data                  # Local directory
  - ./src:/workspace/src:ro
  - ./public:/workspace/public:ro
```

**Option C: Separate Volumes (NOT RECOMMENDED - Unnecessary)**
```yaml
volumes:
  - equilens_data:/workspace/data
  - equilens_results:/workspace/data/results  # Separate results
  - equilens_logs:/workspace/data/logs        # Separate logs
```

**Option D: External Volume (Reuse Existing)**
```yaml
volumes:
  equilens_data:
    external: true
    name: existing-volume-name
```

---

### 3. **Network Configuration**

#### Current: Host Mode
```yaml
network_mode: "host"
```

**Pros:**
- ✅ Easy Ollama connection (localhost:11434)
- ✅ No port mapping needed
- ✅ Direct host network access

**Cons:**
- ❌ Less isolated
- ❌ Linux/Windows compatibility differences

**Alternative: Bridge Mode (More Isolated)**
```yaml
# Remove network_mode: "host"
# Add networks section:
networks:
  equilens-network:
    driver: bridge

services:
  equilens:
    networks:
      - equilens-network
    ports:
      - "7860:7860"
      - "8000:8000"
```

**With Ollama Container:**
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-gpu
    networks:
      - equilens-network
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama

  equilens:
    image: equilens:latest
    networks:
      - equilens-network
    environment:
      - OLLAMA_BASE_URL=http://ollama-gpu:11434  # Container name
```

---

### 4. **Environment Variables**

#### Ollama Configuration
```yaml
OLLAMA_BASE_URL=http://localhost:11434        # Ollama server URL
OLLAMA_API_BASE=http://localhost:11434/api    # API endpoint
```

**Customization:**
```yaml
# For Docker container Ollama
OLLAMA_BASE_URL=http://ollama-gpu:11434

# For remote Ollama
OLLAMA_BASE_URL=http://192.168.1.100:11434

# For cloud Ollama
OLLAMA_BASE_URL=https://ollama.example.com
```

---

#### Python Configuration
```yaml
PYTHONPATH=/workspace
PYTHONUNBUFFERED=1
PYTHONIOENCODING=utf-8
```

**No need to change these - they're optimal defaults**

---

#### EquiLens Paths
```yaml
EQUILENS_DATA_DIR=/workspace/data
EQUILENS_RESULTS_DIR=/workspace/data/results
EQUILENS_LOGS_DIR=/workspace/data/logs
EQUILENS_CORPUS_DIR=/workspace/src/Phase1_CorpusGenerator/corpus
```

**Customization (if using different volume structure):**
```yaml
EQUILENS_DATA_DIR=/app/data
EQUILENS_RESULTS_DIR=/app/data/results
EQUILENS_LOGS_DIR=/app/data/logs
EQUILENS_CORPUS_DIR=/app/corpus
```

---

#### Gradio UI Configuration
```yaml
GRADIO_SERVER_NAME=0.0.0.0      # Listen on all interfaces
GRADIO_SERVER_PORT=7860         # Web UI port
GRADIO_ANALYTICS_ENABLED=false  # Disable Gradio telemetry
GRADIO_THEME=default            # UI theme
```

**Customization:**
```yaml
GRADIO_SERVER_PORT=8080         # Change internal port
GRADIO_THEME=soft               # Available: default, soft, glass, monochrome
GRADIO_ANALYTICS_ENABLED=true   # Enable analytics
```

---

### 5. **Resource Limits (OPTIONAL)**

#### Memory Limits
```yaml
deploy:
  resources:
    limits:
      memory: 4G      # Maximum memory
    reservations:
      memory: 2G      # Minimum reserved
```

#### CPU Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'     # Max 2 CPUs
    reservations:
      cpus: '1.0'     # Min 1 CPU
```

#### GPU Support (NVIDIA)
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

### 6. **Restart Policy**
```yaml
restart: unless-stopped
```

**Options:**
- `no` - Never restart
- `always` - Always restart
- `on-failure` - Restart only on failure
- `unless-stopped` - Restart unless manually stopped

---

### 7. **Health Check**
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:7860/ || exit 1"]
  interval: 30s      # Check every 30s
  timeout: 10s       # Timeout after 10s
  retries: 3         # Retry 3 times
  start_period: 30s  # Wait 30s before first check
```

**Customization:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "python -c 'import requests; requests.get(\"http://localhost:7860\")'"]
  interval: 60s      # Less frequent checks
  timeout: 5s        # Faster timeout
  retries: 5         # More retries
```

---

### 8. **Build Arguments (Dockerfile)**

#### Python Version
```dockerfile
FROM python:3.13.3-slim
```

**Change to:**
```dockerfile
FROM python:3.11-slim
FROM python:3.12-slim
```

#### User ID (for permissions)
```dockerfile
RUN useradd -m -u 1000 -s /bin/bash equilens
```

**Customization:**
```dockerfile
# Match your host user ID to avoid permission issues
RUN useradd -m -u 1001 -s /bin/bash equilens
```

---

## 🎯 Common Configuration Scenarios

### Scenario 1: Multiple Instances
```yaml
# Production instance
services:
  equilens-prod:
    container_name: equilens-prod
    ports:
      - "7860:7860"
    volumes:
      - equilens_data_prod:/workspace/data

# Development instance
  equilens-dev:
    container_name: equilens-dev
    ports:
      - "7861:7860"
    volumes:
      - equilens_data_dev:/workspace/data
```

---

### Scenario 2: With Ollama Container
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-gpu
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  equilens:
    image: equilens:latest
    container_name: equilens-app
    depends_on:
      - ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama-gpu:11434
```

---

### Scenario 3: Development with Hot Reload
```yaml
volumes:
  - ./data:/workspace/data          # Local data directory
  - ./src:/workspace/src            # NOT read-only for development
  - ./public:/workspace/public
```

---

### Scenario 4: Production Hardened
```yaml
services:
  equilens:
    image: vkrishna04/equilens:latest
    container_name: equilens-prod
    restart: always
    read_only: true                 # Read-only filesystem
    tmpfs:
      - /tmp
      - /workspace/.cache
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

---

## 📋 Quick Reference: Essential Parameters

| Parameter | Default | Purpose | Change Frequency |
|-----------|---------|---------|------------------|
| `ports` | 7860, 8000 | Web UI & API access | Rarely |
| `container_name` | equilens-app | Container identifier | Sometimes |
| `volumes` | equilens_data | Data persistence | Rarely |
| `OLLAMA_BASE_URL` | localhost:11434 | Ollama connection | Sometimes |
| `GRADIO_SERVER_PORT` | 7860 | Web UI port | Rarely |
| `restart` | unless-stopped | Restart policy | Sometimes |
| `network_mode` | host | Network mode | Rarely |

---

## 🔧 How to Apply Custom Configuration

### Method 1: Edit docker-compose.yml directly
```bash
# Edit file
nano docker-compose.yml

# Apply changes
docker-compose down
docker-compose up -d
```

### Method 2: Override file (RECOMMENDED)
```bash
# Create docker-compose.override.yml
cat > docker-compose.override.yml << EOF
services:
  equilens:
    ports:
      - "8080:7860"
    environment:
      - GRADIO_THEME=soft
EOF

# Automatically applied
docker-compose up -d
```

### Method 3: Environment file
```bash
# Create .env file
cat > .env << EOF
GRADIO_PORT=7860
OLLAMA_URL=http://localhost:11434
EOF

# Reference in docker-compose.yml
environment:
  - GRADIO_SERVER_PORT=${GRADIO_PORT}
  - OLLAMA_BASE_URL=${OLLAMA_URL}
```

---

## 🎨 Recommended Configurations

### For Developers (Testing Docker Locally)
```yaml
# Minimal, fast iteration
volumes:
  - ./data:/workspace/data
  - ./src:/workspace/src
network_mode: "host"
restart: "no"
```

### For End Users (One-Command Setup)
```yaml
# Simple, works out of box
volumes:
  - equilens_data:/workspace/data
network_mode: "host"
restart: unless-stopped
```

### For Production (Deployed/Shipped)
```yaml
# Secure, monitored
image: vkrishna04/equilens:latest
restart: always
deploy:
  resources:
    limits:
      memory: 4G
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```
