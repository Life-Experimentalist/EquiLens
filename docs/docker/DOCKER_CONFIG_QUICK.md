# Docker Configuration Parameters - Quick Reference

## 🎛️ Essential Parameters You Need to Know

### **1. Ports** (Change if conflicts)
```yaml
ports:
  - "7860:7860"  # Gradio UI - Change left side: "8080:7860"
  - "8000:8000"  # API - Change left side: "9000:8000"
```

### **2. Container Name** (Change for multiple instances)
```yaml
container_name: equilens-app  # → equilens-prod, equilens-dev, etc.
```

### **3. Volume** (ONE volume is sufficient!)
```yaml
volumes:
  equilens_data:/workspace/data  # Contains results, logs, corpus
```

**Inside container:**
- `/workspace/data/results` - Audit results
- `/workspace/data/logs` - Logs
- `/workspace/data/corpus` - Corpus files

**Options:**
```yaml
# Option A: Custom name
name: my-equilens-data

# Option B: Use host directory (dev)
volumes:
  - ./data:/workspace/data

# Option C: External volume (reuse existing)
external: true
name: existing-volume
```

### **4. Ollama Connection**
```yaml
OLLAMA_BASE_URL=http://localhost:11434  # Default for desktop app/host mode
```

**Options:**
```yaml
# For Ollama in another container
OLLAMA_BASE_URL=http://ollama-gpu:11434

# For remote Ollama
OLLAMA_BASE_URL=http://192.168.1.100:11434
```

### **5. Network Mode**
```yaml
network_mode: "host"  # Easy Ollama connection (RECOMMENDED)
```

**Alternative (more isolated):**
```yaml
# Remove network_mode, use bridge:
networks:
  - equilens-network
```

### **6. Restart Policy**
```yaml
restart: unless-stopped  # Auto-restart except manual stop
```

Options: `no`, `always`, `on-failure`, `unless-stopped`

### **7. Gradio UI Theme**
```yaml
GRADIO_THEME=default  # or: soft, glass, monochrome
```

---

## 📋 Configuration Templates

### Template 1: Development (Testing Docker Locally)
```yaml
services:
  equilens:
    build: .
    image: equilens:dev
    container_name: equilens-dev
    ports:
      - "7860:7860"
    volumes:
      - ./data:/workspace/data  # Local directory for easy access
      - ./src:/workspace/src
    network_mode: "host"
    restart: "no"  # Don't auto-restart during dev
```

### Template 2: End User (One-Command Setup)
```yaml
services:
  equilens:
    image: equilens:latest
    container_name: equilens-app
    ports:
      - "7860:7860"
      - "8000:8000"
    volumes:
      - equilens_data:/workspace/data  # Named volume
    network_mode: "host"
    restart: unless-stopped
    environment:
      - OLLAMA_BASE_URL=http://localhost:11434
```

### Template 3: Production (With Ollama Container)
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
    image: vkrishna04/equilens:latest
    container_name: equilens-prod
    depends_on:
      - ollama
    ports:
      - "7860:7860"
    volumes:
      - equilens_data:/workspace/data
    restart: always
    environment:
      - OLLAMA_BASE_URL=http://ollama-gpu:11434
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

volumes:
  ollama-models:
  equilens_data:
```

---

## 🔧 How to Customize

### Method 1: Edit docker-compose.yml
```powershell
# Edit file
notepad docker-compose.yml

# Apply
docker-compose down
docker-compose up -d
```

### Method 2: Override File (RECOMMENDED)
```powershell
# Create override
@"
services:
  equilens:
    ports:
      - "8080:7860"
    environment:
      - GRADIO_THEME=soft
"@ | Out-File docker-compose.override.yml

# Automatically applied
docker-compose up -d
```

### Method 3: Environment Variables
```powershell
# Create .env
@"
GRADIO_PORT=7860
OLLAMA_URL=http://localhost:11434
"@ | Out-File .env

# Reference in docker-compose.yml
# ${GRADIO_PORT}, ${OLLAMA_URL}
```

---

## ⚙️ Most Common Customizations

### Change Ports (Avoid Conflicts)
```yaml
ports:
  - "8080:7860"  # Access UI at http://localhost:8080
  - "9000:8000"  # Access API at http://localhost:9000
```

### Multiple Instances
```yaml
services:
  equilens-prod:
    container_name: equilens-prod
    ports:
      - "7860:7860"
    volumes:
      - equilens_prod:/workspace/data

  equilens-dev:
    container_name: equilens-dev
    ports:
      - "7861:7860"
    volumes:
      - equilens_dev:/workspace/data
```

### Use Host Directory (Easy File Access)
```yaml
volumes:
  - V:\Code\ProjectCode\EquiLens\data:/workspace/data
```

### Different Ollama Location
```yaml
environment:
  - OLLAMA_BASE_URL=http://192.168.1.100:11434
```

---

## 🎨 UI Themes
```yaml
GRADIO_THEME=default  # Clean, modern (default)
GRADIO_THEME=soft     # Soft colors, rounded
GRADIO_THEME=glass    # Glassmorphism style
GRADIO_THEME=monochrome  # Black and white
```

---

## 📊 Resource Limits (Optional)
```yaml
deploy:
  resources:
    limits:
      memory: 4G     # Max 4GB RAM
      cpus: '2.0'    # Max 2 CPU cores
    reservations:
      memory: 2G     # Reserve 2GB
      cpus: '1.0'    # Reserve 1 core
```

---

## 🚀 Quick Start Examples

### Run with Custom Port
```powershell
# Edit docker-compose.yml
ports:
  - "8080:7860"

# Start
docker-compose up -d

# Access at http://localhost:8080
```

### Run with Local Data Directory
```powershell
# Edit docker-compose.yml
volumes:
  - ./data:/workspace/data

# Start
docker-compose up -d

# Data in: .\data\results, .\data\logs
```

### Run Multiple Instances
```powershell
# Copy and modify docker-compose.yml
docker-compose -f docker-compose-prod.yml up -d
docker-compose -f docker-compose-dev.yml up -d
```

---

## 📝 Summary

**ONE volume is enough** → `equilens_data:/workspace/data`

**Key parameters to customize:**
1. **Ports** - Avoid conflicts
2. **Container name** - Multiple instances
3. **Ollama URL** - Where is Ollama?
4. **Volume** - Where to store data?
5. **Theme** - UI appearance

**Most users won't need to change anything!** Default config works great.

**See full guide:** `docs/DOCKER_CONFIG_GUIDE.md`
