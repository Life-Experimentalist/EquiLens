# 🤖 Ollama Setup & Troubleshooting Guide

**Complete guide to setting up and troubleshooting Ollama for EquiLens bias detection**

## 📋 Overview

This guide covers Ollama installation, configuration, and troubleshooting for EquiLens. Includes GPU acceleration setup, connection issue resolution, and concurrent processing optimization.

## 🚀 Installation & Setup

### 1. Install Ollama on Windows Host

1. **Download Ollama**: Go to [https://ollama.ai/download](https://ollama.ai/download) or from the docker hub [https://hub.docker.com/r/ollama/ollama](https://hub.docker.com/r/ollama/ollama).
2. **Install**: Run the installer with Administrator privileges
3. **Verify**: Open PowerShell and run `ollama --version`

### 2. Start Ollama Server

**Option A: Using EquiLens (Recommended)**
```bash
# Start Ollama services through EquiLens
uv run equilens start
```

**Option B: Manual start**
```bash
# In PowerShell as Administrator
ollama serve
```

The server will start on `http://localhost:11434` and be accessible to EquiLens.

### 3. Download Models

**Option A: Using EquiLens**
```bash
# Download recommended models
uv run equilens models pull phi3:mini
uv run equilens models pull llama3.2:1b

# List available models
uv run equilens models list
```

**Option B: Manual download**
```bash
# Download recommended model for bias detection
ollama pull phi3:mini

# Verify
ollama list
```

## 🔍 Understanding Connection Issues

### Why Do Connection Issues Occur?

**1. Resource Limitations**
- **Memory Constraints**: Large language models require significant RAM/VRAM
- **CPU/GPU Bottlenecks**: Models compete for computational resources
- **Disk I/O**: Model loading and context caching strain storage systems

**2. Network-Level Issues**
- **Port Conflicts**: Multiple processes trying to use the same ports
- **Timeout Settings**: Default timeouts may be too aggressive for large models
- **Connection Pooling**: Limited number of simultaneous connections

**3. Ollama Service Management**
- **Startup Time**: Large models take time to load into memory
- **Context Management**: Ollama manages conversation context which uses memory
- **Model Switching**: Loading different models requires resource reallocation

**4. Concurrent Request Challenges**
- **Queue Saturation**: Too many simultaneous requests overwhelm the service
- **Memory Fragmentation**: Multiple concurrent contexts fragment available memory
- **Lock Contention**: Internal Ollama locks can cause request blocking

## 🚀 Concurrent Processing Optimization

### ✅ Benefits of Concurrent Processing

**Performance Gains:**
- **3-5x Speed Improvement**: Multiple requests processed simultaneously
- **Better Resource Utilization**: CPU cores and memory bandwidth used efficiently
- **Reduced Wall-Clock Time**: Large corpus processing completes faster

**When It Works Best:**
- **Small Models** (phi3:mini, qwen2:0.5b): Handle concurrency well
- **Powerful Hardware**: 16GB+ RAM, modern CPUs
- **Fast Storage**: NVMe SSDs for model loading

### ⚠️ Risks & Challenges

**Resource Exhaustion:**
- **Memory Overflow**: Multiple model instances can exceed available RAM
- **GPU Memory Limits**: VRAM exhaustion causes crashes
- **Connection Timeouts**: Overwhelmed service drops connections

**Quality Issues:**
- **Inconsistent Responses**: Overloaded service may return lower quality results
- **Failed Requests**: Increased error rates under high load

## 🛠️ Network Configuration

### Host Network Mode Setup

We've configured EquiLens to use **host network mode** for direct access to Ollama services:

```yaml
# docker-compose.yml
services:
  ollama:
    network_mode: host
    ports:
      - "11434:11434"
```

This provides the most reliable approach for accessing Ollama from EquiLens containers.

### 4. Rebuild Dev Container

1. In VS Code, press `Ctrl+Shift+P`
2. Type "Dev Containers: Rebuild Container"
3. Select it and wait for rebuild

The rebuilt container will use host networking mode, allowing direct access to Ollama.

### 5. Test Connection

After rebuild, test the connection:

```bash
# In the dev container terminal
python tools/test_ollama_connection.py

# Or use the EquiLens CLI
equilens health
```

## GPU Acceleration

Your RTX 2050 will be automatically utilized by Ollama when running on the Windows host. Ollama detects and uses NVIDIA GPUs automatically.

## Configuration Details

### Docker Compose Changes
- **Network Mode**: `host` - allows direct access to host services
- **Extra Hosts**: Maps `host.docker.internal` for cross-platform compatibility
- **Removed**: Separate Ollama container (now runs on host)

### Code Changes
- **audit_model.py**: Enhanced with multi-host detection
- **health_check.py**: Updated to test multiple connection endpoints
- **CLI tools**: Added Ollama connectivity verification

## Troubleshooting

### Common Issues

**"Connection refused" errors:**
- Ensure Ollama server is running: `ollama serve`
- Check Windows Firewall isn't blocking port 11434
- Verify dev container rebuild completed

**"No models available" errors:**
- Download models: `ollama pull phi3:mini`
- Check model list: `ollama list`

**GPU not being used:**
- Ensure NVIDIA drivers are up to date
- Ollama automatically uses GPU when available on Windows
- Check task manager for GPU usage during model inference

### Verification Commands

```bash
# Test connection
curl http://localhost:11434/api/version

# Check available models
curl http://localhost:11434/api/tags

# Test generation
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3:mini", "prompt": "Hello", "stream": false}'
```

## Performance Optimization

### Recommended Models for Bias Detection
1. **phi3:mini** (3.8GB) - Best balance of speed and quality
2. **llama3.2:1b** (1.3GB) - Fastest for rapid testing
3. **gemma2:2b** (1.6GB) - Good reasoning capabilities

### RTX 2050 Optimization
- Models will automatically use GPU acceleration
- Expect 3-5x speedup compared to CPU-only inference
- Monitor GPU usage in Task Manager during operations

## File Locations

- **Windows PowerShell scripts**: `/workspace/tools/*.ps1`
- **Container verification script**: `/workspace/tools/start_ollama.sh`
- **Test script**: `/workspace/tools/test_ollama_connection.py`
- **Configuration**: `/workspace/.devcontainer/docker-compose.yml`
- **Model auditor**: `/workspace/Phase2_ModelAuditor/audit_model.py`

## Next Steps

Once setup is complete, you can:

1. **Run bias audits**: `equilens audit --model phi3:mini --config bias_config.json`
2. **Generate test corpus**: `equilens corpus --type gender`
3. **Analyze results**: `equilens analyze --input results.csv`

## Support

If you encounter issues:

1. Run `equilens health` for diagnostic information
2. Check the logs in `/workspace/logs/`
3. Ensure all prerequisites are met (Ollama installed, models downloaded, container rebuilt)

---

**Note**: This setup provides production-ready bias detection with GPU acceleration while maintaining persistent model storage and reliable connectivity.

# Ollama Setup

EquiLens uses Ollama as the primary LLM runtime. Ollama can run in Docker or locally.

Quick start (Docker)
```powershell
# Start Ollama container
docker compose up -d

# Pull a model (example)
curl -X POST http://localhost:11434/api/pull -d '{"name":"phi3:mini"}'
```

Verify
```powershell
curl http://localhost:11434/api/tags
```

Advanced
- Use GPU-enabled compose file for acceleration
- Configure model options via `--custom-ollama-options` or config files

Troubleshooting
- Check container logs: `docker compose logs -f equilens-ollama-1`
- Ensure Docker GPU runtime is enabled for GPU support
