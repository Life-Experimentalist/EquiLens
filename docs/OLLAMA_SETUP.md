# EquiLens Ollama Setup Guide

## Overview
This guide will help you set up Ollama to work with EquiLens for bias detection using your RTX 2050 GPU acceleration.

## Setup Method: Host Network Mode

We've configured the dev container to use **host network mode**, which allows direct access to services running on your Windows host. This is the most reliable approach for your setup.

## Step-by-Step Setup

### 1. Install Ollama on Windows Host

1. **Download Ollama**: Go to [https://ollama.ai/download](https://ollama.ai/download)
2. **Install**: Run the installer with Administrator privileges
3. **Verify**: Open PowerShell and run `ollama --version`

### 2. Start Ollama Server

**Option A: Using our script**
```powershell
# In PowerShell as Administrator
cd /workspace
./tools/start_ollama_windows.ps1
```

**Option B: Manual start**
```powershell
# In PowerShell as Administrator
ollama serve
```

The server will start on `http://localhost:11434` and be accessible to your dev container.

### 3. Download Models

**Option A: Using our script**
```powershell
# In another PowerShell window (while ollama serve is running)
cd /workspace
./tools/download_models.ps1
```

**Option B: Manual download**
```powershell
# Download recommended model for bias detection
ollama pull phi3:mini

# Verify
ollama list
```

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
