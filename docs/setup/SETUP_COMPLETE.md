# ✅ EquiLens Setup Complete!

## 🎉 Current Status

Your EquiLens installation is **97% complete**! Here's what's working:

### ✓ Working Components
- ✅ Python 3.13.3 installed and configured
- ✅ All required packages installed (typer, rich, textual, requests, attrs)
- ✅ Complete directory structure verified
- ✅ All configuration files present
- ✅ Docker available (version 28.4.0)
- ✅ NVIDIA GPU detected and ready
- ✅ Sufficient system resources (31.7 GB RAM, 30.3 GB disk)

### ⚠️ Pending Setup
- ⏳ Ollama service (needs Docker Desktop running)

---

## 🚀 Quick Start Guide

### 1. Start Docker Desktop
Before using EquiLens, start Docker Desktop:
- **Windows**: Start Docker Desktop from the Start menu
- Wait for Docker to fully initialize (whale icon in system tray)

### 2. Start Ollama Service
```powershell
# Start Ollama with GPU support
docker-compose up -d ollama

# Verify Ollama is running
docker-compose ps
```

### 3. Download AI Models
```powershell
# Download a small test model (2GB)
docker-compose exec ollama ollama pull phi3

# Or download a larger model (4GB)
docker-compose exec ollama ollama pull llama3.2
```

### 4. Verify Everything Works
```powershell
# Run verification again
uv run scripts/setup/verify_setup.py

# Should show 8/8 checks passed!
```

---

## 🎯 Using EquiLens

### Web Interface (Recommended)

```powershell
# Start the web UI
uv run equilens web

# Or with Gradio
uv run python -m equilens.gradio_ui
```
Then open your browser to **http://localhost:7860**

### Command Line Interface

```powershell
# View available commands
uv run equilens --help

# Check system status
uv run equilens status

# Generate bias corpus
uv run equilens generate-corpus --help

# Run bias audit
uv run equilens audit --help

# Analyze results
uv run equilens analyze --help
```

---

## 📊 Project Structure

```
EquiLens/
├── src/
│   ├── equilens/           # Main application package
│   │   ├── cli.py          # Command-line interface
│   │   ├── web_ui.py       # Web interface
│   │   ├── gradio_ui.py    # Gradio interface
│   │   ├── tui.py          # Text UI
│   │   └── core/           # Core functionality
│   ├── Phase1_CorpusGenerator/  # Generate test data
│   ├── Phase2_ModelAuditor/     # Audit AI models
│   └── Phase3_Analysis/         # Analyze results
├── results/                # Audit results storage
├── logs/                   # Application logs
├── docs/                   # Documentation
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

---

## 🔧 Common Commands

### Docker Management
```powershell
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f ollama

# Restart Ollama
docker-compose restart ollama
```

### Model Management
```powershell
# List downloaded models
docker-compose exec ollama ollama list

# Pull a new model
docker-compose exec ollama ollama pull <model-name>

# Remove a model
docker-compose exec ollama ollama rm <model-name>
```

### Development
```powershell
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black .

# Lint code
uv run ruff check .
```

---

## 🐛 Troubleshooting

### Ollama Won't Start
1. Ensure Docker Desktop is running
2. Check Docker has sufficient resources allocated:
   - Memory: 4GB+ recommended
   - Disk: 10GB+ free space
3. Restart Docker Desktop
4. Try: `docker-compose up ollama` (without -d to see logs)

### GPU Not Detected in Docker
```powershell
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
If this fails, ensure NVIDIA Container Toolkit is installed.

### Port Already in Use
```powershell
# Find process using port 11434
netstat -ano | findstr :11434

# Kill the process (replace <PID> with actual process ID)
taskkill /PID <PID> /F
```

### Package Installation Issues
```powershell
# Clear UV cache and reinstall
uv cache clean
uv sync --reinstall
```

---

## 📚 Next Steps

1. **Read the Documentation**
   - `docs/QUICKSTART.md` - Quick start guide
   - `docs/PIPELINE.md` - Understanding the bias detection pipeline
   - `docs/CONFIGURATION_GUIDE.md` - Configuration options

2. **Run Your First Audit**
   - Start with a small model (phi3)
   - Use the web interface for easier setup
   - Generate a small test corpus first

3. **Explore Advanced Features**
   - Custom word lists for bias detection
   - Multi-model comparison
   - Advanced analytics and visualizations

4. **Join the Community**
   - GitHub: https://github.com/Life-Experimentalist/EquiLens
   - Report issues and contribute improvements

---

## 🎓 Additional Resources

- **Website**: https://vkrishna04.github.io
- **Documentation**: `docs/` directory
- **Examples**: `scripts/tools/` for demo scripts
- **Citation**: See `CITATION.cff` for academic use

---

**Happy Bias Hunting! 🔍**

*EquiLens - Making AI Fair and Transparent*
