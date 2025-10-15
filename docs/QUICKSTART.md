# 🚀 EquiLens Quick Start Guide

**Get started with EquiLens AI bias detection - from installation to first audit in under 10 minutes!**

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.11 or newer (3.13 recommended)
- **RAM**: 4GB (8GB+ recommended for better performance)
- **Disk Space**: 2GB free space (additional space for models)
- **Docker**: Latest version with compose support

### Recommended for Optimal Performance
- **RAM**: 16GB+ for large model processing
- **GPU**: NVIDIA GPU with 4GB+ VRAM for acceleration
- **CUDA**: Latest CUDA drivers for GPU support
- **SSD**: For faster model loading and data processing

### Platform Compatibility
- ✅ **Windows 10/11** (WSL2 recommended for Docker)
- ✅ **macOS** (Intel and Apple Silicon supported)
- ✅ **Linux** (Ubuntu 20.04+, Fedora, Arch, CentOS)
- ✅ **Docker Desktop** or **Docker Engine**

## ⚡ Installation

### Option 1: UV Package Manager (Recommended)

```bash
# Clone the repository
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens

# Install UV (fast Python package manager)
pip install uv

# Create virtual environment and install dependencies
uv sync

# Verify installation
python verify_setup.py
```

**Sample verification output:**
```
🔍 EquiLens System Verification

✓ Python 3.13.x detected
✓ Required packages installed
✓ Directory structure validated
✓ Docker available
✓ System resources sufficient (8GB RAM, 15GB disk)
✓ UV package manager ready
✓ Virtual environment configured

🎉 EquiLens is ready for use!

💡 Next steps:
   1. uv run equilens status    # Check system status
   2. uv run equilens start     # Start Ollama services
   3. uv run equilens tui       # Launch interactive interface
```

### Option 2: Traditional Pip Installation

```bash
# Clone the repository
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -e .

# Verify installation
python verify_setup.py
```

## 🐳 Docker Setup

### Install Docker
- **Windows/macOS**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Install Docker Engine and Docker Compose

### Start Services
```bash
# Start Ollama services
docker compose up -d

# Verify services are running
docker compose ps
```

## 🎯 GPU Acceleration (Optional)

### NVIDIA GPU Support

1. **Install NVIDIA Drivers**
   ```bash
   # Check if GPU is detected
   nvidia-smi
   ```

2. **Install CUDA Toolkit** (if not already installed)
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

3. **Install Docker GPU Support**
   ```bash
   # Install nvidia-docker2 (Linux)
   sudo apt-get install nvidia-docker2
   sudo systemctl restart docker
   ```

**Performance Benefits:**
- 🚀 **5-10x faster** model inference with GPU
- ⚡ Automatic GPU detection and configuration
- 🔄 Graceful fallback to CPU-only mode

### 3. Start Your First Audit
```bash
# Interactive audit with auto-discovery
uv run equilens audit

# You'll see a beautiful interface like:
╭─────────────────────────────────────────────────────────╮
│ Step 1: Model Selection                                 │
│ Select or specify the language model to audit for bias. │
╰─────────────────────────────────────────────────────────╯

✓ Found available models:
  1. llama2:latest

Select model number or enter custom name: 1
```
- 🎨 Present beautiful selection panels
- 📊 Guide you through the complete workflow
- 📁 Save results in organized session directories

---

## 🎯 Alternative: One-Command Experience

```bash
# 🚀 Everything in one command
uv run python src/equilens/cli.py && echo "🎉 Bias detection complete!"
```

---

## 🔧 Manual Execution (Advanced)

If you prefer step-by-step control:

### Phase 1: Generate Test Corpus (Optional)
```bash
uv run python src/Phase1_CorpusGenerator/generate_corpus.py
```

### Phase 2: Run Bias Audit
```bash
uv run python src/Phase2_ModelAuditor/audit_model.py \
  --model phi3:mini \
  --corpus src/Phase1_CorpusGenerator/corpus/audit_corpus_gender_bias.csv
```

### Phase 3: Analyze Results
```bash
uv run python src/Phase3_Analysis/analyze_results.py \
  --results_file results/results_phi3_mini_*.csv
```

---

## 🎮 GPU Acceleration Setup

### Check GPU Support
```bash
# 🔍 Check if NVIDIA GPU is available
nvidia-smi

# 🐳 Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Enable GPU in EquiLens
```bash
# 🚀 Start with GPU support
docker compose -f docker-compose.gpu.yml up -d

# ✅ The interactive CLI will automatically detect and use GPU acceleration
uv run python src/equilens/cli.py
```

**GPU Benefits:**
- ⚡ **5-10x faster** model inference
- 🚀 Automatic detection and configuration
- 🔄 Graceful fallback to CPU if GPU unavailable

---

## 📊 Understanding Your Results

After running EquiLens, you'll find organized results:

```
results/
└── 📁 phi3_mini_20250808_123456/          # Session directory
    ├── 📊 results_phi3_mini_*.csv         # Raw audit data
    ├── 📈 bias_report.png                 # Bias visualization
    ├── 📝 summary_*.json                  # Session summary
    ├── 📋 progress_*.json                 # Progress tracking
    └── 📜 session.log                     # Execution log
```

### 📈 Key Files Explained

- **`bias_report.png`** - Visual bias analysis chart
- **`results_*.csv`** - Detailed model responses and bias scores
- **`summary_*.json`** - High-level bias detection summary
- **`session.log`** - Complete execution log for debugging

---

## 🚨 Troubleshooting

### Common Issues & Solutions

| Problem                 | Quick Fix                          |
| ----------------------- | ---------------------------------- |
| **Docker not found**    | Install Docker Desktop and restart |
| **Python not found**    | Install Python 3.13+               |
| **uv not found**        | Run `pip install uv`               |
| **Permission denied**   | Run with admin/sudo privileges     |
| **Port already in use** | Stop other Ollama instances        |

### Quick Diagnostics
```bash
# 🔍 Check all dependencies
python --version      # Should be 3.13+
docker --version      # Should show Docker version
uv --version          # Should show uv version

# 🐳 Check Docker services
docker compose ps     # Should show running services

# 🤖 Test Ollama connectivity
curl http://localhost:11434/api/tags
```

### Reset Everything
```bash
# 🔄 Complete reset if needed
docker compose down
docker compose up -d
uv run python src/equilens/cli.py
```

---

## 🎯 Next Steps

### 1. 🎮 Try Different Models
```bash
# 🤖 List available models
curl http://localhost:11434/api/tags

# 📥 Download new models
docker exec -it equilens-ollama-1 ollama pull llama2
docker exec -it equilens-ollama-1 ollama pull mistral
```

### 2. 📝 Create Custom Test Corpus
Edit `src/Phase1_CorpusGenerator/word_lists.json` to add your own bias categories and test scenarios.

### 3. 📊 Advanced Analysis
Explore the detailed CSV results for custom analysis and integration with your existing tools.

### 4. 🔧 Configuration
Check out [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for advanced customization options.

---

## 🎉 Success Indicators

You know EquiLens is working correctly when:

- ✅ Interactive CLI launches with rich panels
- ✅ Auto-discovery finds corpus files and models
- ✅ Audit completes without errors
- ✅ Bias report is generated with visualizations
- ✅ All files are saved in session directory

---

## � Learn More

- **📖 [PIPELINE.md](PIPELINE.md)** - Complete workflow guide
- **📖 [ARCHITECTURE.md](ARCHITECTURE.md)** - System design details
- **📖 [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Advanced usage
- **📖 [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Customization options

---

**Ready to start?** Run `uv run python src/equilens/cli.py` and begin your AI bias detection journey! 🚀

All setup scripts are now in the `tools/` directory:
- **`tools/start_ollama_windows.ps1`** - Start Ollama on Windows host with GPU support
- **`tools/download_models.ps1`** - Interactive model downloader for Windows
- **`tools/start_ollama.sh`** - Verify Ollama connectivity from dev container
- **`tools/equilens_cli.py`** - Unified EquiLens command-line interface
- **`tools/health_check.py`** - Comprehensive system health verification

## 🔧 EquiLens CLI
After container rebuild, you'll have access to the unified CLI:
```bash
uv run equilens status       # Check system status
uv run equilens validate     # Validate configuration
uv run equilens setup        # Interactive setup
uv run equilens audit        # Run bias audit
uv run equilens analyze      # Analyze results
uv run equilens models       # Manage models
uv run equilens --help       # Show detailed help
```

## 🤖 Automatic Model Download
- **First Run**: Downloads 3 essential models (~3-5GB)
  - `phi3:mini` - Fast, CPU/GPU efficient (2GB)
  - `llama3.2:1b` - Balanced performance (1GB)
  - `gemma2:2b` - Best for bias detection (2GB, GPU only)
- **Subsequent Runs**: Models already available instantly!
- **Persistent Storage**: Models saved forever in Docker volume

## 📊 Recommended Models
- `phi3:mini` - Fast, efficient (CPU/GPU, 2GB)
- `llama3.2:1b` - Balanced performance (CPU/GPU, 1GB)
- `gemma2:2b` - Best for bias detection (CPU/GPU, 2GB)
- `llama3.2:3b` - More capable (better with GPU, 3GB)

## 🔧 Troubleshooting
- **GPU not detected**: System falls back to CPU automatically
- **Ollama not starting**: Check Docker daemon
- **Models not downloading**: Check internet connection
- **Resume interrupted audit**: Use `--resume progress_*.json`

## 💾 Persistent Storage
Models are stored in Docker volume `ollama_data` - survive container rebuilds!

## 🖥️ Hardware Requirements
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB+ RAM, NVIDIA GPU (any model)
- **Optimal**: 16GB+ RAM, modern NVIDIA GPU

# Quickstart — EquiLens

Short, minimal steps to get EquiLens running using the project's preferred `uv` entrypoint.

Prerequisites
- Docker (optional for Ollama / GPU)
- Windows: PowerShell (this project uses PowerShell examples)

1) Create / refresh virtual environment and install deps
```powershell
uv venv        # create or activate the project's virtual environment
uv sync        # install or synchronize dependencies from pyproject.toml
```

2) Launch interactive auditor (recommended)
```powershell
uv run equilens
```

3) Run specific phases manually (non-interactive)
```powershell
# Phase 1: generate corpus
uv run equilens generate --config configs/corpus_config.json

# Phase 2: run audit
uv run equilens audit --model llama2:latest --corpus Phase1_CorpusGenerator/corpus/my_corpus.csv

# Phase 3: analyze results
uv run equilens analyze --results-file results/*/results_*.csv
```

4) Export / reuse configuration
```powershell
uv run equilens export-config --file my_experiment.json
uv run equilens load-config --file my_experiment.json
```

If you prefer containers, start services before running the CLI:
```powershell
docker compose up -d
uv run equilens status
```

This quickstart intentionally uses `uv run equilens` as the single, canonical entrypoint for reproducibility.
