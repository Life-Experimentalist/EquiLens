# 🛠️ EquiLens Tools Directory

**✨ Development and Testing Utilities**

This directory contains essential utility tools for development and testing. All CLI functionality is now available through the modern UV-based EquiLens CLI.

## 📁 Current Tools

### ⚙️ Configuration & Setup
- **`quick_setup.py`** - Interactive bias configuration creator with templates
- **`validate_config.py`** - JSON schema validation for bias configurations

### 🧪 Development & Testing Tools
- **`mock_ollama.py`** - Mock Ollama server for development and testing


## 🚀 EquiLens CLI Usage

### Modern UV-Based CLI (Recommended)
```bash
# Start everything (auto-detects existing Ollama)
uv run equilens start

# Check service status
uv run equilens status

# Download models
uv run equilens models pull phi3:mini

# Run bias audit
uv run equilens audit

# Generate corpus
uv run equilens generate

# Analyze results
uv run equilens analyze results.csv

# Interactive terminal UI
uv run equilens tui

# GPU acceleration check
uv run equilens gpu-check
```

### Development Tools Only
```bash
# Create new bias configuration interactively
python tools/quick_setup.py

# Validate configuration schema
python tools/validate_config.py config.json

# Run mock Ollama for testing
python tools/mock_ollama.py
```

## 📋 Dependencies & Environment

### Host Requirements (One-Time Setup)
- **Python 3.8+** - For running the CLI
- **Docker Desktop** - With Compose V2 support
- **uv** - Fast Python package manager (`pip install uv`)

### Virtual Environment Setup
```bash
# Create isolated environment with uv (recommended)
uv venv                    # Creates .venv directory
uv pip sync requirements.txt   # Installs all dependencies

# Alternative: Standard pip
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Development Tools
- **Python 3.8+** for configuration creation
- **jsonschema** library for validation (installed automatically)
- **No platform-specific dependencies**

## 🎯 Benefits of Consolidation

### ✅ **Simplified User Experience**
- Single entry point: `uv run equilens <command>`
- Platform independence built-in
- Modern UV package management
- Consistent command structure across all platforms

### ✅ **Reduced Maintenance**
- One codebase to maintain
- No duplicate functionality
- Easier testing and debugging
- Clear separation of concerns

### ✅ **Smart Container Integration**
- Auto-detects existing Ollama containers
- Seamless container communication
- Persistent model storage
- GPU acceleration when available

## 📊 Model Management

The tools support these recommended models for bias detection:
> **Note**: Any Model in Ollama is technically supported, but these are optimized for performance and quality.

| Model          | Size  | Speed     | Quality | Best For               |
| -------------- | ----- | --------- | ------- | ---------------------- |
| `phi3:mini`    | 3.8GB | Fast      | High    | General bias detection |
| `llama3.2:1b`  | 1.3GB | Very Fast | Good    | Quick testing          |
| `gemma2:2b`    | 1.6GB | Fast      | High    | Reasoning tasks        |
| `qwen2.5:1.5b` | 1.5GB | Fast      | Good    | Multilingual bias      |


## 🐛 Troubleshooting

### Quick Diagnostics
```bash
# Check everything at once
python equilens.py status

# Test Docker availability
docker --version && docker compose version

# Test Python environment
python --version && python -c "import requests; print('✅ Python OK')"
```

### Common Issues & Solutions

| Issue                    | Solution                                   |
| ------------------------ | ------------------------------------------ |
| `Docker not found`       | Install Docker Desktop                     |
| `Python not found`       | Install Python 3.8+                        |
| `Ollama not responding`  | Run `python equilens.py start`             |
| `Models not downloading` | Check internet connection                  |
| `Container won't start`  | Run `python equilens.py stop` then `start` |

### Development & Testing
```bash
# Test with mock Ollama (for development)
python tools/mock_ollama.py

# Create test configuration
python tools/quick_setup.py

# Validate configuration
python tools/validate_config.py test_config.json
```

## � Getting Started

### Quick Setup (First Time)
```bash
# 1. Install uv (fast Python package manager)
pip install uv

# 2. Create virtual environment and install dependencies
uv venv
uv pip sync requirements.txt

# 3. Activate environment and start EquiLens
# Windows
.venv\Scripts\activate
python equilens.py start

# Linux/macOS
source .venv/bin/activate
python equilens.py start
```

### Daily Usage (After Setup)
```bash
# Activate environment
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS

# Use EquiLens
python equilens.py status
python equilens.py models pull phi3:mini
python equilens.py audit config.json
```
