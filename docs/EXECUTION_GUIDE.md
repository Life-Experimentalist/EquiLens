# 🐳 EquiLens Execution Architecture

## Overview

EquiLens uses a **hybrid execution model** that combines host-based CLI management with containerized processing for optimal user experience and platform independence.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Host Machine  │───▶│   Docker Engine  │───▶│   Containers    │
│                 │    │                  │    │                 │
│ equilens.py     │    │ docker-compose   │    │ • Ollama        │
│ equilens.bat    │    │                  │    │ • EquiLens App  │
│ equilens.sh     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Execution Methods

### Method 1: Direct Python (Recommended)
```bash
# Cross-platform
python equilens.py start
python equilens.py models pull llama3.2
python equilens.py audit config.json
```

### Method 2: Platform Launchers
```bash
# Windows (double-click or command line)
equilens.bat start

# Linux/macOS
./equilens.sh start
```

### Method 3: Container Exec (Advanced)
```bash
# Direct container execution
docker compose exec equilens-app python Phase2_ModelAuditor/audit_model.py config.json
```

## 🔧 How It Works

### 1. **Host CLI Management**
- The `equilens.py` script runs on your **host machine**
- Requires only Python 3.8+ and Docker Desktop
- Handles service orchestration, model management, and user interaction
- **Shebang (`#!/usr/bin/env python3`) is for Unix systems** - ignored on Windows

### 2. **Container Processing**
- Heavy processing happens **inside containers**
- Ollama models are downloaded and stored in Docker volumes
- Python dependencies are isolated in containers
- Results are saved to mounted host directories

### 3. **Smart Ollama Detection**
- Automatically detects existing Ollama containers
- Uses external Ollama if available and accessible
- Creates new Ollama container only if needed
- Preserves model downloads across restarts

## 📁 File Execution Context

| File                       | Execution Location | Purpose             |
| -------------------------- | ------------------ | ------------------- |
| `equilens.py`              | **Host Machine**   | Main CLI interface  |
| `equilens.bat`             | **Host Machine**   | Windows launcher    |
| `equilens.sh`              | **Host Machine**   | Unix/Linux launcher |
| `Phase1_CorpusGenerator/*` | **Container**      | Corpus generation   |
| `Phase2_ModelAuditor/*`    | **Container**      | Bias auditing       |
| `Phase3_Analysis/*`        | **Container**      | Result analysis     |

## 🎯 Benefits of This Architecture

### ✅ **User Experience**
- Single command to start everything: `python equilens.py start`
- No need to understand Docker commands
- Platform independent (Windows, Linux, macOS)
- Models persist across restarts

### ✅ **Developer Experience**
- Clean separation of concerns
- Easy debugging and development
- Minimal host dependencies
- Container isolation for processing

### ✅ **Performance**
- Fast CLI operations (run on host)
- Heavy processing isolated in containers
- Efficient model sharing between runs
- GPU acceleration support (NVIDIA)

## 🛠 Dependencies

### Host Requirements
```
- Python 3.8+
- Docker Desktop
- requests library (auto-installed)
```

### Container Requirements
```
- All dependencies handled automatically
- No manual installation needed
- Models downloaded on-demand
```

## 🔄 Model Persistence

Models are stored in a **Docker volume** (`ollama_data`) that persists across container restarts:

```yaml
# In docker-compose.yml
volumes:
  - ./ollama_data:/root/.ollama
```

This means:
- ✅ Models survive container restarts
- ✅ No re-downloading after `docker compose down`
- ✅ Shared across multiple EquiLens runs
- ✅ Can be backed up by copying `ollama_data/` folder

## 🐛 Troubleshooting

### Python Not Found
```bash
# Windows
winget install Python.Python.3.12

# Linux (Ubuntu)
sudo apt update && sudo apt install python3

# macOS
brew install python@3.12
```

### Docker Not Running
```bash
# Check Docker status
docker --version
docker ps

# Start Docker Desktop if needed
```

### Container Communication Issues
```bash
# Check container network
python equilens.py status

# Reset if needed
python equilens.py stop
python equilens.py start
```

---

**💡 Pro Tip**: The shebang `#!/usr/bin/env python3` allows Unix systems to run `./equilens.py` directly, but on Windows you'll use `python equilens.py` or `equilens.bat`.
