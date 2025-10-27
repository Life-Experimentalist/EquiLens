# Smart Ollama Config - Quick Reference

## 🚀 Quick Start

```bash
# Just run - it auto-configures!
docker compose up -d           # Docker
uv run equilens gui            # Local
```

## 📊 Deployment Scenarios

| Your Setup | Auto-Detected URL |
|------------|-------------------|
| Both in Docker 🐳🐳 | `http://ollama:11434` |
| EquiLens Docker, Ollama Host 🐳💻 | `http://host.docker.internal:11434` |
| EquiLens Local, Ollama Docker 💻🐳 | `http://localhost:11434` |
| Both Local 💻💻 | `http://localhost:11434` |

## 🔧 Manual Override (Optional)

```bash
# Environment variable
export OLLAMA_BASE_URL=http://custom:11434

# Docker Compose
environment:
  - OLLAMA_BASE_URL=http://custom:11434
```

## 🧪 Test Your Setup

```bash
# Quick test
python scripts/tools/test_smart_ollama_config.py

# From inside container
docker exec -it equilens-app python scripts/tools/test_smart_ollama_config.py
```

## 💻 Python API

```python
from equilens.core.ollama_config import get_ollama_url, get_environment_info

# Get URL
url = get_ollama_url()

# Get environment details
env = get_environment_info()
print(f"Scenario: {env['scenario']}")
print(f"URL: {env['ollama_url']}")
```

## 🐛 Troubleshooting

### Connection Failed?
```bash
# Check Ollama is running
docker ps | grep ollama        # Docker
# OR check Docker Desktop UI

# Test all URLs
python scripts/tools/test_smart_ollama_config.py
```

### Force Re-detection?
```python
url = get_ollama_url(force_refresh=True)
```

### Set Explicit URL?
```bash
export OLLAMA_BASE_URL=http://your-url:11434
```

## 📁 Key Files

- **Module**: `src/equilens/core/ollama_config.py`
- **Docs**: `docs/docker/SMART_OLLAMA_CONFIG.md`
- **Test**: `scripts/tools/test_smart_ollama_config.py`
- **Summary**: `SMART_CONFIG_IMPLEMENTATION.md`

## ✅ Benefits

- ✨ **Zero configuration** - works everywhere
- 🔄 **Auto-adapts** to environment changes
- 🚀 **Fast** - cached after first detection
- 🛡️ **Resilient** - smart fallback chain
- 🔧 **Flexible** - easy override if needed

---

**Need help?** See `docs/docker/SMART_OLLAMA_CONFIG.md` for full documentation
