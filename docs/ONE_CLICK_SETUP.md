# 🚀 EquiLens One-Click Setup

**One-liner installation commands for EquiLens on Windows**

## 🎯 Quick Install Commands

### PowerShell (Recommended)
```powershell
irm https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1 | iex
```

### Command Prompt (cmd.exe)
```cmd
curl -fsSL https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.bat | cmd
```

### Git Bash / WSL
```bash
curl -fsSL https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1 | powershell -ExecutionPolicy Bypass -
```

### Universal (works in most Windows shells)
```powershell
powershell -ExecutionPolicy Bypass -Command "irm 'https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1' | iex"
```

## 📋 What the Setup Does

✅ **Prerequisites Check**
- Windows version compatibility
- Python 3.11+ availability
- Git installation
- Disk space (2GB+) and memory (4GB+)

✅ **Environment Setup**
- Installs UV package manager (if needed)
- Clones EquiLens repository
- Creates virtual environment
- Installs all dependencies

✅ **Services & Models**
- Starts Docker services (Ollama)
- Downloads Phi-3 model for testing
- Runs verification checks

✅ **Ready to Use**
- Provides next steps and commands
- Creates desktop shortcuts (optional)
- Opens documentation

## 🎛️ Setup Options

### Custom Install Path
```powershell
irm https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1 | iex -InstallPath "D:\Projects\EquiLens"
```

### Skip Docker Setup
```powershell
irm https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1 | iex -SkipDocker
```

### Skip Model Downloads
```powershell
irm https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1 | iex -SkipModels
```

### Specify Python Version
```powershell
irm https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1 | iex -PythonVersion "3.12"
```

## 🔧 Manual Installation

If the one-liner doesn't work:

1. **Download the setup script:**
   ```powershell
   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Life-Experimentalists/EquiLens/main/setup.ps1" -OutFile "setup.ps1"
   ```

2. **Run the setup:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

## 📊 System Requirements

- **OS**: Windows 10/11
- **Python**: 3.11 or newer
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Disk**: 2GB free space (additional for models)
- **Docker**: Desktop or Engine (optional but recommended)

## 🚨 Troubleshooting

### PowerShell Execution Policy
If you get execution policy errors:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Network/Download Issues
- Ensure internet connection
- Check firewall/proxy settings
- Try using a different command from the list above

### Docker Not Available
The setup will continue but you'll need to:
1. Install Docker Desktop
2. Run: `docker-compose up -d ollama`
3. Download models: `docker-compose exec ollama ollama pull phi3`

### Python Not Found
Install Python from: https://python.org/downloads/

## 🎯 After Installation

Once setup completes, you can:

```powershell
# Check status
uv run equilens status

# Start web interface
uv run equilens gui

# Run bias audit
uv run equilens audit --help

# View documentation
uv run equilens --help
```

## 📚 Documentation

- **Setup Guide**: `SETUP_COMPLETE.md`
- **Quick Start**: `docs/QUICKSTART.md`
- **Full Documentation**: `docs/` directory

## 🆘 Support

- **GitHub Issues**: https://github.com/Life-Experimentalists/EquiLens/issues
- **Documentation**: https://github.com/Life-Experimentalists/EquiLens/docs

---

**Happy Bias Hunting! 🔍**