# 🎯 Quick Start: Using EquiLens with UV Scripts

## ✅ What We Set Up

1. **VS Code Tasks** — 25+ pre-configured tasks in `.vscode/tasks.json`
2. **UV Scripts Guide** — Complete reference in `UV_SCRIPTS_GUIDE.md`
3. **Smart Commands** — All with proper error handling and output formatting

---

## 🚀 Getting Started (3 Steps)

### Step 1: Open Task Menu
Press **`Ctrl+Shift+B`** in VS Code

### Step 2: Pick a Task
- **"Audit: Run with Logprobs (Default)"** — Start a bias audit
- **"Analyze: Advanced with AI Insights"** — Generate full reports
- **"Services: Start (Docker)"** — Launch Ollama container

### Step 3: Watch the Output
Results appear in the integrated terminal, no setup needed!

---

## 📊 Most Common Tasks

### For Running Audits
```powershell
# Via VS Code Task: "Audit: Run with Logprobs (Default)"
# Or terminal:
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv
```

### For Generating Reports
```powershell
# Via VS Code Task: "Analyze: Advanced with AI Insights"
# Or terminal:
uv run equilens analyze --advanced --use-ai
```

### For Development
```powershell
# Via VS Code Task: "Dev: Full Check (Lint + Type)"
# Or terminal:
uv run ruff check src/ && uv run mypy src/
```

### For Services
```powershell
# Start Docker services
uv run equilens start

# Check status
uv run equilens status

# Stop services
uv run equilens stop
```

---

## 🎲 All Available Tasks (By Category)

### 📋 Information
- `Help: Show EquiLens Commands`

### 🔍 Audit Group
- `Audit: Run with Logprobs (Default)` ⭐
- `Audit: Run with Timing Fallback`
- `Audit: Resume Session`

### 📈 Analysis Group
- `Analyze: Generate Reports` ⭐
- `Analyze: Advanced with AI Insights` ⭐

### 📝 Generation
- `Generate: Create Test Corpus`

### 🐳 Services
- `Services: Check Status`
- `Services: Start (Docker)`
- `Services: Stop (Docker)`

### 🛠️ Utilities
- `Utilities: GPU Check`
- `Utilities: List Models`

### 🧪 Development
- `Dev: Run Tests`
- `Dev: Lint Code (Ruff)`
- `Dev: Format Code (Ruff)`
- `Dev: Type Check (MyPy)`
- `Dev: Full Check (Lint + Type)` ⭐

### 🖥️ Web UI
- `Web: Launch Gradio UI`
- `Web: Start Backend API`
- `Web: Full Stack (API + UI)`

---

## 💡 Pro Tips

### Tip 1: Direct Commands
You can also run commands directly in the terminal (same as VS Code task):
```powershell
uv run <command>
```

### Tip 2: Logprobs Default
All audits use **logprobs** by default (requires Ollama >= 0.12.11).
Reports automatically show which method was used.

### Tip 3: Silent Mode
If you see Unicode errors in Windows:
```powershell
uv run equilens audit run --model ... --corpus ... --silent
```

### Tip 4: Resume Sessions
If an audit is interrupted:
```powershell
uv run equilens audit resume
```

### Tip 5: Compare Methods
Run audit twice — once with logprobs, once with `--no-logprobs` — then compare the reports!

---

## 📚 Full Documentation

- **This File**: `UV_SCRIPTS_GUIDE.md` (47+ recipes)
- **Code Files**: `pyproject.toml`, `.vscode/tasks.json`
- **Project Docs**: `docs/README.md`

---

## 🔗 Related Files

| File                  | Purpose                                 |
| --------------------- | --------------------------------------- |
| `.vscode/tasks.json`  | VS Code task definitions                |
| `UV_SCRIPTS_GUIDE.md` | Complete command reference              |
| `pyproject.toml`      | Project config (dependencies, metadata) |
| `src/equilens/cli.py` | CLI entry point                         |

---

## ❓ Troubleshooting

### "Command not found"
Make sure you're in the EquiLens directory:
```powershell
cd V:\Code\ProjectCode\EquiLens
uv run equilens --help
```

### "Docker not running"
Start Docker Desktop or use:
```powershell
uv run equilens start
```

### "Ollama not accessible"
Ollama API needs to be running on port 11434:
```powershell
uv run equilens status  # Check status
uv run equilens start   # Start Docker container
```

### "Unicode encoding errors"
Add `--silent` flag to suppress fancy output:
```powershell
uv run equilens audit run --model ... --corpus ... --silent
```

---

## 🎓 Next Steps

1. **Try a task**: Press `Ctrl+Shift+B`, pick "Audit: Run with Logprobs"
2. **Check results**: Wait for audit to complete
3. **Analyze**: Run "Analyze: Advanced with AI Insights"
4. **Read reports**: Check `results/` directory for HTML/Markdown/PNG outputs

---

**You're all set!** 🚀

For detailed command options, see `UV_SCRIPTS_GUIDE.md`

Last Updated: March 5, 2026
