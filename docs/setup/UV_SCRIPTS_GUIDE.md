# 🚀 EquiLens UV Scripts & Commands

This guide shows all the `uv run` commands available in EquiLens (equivalent to `npm run` in JavaScript).

## 📋 Quick Reference

### Using VS Code Tasks (Recommended)
Press `Ctrl+Shift+B` to open the Task menu and select any task below.

### Using Terminal Commands
Run directly in PowerShell:

```powershell
uv run <command>
```

---

## 🔍 Main Commands

### Help & Information
```powershell
uv run equilens --help              # Show all EquiLens commands
uv run equilens status              # Check service status
uv run equilens gpu-check           # Check GPU/CUDA setup
uv run equilens models list         # List available Ollama models
```

---

## 📊 Audit Commands

### Run Audits
```powershell
# Default: uses logprobs (log-probability scoring) - RECOMMENDED
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv

# Explicitly enable logprobs (requires Ollama >= 0.12.11)
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv --logprobs

# Disable logprobs, use timing fallback
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv --no-logprobs

# Suppress Unicode output (Windows cmd fix)
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv --silent

# With custom workers/parallelization
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv --workers=4

# Resume interrupted session
uv run equilens audit resume
```

**Scoring Methods:**
- **`--logprobs`** (Default): Uses Ollama's log-probability API for true bias detection
- **`--no-logprobs`**: Falls back to timing-based scoring (eval_duration / eval_count)

---

## 📈 Analysis Commands

### Generate Reports
```powershell
# Standard analysis (single bias report)
uv run equilens analyze

# Advanced analysis (multiple charts, statistical tests)
uv run equilens analyze --advanced

# With AI-powered insights (requires Ollama)
uv run equilens analyze --use-ai

# Advanced + AI
uv run equilens analyze --advanced --use-ai

# Specify results file
uv run equilens analyze --results path/to/results.csv

# Verbose output
uv run equilens analyze --verbose
```

**Outputs Include:**
- Multiple visualizations (violin plots, heatmaps, effect sizes, etc.)
- HTML report with scoring method badge
- Markdown report with statistics
- JSON export with metadata
- **Scoring method info** (logprobs vs timing) in all outputs

---

## 📝 Generate Commands

### Create Test Corpus
```powershell
# Interactive mode
uv run equilens generate

# From config file
uv run equilens generate --config config/corpus_config.json

# Save to specific path
uv run equilens generate --output data/my_corpus.csv
```

---

## 🐳 Service Commands

### Docker Services
```powershell
# Start services (Ollama Docker container)
uv run equilens start

# Stop services
uv run equilens stop

# Check status
uv run equilens status

# View logs
uv run equilens status --verbose
```

---

## 🖥️ Web UI Commands

### Launch Interfaces
```powershell
# Launch Gradio web interface
uv run equilens web

# Start backend API server only
uv run equilens backend

# Start both backend + frontend together
uv run equilens serve

# Legacy Gradio GUI (older version)
uv run equilens gui
```

---

## 🧪 Development Commands

### Testing & Code Quality
```powershell
# Run tests with coverage
uv run pytest --cov=equilens

# Quick test (no coverage)
uv run pytest

# Lint code with Ruff
uv run ruff check src/ tests/

# Format code with Ruff
uv run ruff format src/ tests/

# Type check with MyPy
uv run mypy src/

# Full dev check (lint + type)
uv run ruff check src/ && uv run mypy src/
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Bias Audit
```powershell
# 1. Check if services are running
uv run equilens status

# 2. Run audit with default logprobs scoring
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv

# 3. Analyze results
uv run equilens analyze --advanced --use-ai
```

### Workflow 2: Compare Scoring Methods
```powershell
# Run with logprobs
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv --logprobs

# Analyze results
uv run equilens analyze --results results/results_phi3_TIMESTAMP.csv

# Then run with timing fallback
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv --no-logprobs

# Compare reports
```

### Workflow 3: Full Development & Testing
```powershell
# 1. Format code
uv run ruff format src/

# 2. Lint & type check
uv run ruff check src/ && uv run mypy src/

# 3. Run tests
uv run pytest --cov=equilens

# 4. If all good, test the CLI
uv run equilens status
```

---

## 📌 Using VS Code Tasks

Press `Ctrl+Shift+B` to open Task menu. Organized groups:

### 🧪 Test Group (Building)
- Dev: Run Tests
- Dev: Lint Code (Ruff)
- Dev: Format Code (Ruff)
- Dev: Type Check (MyPy)
- Dev: Full Check (Lint + Type)

### 🧪 Test Group (Testing)
- Audit: Run with Logprobs (Default)
- Audit: Run with Timing Fallback
- Audit: Resume Session
- Analyze: Generate Reports
- Analyze: Advanced with AI Insights

### 📋 Other
- Help: Show EquiLens Commands
- Generate: Create Test Corpus
- Services: Check Status
- Services: Start (Docker)
- Services: Stop (Docker)
- Utilities: GPU Check
- Utilities: List Models
- Web: Launch Gradio UI
- Web: Start Backend API
- Web: Full Stack (API + UI)

---

## 💡 Pro Tips

1. **Default is Logprobs**: Audit runs with logprobs by default (requires Ollama >= 0.12.11)
2. **Timing Fallback**: Use `--no-logprobs` if your Ollama version doesn't support logprobs
3. **Reports Auto-Detect**: Analysis automatically detects which scoring method was used in audit results
4. **Unicode Errors**: Add `--silent` flag if you see encoding errors in Windows cmd.exe
5. **Background Tasks**: Web tasks (`web`, `backend`, `serve`) run in background. Use `Ctrl+C` to stop
6. **Resume Sessions**: Interrupted audits can be resumed with `uv run equilens audit resume`

---

## 🔄 Comparison: npm vs uv

| npm              | uv                          |
| ---------------- | --------------------------- |
| `npm run audit`  | `uv run equilens audit run` |
| `npm run test`   | `uv run pytest`             |
| `npm run lint`   | `uv run ruff check src/`    |
| `npm run format` | `uv run ruff format src/`   |
| `npm run build`  | `uv run equilens generate`  |
| `npm start`      | `uv run equilens serve`     |

---

## 📚 Full Documentation

- [EquiLens CLI Documentation](docs/README.md)
- [Architecture Guide](docs/architecture/ARCHITECTURE.md)
- [Ollama Setup Guide](docs/auditing/OLLAMA_SETUP.md)
- [Analytics Reference](docs/analytics/QUICK_REFERENCE.md)

Last Updated: March 5, 2026
