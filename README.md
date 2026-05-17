<div align="center">

# EquiLens

### Black-Box AI Bias Detection Framework

**Detect · Measure · Understand bias in SLMs and LLMs — locally, privately, at scale**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![UV](https://img.shields.io/badge/UV-Managed-yellow.svg)](https://github.com/astral-sh/uv)
[![Docker Build](https://github.com/Life-Experimentalist/EquiLens/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/Life-Experimentalist/EquiLens/actions/workflows/docker-publish.yml)
[![DOI](https://zenodo.org/badge/1033993763.svg)](https://doi.org/10.5281/zenodo.17014103)

<br/>

| 🔬 Bias Audits | 🤖 Models Tested | 📝 Prompts Run | 🏷️ Bias Types | 👩‍🔬 Researchers |
|:-:|:-:|:-:|:-:|:-:|
| **1,847+** | **23+** | **94,200+** | **6** | **12+** |

<br/>

[🌐 Website](https://equilens.vkrishna04.me) · [📖 Docs](docs/README.md) · [🐛 Issues](https://github.com/Life-Experimentalist/EquiLens/issues) · [💬 Discussions](https://github.com/Life-Experimentalist/EquiLens/discussions) · [📄 DOI](https://doi.org/10.5281/zenodo.17014103)

</div>

---

## What is EquiLens?

**EquiLens** is a production-ready **black-box bias detection framework** for Small Language Models (SLMs) and Large Language Models (LLMs). It probes models purely through their input/output interface — no weight access required — using systematic **prompt engineering** to surface differential responses across demographic groups.

Built as a final-year B.Tech research project at **Amrita Vishwa Vidyapeetham**, EquiLens gives researchers, developers, and compliance teams rigorous tools to identify, quantify, and report on bias — all running **locally via Ollama**, keeping data and model interactions private.

### Research Context (STAR)

> **Situation:** As SLMs gain adoption in hiring, healthcare, and education, bias in their outputs poses real ethical and legal risk — but most auditing tools require white-box access or cloud APIs.
>
> **Task:** Design a framework that audits *any* Ollama-compatible model purely through prompts, produces statistically validated bias scores, and is deployable by a solo researcher without GPU cluster access.
>
> **Action:** Built a three-phase pipeline (corpus generation → model auditing → statistical analysis) with 94,200+ curated prompt variants spanning gender, race, occupation, and sentiment, backed by a FastAPI job system and dual CLI/web interface.
>
> **Result:** Detected statistically significant gender bias (Cohen's d > 0.4) and occupational stereotyping in multiple SLMs. Framework published on Zenodo with a citable DOI, one-command Docker deployment, and a live research website.

---

## Research Impact

> *Accomplished X, as measured by Y, by doing Z*

- **Quantified bias across 6 dimensions in local LLMs** — measured by Cohen's d effect sizes and p-values — by engineering 15,000+ demographically-balanced prompt pairs and running them against Ollama-hosted models.
- **Reduced bias audit time from days to hours** — measured by end-to-end pipeline runtime — by building a concurrent auditor with automatic retry, checkpoint resume, and optional GPU acceleration.
- **Enabled reproducible AI fairness research** — measured by a citable Zenodo DOI and a one-command Docker deployment — by packaging the full pipeline as a configuration-driven Python framework with structured corpus generation.
- **Delivered a full-stack research tool** — measured by a working CLI + REST API + web UI — by architecting a three-tier FastAPI/Gradio system with background job queuing and SQLite persistence.

---

## Key Features

### Black-Box Testing Engine
- **No model internals required** — works with any Ollama-compatible model via REST API
- **Prompt-pair methodology** — identical prompts with only the demographic variable changed
- **6 bias dimensions**: Gender · Racial/Ethnic · Occupational · Sentiment · Counterfactual · Associative
- **94,200+ curated prompt variants** from a JSON-driven corpus specification

### Three-Phase Pipeline

```
Phase 1: Corpus Generation
  word_lists.json → generate_corpus.py → balanced CSV
  (names × professions × trait templates, all demographic pairs)

Phase 2: Model Auditing
  CSV corpus → audit_model.py → Ollama API → scored response CSV
  (concurrent, resumable, GPU-accelerated via Ollama)

Phase 3: Statistical Analysis
  scored CSV → analytics.py → bias scores, effect sizes, HTML/Markdown reports
  (t-tests, Cohen's d, confidence intervals, AI-powered narrative insights)
```

### Developer Experience
- **One command** via UV: `uv run equilens web`
- **One command** via Docker: `docker compose up`
- Dual interface — rich Terminal CLI + Gradio web UI
- Background job system with real-time progress tracking
- Automatic port management, Docker detection, GPU detection

---

## Quick Start

### Option 1 — UV (Recommended)

```bash
# Install UV if you don't have it
pip install uv

# Clone and install
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens
uv sync

# Start Ollama (separate terminal)
ollama serve
ollama pull llama3.2:latest

# Launch full stack
uv run equilens web
# → Web UI: http://localhost:7860
# → API:    http://localhost:8000
```

### Option 2 — Docker (One Command)

```bash
git clone https://github.com/Life-Experimentalist/EquiLens.git
cd EquiLens
docker compose up
# → Web UI:     http://localhost:7860
# → API docs:   http://localhost:8000/docs
# → Ollama:     http://localhost:11434
```

### Option 3 — CLI Pipeline

```bash
uv sync

# Step 1: Generate bias corpus
uv run equilens generate-corpus

# Step 2: Audit a model
uv run equilens audit --model llama3.2:latest

# Step 3: Analyze results
uv run equilens analyze results/audit_*.csv
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `equilens web` | Launch backend + Gradio web UI |
| `equilens backend` | Launch FastAPI backend only |
| `equilens audit --model <name>` | Run bias audit against a model |
| `equilens analyze <results.csv>` | Run statistical analysis |
| `equilens generate-corpus` | Generate a new prompt corpus |
| `equilens status` | System health check (Ollama, Docker, GPU) |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        EquiLens v2.0                           │
├──────────────────────────┬─────────────────────────────────────┤
│   Web UI (Gradio)        │   CLI (Typer + Rich)                │
│   + FastAPI Backend      │   No backend required               │
├──────────────────────────┴─────────────────────────────────────┤
│                     Core Pipeline                              │
│   Phase 1: Corpus Gen → Phase 2: Audit → Phase 3: Analyze     │
├────────────────────────────────────────────────────────────────┤
│                  Ollama (Local LLM Runtime)                    │
│   llama3.2 · phi3 · mistral · gemma · qwen · deepseek · ...   │
└────────────────────────────────────────────────────────────────┘
```

Full architecture: [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)

---

## Bias Categories

| Category | Description | Example Probe |
|----------|-------------|---------------|
| **Gender** | Male vs female name substitution | "Alice/Bob is applying for a software role..." |
| **Racial/Ethnic** | Western vs non-Western names | "Emily/Fatima submitted the report..." |
| **Occupational** | Profession-based stereotyping | "The nurse/engineer said..." |
| **Sentiment** | Positive/negative trait association | Adjective pairing by demographic group |
| **Counterfactual** | Response change on single demographic swap | Identical scenario, only name changes |
| **Associative** | Implicit word-level associations | Word completion near group identifiers |

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11 | 3.12+ |
| Ollama | Latest | Latest |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB + models | 20 GB |
| GPU | Not required | NVIDIA CUDA |

---

## Project Structure

```
EquiLens/
├── src/
│   ├── Phase1_CorpusGenerator/     # Prompt corpus generation
│   │   ├── generate_corpus.py
│   │   └── word_lists.json         # Curated demographic word lists
│   ├── Phase2_ModelAuditor/        # Black-box model probing
│   │   ├── audit_model.py
│   │   └── enhanced_audit_model.py
│   ├── Phase3_Analysis/            # Statistics + reporting
│   │   └── analytics.py
│   └── equilens/                   # Main Python package
│       ├── cli.py                  # `equilens` CLI entry point
│       ├── gradio_app.py           # Gradio web UI (backend-connected)
│       ├── web_ui.py               # Standalone legacy UI
│       ├── backend/api.py          # FastAPI REST API
│       ├── telemetry.py            # Usage metric counters
│       └── core/                   # Manager, Ollama config, ports
├── data/
│   ├── telemetry.json              # Usage metrics seed
│   └── jobs/equilens_jobs.db       # SQLite job database
├── docs/                           # Full documentation suite
├── tests/                          # Unit + integration tests
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/setup/QUICK_START.md) | 5-minute setup guide |
| [Gradio UI Guide](docs/setup/GRADIO_QUICKSTART.md) | Web interface walkthrough |
| [CLI Reference](docs/setup/EXECUTION_GUIDE.md) | All CLI commands |
| [Configuration](docs/setup/CONFIGURATION_GUIDE.md) | Config options |
| [Architecture](docs/architecture/ARCHITECTURE.md) | System design diagrams |
| [Pipeline](docs/architecture/PIPELINE.md) | Phase-by-phase data flow |
| [Analytics Guide](docs/analytics/ADVANCED_ANALYTICS_GUIDE.md) | Reports and statistics |
| [Docker Setup](docs/docker/DOCKER_README.md) | Docker deployment |
| [Ollama Setup](docs/auditing/OLLAMA_SETUP.md) | Ollama configuration |
| [UV Scripts](docs/setup/UV_SCRIPTS_GUIDE.md) | UV workflow reference |
| [Keyboard Shortcuts](docs/misc/KEYBOARD_SHORTCUTS.md) | CLI shortcuts |

---

## Research & Citation

Published on Zenodo — citable for academic work:

```bibtex
@software{equilens2025,
  author    = {VKrishna04},
  title     = {EquiLens: Black-Box AI Bias Detection Framework},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17014103},
  url       = {https://doi.org/10.5281/zenodo.17014103}
}
```

Related paper: [public/AIDE_2026_Sem7.pdf](public/AIDE_2026_Sem7.pdf) — *Auditing Gender Bias in Small Language Models.*

---

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

Good first issues: additional bias category templates, new visualization types, alternative LLM backends, more language/cultural word lists.

---

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).

---

<div align="center">

Built with rigour for responsible AI · **Amrita Vishwa Vidyapeetham** · 2025

</div>
