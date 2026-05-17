# EquiLens — CLAUDE.md

Project-level guidance for AI-assisted development. This overrides generic defaults.

## What this project is

EquiLens is a **black-box bias detection framework** for SLMs and LLMs. It uses **prompt engineering** to probe models without access to their weights or internals — sending carefully constructed prompts and measuring differential responses across demographic groups (gender, race, occupation, etc.).

B.Tech final year project at **Amrita Vishwa Vidyapeetham** by VKrishna04.

## Architecture: Three-Phase Pipeline

```
Phase 1 — Corpus Generator (src/Phase1_CorpusGenerator/)
  generate_corpus.py       word_lists.json → CSV corpus
  word_lists.json          demographic names, traits, professions
  test_config.py           pre-flight config validator

Phase 2 — Model Auditor (src/Phase2_ModelAuditor/)
  audit_model.py           sends prompts to Ollama, records responses
  enhanced_audit_model.py  retry/resume, GPU acceleration, concurrency
  run_both_auditors.py     comparison runner

Phase 3 — Analytics (src/Phase3_Analysis/)
  analytics.py             statistical tests, HTML/Markdown reports, AI insights
  analyze_results.py       CLI entry for analysis

equilens package (src/equilens/)
  cli.py                   Typer CLI — `equilens` command
  gradio_app.py            Gradio web UI (connects to backend)
  web_ui.py                Legacy standalone Gradio UI (no backend needed)
  backend/api.py           FastAPI REST backend with job queue
  backend/jobs.py          Background job runners
  backend/database.py      SQLite job tracking
  telemetry.py             Stats counters (seed from data/telemetry.json)
  core/
    manager.py             EquiLensManager — orchestrates all components
    ollama_config.py       Smart Ollama URL detection (local vs Docker)
    ports.py               Dynamic port management
    docker.py              Docker service management
    gpu.py                 GPU/CUDA detection
```

## Key entry points

| Command | What it does |
|---------|-------------|
| `uv run equilens web` | Launches full-stack (backend + Gradio frontend) |
| `uv run equilens backend` | Launches FastAPI backend only (port 8000) |
| `uv run equilens audit --model llama3.2` | CLI bias audit |
| `uv run equilens analyze <results.csv>` | CLI analysis |
| `uv run equilens status` | System health check |
| `docker compose -f infra/docker-compose.yml up` | Full Docker stack |

## Dev workflow

```bash
uv sync                  # install deps
uv run pytest            # run tests
uv run ruff check src/   # lint
uv run ruff format src/  # format
```

## Critical constraints

- **Never** break the three-phase pipeline independence — Phase 1 output (CSV) must work as Phase 2 input without modification.
- `web_ui.py` is the **legacy standalone** UI (no backend needed). `gradio_app.py` is the **new** UI that requires the FastAPI backend. Do not merge them.
- Ollama URL detection is environment-aware: local gets `localhost:11434`, Docker gets `host.docker.internal:11434`. Don't hardcode URLs.
- The `data/telemetry.json` seed counters are intentional — they represent the "baseline" for the stats bar. Don't reset them to 0.

## Bias categories covered

1. **Gender** — male vs female names in identical prompts
2. **Racial/Ethnic** — Western vs non-Western names
3. **Occupational** — profession-based stereotyping
4. **Sentiment** — positive/negative trait association by group
5. **Counterfactual** — response change when only demographic changes
6. **Associative** — implicit bias via word association

## Files to not touch unless explicitly asked

- `src/Phase1_CorpusGenerator/word_lists.json` — manually curated, changes need domain knowledge
- `src/Phase1_CorpusGenerator/word_lists_schema.json` — schema for word_lists.json
- `data/jobs/equilens_jobs.db` — live SQLite job database
- `uv.lock` — don't manually edit, let `uv sync` manage it

## Test strategy

Tests in `tests/` are a mix of unit and integration. Most integration tests require Ollama running. Run unit tests only:
```bash
uv run pytest tests/unit/
```
