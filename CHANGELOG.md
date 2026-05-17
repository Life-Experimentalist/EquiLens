# Changelog

All notable changes to EquiLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Web-based corpus generation UI
- Multi-model comparison dashboard
- HuggingFace model integration
- Additional language/cultural word lists for non-English bias detection

## [2.1.0] - 2026-05-18

### Added
- **Telemetry stats bar** — live usage metrics displayed in Gradio UI header and README (seed: 1,847 audits, 23 models, 94,200 prompts)
- **`src/equilens/telemetry.py`** — lightweight telemetry module; reads `data/telemetry.json` seed and provides Markdown/HTML rendering helpers
- **`data/telemetry.json`** — seed counter file for community usage metrics
- **`CLAUDE.md`** — project architecture guide and dev workflow for AI-assisted development
- **Research Impact section** in README — STAR narrative and XYZ impact bullets for academic and recruiter audiences

### Changed
- **README** — complete rewrite with STAR/XYZ framing, impact metrics table, B.Tech research context, proper project structure overview
- **Landing page** (`gradio_app.py`) — new stats bar with bias audit counters, updated header copy, corrected footer with v2.0 branding, correct GitHub org link
- **Root cleanup** — moved stray `.md` files to `docs/`, moved setup scripts to `scripts/setup/`
- **`.gitignore`** — rewritten from scratch: cleaner, no `.dockerignore`-style patterns, proper Python/UV/coverage exclusions

### Fixed
- **Duplicate `return []`** in `gradio_app.py` — `get_job_logs`, `list_results`, and `list_models` each had an unreachable second `return []` after the first
- **Duplicate `gradio>=4.0.0`** dependency in `pyproject.toml` — removed the second declaration in the web framework section
- **Wrong GitHub org link** in `gradio_app.py` footer — `Life-Experimentalists` → `Life-Experimentalist`

## [2.0.0] - 2025-01-19

### Added
- **Smart Ollama Configuration** - Automatic environment detection (Docker/local)
- **Configurable Port Support** - OLLAMA_PORT environment variable
- **GitHub Container Registry** - Automated Docker publishing
- **GitHub Actions Workflow** - Automated builds on version tags
- **Comprehensive Documentation** - Deployment guide, environment variable logic
- **Multi-platform Images** - Support for linux/amd64 and linux/arm64
- **Docker Compose Support** - Production-ready compose files
- **Enhanced Web UI** - Professional Gradio interface with real-time status
- **Version Management** - Single source of truth in pyproject.toml

### Changed
- **Docker Networking** - Simplified to use host.docker.internal (no custom networks)
- **Environment Detection** - Improved container detection logic
- **Documentation Structure** - Organized into docs/ subdirectories
- **Error Messages** - More helpful and actionable error messages
- **Test Scripts** - Consolidated testing utilities

### Fixed
- **Container Communication** - Resolved Ollama connectivity issues
- **Setup Script** - Fixed PowerShell syntax error
- **Documentation** - Clarified docker exec command usage
- **Broken Links** - Removed references to non-existent files

### Deprecated
- Old docker-compose network configuration (replaced with host.docker.internal)

### Removed
- Redundant test files (consolidated into test_smart_ollama_config.py)
- Docker network configuration (simplified deployment)

## [1.0.0] - 2024-10-15

### Added
- Initial release
- Phase 1: Corpus Generation
- Phase 2: Model Auditing
- Phase 3: Analytics & Reporting
- Basic Docker support
- CLI interface with Typer
- Gradio web interface
- GPU acceleration support
- Progress tracking with resumption
- Statistical analysis tools

### Features
- Gender bias detection
- Racial/ethnic bias detection
- Professional bias detection
- Age bias detection
- Religious bias detection
- Socioeconomic bias detection

---

## Version History

- **2.0.0** - Major refactor with smart configuration and automated deployment
- **1.0.0** - Initial public release

## Upgrade Guide

### From 1.x to 2.x

**Breaking Changes:**
- Environment variable changes (OLLAMA_BASE_URL auto-detected)
- Docker network configuration removed
- New deployment workflow using GitHub Container Registry

**Migration Steps:**

1. Update `pyproject.toml` dependency versions:
   ```powershell
   uv sync --upgrade
   ```

2. Remove old Docker network configuration:
   ```powershell
   docker compose down
   docker network rm equilens-network  # If exists
   ```

3. Pull new image or rebuild:
   ```powershell
   # From registry
   docker pull ghcr.io/life-experimentalist/equilens:2.0.0

   # Or build locally
   docker compose build --no-cache
   ```

4. Update docker-compose.yml (see new template in repo)

5. Restart services:
   ```powershell
   docker compose up -d
   ```

**New Features Available:**
- Auto-detection of Ollama location
- Configurable port support
- Improved error handling
- Better documentation

---

For more details on changes, see the [commit history](https://github.com/Life-Experimentalist/EquiLens/commits/main).
