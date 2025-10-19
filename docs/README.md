# EquiLens Documentation Index

This directory contains the user and developer documentation for EquiLens. The docs have been organized into focused subfolders to make it easier to find installation instructions, Docker guidance, architecture diagrams, analytics and auditing reference material.

## Quick links
- Installer scripts: `../scripts/install/README.md`
- Configuration guide: `./CONFIGURATION_GUIDE.md`
- Execution guide: `./EXECUTION_GUIDE.md`

## Table of contents

### Setup
- `setup/QUICKSTART.md`  Quick start and installation steps (UV, venv, Docker) with verification and example audit commands.
- `setup/SETUP_SCRIPTS_GUIDE.md`  Guide to installer scripts and one-command options.
- `setup/SETUP_IMPROVEMENTS.md`  Notes about recent setup improvements.
- `setup/TROUBLESHOOTING_SETUP.md`  Troubleshooting tips for common setup problems.

### Docker
- `docker/DOCKER_README.md`  Docker quick reference: access points (Gradio/API/Ollama), common commands and volumes.
- `docker/DOCKER_SETUP.md`  Full Docker setup and compose examples.
- `docker/DOCKER_CONFIG_GUIDE.md`  Configurable ports, volumes, container names and environment variables.
- `docker/DOCKER_CHECKLIST.md`  Checklist for building and deploying container images.

### Architecture
- `architecture/ARCHITECTURE.md`  System architecture diagrams (Mermaid) and component interactions.
- `architecture/PIPELINE.md`  Dataflow and phase-to-phase pipeline overview.

### Analytics
- `analytics/QUICK_REFERENCE.md`  Quick reference for analytics workflows and one-command setups.
- `analytics/ADVANCED_ANALYTICS_GUIDE.md`  Advanced analytics guide describing professional visualizations and reports.

### Auditing
- `auditing/AUDITING_MECHANISM.md`  Deep dive into the standard and enhanced auditors, workflows and retry/resume features.
- `auditing/ENHANCED_AUDITOR_DEFAULT.md`  Rationale and notes for making the enhanced auditor the default.
- `auditing/OLLAMA_SETUP.md`  Ollama-specific setup notes and options.

### Misc
- `misc/CODE_ANALYSIS_REPORT.md`  Code analysis and recommended refactors.
- `misc/REPORT.md`  Miscellaneous project reports.
- `misc/SCRIPT_REORGANIZATION_PLAN.md`  The plan used to reorganize docs and scripts.

### Archived
- `archived/ONE_CLICK_SETUP.md`  Archived historical one-click installer instructions.
- `archived/RECOVERY_NOTES.md`  Archived recovery and troubleshooting notes.

## How to use this index
- Click any file path above to open the detailed document.
- If a link points to an old location, please open an issue or request an update via the repository's issue tracker.

## Contributing
- To add or modify docs, create a branch and submit a PR against `main`.
- Keep individual docs focused (one concept per file) and add a one-line summary in this index when adding new files.
- Use relative links when linking between docs so they work both on GitHub and locally.

## Verification checklist (after reorg)
- [ ] All files moved into subfolders under `docs/`.
- [ ] `docs/README.md` updated (this file).
- [ ] Top-level `README.md` and any scripts updated to point to new doc locations.

---

Generated and updated by the project maintainer tooling.
