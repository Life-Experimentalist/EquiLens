# EquiLens Documentation

This directory contains user-facing documentation for EquiLens — an AI bias detection platform.

Canonical docs (kept and maintained):
- `QUICKSTART.md` — Quick start and one-command examples (uv-first)
- `PIPELINE.md` — End-to-end workflow and recommended run patterns
- `ARCHITECTURE.md` — System architecture and component responsibilities
- `CONFIGURATION_GUIDE.md` — How to author and validate corpora/configs
- `EXECUTION_GUIDE.md` — How to run audits locally and in containers
- `ENHANCED_AUDITOR_SUMMARY.md` — Auditor design, customization, and safety guidance
- `SYSTEM_INSTRUCTIONS_BIAS_IMPACT.md` — Guidance on system instructions and bias measurement
- `SCHEMA_SUMMARY.md` — Corpus schema, examples, and validation rules
- `PERFORMANCE_METRICS.md` — Metrics, outputs, and interpretation
- `OLLAMA_SETUP.md` — How to run and connect to Ollama (local/container)
- `DYNAMIC_CONCURRENCY_AUDITING.md` — Advanced auditing concurrency guide
- `INTERRUPTION_RESUMPTION.md` — Session interruption and resume handling

Deprecated or consolidated files
- Several smaller/overlapping documents have been consolidated into the canonical set above. Those files remain in the repo but are marked as deprecated and point to the authoritative document. If you are looking for content previously in one of these files, consult the canonical documents listed above.

Notes
- The authoritative auditor implementations are in `src/Phase2_ModelAuditor/`.
- Use `uv run equilens` to run the pipeline via the CLI (this is the preferred, 'uv-first' workflow used throughout the docs).
- If you want a pointer to where specific content was merged, open the deprecated file — it contains a short note directing you to the canonical location.
