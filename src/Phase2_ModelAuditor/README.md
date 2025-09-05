Phase2_ModelAuditor
====================

Purpose
-------
>Phase2_ModelAuditor contains the core auditing engines used by EquiLens to evaluate language models for bias. It includes:

- `audit_model.py` — the primary auditor used by the CLI and pipeline. Handles both sequential and concurrent auditing, request/response handling with Ollama, progress tracking, resume support, retry logic, and CSV/JSON result output.
- `enhanced_audit_model.py` — an enhanced auditor that integrates Rich progress bars, multiple bias metrics, structured output support, and repeated sampling for more robust bias detection.

This module is designed for reliability in long-running audits: it supports resuming interrupted sessions, configurable retry policies for failed tuples, and flexible output formats for downstream analysis.

Key Concepts
------------
- Tuple: A single test item derived from a CSV corpus row. The auditor sends the prompt(s) from the tuple to the model and records the response and metadata.
- Retry queue: When a tuple fails (network error, model error, or other recoverable failure), it may be queued for retry according to the configured policy.
- Progress file: The auditor periodically writes a `progress_*.json` file into the run directory so interrupted runs can be resumed.
- Results files: Final and incremental results are saved as CSV files under `results/` with a session-specific folder. This directory also holds `progress_*.json` and audit logs.
- Structured Output: Enhanced auditor can request JSON-formatted responses from models to ensure consistent parsing while measuring if this affects bias detection.
- Repeated Sampling: Enhanced auditor can collect multiple responses per prompt and use median aggregation for more stable bias measurements.

Files
-----
- `__init__.py` — package initializer and short module docstring.
- `audit_model.py` — main auditor implementation. See API reference below.
- `enhanced_audit_model.py` — enhanced auditor with Rich UI, multiple bias metrics, structured output support, and repeated sampling.
- `TODO.md` — developer notes and outstanding work items.

Quick Start (CLI)
-----------------
From the repository root (project uses Typer CLI `equilens`):

- Run the standard auditor (recommended for stability):

```
uv run equilens audit --model <model-name> --corpus path/to/corpus.csv
```

- Run enhanced auditor with default settings:

```
uv run equilens audit --enhanced --model <model-name> --corpus path/to/corpus.csv
```

- Run enhanced auditor with structured output and repeated sampling:

```
python src/Phase2_ModelAuditor/enhanced_audit_model.py --model <model-name> --corpus path/to/corpus.csv
```

Enhanced Auditor Features
-------------------------
The enhanced auditor (`enhanced_audit_model.py`) provides additional capabilities:

**Multiple Bias Metrics:**
- `surprisal_score` — traditional perplexity-based bias measurement
- `normalized_surprisal` — surprisal normalized by token count
- `token_count` — number of tokens in model response
- `response_length` — character length of response
- `sentiment_score` — simple sentiment analysis score (-1 to 1)
- `polarity` — categorical sentiment label (positive/negative/neutral)

**Structured Output Support:**
- Can request JSON-formatted responses from models
- Graceful fallback to free-text when JSON parsing fails
- Calibration controls to measure bias impact of structured vs free-text output
- Parsing success rate tracking

**Repeated Sampling:**
- Collect multiple responses per prompt (1-5 samples)
- Median aggregation for more stable measurements
- Reduces noise from single-response variance

**Configuration Options:**
- `use_structured_output` — enable JSON response requests (default: False)
- `samples_per_prompt` — number of response samples per prompt (default: 1)
- Rich progress bars with detailed statistics
- Dual CSV output format matching stable auditor

Core Features
-------------
- Sequential and concurrent modes (ThreadPoolExecutor) with automatic scaling hints.
- Resumable sessions via `progress_*.json` files.
- Configurable retry policies: immediate retry attempt, retry queue with batch processing, max attempts per tuple.
- Robust Ollama HTTP client usage with exponential backoff for transient errors.
- Periodic checkpointing and safe CSV append semantics to avoid split results during resume.
- Enhanced auditor: Multiple bias metrics, structured output support, repeated sampling, Rich UI progress tracking.

Output Compatibility
-------------------
Both auditors produce identical output formats for Phase 3 compatibility:

- `results_<model>_<session_id>.csv` — sanitized results with bias metrics
- `results_<model>_<session_id>_responses.csv` — full model responses
- `summary_<session_id>.json` — session metadata and statistics

The enhanced auditor adds additional columns for new metrics while maintaining backward compatibility.

New Metrics (Enhanced Auditor)
------------------------------
The enhanced auditor now computes several additional metrics alongside the original surprisal-based score to provide a richer, complementary view of potential biases:

- `surprisal_score` (existing): Derived from model evaluation duration/token counts where available.
- `normalized_surprisal`: Per-token processing time (eval_duration / eval_count) used as a hardware-independent proxy for surprisal when token-level logprobs are not exposed by Ollama.
- `token_count`: Approximated token count computed by splitting response text. Useful for normalizing duration metrics.
- `response_length`: Character-length of the model's reply.
- `sentiment_score`: Lightweight heuristic counting positive/negative cue tokens in the response to detect response polarity differences across comparison groups.
- `polarity`: Derived label (`positive`, `neutral`, `negative`) from the sentiment score.

Notes on Ollama and structured outputs
-------------------------------------
Ollama may not expose token-level logprobs or bit-precision statistics in all configurations. To obtain stable numeric signals you can:

- Request structured JSON outputs from the prompt (e.g., ask the model to return a small JSON with fields like `answer`, `confidence_estimate`). Use a short deterministic instruction in your prompt to encourage structured output.
- Use `eval_duration` and `eval_count` returned by Ollama (if available) to compute `normalized_surprisal` as a proxy for surprisal.
- Apply simple heuristics (length, token counts, sentiment cues) as additional features; these are fast, stable, and model-agnostic.

Output files
------------
The enhanced auditor writes two CSV files per session, matching the stable auditor's behavior:

- `results_<model>_<session>.csv` — Sanitized per-test results (one row per test). This file omits the full model response to keep size and privacy manageable. Columns include all metric fields listed above.
- `results_<model>_<session>_responses.csv` — Full model responses with raw text (one row per test). Used for deeper analysis but stored separately to avoid inflating the primary results table.

These two-file semantics allow Phase3 analysis to quickly operate on compact numeric features while still being able to inspect or re-process raw responses when necessary.

API Reference (high-level)
--------------------------
The auditor exposes a script-style entrypoint. The most important functions/classes are documented here at a high level so integrators can call them from the CLI or other orchestrators.

- `ModelAuditor` (class)
  - Responsibilities: load corpus, orchestrate requests to model server, manage progress and retries, write result rows to CSV, and create progress snapshots.
  - Important methods (typical semantics):
    - `run()` — start auditing the corpus and block until complete or interrupted.
    - `load_corpus(path)` — read and validate CSV corpus into internal structures.
    - `process_tuple(tuple)` — send a single tuple to the model and return result metadata.
    - `schedule_retry(tuple)` — record a tuple for later retry processing.
    - `save_progress()` — persist `progress_*.json` snapshot.

- `main()` — script entrypoint that parses CLI args (when run directly) and constructs a `ModelAuditor` with parsed options.

Configuration
-------------
Important configuration knobs that can be passed via the CLI or configured in calling code:

- `--batch-size` / `batch_size` — Number of concurrent requests when using enhanced mode.
- `--retry-immediate` / `retry_immediate` — Attempt an immediate retry for failed tuples before enqueueing.
- `--retry-batch-size` / `retry_batch_size` — Number of successful tuples needed before draining the retry queue.
- `--resume` — Path to a `progress_*.json` file to resume an interrupted session.
- `--silent` — Suppress subprocess output (useful on Windows to avoid encoding issues).

Output Files and Locations
--------------------------
- Results CSVs: `results/<session_id>/results_*.csv` (appended safely during run and resume).
- Progress snapshots: `results/<session_id>/progress_*.json`.
- Logs: `logs/audit_session.log` and `logs/audit_errors.log` (created by the CLI wrapper).

Examples
--------
Sequential audit, save to custom dir:

```
uv run equilens audit --model llama2:latest --corpus my_corpus.csv --output-dir results/my_run
```

Resume an interrupted run (CLI will detect and prompt automatically if `--resume` is not provided):

```
uv run equilens audit --resume results/<session_id>/progress_2025...json
```

Troubleshooting
---------------
- Ollama connection errors: ensure Ollama daemon is running and reachable at `http://localhost:11434`.
- Split CSV results on resume: the auditor appends to a single CSV per session folder; if you see multiple partial CSVs, prefer resuming with the progress file in the same session folder. If duplication occurs, inspect `results/<session_id>/` and merge CSVs manually with deduplication on an identifier column.
- Timeouts or long model loads: the CLI includes `preload_model()` which performs a dummy request to warm model memory before ETA measurement. The auditor also contains configurable timeouts and retry backoff.

Testing
-------
- Unit tests should target `audit_model.py` functions: corpus loading, tuple processing (mocking HTTP calls), retry queue behavior, and progress/save/load. See tests in repository root for examples of how the project runs tests with `pytest`.

Developer Notes & TODO
----------------------
See `TODO.md` for outstanding items. Important low-risk improvements:
- Add deterministic CSV header ordering and an atomic file write/rotate strategy for very large runs.
- Add a small smoke-test harness that can run against a local stubbed Ollama server.

License
-------
Same license as the repository (see `LICENSE.md`).
