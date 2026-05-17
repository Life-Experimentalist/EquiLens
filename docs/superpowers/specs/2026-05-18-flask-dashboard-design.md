# EquiLens Dashboard & Root Cleanup — Design Spec
**Date:** 2026-05-18  
**Status:** Approved by user

---

## 1. Goals

1. Replace duplicate Gradio UIs (`web_ui.py`, `gradio_app.py`, `start_all.py`) with a single, clean HTML dashboard served directly from the existing FastAPI backend using Jinja2 templates.
2. Remove `gradio` dependency entirely.
3. Add periodic backup, auto-recovery, SSE-based real-time job progress, and inline results display.
4. Add a complete, explicit OpenAPI 3.1 contract to every endpoint.
5. Clean up the project root by moving all Docker/infrastructure files to `infra/`.

---

## 2. Architecture

Single FastAPI process serves everything — no second server, no browser auto-open:

```
localhost:8000/                    → HTML Dashboard (Jinja2-rendered)
localhost:8000/api/...             → REST API (existing, extended)
localhost:8000/api/events/{job_id} → SSE stream (real-time job logs)
localhost:8000/api/backups/...     → Backup management endpoints
localhost:8000/docs                → Swagger UI (FastAPI auto-generated)
localhost:8000/openapi.json        → OpenAPI 3.1 contract (machine-readable)
```

`equilens web` (CLI) → starts this single server, prints URL, does NOT open browser.

---

## 3. Dashboard Pages

All pages share a sidebar nav + top error banner. **Alpine.js** (CDN, ~15KB, no build step) for reactive UI state — same paradigm as Vue (`x-data`, `x-bind`, `x-on`, `x-show`) but works directly with Jinja2 templates without delimiter conflicts. Zero tooling: no npm, no vite, no webpack. Plus **Chart.js** (CDN) for bias score visualizations.

### 3.1 Dashboard (Home)
- System health cards: Ollama status (green/red), GPU detected (yes/no), Docker mode
- Stats bar: telemetry counters (from `data/telemetry.json`)
- Active jobs count with link to Jobs tab
- Last backup timestamp + next scheduled backup

### 3.2 Audit
- Model selector: dropdown populated from `/api/models` (auto-refreshes)
- Corpus picker: dropdown of discovered corpus CSVs (from backend scan)
- Advanced options (collapsible): `enhanced`, `logprobs`, `temperature`, `batch_size`, `request_timeout`, `max_retries`, `num_predict`, `silent`
- Submit → creates job → shows inline live log stream via SSE
- Resume interrupted session: shows detected incomplete sessions with % completion

### 3.3 Generate
- Config file path input (optional)
- Submit → creates corpus generation job → live SSE log stream

### 3.4 Analyze
- Dropdown: pick from existing results directories
- Options: analysis flags if any
- Submit → creates analysis job → on completion, renders HTML report inline in an iframe
- Download ZIP export button

### 3.5 Jobs
- Table: job_id, type, status (badge), progress bar, created_at, actions (cancel, delete)
- Click row → expand log panel (last 100 lines, auto-scroll)
- Filter by status (all / running / completed / failed / cancelled)
- Auto-refreshes every 5 seconds via polling

### 3.6 Results
- List of result directories sorted by date
- Per-result: view HTML report (inline iframe), download ZIP, view raw CSV link
- Delete result button (with confirmation)

---

## 4. Error Handling & Auto-Recovery

### Backend startup
- On `equilens web`, health-check Ollama with exponential backoff (1s, 2s, 4s, max 30s)
- If Ollama unreachable after 3 attempts, log warning but continue (dashboard shows red indicator)
- Catch and log all unhandled exceptions via FastAPI exception handler → return JSON error with request_id

### Job failures
- All job exceptions caught in `jobs.py`, stored in DB as `status=failed` with full traceback in `error_message`
- Jobs tab surfaces failures with expandable error detail
- Failed jobs can be retried (new job, same config) via "Retry" button

### Dashboard client
- Every `fetch()` call has 10s timeout, retries up to 2 times with 1s delay
- SSE (`EventSource`) auto-reconnects on drop (browser native behavior)
- Top-of-page dismissible error banner shown on any fetch failure
- Stale job polling: if a job stays `running` for >30 min with no log activity, dashboard shows a warning

---

## 5. Periodic Backups

### Scheduler
- APScheduler `BackgroundScheduler` started on FastAPI startup event
- Schedule: every 30 minutes (configurable via env var `EQUILENS_BACKUP_INTERVAL_MINUTES`, default 30)
- Backup target: `results/` directory + `data/jobs/equilens_jobs.db`
- Output: `backups/backup_YYYYMMDD_HHMMSS.zip`
- Retention: keep last 10 backups (delete oldest when exceeded), configurable via `EQUILENS_BACKUP_RETENTION`, default 10

### Backup endpoints (new)
```
GET  /api/backups              → list all backups (name, size, created_at)
POST /api/backups              → trigger manual backup immediately
GET  /api/backups/{name}       → download backup zip
DELETE /api/backups/{name}     → delete a backup
```

### Dashboard integration
- Dashboard home shows last backup time, next scheduled time, backup count
- Manual "Backup Now" button

---

## 6. OpenAPI Contract

Every endpoint gets explicit annotations:
- `tags`: groups (System, Jobs, Results, Models, Backups)
- `summary`: one-line description
- `description`: fuller explanation
- `response_model`: typed Pydantic model
- `responses`: explicit error codes (400, 404, 500) with descriptions

FastAPI auto-publishes to `/openapi.json` and `/docs` (Swagger UI).

New endpoints added in this spec:
- `GET /api/events/{job_id}` — SSE stream, documented with note about text/event-stream content type
- `GET /api/corpus` — list discovered corpus files
- `GET /api/backups`, `POST /api/backups`, `GET /api/backups/{name}`, `DELETE /api/backups/{name}`
- `POST /api/jobs/{job_id}/retry` — retry a failed job with the same config

---

## 7. New File Structure

```
src/equilens/
  dashboard/
    __init__.py
    routes.py              # HTML route handlers returning Jinja2 TemplateResponse
    templates/
      base.html            # layout: sidebar, nav, error banner, footer
      dashboard.html       # home/status page
      audit.html           # audit form + SSE log viewer
      generate.html        # corpus generation form + SSE log viewer
      analyze.html         # analysis form + results iframe
      jobs.html            # jobs table with filter + log expansion
      results.html         # results browser
    static/
      app.js               # SSE client, fetch wrappers, auto-retry, polling, Alpine.js stores
      style.css            # clean minimal CSS (CSS variables, dark/light via prefers-color-scheme)
      # Alpine.js + Chart.js loaded from CDN in base.html — no build step
  backup.py                # APScheduler backup task + retention cleanup
```

---

## 8. Files Deleted (Duplicates Removed)

| File | Reason |
|------|--------|
| `src/equilens/web_ui.py` | Legacy standalone Gradio UI |
| `src/equilens/gradio_app.py` | Gradio frontend that requires backend |
| `src/equilens/start_all.py` | Gradio+backend launcher |

---

## 9. Project Root Cleanup

Move to `infra/` directory:

| File | Destination |
|------|-------------|
| `Dockerfile` | `infra/Dockerfile` |
| `docker-compose.yml` | `infra/docker-compose.yml` |
| `docker-compose.full-stack.yml` | `infra/docker-compose.full-stack.yml` |
| `.dockerignore` | `infra/.dockerignore` |
| `equilens_cli.py` | `infra/equilens_cli.py` |

After move:
- Update `Dockerfile` → `CMD` to reference new path of `equilens_cli.py`
- Update `docker-compose.yml` → `build.context` to `..` (one level up from `infra/`)
- Update `docker-compose.yml` → `build.dockerfile` to `infra/Dockerfile`
- Update `CLAUDE.md` and `README.md` docker references
- Add `infra/README.md` explaining the directory

Root after cleanup contains only: `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `CHANGELOG.md`, `LICENSE.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `.gitignore`, `.editorconfig`, `.gitattributes`, `.pre-commit-config.yaml`, `pyrightconfig.json`, `src/`, `tests/`, `data/`, `docs/`, `scripts/`, `results/`, `logs/`, `infra/`

---

## 10. Dependency Changes

```bash
uv add apscheduler          # periodic backup scheduler
uv remove gradio            # no longer needed
# jinja2 already present, fastapi already present
```

---

## 11. CLI Changes

`equilens web` command:
- Currently: starts backend + Gradio in two processes
- After: starts single FastAPI server with embedded dashboard
- Prints: `Dashboard running at http://localhost:8000`
- Does NOT open browser automatically

`equilens backend` command:
- Unchanged — still starts API-only (no dashboard routes loaded)

---

## 12. Success Criteria

- [ ] `uv run equilens web` starts one server, prints URL, no browser auto-open
- [ ] All 6 dashboard pages render correctly and are functional
- [ ] Audit job submits, SSE stream shows live logs, results appear in Results tab on completion
- [ ] Periodic backup runs every 30 min, old backups pruned, dashboard home shows status
- [ ] `/openapi.json` returns valid OpenAPI 3.1 contract covering all endpoints
- [ ] `/docs` Swagger UI works and lists all endpoints with descriptions
- [ ] `web_ui.py`, `gradio_app.py`, `start_all.py` deleted, `gradio` removed from deps
- [ ] `infra/` contains all Docker files, root is clean
- [ ] Docker builds still work with updated paths (`docker compose -f infra/docker-compose.yml up`)
