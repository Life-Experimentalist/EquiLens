# EquiLens Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicate Gradio UIs with a single FastAPI+Jinja2+Alpine.js dashboard served from the existing backend, add periodic backups, SSE job streaming, full OpenAPI annotations, and clean up the project root into `infra/`.

**Architecture:** The existing FastAPI app (`src/equilens/backend/api.py`) gains Jinja2 dashboard routes, a `/static` mount, SSE streaming, and backup endpoints — all served from one process on port 8000. Alpine.js (CDN) and Chart.js (CDN) provide reactive UI with zero build tooling. APScheduler runs periodic backups as a background task inside the same process.

**Tech Stack:** FastAPI, Jinja2, Alpine.js 3.x (CDN), Chart.js 4.x (CDN), APScheduler 3.x, aiofiles, uvicorn

---

## File Map

**Created:**
- `infra/Dockerfile` — moved + updated (CMD, EXPOSE, ENV)
- `infra/docker-compose.yml` — moved + updated (build context, healthcheck)
- `infra/docker-compose.full-stack.yml` — moved as-is
- `infra/.dockerignore` — moved as-is
- `infra/equilens_cli.py` — moved as-is
- `infra/README.md` — explains infra/ directory
- `src/equilens/backup.py` — APScheduler backup task + retention
- `src/equilens/dashboard/__init__.py` — package marker
- `src/equilens/dashboard/routes.py` — six Jinja2 HTML page routes
- `src/equilens/dashboard/templates/base.html` — layout, sidebar, nav, error banner
- `src/equilens/dashboard/templates/dashboard.html` — system status home
- `src/equilens/dashboard/templates/audit.html` — audit form + SSE log viewer
- `src/equilens/dashboard/templates/generate.html` — corpus generation form + SSE
- `src/equilens/dashboard/templates/analyze.html` — analysis form + inline iframe
- `src/equilens/dashboard/templates/jobs.html` — jobs table with expand/cancel
- `src/equilens/dashboard/templates/results.html` — results browser + report viewer
- `src/equilens/dashboard/static/app.js` — fetch wrappers, SSE client, Alpine stores
- `src/equilens/dashboard/static/style.css` — minimal CSS with dark/light variables
- `tests/unit/test_backup.py` — backup unit tests
- `tests/unit/test_dashboard_routes.py` — dashboard route integration tests

**Modified:**
- `src/equilens/backend/api.py` — add SSE, corpus, sessions, retry, backup endpoints; lifespan; static mount; dashboard router; full OpenAPI annotations
- `src/equilens/cli.py` — update `web`, `gui`, `serve`, `backend` commands
- `src/equilens/backend_server.py` — update startup banner
- `pyproject.toml` — add apscheduler, aiofiles, jinja2 to core; remove gradio

**Deleted:**
- `src/equilens/web_ui.py`
- `src/equilens/gradio_app.py`
- `src/equilens/start_all.py`
- `Dockerfile` (moved to infra/)
- `docker-compose.yml` (moved to infra/)
- `docker-compose.full-stack.yml` (moved to infra/)
- `.dockerignore` (moved to infra/)
- `equilens_cli.py` (moved to infra/)

---

## Task 1: Root cleanup — move Docker/infra files

**Files:**
- Create: `infra/` directory with all Docker files

- [ ] **Step 1: Create infra/ and move files**

```bash
mkdir infra
cp Dockerfile infra/Dockerfile
cp docker-compose.yml infra/docker-compose.yml
cp docker-compose.full-stack.yml infra/docker-compose.full-stack.yml
cp .dockerignore infra/.dockerignore
cp equilens_cli.py infra/equilens_cli.py
```

- [ ] **Step 2: Update `infra/Dockerfile`**

Replace the entire file content with the updated version — build context is now `..` (project root), dashboard port, no Gradio env vars:

```dockerfile
FROM python:3.13.3-slim-bullseye

LABEL author="VKrishna04"
LABEL org.opencontainers.image.source="https://github.com/Life-Experimentalist/EquiLens"
LABEL org.opencontainers.image.description="EquiLens AI Bias Detection Platform"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="2.2.0"

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && apt-get autoremove -y

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir uv

RUN useradd -m -u 1000 -s /bin/bash equilens && \
    mkdir -p /workspace/data/results /workspace/data/logs /workspace/data/corpus /workspace/backups && \
    chown -R equilens:equilens /workspace

USER equilens
WORKDIR /workspace

COPY --chown=equilens:equilens pyproject.toml README.md ./
COPY --chown=equilens:equilens uv.lock* ./

RUN --mount=type=cache,target=/home/equilens/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev || uv sync --no-dev

COPY --chown=equilens:equilens . .

RUN mkdir -p data/results data/logs data/corpus src/Phase1_CorpusGenerator/corpus public backups && \
    chmod -R 755 data backups && \
    find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/workspace/src:/workspace \
    OLLAMA_BASE_URL=http://localhost:11434 \
    EQUILENS_BACKUP_INTERVAL_MINUTES=30 \
    EQUILENS_BACKUP_RETENTION=10

CMD [".venv/bin/equilens", "web"]
```

- [ ] **Step 3: Update `infra/docker-compose.yml`**

Replace entire file — build context moves to `..`, healthcheck updated:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s

  equilens:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    image: equilens-app
    container_name: equilens-app
    ports:
      - "8000:8000"
    volumes:
      - equilens_data:/workspace/data
      - equilens_backups:/workspace/backups
      - ./public:/workspace/public:ro
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_API_BASE=http://ollama:11434/api
      - EQUILENS_DATA_DIR=/workspace/data
      - EQUILENS_RESULTS_DIR=/workspace/data/results
      - EQUILENS_LOGS_DIR=/workspace/data/logs
      - EQUILENS_CORPUS_DIR=/workspace/src/Phase1_CorpusGenerator/corpus
      - EQUILENS_BACKUP_INTERVAL_MINUTES=30
      - EQUILENS_BACKUP_RETENTION=10
    restart: unless-stopped
    depends_on:
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  ollama_data:
    external: true
  equilens_data:
    driver: local
    name: equilens-data
  equilens_backups:
    driver: local
    name: equilens-backups
```

- [ ] **Step 4: Create `infra/README.md`**

```markdown
# infra/

Infrastructure files for running EquiLens in Docker.

## Usage

```bash
# From project root:
docker compose -f infra/docker-compose.yml up

# Full stack with GPU:
docker compose -f infra/docker-compose.full-stack.yml up
```

## Files

- `Dockerfile` — EquiLens container image
- `docker-compose.yml` — Standard stack (Ollama + EquiLens)
- `docker-compose.full-stack.yml` — Extended stack
- `.dockerignore` — Files excluded from build context
- `equilens_cli.py` — Docker entry point shim (adds src/ to PYTHONPATH)
```

- [ ] **Step 5: Delete original root-level files**

```bash
rm Dockerfile docker-compose.yml docker-compose.full-stack.yml .dockerignore equilens_cli.py
```

- [ ] **Step 6: Commit**

```bash
git add infra/ Dockerfile docker-compose.yml docker-compose.full-stack.yml .dockerignore equilens_cli.py
git commit -m "chore: move Docker/infra files to infra/ directory"
```

---

## Task 2: Update dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Remove gradio, add new deps**

```bash
uv remove gradio
uv add apscheduler aiofiles "jinja2>=3.1.0"
```

- [ ] **Step 2: Move jinja2 from optional to core in `pyproject.toml`**

In `pyproject.toml`, the `viz` optional group still lists jinja2. Remove the duplicate entry from `[project.optional-dependencies]` viz section:

Find and remove `"jinja2>=3.1.0",` from the `viz` optional group (jinja2 is now in core deps via the `uv add` above).

- [ ] **Step 3: Verify sync**

```bash
uv sync
```

Expected: completes without errors, no gradio in output.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: remove gradio, add apscheduler+aiofiles+jinja2 to core"
```

---

## Task 3: Delete duplicate UI files

**Files:**
- Delete: `src/equilens/web_ui.py`, `src/equilens/gradio_app.py`, `src/equilens/start_all.py`

- [ ] **Step 1: Delete files**

```bash
rm src/equilens/web_ui.py src/equilens/gradio_app.py src/equilens/start_all.py
```

- [ ] **Step 2: Verify no other imports**

```bash
grep -r "web_ui\|gradio_app\|start_all" src/ --include="*.py"
```

Expected: matches only in `src/equilens/cli.py` (the `gui`, `web`, and `serve` commands — handled in Task 16).

- [ ] **Step 3: Commit**

```bash
git add src/equilens/web_ui.py src/equilens/gradio_app.py src/equilens/start_all.py
git commit -m "chore: remove legacy Gradio UI files"
```

---

## Task 4: Implement backup system (TDD)

**Files:**
- Create: `tests/unit/test_backup.py`
- Create: `src/equilens/backup.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_backup.py`:

```python
"""Tests for the backup module."""

import zipfile
from pathlib import Path

import pytest


@pytest.fixture()
def work_dir(tmp_path, monkeypatch):
    """Change cwd to tmp_path and return it."""
    monkeypatch.chdir(tmp_path)
    # Ensure patched BACKUP_DIR points inside tmp_path
    import equilens.backup as bk
    monkeypatch.setattr(bk, "BACKUP_DIR", tmp_path / "backups")
    return tmp_path


def _seed(work_dir: Path) -> None:
    (work_dir / "results" / "run1").mkdir(parents=True)
    (work_dir / "results" / "run1" / "audit.csv").write_text("a,b\n1,2")
    (work_dir / "data" / "jobs").mkdir(parents=True)
    (work_dir / "data" / "jobs" / "equilens_jobs.db").write_bytes(b"SQLite")


def test_create_backup_returns_zip_path(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    assert path.exists()
    assert path.suffix == ".zip"


def test_create_backup_zip_contains_csv(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert any("audit.csv" in n for n in names)


def test_create_backup_zip_contains_db(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert any("equilens_jobs.db" in n for n in names)


def test_prune_backups_keeps_retention(work_dir):
    from equilens.backup import BACKUP_DIR, _prune_backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        (BACKUP_DIR / f"backup_{i:04d}.zip").write_bytes(b"x")

    _prune_backups(retention=10)

    remaining = list(BACKUP_DIR.glob("backup_*.zip"))
    assert len(remaining) == 10


def test_list_backups_empty_when_no_dir(work_dir):
    from equilens.backup import list_backups

    result = list_backups()
    assert result == []


def test_list_backups_returns_sorted_newest_first(work_dir):
    import time
    from equilens.backup import BACKUP_DIR, list_backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["backup_0001.zip", "backup_0002.zip", "backup_0003.zip"]:
        p = BACKUP_DIR / name
        p.write_bytes(b"x")
        time.sleep(0.01)

    result = list_backups()
    names = [b["name"] for b in result]
    assert names == ["backup_0003.zip", "backup_0002.zip", "backup_0001.zip"]


def test_get_scheduler_status_not_running(work_dir):
    from equilens.backup import get_scheduler_status

    status = get_scheduler_status()
    assert status["running"] is False
    assert status["next_run"] is None
```

- [ ] **Step 2: Run tests — expect failures**

```bash
uv run pytest tests/unit/test_backup.py -v
```

Expected: `ModuleNotFoundError: No module named 'equilens.backup'`

- [ ] **Step 3: Implement `src/equilens/backup.py`**

```python
"""Periodic backup scheduler for EquiLens results and database."""

import os
import zipfile
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

BACKUP_DIR = Path(os.getenv("EQUILENS_BACKUP_DIR", "backups"))
DEFAULT_INTERVAL_MINUTES = int(os.getenv("EQUILENS_BACKUP_INTERVAL_MINUTES", "30"))
DEFAULT_RETENTION = int(os.getenv("EQUILENS_BACKUP_RETENTION", "10"))

_scheduler: BackgroundScheduler | None = None


def create_backup() -> Path:
    """Zip results/ and the job database into backups/backup_YYYYMMDD_HHMMSS.zip."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = BACKUP_DIR / name

    targets = [
        Path("results"),
        Path("data/jobs/equilens_jobs.db"),
    ]

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for target in targets:
                if target.is_file():
                    zf.write(target, target.name)
                elif target.is_dir():
                    for file in target.rglob("*"):
                        if file.is_file():
                            zf.write(file, file.relative_to(target.parent))
        _prune_backups()
        return zip_path
    except Exception as exc:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Backup failed: {exc}") from exc


def _prune_backups(retention: int = DEFAULT_RETENTION) -> None:
    """Remove oldest backups beyond retention limit."""
    if not BACKUP_DIR.exists():
        return
    backups = sorted(
        BACKUP_DIR.glob("backup_*.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in backups[:-retention]:
        old.unlink(missing_ok=True)


def list_backups() -> list[dict]:
    """Return all backups sorted newest-first."""
    if not BACKUP_DIR.exists():
        return []
    backups = sorted(
        BACKUP_DIR.glob("backup_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": p.name,
            "size": p.stat().st_size,
            "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
        for p in backups
    ]


def start_scheduler(interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> None:
    """Start the background backup scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        create_backup,
        "interval",
        minutes=interval_minutes,
        id="periodic_backup",
        replace_existing=True,
    )
    _scheduler.start()


def stop_scheduler() -> None:
    """Stop the backup scheduler gracefully."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def get_scheduler_status() -> dict:
    """Return scheduler state and next run time."""
    if not _scheduler or not _scheduler.running:
        return {"running": False, "next_run": None}
    job = _scheduler.get_job("periodic_backup")
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
    return {"running": True, "next_run": next_run}
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
uv run pytest tests/unit/test_backup.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/equilens/backup.py tests/unit/test_backup.py
git commit -m "feat: add periodic backup system with APScheduler"
```

---

## Task 5: Add new API endpoints to api.py

**Files:**
- Modify: `src/equilens/backend/api.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_dashboard_routes.py`:

```python
"""Integration tests for dashboard HTML routes and new API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from equilens.backend.api import app
    with TestClient(app) as c:
        yield c


# ── HTML routes ──────────────────────────────────────────────────────────────

def test_dashboard_home_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "EquiLens" in r.text


def test_audit_page_returns_html(client):
    r = client.get("/audit")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_generate_page_returns_html(client):
    r = client.get("/generate")
    assert r.status_code == 200


def test_analyze_page_returns_html(client):
    r = client.get("/analyze")
    assert r.status_code == 200


def test_jobs_page_returns_html(client):
    r = client.get("/jobs")
    assert r.status_code == 200


def test_results_page_returns_html(client):
    r = client.get("/results")
    assert r.status_code == 200


def test_static_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200


def test_static_css_served(client):
    r = client.get("/static/style.css")
    assert r.status_code == 200


# ── New JSON endpoints ────────────────────────────────────────────────────────

def test_corpus_endpoint_returns_list(client):
    r = client.get("/api/corpus")
    assert r.status_code == 200
    data = r.json()
    assert "corpus_files" in data
    assert isinstance(data["corpus_files"], list)


def test_sessions_endpoint_returns_list(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    data = r.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_backups_endpoint_returns_list(client):
    r = client.get("/api/backups")
    assert r.status_code == 200
    data = r.json()
    assert "backups" in data


def test_retry_nonexistent_job_returns_404(client):
    r = client.post("/api/jobs/nonexistent_job_id/retry")
    assert r.status_code == 404


def test_openapi_schema_valid(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "EquiLens Backend API"
    assert "paths" in schema
```

- [ ] **Step 2: Run tests — expect failures**

```bash
uv run pytest tests/unit/test_dashboard_routes.py -v 2>&1 | head -40
```

Expected: most tests fail — routes don't exist yet.

- [ ] **Step 3: Add SSE endpoint, corpus endpoint, sessions endpoint to `src/equilens/backend/api.py`**

Add these imports at the top of `api.py` (after existing imports):

```python
import asyncio
import json as json_lib

from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
```

Add SSE endpoint after the existing job endpoints section:

```python
@app.get(
    "/api/events/{job_id}",
    tags=["Jobs"],
    summary="Stream job logs via Server-Sent Events",
    description=(
        "Returns a text/event-stream of log lines for the given job. "
        "Each event is a JSON object with 'level', 'message', 'timestamp'. "
        "A final event with 'event': 'done' or 'event': 'timeout' signals stream end. "
        "Clients should reconnect automatically on disconnect."
    ),
)
async def stream_job_events(job_id: str):
    """SSE stream of log lines for a running job."""

    async def event_generator():
        last_index = 0
        idle_ticks = 0
        max_idle = 180  # 3 minutes of no new logs before timeout

        while True:
            logs = JobDatabase.get_logs(job_id, limit=500)
            new_logs = logs[last_index:]
            for log in new_logs:
                payload = json_lib.dumps(
                    {
                        "level": log["level"],
                        "message": log["message"],
                        "timestamp": log.get("timestamp", ""),
                    }
                )
                yield f"data: {payload}\n\n"
                last_index += 1
                idle_ticks = 0

            job = JobDatabase.get_job(job_id)
            if job and job["status"] in ("completed", "failed", "cancelled"):
                yield f"data: {json_lib.dumps({'event': 'done', 'status': job['status']})}\n\n"
                break

            idle_ticks += 1
            if idle_ticks > max_idle:
                yield f"data: {json_lib.dumps({'event': 'timeout'})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Add corpus and sessions endpoints:

```python
@app.get(
    "/api/corpus",
    tags=["Configuration"],
    summary="List discovered corpus CSV files",
    description="Scans known corpus directories and returns available CSV files with row counts.",
)
async def list_corpus_files():
    """Discover corpus CSV files from standard locations."""
    import csv

    search_paths = [
        Path("corpus"),
        Path("src/Phase1_CorpusGenerator/corpus"),
    ]

    found: list[dict] = []
    seen: set[str] = set()

    for sp in search_paths:
        if not sp.exists():
            continue
        for csv_path in sorted(sp.glob("*.csv")):
            key = str(csv_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            row_count = 0
            try:
                with csv_path.open(encoding="utf-8") as f:
                    row_count = sum(1 for _ in csv.reader(f)) - 1
            except Exception:
                pass
            found.append(
                {
                    "name": csv_path.name,
                    "path": str(csv_path),
                    "rows": row_count,
                    "size": csv_path.stat().st_size,
                }
            )

    return {"corpus_files": found}


@app.get(
    "/api/sessions",
    tags=["Jobs"],
    summary="List interruptible audit sessions that can be resumed",
    description="Scans results/ for progress JSON files belonging to incomplete audits.",
)
async def list_interrupted_sessions():
    """Find audit sessions that can be resumed."""
    sessions: list[dict] = []
    results_dir = Path("results")

    if not results_dir.exists():
        return {"sessions": []}

    for session_dir in results_dir.iterdir():
        if not session_dir.is_dir():
            continue
        for progress_file in session_dir.glob("progress_*.json"):
            try:
                with progress_file.open(encoding="utf-8") as f:
                    data = json_lib.load(f)
                completed = data.get("completed_tests", 0)
                total = data.get("total_tests", 0)
                if completed < total:
                    sessions.append(
                        {
                            "progress_file": str(progress_file),
                            "model_name": data.get("model_name", "Unknown"),
                            "completed_tests": completed,
                            "total_tests": total,
                            "completion_pct": round(completed / total * 100)
                            if total
                            else 0,
                            "started_at": data.get("start_time", ""),
                        }
                    )
            except Exception:
                continue

    sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return {"sessions": sessions}
```

- [ ] **Step 4: Add job retry endpoint**

Add after the delete job endpoint in `api.py`:

```python
@app.post(
    "/api/jobs/{job_id}/retry",
    tags=["Jobs"],
    summary="Retry a failed job with the same configuration",
    description="Creates a new job using the same config as the specified failed job.",
    responses={
        200: {"description": "New job created"},
        400: {"description": "Job is not in a failed state"},
        404: {"description": "Job not found"},
    },
)
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Retry a failed job by re-queuing it with original config."""
    job_data = JobDatabase.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_data["status"] not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed/cancelled jobs, current status: {job_data['status']}",
        )

    import json as _json

    original_config = _json.loads(job_data.get("config") or "{}")
    job_type = job_data["job_type"]

    new_job_id = f"{job_type}_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    JobDatabase.create_job(job_id=new_job_id, job_type=job_type, config=original_config)

    if job_type == "corpus_generation":
        background_tasks.add_task(run_corpus_generation_job, new_job_id, original_config)
    elif job_type == "audit":
        background_tasks.add_task(run_audit_job, new_job_id, original_config)
    elif job_type == "analysis":
        background_tasks.add_task(run_analysis_job, new_job_id, original_config)

    return {"original_job_id": job_id, "new_job_id": new_job_id}
```

- [ ] **Step 5: Add backup endpoints**

Add after the results section in `api.py`:

```python
# ===== Backup Endpoints =====


@app.get(
    "/api/backups",
    tags=["Backups"],
    summary="List all available backups",
    description="Returns backups sorted newest-first with name, size, and creation timestamp.",
)
async def list_backups_endpoint():
    """List all backup archives."""
    from equilens.backup import list_backups

    return {"backups": list_backups()}


@app.post(
    "/api/backups",
    tags=["Backups"],
    summary="Trigger an immediate backup",
    description="Creates a backup zip of results/ and the job database right now.",
    responses={
        200: {"description": "Backup created successfully"},
        500: {"description": "Backup failed"},
    },
)
async def trigger_backup():
    """Create a backup immediately."""
    from equilens.backup import create_backup

    try:
        path = create_backup()
        return {"message": "Backup created", "path": str(path), "name": path.name}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/api/backups/{name}",
    tags=["Backups"],
    summary="Download a backup archive",
    responses={
        200: {"description": "ZIP file download"},
        404: {"description": "Backup not found"},
    },
)
async def download_backup(name: str):
    """Download a backup ZIP by filename."""
    from equilens.backup import BACKUP_DIR

    path = BACKUP_DIR / name
    if not path.exists() or path.suffix != ".zip":
        raise HTTPException(status_code=404, detail="Backup not found")
    return FileResponse(path, media_type="application/zip", filename=name)


@app.delete(
    "/api/backups/{name}",
    tags=["Backups"],
    summary="Delete a backup archive",
    responses={
        200: {"description": "Deleted"},
        404: {"description": "Backup not found"},
    },
)
async def delete_backup(name: str):
    """Delete a specific backup ZIP."""
    from equilens.backup import BACKUP_DIR

    path = BACKUP_DIR / name
    if not path.exists() or path.suffix != ".zip":
        raise HTTPException(status_code=404, detail="Backup not found")
    path.unlink()
    return {"message": f"Backup {name} deleted"}
```

- [ ] **Step 6: Add a summary endpoint for the dashboard home**

```python
@app.get(
    "/api/dashboard",
    tags=["System"],
    summary="Dashboard summary — system status, active jobs, backup state",
    description="Single endpoint returning all data needed to render the dashboard home page.",
)
async def dashboard_summary():
    """Aggregate status for the dashboard home page."""
    from equilens.backup import get_scheduler_status, list_backups

    # System status (reuse existing logic)
    docker_detected = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"
    ollama_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://ollama:11434" if docker_detected else "http://localhost:11434",
    )
    ollama_available = False
    try:
        import requests as _req

        ollama_available = _req.get(f"{ollama_url}/api/tags", timeout=3).status_code == 200
    except Exception:
        pass

    active_jobs = JobDatabase.list_jobs(status="running", limit=100)
    recent_jobs = JobDatabase.list_jobs(limit=5)
    backups = list_backups()
    scheduler = get_scheduler_status()

    return {
        "ollama_available": ollama_available,
        "ollama_url": ollama_url,
        "docker_detected": docker_detected,
        "active_jobs_count": len(active_jobs),
        "recent_jobs": recent_jobs,
        "backup_count": len(backups),
        "last_backup": backups[0] if backups else None,
        "scheduler": scheduler,
    }
```

- [ ] **Step 7: Run tests — expect most pass (HTML routes still fail)**

```bash
uv run pytest tests/unit/test_dashboard_routes.py::test_corpus_endpoint_returns_list tests/unit/test_dashboard_routes.py::test_sessions_endpoint_returns_list tests/unit/test_dashboard_routes.py::test_backups_endpoint_returns_list tests/unit/test_dashboard_routes.py::test_retry_nonexistent_job_returns_404 tests/unit/test_dashboard_routes.py::test_openapi_schema_valid -v
```

Expected: 5 tests PASSED.

- [ ] **Step 8: Commit**

```bash
git add src/equilens/backend/api.py
git commit -m "feat: add SSE, corpus, sessions, retry, backup, dashboard API endpoints"
```

---

## Task 6: Wire dashboard into api.py (static files + router + lifespan)

**Files:**
- Modify: `src/equilens/backend/api.py`
- Create: `src/equilens/dashboard/__init__.py`
- Create: `src/equilens/dashboard/routes.py`

- [ ] **Step 1: Create dashboard package**

Create `src/equilens/dashboard/__init__.py` (empty):

```python
```

Create `src/equilens/dashboard/routes.py`:

```python
"""Jinja2 HTML page routes for the EquiLens dashboard."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/audit", response_class=HTMLResponse)
async def page_audit(request: Request):
    return templates.TemplateResponse("audit.html", {"request": request})


@router.get("/generate", response_class=HTMLResponse)
async def page_generate(request: Request):
    return templates.TemplateResponse("generate.html", {"request": request})


@router.get("/analyze", response_class=HTMLResponse)
async def page_analyze(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})


@router.get("/jobs", response_class=HTMLResponse)
async def page_jobs(request: Request):
    return templates.TemplateResponse("jobs.html", {"request": request})


@router.get("/results", response_class=HTMLResponse)
async def page_results(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})
```

- [ ] **Step 2: Update `api.py` — replace startup event with lifespan, mount static, include dashboard router**

At the top of `api.py`, add these imports after existing imports:

```python
from contextlib import asynccontextmanager

from fastapi.staticfiles import StaticFiles
```

Replace the existing `@app.on_event("startup")` block and the `app = FastAPI(...)` call with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    import os as _os
    from equilens.backup import start_scheduler, stop_scheduler

    interval = int(_os.getenv("EQUILENS_BACKUP_INTERVAL_MINUTES", "30"))
    start_scheduler(interval_minutes=interval)
    print("✅ EquiLens Backend API started — Dashboard at http://localhost:8000")
    yield
    # Shutdown
    stop_scheduler()


app = FastAPI(
    title="EquiLens Backend API",
    description=(
        "REST API and dashboard for the EquiLens AI bias detection platform. "
        "Dashboard: http://localhost:8000 | API docs: http://localhost:8000/docs"
    ),
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and include dashboard router BEFORE API routes
_STATIC_DIR = Path(__file__).parent.parent / "dashboard" / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

from equilens.dashboard.routes import router as _dashboard_router  # noqa: E402

app.include_router(_dashboard_router)
```

- [ ] **Step 3: Create placeholder templates so tests don't error on missing files**

Create required template directories:

```bash
mkdir -p src/equilens/dashboard/templates
mkdir -p src/equilens/dashboard/static
```

Create a minimal `src/equilens/dashboard/templates/base.html` (will be replaced in Task 9):

```html
<!DOCTYPE html>
<html><head><title>EquiLens</title></head>
<body>{% block content %}{% endblock %}</body></html>
```

Create stub templates for each page (will be replaced in Tasks 10-15):

For each of: `dashboard.html`, `audit.html`, `generate.html`, `analyze.html`, `jobs.html`, `results.html` — create:

```html
{% extends "base.html" %}
{% block content %}<h1>EquiLens</h1>{% endblock %}
```

Create empty `src/equilens/dashboard/static/app.js` and `src/equilens/dashboard/static/style.css`.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/unit/test_dashboard_routes.py -v
```

Expected: all 15 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/equilens/dashboard/ tests/unit/test_dashboard_routes.py
git commit -m "feat: wire dashboard router and static mount into FastAPI app"
```

---

## Task 7: OpenAPI annotations on all existing endpoints

**Files:**
- Modify: `src/equilens/backend/api.py`

- [ ] **Step 1: Add tags to all existing endpoints**

In `api.py`, add `tags`, `summary`, and `description` to the following endpoints that lack them. Apply the pattern below to each:

```python
# Root
@app.get("/", tags=["System"], summary="API root", description="Returns service name and version.")

# Health
@app.get("/api/health", tags=["System"], summary="Health check", description="Returns healthy status and current timestamp.")

# Status
@app.get("/api/status", tags=["System"], summary="System status", description="Checks Docker presence, Ollama availability and URL.")

# POST /api/jobs
@app.post("/api/jobs", tags=["Jobs"], summary="Create and queue a new job", description="Accepts job_type (corpus_generation|audit|analysis) and config dict. Returns job details immediately while running in background.")

# GET /api/jobs/{job_id}
@app.get("/api/jobs/{job_id}", tags=["Jobs"], summary="Get job by ID", responses={404: {"description": "Job not found"}})

# GET /api/jobs
@app.get("/api/jobs", tags=["Jobs"], summary="List jobs", description="Returns up to `limit` jobs, optionally filtered by status (pending|running|completed|failed|cancelled).")

# POST /api/jobs/{job_id}/cancel
@app.post("/api/jobs/{job_id}/cancel", tags=["Jobs"], summary="Cancel a running job", responses={400: {"description": "Cancel failed"}})

# DELETE /api/jobs/{job_id}
@app.delete("/api/jobs/{job_id}", tags=["Jobs"], summary="Delete job and its logs", responses={404: {"description": "Job not found"}})

# GET /api/jobs/{job_id}/logs
@app.get("/api/jobs/{job_id}/logs", tags=["Jobs"], summary="Get job log lines", description="Returns last `limit` log lines for the job.")

# GET /api/results
@app.get("/api/results", tags=["Results"], summary="List result directories", description="Scans results/ for directories containing CSV output.")

# GET /api/results/{result_name}/export
@app.get("/api/results/{result_name}/export", tags=["Results"], summary="Download results as ZIP", responses={404: {"description": "Results not found"}, 500: {"description": "Export creation failed"}})

# GET /api/results/{result_name}/html
@app.get("/api/results/{result_name}/html", tags=["Results"], summary="Get HTML bias report", responses={404: {"description": "HTML report not found"}})

# GET /api/models
@app.get("/api/models", tags=["Models"], summary="List available Ollama models", description="Queries Ollama /api/tags and returns model name, size, and modified date.")

# POST /api/models/pull
@app.post("/api/models/pull", tags=["Models"], summary="Pull an Ollama model", description="Downloads the specified model in the background. Returns a job_id to track progress.")
```

- [ ] **Step 2: Verify OpenAPI schema test still passes**

```bash
uv run pytest tests/unit/test_dashboard_routes.py::test_openapi_schema_valid -v
```

Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/equilens/backend/api.py
git commit -m "docs: add full OpenAPI tags, summaries, descriptions to all endpoints"
```

---

## Task 8: style.css — design system

**Files:**
- Create: `src/equilens/dashboard/static/style.css`

- [ ] **Step 1: Write `style.css`**

```css
/* EquiLens Dashboard — CSS custom properties + minimal component styles */

:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface-2: #22263a;
  --border: #2d3050;
  --text: #e2e8f0;
  --text-muted: #8892a4;
  --primary: #6366f1;
  --primary-h: #4f52d8;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
  --radius: 8px;
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
  }
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.6;
}

/* ── Layout ── */
.layout { display: grid; grid-template-columns: 220px 1fr; min-height: 100vh; }

.sidebar {
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 20px 12px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 8px 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 12px;
  font-weight: 700;
  font-size: 16px;
}

.nav-links { list-style: none; flex: 1; }
.nav-links li { margin-bottom: 2px; }

.nav-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--radius);
  color: var(--text-muted);
  text-decoration: none;
  font-size: 13px;
  transition: background 0.1s, color 0.1s;
}
.nav-link:hover { background: var(--surface-2); color: var(--text); }
.nav-link.active { background: var(--surface-2); color: var(--primary); font-weight: 600; }

.sidebar-footer { border-top: 1px solid var(--border); padding-top: 12px; margin-top: auto; }

.main-content { padding: 32px; max-width: 1100px; }

/* ── Page header ── */
.page-header { margin-bottom: 28px; }
.page-title { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
.page-subtitle { color: var(--text-muted); }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
}
.card-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 14px;
}

/* ── Status grid ── */
.status-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
.status-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
.status-value { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.status-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Badges ── */
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.badge-running  { background: rgba(99,102,241,.18); color: var(--primary); }
.badge-completed{ background: rgba(34,197,94,.18);  color: var(--success); }
.badge-failed   { background: rgba(239,68,68,.18);  color: var(--danger); }
.badge-pending  { background: rgba(245,158,11,.18); color: var(--warning); }
.badge-cancelled{ background: rgba(139,148,158,.18);color: var(--text-muted); }

/* ── Buttons ── */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: var(--radius);
  border: 1px solid transparent;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.15s;
  background: none;
  color: var(--text);
}
.btn-primary { background: var(--primary); color: #fff; border-color: var(--primary); }
.btn-primary:hover:not(:disabled) { background: var(--primary-h); }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; }
.btn-secondary { background: var(--surface-2); border-color: var(--border); }
.btn-secondary:hover { border-color: var(--primary); color: var(--primary); }
.btn-danger { color: var(--danger); border-color: var(--danger); }
.btn-danger:hover { background: rgba(239,68,68,.1); }
.btn-sm { padding: 4px 10px; font-size: 12px; }

/* ── Forms ── */
.form-group { margin-bottom: 16px; }
.form-label {
  display: block;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.form-control {
  width: 100%;
  padding: 8px 12px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}
.form-control:focus { border-color: var(--primary); }
select.form-control option { background: var(--surface-2); }
.form-hint { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

/* ── Log viewer ── */
.log-viewer {
  background: #080a12;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  font-family: 'Cascadia Code', 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  height: 320px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.log-line { margin-bottom: 1px; }
.log-error   { color: var(--danger); }
.log-warning { color: var(--warning); }
.log-info    { color: #94a3b8; }

/* ── Progress bar ── */
.progress-wrap { background: var(--surface-2); border-radius: 999px; height: 6px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--primary); border-radius: 999px; transition: width .4s ease; }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); font-weight: 700; }
tr:hover td { background: var(--surface-2); }

/* ── Error banner ── */
.error-banner {
  position: fixed; top: 0; left: 0; right: 0;
  background: var(--danger); color: #fff;
  padding: 10px 20px;
  display: flex; justify-content: space-between; align-items: center;
  z-index: 1000; font-size: 13px;
}
.error-banner button { background: none; border: none; color: #fff; cursor: pointer; font-size: 20px; line-height: 1; }

/* ── Indicator dot ── */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.dot-green  { background: var(--success); box-shadow: 0 0 6px var(--success); }
.dot-red    { background: var(--danger); }
.dot-yellow { background: var(--warning); }

/* ── Collapsible ── */
.collapsible-trigger {
  display: flex; align-items: center; justify-content: space-between;
  cursor: pointer; font-size: 12px; color: var(--text-muted);
  padding: 8px 0; border-top: 1px solid var(--border); margin-top: 12px;
  user-select: none;
}

/* ── Checkbox toggle ── */
.toggle-row { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 10px; }
.toggle-label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; }

/* ── iframe ── */
.report-frame { width: 100%; height: 600px; border: 1px solid var(--border); border-radius: var(--radius); background: white; }

/* ── Utility ── */
.flex { display: flex; } .gap-2 { gap: 8px; } .gap-3 { gap: 12px; }
.mt-2 { margin-top: 8px; } .mt-3 { margin-top: 12px; } .mt-4 { margin-top: 16px; }
.text-muted { color: var(--text-muted); } .text-sm { font-size: 12px; }
.text-success { color: var(--success); } .text-danger { color: var(--danger); }
[x-cloak] { display: none !important; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.pulse { animation: pulse 1.5s ease-in-out infinite; }
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/static/style.css
git commit -m "feat: add dashboard CSS design system with dark/light theme"
```

---

## Task 9: app.js — Alpine.js stores and utilities

**Files:**
- Create: `src/equilens/dashboard/static/app.js`

- [ ] **Step 1: Write `app.js`**

```javascript
// EquiLens Dashboard — Alpine.js utilities and fetch wrappers

// ── Fetch with timeout + retry ────────────────────────────────────────────────

async function apiFetch(url, options = {}, retries = 2) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), 10000);
  try {
    const r = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    if (!r.ok) {
      const body = await r.text().catch(() => r.statusText);
      throw new Error(`HTTP ${r.status}: ${body}`);
    }
    return r;
  } catch (err) {
    clearTimeout(id);
    if (retries > 0 && err.name !== 'AbortError') {
      await new Promise(res => setTimeout(res, 1000));
      return apiFetch(url, options, retries - 1);
    }
    throw err;
  }
}

async function apiGet(url) {
  const r = await apiFetch(url);
  return r.json();
}

async function apiPost(url, body) {
  const r = await apiFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function apiDelete(url) {
  const r = await apiFetch(url, { method: 'DELETE' });
  return r.json();
}

// ── SSE job log streamer ──────────────────────────────────────────────────────
// Returns a stop() function. Calls onLog(logObj) for each log line.
// Calls onDone(status) when the job finishes or times out.

function streamJobLogs(jobId, onLog, onDone) {
  const es = new EventSource(`/api/events/${jobId}`);

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.event === 'done' || data.event === 'timeout') {
        es.close();
        onDone(data.status || data.event);
      } else {
        onLog(data);
      }
    } catch (_) {}
  };

  es.onerror = () => {
    // EventSource reconnects automatically — just log silently
    console.debug('SSE reconnecting for job', jobId);
  };

  return () => es.close();
}

// ── UI helpers ───────────────────────────────────────────────────────────────

function badgeClass(status) {
  return {
    running:   'badge-running',
    completed: 'badge-completed',
    failed:    'badge-failed',
    pending:   'badge-pending',
    cancelled: 'badge-cancelled',
  }[status] || 'badge-pending';
}

function dotClass(status) {
  return {
    running:   'dot-green pulse',
    completed: 'dot-green',
    failed:    'dot-red',
    pending:   'dot-yellow',
    cancelled: 'dot-yellow',
  }[status] || 'dot-yellow';
}

function fmtBytes(bytes) {
  if (!bytes) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' });
}

function fmtDuration(startIso, endIso) {
  if (!startIso) return '—';
  const s = new Date(startIso);
  const e = endIso ? new Date(endIso) : new Date();
  const sec = Math.round((e - s) / 1000);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

// Auto-scroll a container to bottom
function scrollBottom(el) {
  if (el) el.scrollTop = el.scrollHeight;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/static/app.js
git commit -m "feat: add Alpine.js utilities, SSE client, fetch wrappers"
```

---

## Task 10: base.html — layout template

**Files:**
- Create: `src/equilens/dashboard/templates/base.html`

- [ ] **Step 1: Write `base.html`**

```html
<!DOCTYPE html>
<html lang="en" x-cloak>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EquiLens — {% block title %}Dashboard{% endblock %}</title>
  <link rel="stylesheet" href="/static/style.css">
  <!-- Alpine.js — reactive UI, no build step -->
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
  <!-- Chart.js — bias score charts -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.x/dist/chart.umd.min.js"></script>
  <script src="/static/app.js" defer></script>
</head>
<body>

<!-- Global error banner -->
<div id="error-banner" class="error-banner" style="display:none">
  <span id="error-msg"></span>
  <button onclick="document.getElementById('error-banner').style.display='none'">×</button>
</div>
<script>
  window.showError = function(msg) {
    document.getElementById('error-msg').textContent = msg;
    document.getElementById('error-banner').style.display = 'flex';
    setTimeout(() => document.getElementById('error-banner').style.display='none', 8000);
  };
  // Patch apiFetch to surface errors globally
  window._origApiFetch = window.apiFetch;
</script>

<div class="layout">
  <!-- Sidebar -->
  <nav class="sidebar">
    <div class="logo">
      <span>🔍</span>
      <span>EquiLens</span>
    </div>

    <ul class="nav-links">
      <li><a href="/"        class="nav-link {% block nav_home %}{% endblock %}">📊 Dashboard</a></li>
      <li><a href="/audit"   class="nav-link {% block nav_audit %}{% endblock %}">🔍 Audit</a></li>
      <li><a href="/generate"class="nav-link {% block nav_generate %}{% endblock %}">⚙️ Generate</a></li>
      <li><a href="/analyze" class="nav-link {% block nav_analyze %}{% endblock %}">📈 Analyze</a></li>
      <li><a href="/jobs"    class="nav-link {% block nav_jobs %}{% endblock %}">🗃️ Jobs</a></li>
      <li><a href="/results" class="nav-link {% block nav_results %}{% endblock %}">📁 Results</a></li>
    </ul>

    <div class="sidebar-footer">
      <a href="/docs" target="_blank" class="nav-link">📖 API Docs</a>
      <a href="/openapi.json" target="_blank" class="nav-link">📄 OpenAPI</a>
    </div>
  </nav>

  <!-- Main -->
  <main class="main-content">
    {% block content %}{% endblock %}
  </main>
</div>
</body>
</html>
```

- [ ] **Step 2: Run route tests to confirm base template still works**

```bash
uv run pytest tests/unit/test_dashboard_routes.py -v -k "returns_html"
```

Expected: 6 PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/equilens/dashboard/templates/base.html
git commit -m "feat: add dashboard base layout template with Alpine.js + Chart.js CDN"
```

---

## Task 11: dashboard.html — system status home page

**Files:**
- Create: `src/equilens/dashboard/templates/dashboard.html`

- [ ] **Step 1: Write `dashboard.html`**

```html
{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block nav_home %}active{% endblock %}

{% block content %}
<div class="page-header">
  <h1 class="page-title">📊 Dashboard</h1>
  <p class="page-subtitle">System health and recent activity</p>
</div>

<div x-data="dashPage()" x-init="init()">

  <!-- Status grid -->
  <div class="status-grid">
    <div class="status-card">
      <div class="status-value">
        <span class="dot" :class="ollama ? 'dot-green' : 'dot-red'"></span>
        <span x-text="ollama ? 'Online' : 'Offline'"></span>
      </div>
      <div class="status-label">Ollama</div>
    </div>
    <div class="status-card">
      <div class="status-value" x-text="activeJobs"></div>
      <div class="status-label">Active Jobs</div>
    </div>
    <div class="status-card">
      <div class="status-value" x-text="backupCount"></div>
      <div class="status-label">Backups Stored</div>
    </div>
    <div class="status-card">
      <div class="status-value">
        <span class="dot" :class="docker ? 'dot-green' : 'dot-yellow'"></span>
        <span x-text="docker ? 'Docker' : 'Local'"></span>
      </div>
      <div class="status-label">Environment</div>
    </div>
  </div>

  <!-- Ollama URL -->
  <div class="card">
    <div class="card-title">Ollama Endpoint</div>
    <code x-text="ollamaUrl" style="font-size:13px; color: var(--primary);"></code>
  </div>

  <!-- Backup status -->
  <div class="card">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
      <div>
        <div class="card-title">Backup Status</div>
        <p class="text-sm">
          Scheduler:
          <span x-text="schedulerRunning ? '✅ Running' : '⚠️ Stopped'"></span>
        </p>
        <p class="text-sm text-muted mt-2">
          Next backup: <span x-text="nextBackup || '—'"></span>
        </p>
        <p class="text-sm text-muted">
          Last backup: <span x-text="lastBackup ? fmtDate(lastBackup.created_at) : 'None yet'"></span>
          <span x-show="lastBackup" x-text="lastBackup ? '(' + fmtBytes(lastBackup.size) + ')' : ''"></span>
        </p>
      </div>
      <button class="btn btn-secondary btn-sm" @click="triggerBackup()" :disabled="backingUp">
        <span x-show="!backingUp">💾 Backup Now</span>
        <span x-show="backingUp">Creating...</span>
      </button>
    </div>
    <p x-show="backupMsg" x-text="backupMsg" class="text-sm text-success mt-2"></p>
    <p x-show="backupErr" x-text="backupErr" class="text-sm text-danger mt-2"></p>
  </div>

  <!-- Recent jobs -->
  <div class="card">
    <div class="card-title">Recent Jobs</div>
    <div x-show="recentJobs.length === 0" class="text-muted text-sm">No jobs yet.</div>
    <table x-show="recentJobs.length > 0">
      <thead><tr><th>ID</th><th>Type</th><th>Status</th><th>Created</th></tr></thead>
      <tbody>
        <template x-for="j in recentJobs" :key="j.job_id">
          <tr style="cursor:pointer" @click="window.location='/jobs'">
            <td><code style="font-size:11px;" x-text="j.job_id.slice(0,24) + '…'"></code></td>
            <td x-text="j.job_type"></td>
            <td><span class="badge" :class="badgeClass(j.status)" x-text="j.status"></span></td>
            <td x-text="fmtDate(j.created_at)" class="text-muted text-sm"></td>
          </tr>
        </template>
      </tbody>
    </table>
    <div class="mt-3"><a href="/jobs" class="btn btn-secondary btn-sm">View All Jobs →</a></div>
  </div>
</div>

<script>
function dashPage() {
  return {
    ollama: false, ollamaUrl: '—', docker: false,
    activeJobs: 0, backupCount: 0,
    lastBackup: null, schedulerRunning: false, nextBackup: null,
    recentJobs: [],
    backingUp: false, backupMsg: '', backupErr: '',

    async init() {
      await this.load();
      setInterval(() => this.load(), 30000);
    },

    async load() {
      try {
        const d = await apiGet('/api/dashboard');
        this.ollama = d.ollama_available;
        this.ollamaUrl = d.ollama_url;
        this.docker = d.docker_detected;
        this.activeJobs = d.active_jobs_count;
        this.backupCount = d.backup_count;
        this.lastBackup = d.last_backup;
        this.schedulerRunning = d.scheduler.running;
        this.nextBackup = d.scheduler.next_run ? fmtDate(d.scheduler.next_run) : null;
        this.recentJobs = d.recent_jobs || [];
      } catch (e) {
        showError('Failed to load dashboard: ' + e.message);
      }
    },

    async triggerBackup() {
      this.backingUp = true; this.backupMsg = ''; this.backupErr = '';
      try {
        const r = await apiPost('/api/backups', {});
        this.backupMsg = '✅ Backup created: ' + r.name;
        await this.load();
      } catch (e) {
        this.backupErr = '❌ ' + e.message;
      } finally {
        this.backingUp = false;
      }
    },

    badgeClass, fmtDate, fmtBytes,
  };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/templates/dashboard.html
git commit -m "feat: add dashboard home page (status, backups, recent jobs)"
```

---

## Task 12: audit.html — audit form with live SSE log viewer

**Files:**
- Create: `src/equilens/dashboard/templates/audit.html`

- [ ] **Step 1: Write `audit.html`**

```html
{% extends "base.html" %}
{% block title %}Audit{% endblock %}
{% block nav_audit %}active{% endblock %}

{% block content %}
<div class="page-header">
  <h1 class="page-title">🔍 Bias Audit</h1>
  <p class="page-subtitle">Run a bias audit against an Ollama model</p>
</div>

<div x-data="auditPage()" x-init="init()">

  <!-- Active job banner -->
  <div x-show="jobId && !['completed','failed','cancelled'].includes(jobStatus)" class="card" style="border-color:var(--primary);">
    <div class="flex gap-3" style="align-items:center;">
      <span class="dot dot-green pulse"></span>
      <span>Running job: <code x-text="jobId" style="font-size:12px;"></code></span>
      <span class="badge" :class="badgeClass(jobStatus)" x-text="jobStatus"></span>
      <button class="btn btn-danger btn-sm" @click="cancelJob()">Cancel</button>
    </div>
  </div>

  <!-- Configuration form (hidden while job is running) -->
  <div class="card" x-show="!jobId || ['completed','failed','cancelled'].includes(jobStatus)">
    <div class="card-title">Audit Configuration</div>

    <div class="form-group">
      <label class="form-label">Model *</label>
      <select class="form-control" x-model="cfg.model_name">
        <option value="" disabled>Select a model…</option>
        <template x-for="m in models" :key="m.name">
          <option :value="m.name" x-text="m.name + ' (' + fmtBytes(m.size) + ')'"></option>
        </template>
      </select>
      <p class="form-hint" x-show="models.length === 0">No models found — is Ollama running?</p>
    </div>

    <div class="form-group">
      <label class="form-label">Corpus File</label>
      <select class="form-control" x-model="cfg.corpus_file">
        <option value="">Auto-detect</option>
        <template x-for="c in corpus" :key="c.path">
          <option :value="c.path" x-text="c.name + ' — ' + c.rows + ' rows'"></option>
        </template>
      </select>
    </div>

    <div class="form-group">
      <label class="form-label">Output Directory</label>
      <input class="form-control" type="text" x-model="cfg.output_dir">
    </div>

    <!-- Interrupted sessions -->
    <div x-show="sessions.length > 0" class="card mt-3" style="border-color:var(--warning);">
      <div class="card-title">⚠️ Interrupted Sessions (can be resumed)</div>
      <template x-for="s in sessions" :key="s.progress_file">
        <div class="flex gap-3 mt-2" style="align-items:center; justify-content:space-between;">
          <div>
            <strong x-text="s.model_name"></strong>
            <span class="badge badge-pending" x-text="s.completion_pct + '%'"></span>
            <p class="text-sm text-muted" x-text="fmtDate(s.started_at)"></p>
          </div>
          <button class="btn btn-secondary btn-sm" @click="resume(s.progress_file)">Resume</button>
        </div>
      </template>
    </div>

    <!-- Advanced options -->
    <div class="collapsible-trigger" @click="adv = !adv">
      <span>Advanced Options</span>
      <span x-text="adv ? '▲' : '▼'"></span>
    </div>
    <div x-show="adv" x-transition style="padding-top:12px;">
      <div class="form-row">
        <div class="form-group">
          <label class="form-label">Temperature</label>
          <input class="form-control" type="number" step="0.05" min="0" max="2" x-model.number="cfg.temperature">
        </div>
        <div class="form-group">
          <label class="form-label">Batch Size</label>
          <input class="form-control" type="number" min="1" max="20" x-model.number="cfg.batch_size">
        </div>
        <div class="form-group">
          <label class="form-label">Max Retries</label>
          <input class="form-control" type="number" min="0" max="10" x-model.number="cfg.max_retries">
        </div>
        <div class="form-group">
          <label class="form-label">Request Timeout (s)</label>
          <input class="form-control" type="number" min="10" max="300" x-model.number="cfg.request_timeout">
        </div>
        <div class="form-group">
          <label class="form-label">Num Predict</label>
          <input class="form-control" type="number" min="8" max="512" x-model.number="cfg.num_predict">
        </div>
        <div class="form-group">
          <label class="form-label">Retry Batch Size</label>
          <input class="form-control" type="number" min="1" max="20" x-model.number="cfg.retry_batch_size">
        </div>
      </div>
      <div class="toggle-row">
        <label class="toggle-label"><input type="checkbox" x-model="cfg.enhanced"> Enhanced mode</label>
        <label class="toggle-label"><input type="checkbox" x-model="cfg.logprobs"> Logprobs</label>
        <label class="toggle-label"><input type="checkbox" x-model="cfg.silent"> Silent</label>
        <label class="toggle-label"><input type="checkbox" x-model="cfg.retry_immediate"> Retry immediate</label>
      </div>
    </div>

    <div class="flex gap-2 mt-4">
      <button class="btn btn-primary" @click="start()" :disabled="!cfg.model_name || busy">
        <span x-show="!busy">🔍 Start Audit</span>
        <span x-show="busy">Starting…</span>
      </button>
    </div>
    <p x-show="err" x-text="err" class="text-danger text-sm mt-2"></p>
  </div>

  <!-- Live log viewer -->
  <div class="card" x-show="jobId">
    <div class="card-title">Live Output</div>
    <div class="log-viewer" id="audit-log">
      <template x-for="(l, i) in logs" :key="i">
        <div class="log-line" :class="'log-' + (l.level || 'info')" x-text="l.message"></div>
      </template>
      <div x-show="logs.length === 0" class="text-muted">Waiting for output…</div>
    </div>
    <div x-show="jobStatus === 'completed'" class="flex gap-2 mt-3">
      <a href="/results" class="btn btn-secondary">📁 View Results</a>
      <a href="/jobs" class="btn btn-secondary">🗃️ View Jobs</a>
    </div>
    <div x-show="jobStatus === 'failed'" class="mt-3">
      <p class="text-danger text-sm">Job failed. <a href="/jobs" style="color:var(--primary);">View logs in Jobs tab.</a></p>
      <button class="btn btn-secondary btn-sm mt-2" @click="retryJob()">🔄 Retry</button>
    </div>
  </div>
</div>

<script>
function auditPage() {
  return {
    models: [], corpus: [], sessions: [],
    adv: false, busy: false, err: '',
    jobId: null, jobStatus: null, logs: [],
    stopStream: null,
    cfg: {
      model_name: '', corpus_file: '', output_dir: 'results',
      enhanced: true, logprobs: true, silent: false, retry_immediate: false,
      temperature: 0.2, batch_size: 5, max_retries: 2,
      request_timeout: 45, num_predict: 32, retry_batch_size: 5,
    },

    async init() {
      try {
        const [md, cd, sd] = await Promise.all([
          apiGet('/api/models'),
          apiGet('/api/corpus'),
          apiGet('/api/sessions'),
        ]);
        this.models = md.models || [];
        this.corpus = cd.corpus_files || [];
        this.sessions = sd.sessions || [];
        if (this.models.length) this.cfg.model_name = this.models[0].name;
      } catch (e) { showError('Init failed: ' + e.message); }
    },

    async start(resumePath) {
      this.busy = true; this.err = ''; this.logs = [];
      const config = resumePath
        ? { ...this.cfg, resume: resumePath }
        : { ...this.cfg };
      try {
        const job = await apiPost('/api/jobs', { job_type: 'audit', config });
        this.jobId = job.job_id;
        this.jobStatus = 'running';
        this.stopStream = streamJobLogs(
          this.jobId,
          (l) => { this.logs.push(l); this.$nextTick(() => scrollBottom(document.getElementById('audit-log'))); },
          (s) => { this.jobStatus = s; }
        );
      } catch (e) {
        this.err = e.message;
      } finally {
        this.busy = false;
      }
    },

    resume(path) { this.start(path); },

    async cancelJob() {
      try {
        await apiFetch(`/api/jobs/${this.jobId}/cancel`, { method: 'POST' });
        this.jobStatus = 'cancelled';
        if (this.stopStream) this.stopStream();
      } catch (e) { showError('Cancel failed: ' + e.message); }
    },

    async retryJob() {
      try {
        const r = await apiPost(`/api/jobs/${this.jobId}/retry`, {});
        this.jobId = r.new_job_id;
        this.jobStatus = 'running';
        this.logs = [];
        this.stopStream = streamJobLogs(
          this.jobId,
          (l) => { this.logs.push(l); this.$nextTick(() => scrollBottom(document.getElementById('audit-log'))); },
          (s) => { this.jobStatus = s; }
        );
      } catch (e) { showError('Retry failed: ' + e.message); }
    },

    badgeClass, fmtBytes, fmtDate,
  };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/templates/audit.html
git commit -m "feat: add audit page with SSE log viewer, resume, and retry"
```

---

## Task 13: generate.html — corpus generation page

**Files:**
- Create: `src/equilens/dashboard/templates/generate.html`

- [ ] **Step 1: Write `generate.html`**

```html
{% extends "base.html" %}
{% block title %}Generate{% endblock %}
{% block nav_generate %}active{% endblock %}

{% block content %}
<div class="page-header">
  <h1 class="page-title">⚙️ Generate Corpus</h1>
  <p class="page-subtitle">Create bias test prompts from word lists</p>
</div>

<div x-data="genPage()" x-init="init()">

  <div class="card" x-show="!jobId || ['completed','failed','cancelled'].includes(jobStatus)">
    <div class="card-title">Configuration</div>

    <div class="form-group">
      <label class="form-label">Config File (optional)</label>
      <input class="form-control" type="text" x-model="cfg.config_file" placeholder="Leave blank to use defaults">
      <p class="form-hint">Path to a custom word_lists.json or config override.</p>
    </div>

    <div class="flex gap-2 mt-3">
      <button class="btn btn-primary" @click="start()" :disabled="busy">
        <span x-show="!busy">⚙️ Generate</span>
        <span x-show="busy">Starting…</span>
      </button>
    </div>
    <p x-show="err" x-text="err" class="text-danger text-sm mt-2"></p>
  </div>

  <div class="card" x-show="jobId">
    <div class="flex gap-3 mb-3" style="align-items:center;" x-show="jobStatus === 'running'">
      <span class="dot dot-green pulse"></span>
      <span>Job: <code x-text="jobId" style="font-size:12px;"></code></span>
    </div>
    <div class="card-title">Live Output</div>
    <div class="log-viewer" id="gen-log">
      <template x-for="(l, i) in logs" :key="i">
        <div class="log-line" :class="'log-' + (l.level || 'info')" x-text="l.message"></div>
      </template>
      <div x-show="logs.length === 0" class="text-muted">Waiting…</div>
    </div>
    <div x-show="jobStatus === 'completed'" class="flex gap-2 mt-3">
      <a href="/audit" class="btn btn-primary">🔍 Audit Now →</a>
    </div>
    <div x-show="jobStatus === 'failed'" class="mt-3">
      <p class="text-danger text-sm">Generation failed.</p>
      <button class="btn btn-secondary btn-sm mt-2" @click="retry()">🔄 Retry</button>
    </div>
  </div>
</div>

<script>
function genPage() {
  return {
    busy: false, err: '', jobId: null, jobStatus: null, logs: [],
    stopStream: null,
    cfg: { config_file: '' },

    async init() {},

    async start() {
      this.busy = true; this.err = ''; this.logs = [];
      try {
        const job = await apiPost('/api/jobs', { job_type: 'corpus_generation', config: this.cfg });
        this.jobId = job.job_id;
        this.jobStatus = 'running';
        this.stopStream = streamJobLogs(
          this.jobId,
          (l) => { this.logs.push(l); this.$nextTick(() => scrollBottom(document.getElementById('gen-log'))); },
          (s) => { this.jobStatus = s; }
        );
      } catch (e) {
        this.err = e.message;
      } finally {
        this.busy = false;
      }
    },

    async retry() {
      if (!this.jobId) return;
      try {
        const r = await apiPost(`/api/jobs/${this.jobId}/retry`, {});
        this.jobId = r.new_job_id; this.jobStatus = 'running'; this.logs = [];
        this.stopStream = streamJobLogs(this.jobId,
          (l) => { this.logs.push(l); this.$nextTick(() => scrollBottom(document.getElementById('gen-log'))); },
          (s) => { this.jobStatus = s; }
        );
      } catch (e) { showError('Retry failed: ' + e.message); }
    },
  };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/templates/generate.html
git commit -m "feat: add corpus generation page with SSE and retry"
```

---

## Task 14: analyze.html — analysis and inline report viewer

**Files:**
- Create: `src/equilens/dashboard/templates/analyze.html`

- [ ] **Step 1: Write `analyze.html`**

```html
{% extends "base.html" %}
{% block title %}Analyze{% endblock %}
{% block nav_analyze %}active{% endblock %}

{% block content %}
<div class="page-header">
  <h1 class="page-title">📈 Analyze Results</h1>
  <p class="page-subtitle">Generate bias analysis report from audit output</p>
</div>

<div x-data="analyzePage()" x-init="init()">

  <div class="card" x-show="!jobId || ['completed','failed','cancelled'].includes(jobStatus)">
    <div class="card-title">Select Results</div>

    <div class="form-group">
      <label class="form-label">Results Directory *</label>
      <select class="form-control" x-model="cfg.result_dir">
        <option value="">Choose a result set…</option>
        <template x-for="r in results" :key="r.name">
          <option :value="r.path" x-text="r.name + ' — ' + fmtDate(r.created)"></option>
        </template>
      </select>
      <p class="form-hint" x-show="results.length === 0">No results found. Run an audit first.</p>
    </div>

    <div class="flex gap-2 mt-3">
      <button class="btn btn-primary" @click="start()" :disabled="!cfg.result_dir || busy">
        <span x-show="!busy">📈 Analyze</span>
        <span x-show="busy">Starting…</span>
      </button>
    </div>
    <p x-show="err" x-text="err" class="text-danger text-sm mt-2"></p>
  </div>

  <!-- Log viewer -->
  <div class="card" x-show="jobId && jobStatus !== 'completed'">
    <div class="card-title">Analysis Progress</div>
    <div class="log-viewer" id="analyze-log">
      <template x-for="(l, i) in logs" :key="i">
        <div class="log-line" :class="'log-' + (l.level || 'info')" x-text="l.message"></div>
      </template>
      <div x-show="logs.length === 0" class="text-muted">Waiting…</div>
    </div>
    <div x-show="jobStatus === 'failed'" class="mt-3">
      <p class="text-danger text-sm">Analysis failed.</p>
      <button class="btn btn-secondary btn-sm" @click="retry()">🔄 Retry</button>
    </div>
  </div>

  <!-- Inline report iframe -->
  <div class="card" x-show="reportUrl">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
      <div class="card-title" style="margin:0;">Bias Analysis Report</div>
      <div class="flex gap-2">
        <a :href="reportUrl" target="_blank" class="btn btn-secondary btn-sm">Open in new tab ↗</a>
        <a :href="exportUrl" class="btn btn-secondary btn-sm" download>⬇ Download ZIP</a>
      </div>
    </div>
    <iframe :src="reportUrl" class="report-frame" title="Bias Analysis Report"></iframe>
  </div>
</div>

<script>
function analyzePage() {
  return {
    results: [], busy: false, err: '',
    jobId: null, jobStatus: null, logs: [], stopStream: null,
    reportUrl: '', exportUrl: '',
    cfg: { result_dir: '' },

    async init() {
      try {
        const d = await apiGet('/api/results');
        this.results = d.results || [];
        if (this.results.length) this.cfg.result_dir = this.results[0].path;
      } catch (e) { showError('Failed to load results: ' + e.message); }
    },

    async start() {
      this.busy = true; this.err = ''; this.logs = ''; this.reportUrl = '';
      try {
        const job = await apiPost('/api/jobs', { job_type: 'analysis', config: this.cfg });
        this.jobId = job.job_id; this.jobStatus = 'running';
        this.stopStream = streamJobLogs(
          this.jobId,
          (l) => { this.logs.push(l); this.$nextTick(() => scrollBottom(document.getElementById('analyze-log'))); },
          async (s) => {
            this.jobStatus = s;
            if (s === 'completed') await this.loadReport();
          }
        );
      } catch (e) {
        this.err = e.message;
      } finally {
        this.busy = false;
      }
    },

    async loadReport() {
      // Find result name from selected path
      const sel = this.results.find(r => r.path === this.cfg.result_dir);
      if (!sel) return;
      this.reportUrl = `/api/results/${sel.name}/html`;
      this.exportUrl = `/api/results/${sel.name}/export`;
    },

    async retry() {
      if (!this.jobId) return;
      try {
        const r = await apiPost(`/api/jobs/${this.jobId}/retry`, {});
        this.jobId = r.new_job_id; this.jobStatus = 'running'; this.logs = [];
        this.stopStream = streamJobLogs(this.jobId,
          (l) => { this.logs.push(l); this.$nextTick(() => scrollBottom(document.getElementById('analyze-log'))); },
          async (s) => { this.jobStatus = s; if (s === 'completed') await this.loadReport(); }
        );
      } catch (e) { showError('Retry failed: ' + e.message); }
    },

    fmtDate, fmtBytes,
  };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/templates/analyze.html
git commit -m "feat: add analyze page with inline iframe report viewer"
```

---

## Task 15: jobs.html — job management table

**Files:**
- Create: `src/equilens/dashboard/templates/jobs.html`

- [ ] **Step 1: Write `jobs.html`**

```html
{% extends "base.html" %}
{% block title %}Jobs{% endblock %}
{% block nav_jobs %}active{% endblock %}

{% block content %}
<div class="page-header">
  <h1 class="page-title">🗃️ Jobs</h1>
  <p class="page-subtitle">All background jobs — click a row to expand logs</p>
</div>

<div x-data="jobsPage()" x-init="init()">

  <!-- Filter bar -->
  <div class="flex gap-2" style="margin-bottom:16px;">
    <template x-for="f in ['all','running','completed','failed','pending','cancelled']" :key="f">
      <button class="btn btn-sm"
        :class="filter === f ? 'btn-primary' : 'btn-secondary'"
        @click="filter = f; load()"
        x-text="f.charAt(0).toUpperCase() + f.slice(1)">
      </button>
    </template>
    <div style="margin-left:auto;">
      <button class="btn btn-secondary btn-sm" @click="load()">↻ Refresh</button>
    </div>
  </div>

  <div class="card" x-show="jobs.length === 0">
    <p class="text-muted text-sm">No jobs found.</p>
  </div>

  <div class="card" x-show="jobs.length > 0" style="padding:0; overflow:hidden;">
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Type</th><th>Status</th>
          <th>Progress</th><th>Created</th><th>Duration</th><th>Actions</th>
        </tr>
      </thead>
      <tbody>
        <template x-for="j in jobs" :key="j.job_id">
          <!-- Job row -->
          <tr style="cursor:pointer;" @click="toggle(j.job_id)">
            <td><code style="font-size:11px;" x-text="j.job_id.slice(0,20) + '…'"></code></td>
            <td x-text="j.job_type"></td>
            <td><span class="badge" :class="badgeClass(j.status)" x-text="j.status"></span></td>
            <td style="width:120px;">
              <div class="progress-wrap">
                <div class="progress-fill" :style="'width:' + (j.progress || 0) + '%'"></div>
              </div>
              <div class="text-sm text-muted" x-text="(j.progress || 0) + '%'"></div>
            </td>
            <td class="text-muted text-sm" x-text="fmtDate(j.created_at)"></td>
            <td class="text-muted text-sm" x-text="fmtDuration(j.started_at, j.completed_at)"></td>
            <td class="flex gap-2" @click.stop="">
              <button x-show="j.status === 'running'" class="btn btn-danger btn-sm"
                @click="cancel(j.job_id)">Cancel</button>
              <button x-show="['failed','cancelled'].includes(j.status)" class="btn btn-secondary btn-sm"
                @click="retry(j.job_id)">Retry</button>
              <button class="btn btn-danger btn-sm" @click="del(j.job_id)">✕</button>
            </td>
          </tr>
          <!-- Expanded log panel -->
          <tr x-show="expanded === j.job_id">
            <td colspan="7" style="padding:0;">
              <div style="padding:12px; background:var(--surface-2);">
                <div x-show="j.error_message" class="text-danger text-sm" style="margin-bottom:8px;" x-text="j.error_message"></div>
                <div class="log-viewer" style="height:200px;" :id="'log-' + j.job_id">
                  <template x-if="logs[j.job_id]">
                    <template x-for="(l, i) in logs[j.job_id]" :key="i">
                      <div class="log-line" :class="'log-' + (l.level || 'info')" x-text="l.message"></div>
                    </template>
                  </template>
                </div>
              </div>
            </td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>
</div>

<script>
function jobsPage() {
  return {
    jobs: [], filter: 'all', expanded: null, logs: {},

    async init() {
      await this.load();
      setInterval(() => this.load(), 5000);
    },

    async load() {
      try {
        const params = this.filter === 'all' ? '' : `?status=${this.filter}`;
        this.jobs = await apiGet('/api/jobs' + params);
      } catch (e) { showError('Failed to load jobs: ' + e.message); }
    },

    async toggle(jobId) {
      if (this.expanded === jobId) { this.expanded = null; return; }
      this.expanded = jobId;
      if (!this.logs[jobId]) {
        try {
          const d = await apiGet(`/api/jobs/${jobId}/logs?limit=100`);
          this.logs[jobId] = d.logs || [];
          this.$nextTick(() => scrollBottom(document.getElementById('log-' + jobId)));
        } catch (e) { showError('Failed to load logs: ' + e.message); }
      }
    },

    async cancel(jobId) {
      try {
        await apiFetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
        await this.load();
      } catch (e) { showError('Cancel failed: ' + e.message); }
    },

    async retry(jobId) {
      try {
        await apiPost(`/api/jobs/${jobId}/retry`, {});
        await this.load();
      } catch (e) { showError('Retry failed: ' + e.message); }
    },

    async del(jobId) {
      if (!confirm('Delete this job and its logs?')) return;
      try {
        await apiDelete(`/api/jobs/${jobId}`);
        this.jobs = this.jobs.filter(j => j.job_id !== jobId);
        if (this.expanded === jobId) this.expanded = null;
      } catch (e) { showError('Delete failed: ' + e.message); }
    },

    badgeClass, fmtDate, fmtDuration,
  };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/templates/jobs.html
git commit -m "feat: add jobs page with filtering, expand logs, cancel, retry, delete"
```

---

## Task 16: results.html — results browser and report viewer

**Files:**
- Create: `src/equilens/dashboard/templates/results.html`

- [ ] **Step 1: Write `results.html`**

```html
{% extends "base.html" %}
{% block title %}Results{% endblock %}
{% block nav_results %}active{% endblock %}

{% block content %}
<div class="page-header">
  <h1 class="page-title">📁 Results</h1>
  <p class="page-subtitle">Browse audit results, view reports, download exports</p>
</div>

<div x-data="resultsPage()" x-init="init()">

  <div x-show="results.length === 0 && !loading" class="card">
    <p class="text-muted text-sm">No results yet. <a href="/audit" style="color:var(--primary);">Run an audit →</a></p>
  </div>

  <div x-show="loading" class="text-muted text-sm">Loading…</div>

  <template x-for="r in results" :key="r.name">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
          <strong x-text="r.name" style="font-size:15px;"></strong>
          <div class="text-sm text-muted mt-2">
            <span x-text="fmtDate(r.created)"></span>
            &nbsp;·&nbsp;
            <span x-text="r.csv_file"></span>
          </div>
        </div>
        <div class="flex gap-2">
          <button class="btn btn-secondary btn-sm" @click="viewReport(r)">📊 View Report</button>
          <a :href="'/api/results/' + r.name + '/export'" class="btn btn-secondary btn-sm" download>⬇ ZIP</a>
        </div>
      </div>

      <!-- Inline report -->
      <div x-show="activeReport === r.name" x-transition style="margin-top:14px;">
        <iframe :src="'/api/results/' + r.name + '/html'" class="report-frame" :title="r.name + ' report'"></iframe>
      </div>
    </div>
  </template>
</div>

<script>
function resultsPage() {
  return {
    results: [], loading: true, activeReport: null,

    async init() {
      try {
        const d = await apiGet('/api/results');
        this.results = d.results || [];
      } catch (e) { showError('Failed to load results: ' + e.message); }
      finally { this.loading = false; }
    },

    viewReport(r) {
      this.activeReport = this.activeReport === r.name ? null : r.name;
    },

    fmtDate,
  };
}
</script>
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add src/equilens/dashboard/templates/results.html
git commit -m "feat: add results page with inline report viewer and ZIP download"
```

---

## Task 17: Update CLI commands

**Files:**
- Modify: `src/equilens/cli.py`
- Modify: `src/equilens/backend_server.py`

- [ ] **Step 1: Update `web` command in `cli.py`**

Find the `web` command (line ~2001) and replace its body:

```python
@app.command()
def web():
    """🌐 Launch the EquiLens dashboard (backend API + web interface)"""
    try:
        from equilens.backend_server import main as backend_main

        console.print("🚀 [green]Starting EquiLens Dashboard...[/green]")
        console.print("🌐 Dashboard: [cyan]http://localhost:8000[/cyan]")
        console.print("📖 API docs:  [cyan]http://localhost:8000/docs[/cyan]")
        console.print("📄 OpenAPI:   [cyan]http://localhost:8000/openapi.json[/cyan]")
        console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
        backend_main()
    except ImportError as e:
        console.print("[red]❌ Backend dependencies not available[/red]")
        console.print(f"[dim]Error: {e}[/dim]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Failed to start dashboard: {e}[/red]")
        raise typer.Exit(1) from e
```

- [ ] **Step 2: Update `gui` command — show deprecation, redirect to `web`**

Find the `gui` command (~line 1954) and replace its body:

```python
@app.command()
def gui():
    """[REMOVED] Legacy Gradio UI — use `equilens web` instead"""
    console.print("[yellow]⚠️  The 'gui' command has been removed.[/yellow]")
    console.print("The new dashboard is served from the same server.")
    console.print("Run: [bold cyan]equilens web[/bold cyan]")
    raise typer.Exit(0)
```

- [ ] **Step 3: Update `serve` command — redirect to `web`**

Find the `serve` command (~line 2022) and replace its body:

```python
@app.command()
def serve():
    """🚀 Alias for `equilens web` — starts dashboard + backend in one server"""
    web()
```

- [ ] **Step 4: Update `backend_server.py` startup banner**

In `src/equilens/backend_server.py`, update the `main()` function print block:

```python
def main():
    """Launch the EquiLens backend API with embedded dashboard."""
    from equilens.core.ports import get_backend_port

    port = get_backend_port()

    print("\n" + "=" * 70)
    print("🔍 EquiLens — AI Bias Detection Platform")
    print("=" * 70)
    print(f"\n  Dashboard:  http://localhost:{port}")
    print(f"  API:        http://localhost:{port}/api/")
    print(f"  Swagger UI: http://localhost:{port}/docs")
    print(f"  OpenAPI:    http://localhost:{port}/openapi.json")
    print(f"\n  Health:     http://localhost:{port}/api/health")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
```

- [ ] **Step 5: Smoke test the CLI**

```bash
uv run equilens --help
```

Expected: shows `web`, `backend`, `audit`, `generate`, `analyze`, `status`, `models`, `gpu-check`. No `gui`-related errors.

- [ ] **Step 6: Commit**

```bash
git add src/equilens/cli.py src/equilens/backend_server.py
git commit -m "feat: update CLI web/gui/serve commands for new single-server dashboard"
```

---

## Task 18: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/unit/ -v --no-header 2>&1 | tail -20
```

Expected: all tests PASS, no import errors.

- [ ] **Step 2: Start the server and verify dashboard loads**

```bash
uv run equilens web
```

In a second terminal:

```bash
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/ | grep -i "EquiLens"
curl -s http://localhost:8000/static/app.js | head -3
curl -s http://localhost:8000/openapi.json | python -c "import sys,json; d=json.load(sys.stdin); print('Paths:', len(d['paths']))"
```

Expected:
- Health: `{"status":"healthy","timestamp":"..."}`
- Dashboard: contains `EquiLens`
- app.js: first lines of the JS file
- OpenAPI: `Paths: 20+` (all endpoints documented)

- [ ] **Step 3: Verify Docker build still works (from project root)**

```bash
docker build -f infra/Dockerfile -t equilens-test .
```

Expected: build completes without error.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete dashboard v2.2.0 — single-server FastAPI+Alpine.js UI"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Replace Gradio UIs → Tasks 3, 6, 17
- [x] One server, no browser auto-open → Tasks 6, 17
- [x] 6 dashboard pages → Tasks 11-16
- [x] SSE job streaming → Task 5, used in Tasks 12-14
- [x] Error handling + auto-recovery → app.js fetch retry, SSE reconnect, error banner (Task 9)
- [x] Periodic backups + retention → Task 4
- [x] Backup dashboard view → Task 11
- [x] Manual backup trigger → Task 11
- [x] OpenAPI annotations → Task 7
- [x] Root cleanup → Task 1
- [x] uv dep changes → Task 2
- [x] Resume interrupted sessions → Task 12

**No placeholders found.**

**Type consistency:** `apiGet`, `apiPost`, `apiDelete`, `apiFetch`, `streamJobLogs`, `badgeClass`, `fmtDate`, `fmtBytes`, `fmtDuration`, `scrollBottom` defined in Task 9 (app.js) and used consistently in Tasks 11-16.
