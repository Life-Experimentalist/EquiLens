# EquiLens Dashboard

The EquiLens dashboard is a built-in web interface served directly by the FastAPI backend. No separate process, no browser auto-open, no build step.

## Quick Start

```bash
uv run equilens web
```

Open **http://localhost:8000** in your browser.

That's it. One command, one server.

---

## Architecture

```
localhost:8000/                    → Dashboard home (Jinja2-rendered HTML)
localhost:8000/audit               → Run a bias audit
localhost:8000/generate            → Generate a corpus
localhost:8000/analyze             → Analyze results
localhost:8000/jobs                → Job queue management
localhost:8000/results             → Browse past results
localhost:8000/api/...             → REST API
localhost:8000/api/events/{job_id} → SSE real-time job logs
localhost:8000/docs                → Swagger UI
localhost:8000/openapi.json        → OpenAPI 3.1 contract
```

Single FastAPI process serves both the HTML dashboard and the REST API. Static assets (CSS/JS) are served from `/static/`. Alpine.js and Chart.js are loaded from CDN — no npm, no build step.

---

## Pages

### Dashboard (Home)
System health at a glance:
- Ollama connection status (green/red)
- Docker mode detection
- Active jobs count
- Backup count and last backup time
- Recent jobs table (auto-refreshes every 5 seconds)
- "Backup Now" button

### Audit
Run a bias audit against an Ollama model:
1. Select model from dropdown (populated from Ollama)
2. Select corpus CSV file (auto-discovered from `corpus/` and `src/Phase1_CorpusGenerator/corpus/`)
3. Optional: toggle Enhanced mode, Silent mode (Advanced options)
4. Click **Start Audit** — live log stream appears immediately via SSE
5. Interrupted sessions (from previous incomplete audits) are listed below with a **Resume** button

### Generate
Generate a new bias corpus:
1. Optionally specify a config file path (leave blank for defaults)
2. Click **Generate Corpus** — live log stream shows progress

### Analyze
Run statistical analysis on audit results:
1. Select a results directory from the dropdown
2. Optionally enable "Skip AI insights" or "Advanced statistical tests"
3. Click **Run Analysis** — live log stream shows progress
4. On completion: HTML bias report renders inline in an iframe, with a **Download Report** ZIP button

### Jobs
Full job management:
- Filter by status: All / Running / Completed / Failed / Cancelled
- Table shows: Job ID, Type, Status badge, Progress bar, Created time, Duration, Actions
- Per-job actions: **Logs** (expand inline log panel), **Cancel** (running jobs), **Retry** (failed/cancelled), **Delete**
- Auto-refreshes every 5 seconds

### Results
Browse past audit results:
- Two-column layout: results list on left, content on right
- Click a result to select it
- **View Report** loads the HTML analysis report in an inline iframe
- **Download ZIP** downloads the full result export

---

## Real-Time Job Logs (SSE)

Job progress is streamed via Server-Sent Events from `/api/events/{job_id}`. The browser's native `EventSource` API handles reconnection automatically.

Each event is a JSON object:
```json
{"level": "info", "message": "Testing prompt 42/100", "timestamp": "2026-05-18T14:32:01"}
```

The stream ends with a final event:
```json
{"event": "done", "status": "completed"}
```

---

## Periodic Backups

The dashboard automatically backs up `results/` and the jobs database every 30 minutes.

**Configuration** (environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `EQUILENS_BACKUP_INTERVAL_MINUTES` | `30` | How often to run automatic backups |
| `EQUILENS_BACKUP_RETENTION` | `10` | How many backups to keep (oldest deleted) |
| `EQUILENS_BACKUP_DIR` | `<project>/backups/` | Where to store backup ZIPs |

**Manual backup:** Click "Backup Now" on the Dashboard home page, or:
```bash
curl -X POST http://localhost:8000/api/backups
```

**List backups:**
```bash
curl http://localhost:8000/api/backups
```

**Download a backup:**
```bash
curl http://localhost:8000/api/backups/backup_20260518_143200.zip -o backup.zip
```

---

## API Reference

Full OpenAPI docs: **http://localhost:8000/docs**

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | System status (Ollama, Docker) |
| GET | `/api/dashboard` | Home page summary data |
| GET | `/api/models` | List Ollama models |
| GET | `/api/corpus` | List corpus CSV files |
| GET | `/api/sessions` | List interruptible audit sessions |
| POST | `/api/jobs` | Create a new job |
| GET | `/api/jobs` | List jobs (optional `?status=running`) |
| GET | `/api/jobs/{id}` | Get job details |
| POST | `/api/jobs/{id}/cancel` | Cancel a running job |
| POST | `/api/jobs/{id}/retry` | Retry a failed/cancelled job |
| DELETE | `/api/jobs/{id}` | Delete a job |
| GET | `/api/jobs/{id}/logs` | Get job log lines |
| GET | `/api/events/{id}` | SSE stream of live job logs |
| GET | `/api/results` | List result directories |
| GET | `/api/results/{name}/html` | Get HTML bias report |
| GET | `/api/results/{name}/export` | Download results as ZIP |
| GET | `/api/backups` | List backups |
| POST | `/api/backups` | Trigger manual backup |
| GET | `/api/backups/{name}` | Download a backup ZIP |
| DELETE | `/api/backups/{name}` | Delete a backup |

---

## Docker

```bash
docker compose -f infra/docker-compose.yml up
```

The container exposes port 8000. Environment variables for backup configuration can be set in `docker-compose.yml` or passed via `-e`:

```bash
docker run -e EQUILENS_BACKUP_INTERVAL_MINUTES=60 -p 8000:8000 equilens
```

---

## Theming

The dashboard respects `prefers-color-scheme`. Dark mode is the default. Light mode activates automatically if your OS/browser is set to light.

No manual toggle — it follows your system preference.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + uvicorn |
| Templates | Jinja2 |
| Reactivity | Alpine.js 3.x (CDN) |
| Charts | Chart.js 4.x (CDN) |
| Scheduling | APScheduler |
| Storage | SQLite (jobs), filesystem (backups/results) |
| Build tooling | **none** |
