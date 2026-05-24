"""
EquiLens FastAPI Backend

REST API and dashboard for managing EquiLens operations with background job support.
"""

import asyncio
import json as json_lib
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import JobDatabase, init_db
from .export import create_results_export
from .jobs import (
    cancel_job,
    run_analysis_job,
    run_audit_job,
    run_corpus_generation_job,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    import os as _os

    from equilens.backup import start_scheduler, stop_scheduler

    interval = int(_os.getenv("EQUILENS_BACKUP_INTERVAL_MINUTES", "30"))
    start_scheduler(interval_minutes=interval)
    print("✅ EquiLens Dashboard started — http://localhost:8000")
    yield
    # Shutdown
    stop_scheduler()


app = FastAPI(
    title="EquiLens Backend API",
    description=(
        "REST API and dashboard for EquiLens AI bias detection. "
        "Dashboard: http://localhost:8000 | Docs: http://localhost:8000/docs"
    ),
    version="2.2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
_STATIC_DIR = Path(__file__).parent.parent / "dashboard" / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Include dashboard HTML routes
from equilens.dashboard.routes import router as _dashboard_router  # noqa: E402

app.include_router(_dashboard_router)


# ===== Models =====


class JobCreate(BaseModel):
    job_type: str
    config: dict[str, Any] = {}


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: int = 0
    total: int = 100
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result_path: str | None = None
    error_message: str | None = None


class SystemStatus(BaseModel):
    backend_status: str
    docker_detected: bool
    ollama_available: bool
    ollama_url: str


# ===== Health & Status Endpoints =====


@app.get(
    "/api/health",
    tags=["System"],
    summary="Health check",
)
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get(
    "/api/status",
    response_model=SystemStatus,
    tags=["System"],
    summary="System status",
)
async def get_system_status():
    """Get system status including Docker and Ollama availability."""

    # Check if running in Docker
    docker_detected = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"

    # Determine Ollama URL
    if docker_detected:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    else:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Check Ollama availability
    ollama_available = False
    try:
        import requests

        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        ollama_available = response.status_code == 200
    except Exception:
        pass

    return SystemStatus(
        backend_status="running",
        docker_detected=docker_detected,
        ollama_available=ollama_available,
        ollama_url=ollama_url,
    )


# ===== Job Management Endpoints =====


@app.post(
    "/api/jobs",
    response_model=JobResponse,
    tags=["Jobs"],
    summary="Create and queue a new job",
)
async def create_job(job: JobCreate, background_tasks: BackgroundTasks):
    """Create and queue a new job."""

    job_id = f"{job.job_type}_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create job in database
    success = JobDatabase.create_job(
        job_id=job_id,
        job_type=job.job_type,
        config=job.config,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Job creation failed")

    # Queue background task based on job type
    if job.job_type == "corpus_generation":
        background_tasks.add_task(run_corpus_generation_job, job_id, job.config)
    elif job.job_type == "audit":
        background_tasks.add_task(run_audit_job, job_id, job.config)
    elif job.job_type == "analysis":
        background_tasks.add_task(run_analysis_job, job_id, job.config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job.job_type}")

    JobDatabase.add_log(job_id, "info", f"Job {job_id} created and queued")

    job_data = JobDatabase.get_job(job_id)
    return JobResponse(**job_data) if job_data else None


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobResponse,
    tags=["Jobs"],
    summary="Get job by ID",
)
async def get_job(job_id: str):
    """Get job details by ID."""
    job_data = JobDatabase.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job_data)


@app.get(
    "/api/jobs",
    response_model=list[JobResponse],
    tags=["Jobs"],
    summary="List jobs",
)
async def list_jobs(status: str | None = None, limit: int = 50):
    """List all jobs with optional status filter."""
    jobs = JobDatabase.list_jobs(limit=limit, status=status)
    return [JobResponse(**job) for job in jobs]


@app.post(
    "/api/jobs/{job_id}/cancel",
    tags=["Jobs"],
    summary="Cancel a running job",
)
async def cancel_job_endpoint(job_id: str):
    """Cancel a running job."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel job")
    return {"message": f"Job {job_id} cancelled successfully"}


@app.delete(
    "/api/jobs/{job_id}",
    tags=["Jobs"],
    summary="Delete job and its logs",
)
async def delete_job(job_id: str):
    """Delete a job and its logs."""
    success = JobDatabase.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": f"Job {job_id} deleted successfully"}


@app.get(
    "/api/jobs/{job_id}/logs",
    tags=["Jobs"],
    summary="Get job log lines",
)
async def get_job_logs(job_id: str, limit: int = 100):
    """Get logs for a specific job."""
    logs = JobDatabase.get_logs(job_id, limit=limit)
    return {"job_id": job_id, "logs": logs}


@app.post(
    "/api/jobs/{job_id}/retry",
    tags=["Jobs"],
    summary="Retry a failed or cancelled job with the same config",
    responses={
        200: {"description": "New job created"},
        400: {"description": "Job is not in a retriable state"},
        404: {"description": "Job not found"},
    },
)
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Re-queue a failed or cancelled job using its original configuration."""
    import json as _json

    job_data = JobDatabase.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_data["status"] not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Only failed/cancelled jobs can be retried. Current status: {job_data['status']}",
        )

    original_config = _json.loads(job_data.get("config") or "{}")
    job_type = job_data["job_type"]
    new_job_id = (
        f"{job_type}_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    JobDatabase.create_job(job_id=new_job_id, job_type=job_type, config=original_config)

    if job_type == "corpus_generation":
        background_tasks.add_task(
            run_corpus_generation_job, new_job_id, original_config
        )
    elif job_type == "audit":
        background_tasks.add_task(run_audit_job, new_job_id, original_config)
    elif job_type == "analysis":
        background_tasks.add_task(run_analysis_job, new_job_id, original_config)

    return {"original_job_id": job_id, "new_job_id": new_job_id}


@app.get(
    "/api/events/{job_id}",
    tags=["Jobs"],
    summary="Stream job logs via Server-Sent Events",
    description=(
        "Returns a text/event-stream of JSON log objects for the given job. "
        "Each event has 'level', 'message', 'timestamp'. "
        "A final event with 'event': 'done' signals completion. "
        "Clients reconnect automatically on disconnect."
    ),
)
async def stream_job_events(job_id: str):
    """SSE stream of log lines for a running or recently completed job."""

    async def event_generator():
        # If job doesn't exist, immediately signal done
        if not JobDatabase.get_job(job_id):
            yield f"data: {json_lib.dumps({'event': 'done', 'status': 'not_found'})}\n\n"
            return

        last_index = 0
        idle_ticks = 0
        max_idle = 180  # 3-minute timeout with no new logs

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


# ===== Results & Export Endpoints =====


@app.get(
    "/api/results",
    tags=["Results"],
    summary="List result directories",
)
async def list_results():
    """List all available result directories."""
    results_dir = Path("results")
    if not results_dir.exists():
        return {"results": []}

    results = []
    for item in results_dir.iterdir():
        if item.is_dir():
            # Get CSV file if exists
            csv_files = list(item.glob("*.csv"))
            if csv_files:
                results.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "created": datetime.fromtimestamp(
                            item.stat().st_ctime
                        ).isoformat(),
                        "csv_file": csv_files[0].name,
                    }
                )

    return {"results": sorted(results, key=lambda x: x["created"], reverse=True)}


@app.get(
    "/api/results/{result_name}/export",
    tags=["Results"],
    summary="Download results as ZIP",
)
async def export_results(result_name: str):
    """Export results as a .zip file."""
    results_dir = Path("results") / result_name

    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    # Create export zip
    zip_path = create_results_export(results_dir)

    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=500, detail="Export creation failed")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{result_name}_export.zip",
    )


@app.get(
    "/api/results/{result_name}/html",
    tags=["Results"],
    summary="Get HTML bias report",
)
async def get_html_report(result_name: str):
    """Get HTML report for results."""
    html_path = Path("results") / result_name / "bias_analysis_report.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")

    return FileResponse(html_path, media_type="text/html")


# ===== Configuration Endpoints =====


@app.get(
    "/api/corpus",
    tags=["Configuration"],
    summary="List discovered corpus CSV files",
    description="Scans known corpus directories and returns available CSV files.",
)
async def list_corpus_files():
    import csv as _csv

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
                    row_count = sum(1 for _ in _csv.reader(f)) - 1
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
    description="Scans results/ for progress JSON files from incomplete audits.",
)
async def list_interrupted_sessions():
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


@app.get(
    "/api/models",
    tags=["Models"],
    summary="List available Ollama models",
)
async def list_ollama_models():
    """List available Ollama models."""
    status = await get_system_status()

    if not status.ollama_available:
        return {"models": [], "error": "Ollama not available"}

    try:
        import requests

        response = requests.get(f"{status.ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return {
                "models": [
                    {
                        "name": m["name"],
                        "size": m.get("size", 0),
                        "modified": m.get("modified_at", ""),
                    }
                    for m in models
                ]
            }
    except Exception as e:
        return {"models": [], "error": str(e)}

    return {"models": []}


@app.post(
    "/api/models/pull",
    tags=["Models"],
    summary="Pull an Ollama model",
)
async def pull_ollama_model(model_name: str, background_tasks: BackgroundTasks):
    """Pull/download an Ollama model."""
    job_id = f"model_pull_{uuid.uuid4().hex[:8]}"

    JobDatabase.create_job(
        job_id=job_id,
        job_type="model_pull",
        config={"model_name": model_name},
    )

    async def pull_model():
        try:
            JobDatabase.update_job(job_id, status="running")
            JobDatabase.add_log(job_id, "info", f"Pulling model: {model_name}")

            proc = await asyncio.create_subprocess_exec(
                "ollama",
                "pull",
                model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                JobDatabase.update_job(job_id, status="completed", progress=100)
                JobDatabase.add_log(
                    job_id, "info", f"Model {model_name} pulled successfully"
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                JobDatabase.update_job(job_id, status="failed", error_message=error_msg)
                JobDatabase.add_log(
                    job_id, "error", f"Failed to pull model: {error_msg}"
                )

        except Exception as e:
            JobDatabase.update_job(job_id, status="failed", error_message=str(e))
            JobDatabase.add_log(job_id, "error", f"Exception: {str(e)}")

    background_tasks.add_task(pull_model)

    return {"job_id": job_id, "message": f"Pulling model {model_name}"}


# ===== Backup Endpoints =====


@app.get(
    "/api/backups",
    tags=["Backups"],
    summary="List all backups (newest-first)",
)
async def list_backups_endpoint():
    from equilens.backup import list_backups

    return {"backups": list_backups()}


@app.post(
    "/api/backups",
    tags=["Backups"],
    summary="Trigger an immediate backup",
    responses={
        200: {"description": "Backup created"},
        500: {"description": "Backup failed"},
    },
)
async def trigger_backup():
    from equilens.backup import create_backup

    try:
        path = create_backup()
        return {"message": "Backup created", "path": str(path), "name": path.name}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/api/backups/{name}",
    tags=["Backups"],
    summary="Download a backup ZIP by filename",
    responses={200: {"description": "ZIP file"}, 404: {"description": "Not found"}},
)
async def download_backup(name: str):
    from equilens.backup import BACKUP_DIR

    path = BACKUP_DIR / name
    if not path.exists() or path.suffix != ".zip":
        raise HTTPException(status_code=404, detail="Backup not found")
    return FileResponse(path, media_type="application/zip", filename=name)


@app.delete(
    "/api/backups/{name}",
    tags=["Backups"],
    summary="Delete a backup ZIP by filename",
    responses={200: {"description": "Deleted"}, 404: {"description": "Not found"}},
)
async def delete_backup(name: str):
    from equilens.backup import BACKUP_DIR

    path = BACKUP_DIR / name
    if not path.exists() or path.suffix != ".zip":
        raise HTTPException(status_code=404, detail="Backup not found")
    path.unlink()
    return {"message": f"Backup {name} deleted"}


@app.get(
    "/api/dashboard",
    tags=["System"],
    summary="Dashboard summary — single endpoint for home page data",
)
async def dashboard_summary():
    from equilens.backup import get_scheduler_status, list_backups

    docker_detected = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"
    ollama_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://ollama:11434" if docker_detected else "http://localhost:11434",
    )
    ollama_available = False
    try:
        import requests as _req

        ollama_available = (
            _req.get(f"{ollama_url}/api/tags", timeout=3).status_code == 200
        )
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
