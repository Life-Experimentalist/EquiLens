"""
EquiLens FastAPI Backend

REST API for managing EquiLens operations with background job support.
"""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .database import JobDatabase, init_db
from .export import create_results_export
from .jobs import (
    cancel_job,
    run_analysis_job,
    run_audit_job,
    run_corpus_generation_job,
)

# Initialize FastAPI app
app = FastAPI(
    title="EquiLens Backend API",
    description="Backend service for EquiLens bias detection platform",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("✅ EquiLens Backend API started")


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


@app.get("/")
async def root():
    return {"service": "EquiLens Backend API", "status": "online", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/status", response_model=SystemStatus)
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


@app.post("/api/jobs", response_model=JobResponse)
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


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job details by ID."""
    job_data = JobDatabase.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job_data)


@app.get("/api/jobs", response_model=list[JobResponse])
async def list_jobs(status: str | None = None, limit: int = 50):
    """List all jobs with optional status filter."""
    jobs = JobDatabase.list_jobs(limit=limit, status=status)
    return [JobResponse(**job) for job in jobs]


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str):
    """Cancel a running job."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel job")
    return {"message": f"Job {job_id} cancelled successfully"}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its logs."""
    success = JobDatabase.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = 100):
    """Get logs for a specific job."""
    logs = JobDatabase.get_logs(job_id, limit=limit)
    return {"job_id": job_id, "logs": logs}


# ===== Results & Export Endpoints =====


@app.get("/api/results")
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


@app.get("/api/results/{result_name}/export")
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


@app.get("/api/results/{result_name}/html")
async def get_html_report(result_name: str):
    """Get HTML report for results."""
    html_path = Path("results") / result_name / "bias_analysis_report.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found")

    return FileResponse(html_path, media_type="text/html")


# ===== Configuration Endpoints =====


@app.get("/api/models")
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


@app.post("/api/models/pull")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
