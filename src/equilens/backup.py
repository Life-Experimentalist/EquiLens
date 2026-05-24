"""Periodic backup scheduler — zips results/ and job DB on a schedule."""

import os
import zipfile
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

# Project root = three levels up from this file (src/equilens/backup.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BACKUP_DIR = Path(os.getenv("EQUILENS_BACKUP_DIR", str(_PROJECT_ROOT / "backups")))
DEFAULT_INTERVAL_MINUTES: int = int(os.getenv("EQUILENS_BACKUP_INTERVAL_MINUTES", "30"))
DEFAULT_RETENTION: int = int(os.getenv("EQUILENS_BACKUP_RETENTION", "10"))

_scheduler: BackgroundScheduler | None = None


def create_backup() -> Path:
    """Zip results/ and data/jobs/equilens_jobs.db into BACKUP_DIR.

    Returns the path to the created zip file.
    Raises RuntimeError if zipping fails; the partial zip file is removed on exception
    (not guaranteed on SIGKILL or power loss).
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = BACKUP_DIR / name

    targets = [
        _PROJECT_ROOT / "results",
        _PROJECT_ROOT / "data" / "jobs" / "equilens_jobs.db",
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
            zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Backup failed: {exc}") from exc


def _prune_backups(retention: int = DEFAULT_RETENTION) -> None:
    """Delete oldest backups beyond the retention limit."""
    if not BACKUP_DIR.exists():
        return
    backups = sorted(
        BACKUP_DIR.glob("backup_*.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in backups[:-retention] if retention > 0 else backups:
        old.unlink(missing_ok=True)


def list_backups() -> list[dict]:
    """Return all backups as dicts sorted newest-first."""
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
    """Start the APScheduler background scheduler for periodic backups."""
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
    """Return {'running': bool, 'next_run': ISO string or None}."""
    if not _scheduler or not _scheduler.running:
        return {"running": False, "next_run": None}
    job = _scheduler.get_job("periodic_backup")
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
    return {"running": True, "next_run": next_run}
