"""
Database module for job tracking and persistence.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

# Thread-local storage for database connections
_thread_local = threading.local()


def get_db_path() -> Path:
    """Get database file path."""
    db_dir = Path("data/jobs")
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "equilens_jobs.db"


def get_connection() -> sqlite3.Connection:
    """Get thread-local database connection."""
    if not hasattr(_thread_local, "connection"):
        _thread_local.connection = sqlite3.connect(
            get_db_path(), check_same_thread=False
        )
        _thread_local.connection.row_factory = sqlite3.Row
    return _thread_local.connection


def init_db():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Jobs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            total INTEGER DEFAULT 100,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            config TEXT,
            result_path TEXT,
            error_message TEXT,
            pid INTEGER
        )
    """
    )

    # Job logs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS job_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        )
    """
    )

    conn.commit()
    print(f"✅ Database initialized at: {get_db_path()}")


class JobDatabase:
    """Database interface for job management."""

    @staticmethod
    def create_job(
        job_id: str,
        job_type: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """Create a new job entry."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, created_at, config)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    job_id,
                    job_type,
                    "queued",
                    datetime.now().isoformat(),
                    json.dumps(config) if config else None,
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    @staticmethod
    def get_job(job_id: str) -> dict[str, Any] | None:
        """Get job details by ID."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()

        if row:
            job = dict(row)
            if job.get("config"):
                job["config"] = json.loads(job["config"])
            return job
        return None

    @staticmethod
    def update_job(
        job_id: str,
        status: str | None = None,
        progress: int | None = None,
        total: int | None = None,
        result_path: str | None = None,
        error_message: str | None = None,
        pid: int | None = None,
    ) -> bool:
        """Update job status and details."""
        conn = get_connection()
        cursor = conn.cursor()

        updates = []
        values = []

        if status:
            updates.append("status = ?")
            values.append(status)
            current_job = JobDatabase.get_job(job_id)
            if (
                status == "running"
                and current_job
                and not current_job.get("started_at")
            ):
                updates.append("started_at = ?")
                values.append(datetime.now().isoformat())
            elif status in ["completed", "failed", "cancelled"]:
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())

        if progress is not None:
            updates.append("progress = ?")
            values.append(progress)

        if total is not None:
            updates.append("total = ?")
            values.append(total)

        if result_path:
            updates.append("result_path = ?")
            values.append(result_path)

        if error_message:
            updates.append("error_message = ?")
            values.append(error_message)

        if pid is not None:
            updates.append("pid = ?")
            values.append(pid)

        if not updates:
            return False

        values.append(job_id)
        query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"

        cursor.execute(query, values)
        conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def list_jobs(limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        """List jobs with optional filtering."""
        conn = get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (status, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

        jobs = []
        for row in cursor.fetchall():
            job = dict(row)
            if job.get("config"):
                job["config"] = json.loads(job["config"])
            jobs.append(job)

        return jobs

    @staticmethod
    def add_log(job_id: str, level: str, message: str):
        """Add a log entry for a job."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO job_logs (job_id, timestamp, level, message)
            VALUES (?, ?, ?, ?)
        """,
            (job_id, datetime.now().isoformat(), level, message),
        )
        conn.commit()

    @staticmethod
    def get_logs(job_id: str, limit: int = 100) -> list[dict[str, str]]:
        """Get logs for a specific job."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, level, message
            FROM job_logs
            WHERE job_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (job_id, limit),
        )

        return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def delete_job(job_id: str) -> bool:
        """Delete a job and its logs."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()

        return cursor.rowcount > 0
