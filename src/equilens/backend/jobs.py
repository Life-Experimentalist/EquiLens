"""
Job execution module for background tasks.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from equilens.backend.database import JobDatabase


def run_corpus_generation_job(job_id: str, config: dict[str, Any]):
    """Execute corpus generation job."""
    try:
        JobDatabase.update_job(job_id, status="running", progress=0)
        JobDatabase.add_log(job_id, "info", "Starting corpus generation...")

        # Build command
        cmd = [sys.executable, "-m", "equilens", "generate"]

        if config.get("config_file"):
            cmd.extend(["--config", config["config_file"]])

        # Execute
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Store PID for cancellation
        JobDatabase.update_job(job_id, pid=process.pid)

        # Stream output
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if line:
                    JobDatabase.add_log(job_id, "info", line)
                    # Update progress based on output patterns
                    if "Generating" in line or "Creating" in line:
                        JobDatabase.update_job(job_id, progress=50)

        process.wait()

        if process.returncode == 0:
            JobDatabase.update_job(
                job_id,
                status="completed",
                progress=100,
            )
            JobDatabase.add_log(
                job_id, "info", "Corpus generation completed successfully"
            )
        else:
            JobDatabase.update_job(
                job_id,
                status="failed",
                error_message=f"Process exited with code {process.returncode}",
            )
            JobDatabase.add_log(
                job_id, "error", f"Failed with exit code {process.returncode}"
            )

    except Exception as e:
        JobDatabase.update_job(job_id, status="failed", error_message=str(e))
        JobDatabase.add_log(job_id, "error", f"Exception: {str(e)}")


def run_audit_job(job_id: str, config: dict[str, Any]):
    """Execute audit job."""
    try:
        JobDatabase.update_job(job_id, status="running", progress=0)
        JobDatabase.add_log(job_id, "info", "Starting bias audit...")

        # Build command
        cmd = [sys.executable, "-m", "equilens", "audit"]

        if config.get("model"):
            cmd.extend(["--model", config["model"]])
        if config.get("corpus"):
            cmd.extend(["--corpus", config["corpus"]])
        if config.get("output_dir"):
            cmd.extend(["--output-dir", config["output_dir"]])
        if config.get("silent"):
            cmd.append("--silent")

        # Execute
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Store PID for cancellation
        JobDatabase.update_job(job_id, pid=process.pid)

        # Stream output and track progress
        total_tests = 0
        completed_tests = 0

        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if line:
                    JobDatabase.add_log(job_id, "info", line)

                    # Parse progress from output
                    if "Total tests:" in line or "Running" in line:
                        try:
                            parts = line.split()
                            for part in parts:
                                if part.isdigit():
                                    total_tests = int(part)
                                    JobDatabase.update_job(job_id, total=total_tests)
                                    break
                        except Exception as e:
                            JobDatabase.add_log(
                                job_id,
                                "error",
                                f"Failed to parse total tests: {str(e)}",
                            )

                    if "Completed" in line or "Testing" in line:
                        completed_tests += 1
                        if total_tests > 0:
                            progress = int((completed_tests / total_tests) * 100)
                            JobDatabase.update_job(job_id, progress=min(progress, 95))

        process.wait()

        if process.returncode == 0:
            # Find result file
            result_path = None
            if config.get("output_dir"):
                results_dir = Path(config["output_dir"])
                csv_files = list(results_dir.glob("*.csv"))
                if csv_files:
                    result_path = str(csv_files[0])

            JobDatabase.update_job(
                job_id,
                status="completed",
                progress=100,
                result_path=result_path,
            )
            JobDatabase.add_log(job_id, "info", "Audit completed successfully")
        else:
            JobDatabase.update_job(
                job_id,
                status="failed",
                error_message=f"Process exited with code {process.returncode}",
            )
            JobDatabase.add_log(
                job_id, "error", f"Failed with exit code {process.returncode}"
            )

    except Exception as e:
        JobDatabase.update_job(job_id, status="failed", error_message=str(e))
        JobDatabase.add_log(job_id, "error", f"Exception: {str(e)}")


def run_analysis_job(job_id: str, config: dict[str, Any]):
    """Execute analysis job."""
    try:
        JobDatabase.update_job(job_id, status="running", progress=0)
        JobDatabase.add_log(job_id, "info", "Starting bias analysis...")

        # Build command
        cmd = [sys.executable, "-m", "equilens", "analyze"]

        if config.get("results_file"):
            cmd.append(config["results_file"])
        if config.get("no_ai"):
            cmd.append("--no-ai")
        if config.get("advanced"):
            cmd.append("--advanced")

        # Execute
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Store PID for cancellation
        JobDatabase.update_job(job_id, pid=process.pid)

        # Stream output
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if line:
                    JobDatabase.add_log(job_id, "info", line)

                    # Update progress based on keywords
                    if "Loading" in line:
                        JobDatabase.update_job(job_id, progress=10)
                    elif "Statistical" in line:
                        JobDatabase.update_job(job_id, progress=30)
                    elif "Creating" in line or "Generating" in line:
                        JobDatabase.update_job(job_id, progress=60)
                    elif "report" in line.lower():
                        JobDatabase.update_job(job_id, progress=90)

        process.wait()

        if process.returncode == 0:
            # Find HTML report
            result_path = None
            if config.get("results_file"):
                results_dir = Path(config["results_file"]).parent
                html_file = results_dir / "bias_analysis_report.html"
                if html_file.exists():
                    result_path = str(html_file)

            JobDatabase.update_job(
                job_id,
                status="completed",
                progress=100,
                result_path=result_path,
            )
            JobDatabase.add_log(job_id, "info", "Analysis completed successfully")
        else:
            JobDatabase.update_job(
                job_id,
                status="failed",
                error_message=f"Process exited with code {process.returncode}",
            )
            JobDatabase.add_log(
                job_id, "error", f"Failed with exit code {process.returncode}"
            )

    except Exception as e:
        JobDatabase.update_job(job_id, status="failed", error_message=str(e))
        JobDatabase.add_log(job_id, "error", f"Exception: {str(e)}")


def cancel_job(job_id: str) -> bool:
    """Cancel a running job by terminating its process."""
    job = JobDatabase.get_job(job_id)

    if not job:
        return False

    if job["status"] not in ["running", "queued"]:
        return False

    # Try to terminate process if PID exists
    if job.get("pid"):
        try:
            if os.name == "nt":  # Windows
                subprocess.run(["taskkill", "/F", "/PID", str(job["pid"])], check=True)
            else:  # Unix-like
                os.kill(job["pid"], signal.SIGTERM)

            JobDatabase.add_log(job_id, "warning", "Job cancelled by user")

        except Exception as e:
            JobDatabase.add_log(job_id, "error", f"Failed to cancel job: {str(e)}")

    JobDatabase.update_job(job_id, status="cancelled")
    return True
