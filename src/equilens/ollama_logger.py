#!/usr/bin/env python3
"""
Ollama Server Log Monitor

Captures Ollama server logs from Docker containers or
Windows desktop application to a local file for diagnostics.

Supports:
- Docker container log collection via `docker logs`
- Windows app log tailing from %LOCALAPPDATA%\\Ollama\\logs\\server.log
- Background thread operation with clean start/stop lifecycle

Usage:
    monitor = OllamaLogMonitor(output_dir=Path("logs"))
    monitor.start(session_id="20250101_120000")
    # ... run audit ...
    monitor.stop()
    print(f"Logs saved to: {monitor.log_path}")
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaLogMonitor:
    """
    Background monitor that captures Ollama server logs to a file.

    Auto-detects whether Ollama is running in a Docker container or as a
    native Windows application and captures logs accordingly.
    """

    def __init__(self, output_dir: Path | str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._log_path: Path | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._source: str = "unknown"  # "docker" or "windows" or "unknown"
        self._container_name: str | None = None

    @property
    def log_path(self) -> Path | None:
        """Path to the captured log file, or None if not started."""
        return self._log_path

    @property
    def source(self) -> str:
        """Log source type: 'docker', 'windows', or 'unknown'."""
        return self._source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, session_id: str) -> bool:
        """
        Start log collection in a background thread.

        Args:
            session_id: Audit session identifier used in the log filename.

        Returns:
            True if monitoring started successfully, False otherwise.
        """
        if self._thread and self._thread.is_alive():
            logger.warning("OllamaLogMonitor is already running")
            return True

        self._stop_event.clear()
        self._log_path = self.output_dir / f"ollama_server_{session_id}.log"

        # Detect source
        container = self._detect_docker_container()
        if container:
            self._source = "docker"
            self._container_name = container
            logger.info(
                f"Ollama log source: Docker container '{container}' -> {self._log_path}"
            )
        elif self._detect_windows_app_log():
            self._source = "windows"
            logger.info(f"Ollama log source: Windows app -> {self._log_path}")
        else:
            self._source = "unknown"
            logger.warning(
                "Could not detect Ollama log source (Docker container or Windows app). "
                "Log collection will be skipped."
            )
            return False

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="OllamaLogMonitor"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop log collection and wait for the background thread to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_docker_container(self) -> str | None:
        """
        Detect a running Ollama Docker container.

        Returns:
            Container name/ID if found, None otherwise.
        """
        if not shutil.which("docker"):
            return None

        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "ancestor=ollama/ollama",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            names = result.stdout.strip().splitlines()
            if names:
                return names[0]

            # Fallback: check for containers with "ollama" in the name
            result2 = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=ollama",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            names2 = result2.stdout.strip().splitlines()
            if names2:
                return names2[0]

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return None

    def _detect_windows_app_log(self) -> bool:
        """Check if the Ollama Windows app server log exists."""
        log_path = self._windows_log_path()
        return log_path is not None and log_path.exists()

    def _windows_log_path(self) -> Path | None:
        """
        Return the path to the Ollama Windows app server log.

        Checks:
        1. %LOCALAPPDATA%\\Ollama\\logs\\server.log  (primary)
        2. %APPDATA%\\Ollama\\logs\\server.log        (fallback)
        """
        for env_var in ("LOCALAPPDATA", "APPDATA"):
            base = os.environ.get(env_var)
            if base:
                candidate = Path(base) / "Ollama" / "logs" / "server.log"
                if candidate.exists():
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the background thread."""
        try:
            if self._source == "docker":
                self._tail_docker()
            elif self._source == "windows":
                self._tail_windows()
        except Exception as e:
            logger.error(f"OllamaLogMonitor error: {e}")

    def _tail_docker(self) -> None:
        """Stream Docker container logs to the output file."""
        if not self._container_name or not self._log_path:
            return

        try:
            proc = subprocess.Popen(
                [
                    "docker",
                    "logs",
                    "--follow",
                    "--since",
                    "0s",
                    self._container_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Failed to start docker logs: {e}")
            return

        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"=== Ollama Docker log capture started ({self._container_name}) ===\n"
                )
                while not self._stop_event.is_set():
                    if proc.stdout is None:
                        break
                    line = proc.stdout.readline()
                    if line:
                        f.write(line)
                        f.flush()
                    elif proc.poll() is not None:
                        # Process exited
                        break
                    else:
                        # No output, short sleep to avoid busy-wait
                        time.sleep(0.1)
                f.write("=== Ollama Docker log capture stopped ===\n")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _tail_windows(self) -> None:
        """Tail the Ollama Windows app server log to the output file."""
        source_path = self._windows_log_path()
        if not source_path or not self._log_path:
            return

        try:
            with (
                source_path.open(encoding="utf-8", errors="replace") as src,
                self._log_path.open("a", encoding="utf-8") as dst,
            ):
                dst.write(
                    f"=== Ollama Windows app log capture started ({source_path}) ===\n"
                )

                # Seek to end of file so we only capture new log entries
                src.seek(0, 2)

                while not self._stop_event.is_set():
                    line = src.readline()
                    if line:
                        dst.write(line)
                        dst.flush()
                    else:
                        # No new line yet; wait a bit
                        time.sleep(0.25)

                dst.write("=== Ollama Windows app log capture stopped ===\n")
        except OSError as e:
            logger.error(f"Failed to tail Windows Ollama log: {e}")
