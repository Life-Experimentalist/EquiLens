#!/usr/bin/env python3
#
# Copyright 2025 Krishna GSVV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
EquiLens Model Auditor with Robust Error Handling and GPU Acceleration

This script provides:
- Automatic service setup and model downloading with GPU acceleration
- Robust error handling with exponential backoff
- Progress tracking and resumption capabilities
- Persistent model storage to avoid re-downloads
- RTX 2050 GPU acceleration for faster inference
"""

import argparse
import io
import json
import logging
import os
import signal
import subprocess

# Configure logging with Unicode-safe console output
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

# Color support with fallback
try:
    import colorama

    colorama.init(autoreset=True)  # Initialize colorama for Windows compatibility
    COLORS_AVAILABLE = True
    # Color constants
    COLOR_CYAN = colorama.Fore.CYAN
    COLOR_GREEN = colorama.Fore.GREEN
    COLOR_RED = colorama.Fore.RED
    COLOR_YELLOW = colorama.Fore.YELLOW
    COLOR_BLUE = colorama.Fore.BLUE
    COLOR_MAGENTA = colorama.Fore.MAGENTA
    COLOR_WHITE = colorama.Fore.WHITE
    STYLE_BRIGHT = colorama.Style.BRIGHT
    STYLE_RESET = colorama.Style.RESET_ALL
except ImportError:
    # Fallback if colorama is not available
    COLORS_AVAILABLE = False
    COLOR_CYAN = COLOR_GREEN = COLOR_RED = COLOR_YELLOW = ""
    COLOR_BLUE = COLOR_MAGENTA = COLOR_WHITE = ""
    STYLE_BRIGHT = STYLE_RESET = ""

# Ensure logs directory exists
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

# Create a custom stream handler that handles Unicode properly
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Set encoding to UTF-8 with error handling for Windows compatibility
try:
    # Try to reconfigure stdout for UTF-8 on Windows
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    # If reconfiguration fails, the handler will still work with error replacement
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/audit_session.log", encoding="utf-8"),
        console_handler,
    ],
)
logger = logging.getLogger(__name__)
class TqdmLoggingHandler(logging.Handler):
    """Use tqdm.write so log lines appear above an active progress bar."""

    def emit(self, record):  # type: ignore[override]
        try:
            from tqdm import tqdm as _tqdm

            msg = self.format(record)
            _tqdm.write(msg)
        except Exception:
            try:
                print(self.format(record))
            except Exception:
                pass


@dataclass
class AuditProgress:
    """Track audit progress for resumption"""

    total_tests: int = 0
    completed_tests: int = 0
    failed_tests: int = 0
    current_index: int = 0
    session_id: str = ""
    model_name: str = ""
    corpus_file: str = ""
    results_file: str = ""
    start_time: str = ""
    last_checkpoint: str = ""


class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.kill_now = True


class PauseController:
    """Handle pause/resume functionality with 'p' key"""

    def __init__(self):
        self.paused = False
        self.pause_lock = threading.Lock()
        self.listener_thread = None
        self.stop_listener = False

    def start_listener(self):
        """Start the keyboard listener thread"""
        self.stop_listener = False
        self.listener_thread = threading.Thread(
            target=self._keyboard_listener, daemon=True
        )
        self.listener_thread.start()

    def stop_listener_thread(self):
        """Stop the keyboard listener thread"""
        self.stop_listener = True
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=0.5)

    def _keyboard_listener(self):
        """Listen for 'p' key presses to toggle pause"""
        try:
            while not self.stop_listener:
                try:
                    # Windows-compatible approach using msvcrt
                    if sys.platform == "win32":
                        import msvcrt

                        if msvcrt.kbhit():
                            char = (
                                msvcrt.getch().decode("utf-8", errors="ignore").lower()
                            )
                            if char == "p":
                                self.toggle_pause()
                    else:
                        # Unix-like systems
                        import select

                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready:
                            char = sys.stdin.read(1).lower()
                            if char == "p":
                                self.toggle_pause()

                    time.sleep(0.1)  # Small delay to prevent busy waiting

                except Exception as e:
                    logger.debug(f"Keyboard input error: {e}")
                    time.sleep(0.5)

        except Exception as e:
            logger.debug(f"Keyboard listener error: {e}")

    def toggle_pause(self):
        """Toggle pause state"""
        with self.pause_lock:
            self.paused = not self.paused
            if self.paused:
                print("\n⏸️  AUDIT PAUSED")
                print("   📊 Current progress is automatically saved")
                print("   ▶️  Press 'p' again to resume")
                print("   🔥 Press Ctrl+C to save and exit")
            else:
                print("\n▶️  AUDIT RESUMED - continuing processing...")

    def wait_if_paused(self):
        """Block execution if currently paused"""
        with self.pause_lock:
            while self.paused:
                time.sleep(0.1)

    def is_paused(self):
        """Check if currently paused"""
        with self.pause_lock:
            return self.paused


class ModelAuditor:
    """Enhanced model auditor with robust error handling"""

    def __init__(
        self,
        model_name: str,
        corpus_file: str,
        output_dir: str = "results",
        eta_per_test: float | None = None,
        max_workers: int = 1,
    ):
        self.model_name = model_name
        self.corpus_file = corpus_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # ETA tracking
        self.user_eta_per_test = eta_per_test  # User-provided estimate
        self.actual_response_times = []  # Track actual response times for dynamic ETA
        self.session_start_time = None  # Track when processing starts

        # Concurrency settings with dynamic scaling
        self.max_workers = max_workers  # Number of concurrent threads
        self.current_workers = max_workers  # Current active workers (can be scaled down)
        self.consecutive_errors = 0  # Track consecutive errors for fallback
        self.consecutive_successes = 0  # Track consecutive successes for recovery
        self.max_consecutive_errors = 3  # Reduce workers after 3 consecutive errors
        self.recovery_threshold = 10  # Increase workers after 10 consecutive successes
        self.original_max_workers = max_workers  # Store original setting for recovery
        self.error_fallback_active = False  # Track if we're in fallback mode

        # Retry tracking system
        self.failed_tests = {}  # Track failed tests: {idx: (row, failure_count)}
        self.retry_queue = []  # Tests ready for retry
        self.success_count_since_last_retry = 0  # Count successes between retries
        self.retry_batch_size = 5  # Retry after this many successes (reduced from 10)

        # Error tracking for diagnostics
        self.error_counts = {
            "timeout": 0,
            "server_error_500": 0,
            "connection_error": 0,
            "other_error": 0,
        }

        # Setup session management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize model name for Windows filesystem (replace invalid characters)
        safe_model_name = (
            model_name.replace(":", "_").replace("/", "_").replace("\\", "_")
        )
        # Create model-specific directory for all session files
        self.model_session_dir = (
            self.output_dir / f"{safe_model_name}_{self.session_id}"
        )
        self.model_session_dir.mkdir(exist_ok=True)

        # All files go in the model session directory
        self.progress_file = self.model_session_dir / f"progress_{self.session_id}.json"
        self.results_file = (
            self.model_session_dir / f"results_{safe_model_name}_{self.session_id}.csv"
        )

        # Initialize progress tracking
        self.progress = AuditProgress(
            session_id=self.session_id,
            model_name=model_name,
            corpus_file=corpus_file,
            results_file=str(self.results_file),
            start_time=datetime.now().isoformat(),
        )

        # Graceful shutdown handler
        self.killer = GracefulKiller()

        # Pause/resume controller
        self.pause_controller = PauseController()

        # API configuration - try multiple host options for multi-container setup
        self.ollama_hosts = [
            "http://ollama:11434",  # Docker compose service name (primary)
            "http://localhost:11434",  # Local container
            "http://127.0.0.1:11434",  # Local loopback
        ]
        self.ollama_url = None
        self.max_retries = 5
        self.base_delay = 1.0
        self.max_delay = 60.0

        logger.info(f"Initialized audit session {self.session_id}")

    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running on any available host"""
        for host in self.ollama_hosts:
            try:
                response = requests.get(f"{host}/api/tags", timeout=10)
                if response.status_code == 200:
                    self.ollama_url = host
                    logger.info(f"✅ Ollama service found at {host}")
                    return True
                else:
                    logger.debug(
                        f"❌ Ollama not available at {host}: {response.status_code}"
                    )
            except Exception as e:
                logger.debug(f"❌ Cannot connect to Ollama at {host}: {e}")

        logger.error("❌ Ollama service not accessible on any host")
        return False

    def start_ollama_service(self) -> bool:
        """Start Ollama service with automatic GPU detection"""
        logger.info("🚀 Starting Ollama service with automatic GPU detection...")

        try:
            # Start with docker compose from devcontainer
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "/workspace/.devcontainer/docker-compose.yml",
                    "up",
                    "-d",
                    "ollama",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                logger.info("✅ Ollama service started successfully")
                time.sleep(15)  # Wait for service initialization

                # Check for GPU availability
                self._check_gpu_availability()

                return self.check_ollama_service()
            else:
                logger.warning(f"Docker compose failed: {result.stderr}")
                logger.info(
                    "💡 Try running: docker compose -f .devcontainer/docker-compose.yml up -d ollama"
                )

        except subprocess.TimeoutExpired:
            logger.warning("Docker compose start timed out")
        except Exception as e:
            logger.warning(f"Failed to start with docker compose: {e}")
            logger.info(
                "💡 Manual start: docker compose -f .devcontainer/docker-compose.yml up -d ollama"
            )

        return False

    def _check_gpu_availability(self):
        """Check and report GPU availability"""
        try:
            # Try to get GPU info from the Ollama container
            result = subprocess.run(
                ["docker", "exec", "bias_auditor-ollama-1", "nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                gpu_lines = [
                    line.strip()
                    for line in result.stdout.split("\n")
                    if line.strip() and "GPU" in line
                ]
                if gpu_lines:
                    logger.info("🎮 GPU Acceleration ENABLED:")
                    for gpu_line in gpu_lines:
                        logger.info(f"   ⚡ {gpu_line}")
                    logger.info("   📈 Expect 3-5x faster inference than CPU-only")
                    return True

            # If nvidia-smi fails, try alternative detection
            result = subprocess.run(
                ["docker", "exec", "bias_auditor-ollama-1", "ls", "/dev/nvidia*"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                logger.info("🎮 GPU detected but nvidia-smi not available")
                logger.info("   ⚡ GPU acceleration likely available")
                return True

        except Exception as e:
            logger.debug(f"GPU check failed: {e}")

        # No GPU detected
        logger.info("💻 CPU-ONLY mode:")
        logger.info("   🔄 No GPU detected - using CPU for inference")
        logger.info("   ⏱️  Inference will be slower but still functional")
        logger.info(
            "   💡 For faster performance, ensure NVIDIA drivers and Docker GPU support are installed"
        )
        return False

    def list_available_models(self) -> list[str]:
        """List models available in Ollama"""
        if not self.ollama_url:
            if not self.check_ollama_service():
                return []

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])]
                logger.info(f"Available models: {models}")
                return models
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def download_model(self, model_name: str) -> bool:
        """Download a model with progress tracking"""
        logger.info(f"📥 Downloading model: {model_name}")

        try:
            # Start the pull request
            pull_url = f"{self.ollama_url}/api/pull"
            response = requests.post(
                pull_url,
                json={"name": model_name},
                stream=True,
                timeout=3600,  # 1 hour timeout for download
            )

            if response.status_code != 200:
                logger.error(f"Failed to start download: {response.status_code}")
                return False

            # Track download progress
            for line in response.iter_lines():
                if self.killer.kill_now:
                    logger.info("Download interrupted by user")
                    return False

                if line:
                    try:
                        progress_data = json.loads(line)
                        status = progress_data.get("status", "")

                        if "downloading" in status.lower():
                            completed = progress_data.get("completed", 0)
                            total = progress_data.get("total", 0)
                            if total > 0:
                                percent = (completed / total) * 100
                                logger.info(
                                    f"📥 Download progress: {percent:.1f}% ({completed}/{total})"
                                )
                        elif (
                            "success" in status.lower() or "complete" in status.lower()
                        ):
                            logger.info(
                                f"✅ Model {model_name} downloaded successfully"
                            )
                            return True
                        elif "error" in status.lower():
                            logger.error(f"❌ Download error: {status}")
                            return False

                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            logger.error("⏰ Model download timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False

        return False

    def ensure_model_available(self) -> bool:
        """Ensure the model is available, download if necessary"""
        available_models = self.list_available_models()

        if self.model_name in available_models:
            logger.info(f"✅ Model {self.model_name} is already available")
            return True

        logger.info(f"📥 Model {self.model_name} not found, attempting download...")

        # Suggest optimized models based on available hardware
        optimized_models = [
            "phi3:mini",  # Fast and efficient, works well on both GPU and CPU
            "llama3.2:1b",  # Good balance of speed and capability
            "gemma2:2b",  # Excellent for bias detection
            "qwen2:0.5b",  # Very fast inference, minimal resources
            "llama3.2:3b",  # More capable, better with GPU
        ]

        if self.download_model(self.model_name):
            return True

        if self.model_name not in optimized_models:
            logger.info("💡 Suggested models (CPU/GPU compatible):")
            for model in optimized_models:
                logger.info(f"   - {model}")

            # Try downloading a small optimized model automatically
            logger.info("🎯 Attempting to download phi3:mini (CPU/GPU compatible)...")
            if self.download_model("phi3:mini"):
                logger.info(
                    "✅ Using phi3:mini - works efficiently on both CPU and GPU"
                )
                self.model_name = "phi3:mini"
                self.progress.model_name = "phi3:mini"
                return True

        return False

    def add_failed_test(self, idx: int, row: Any) -> None:
        """Add a failed test to retry tracking"""
        if idx in self.failed_tests:
            _, failure_count = self.failed_tests[idx]
            self.failed_tests[idx] = (row, failure_count + 1)
        else:
            self.failed_tests[idx] = (row, 1)

        # Add to retry queue if failed 3 times (reduced from 5)
        if self.failed_tests[idx][1] >= 3:
            if idx not in [test_idx for test_idx, _ in self.retry_queue]:
                self.retry_queue.append((idx, row))
                logger.debug(
                    f"Added test {idx} to retry queue after {self.failed_tests[idx][1]} failures"
                )
            del self.failed_tests[idx]  # Remove from failed tracking

    def should_process_retries(self) -> bool:
        """Check if we should process retry queue"""
        return (
            self.success_count_since_last_retry >= self.retry_batch_size
            and len(self.retry_queue) > 0
        )

    def process_retry_batch(self, pbar, results: list) -> None:
        """Process a batch of retry tests with clean logging above progress bar"""
        if not self.retry_queue:
            return

        # Print retry messages above progress bar
        pbar.write(
            f"\n🔄 Processing {len(self.retry_queue)} failed tests (retry batch)"
        )
        pbar.write("─" * 60)

        retry_batch = self.retry_queue.copy()
        self.retry_queue.clear()
        self.success_count_since_last_retry = 0
        retry_successes = 0
        retry_failures = 0

        for retry_idx, (_test_idx, row) in enumerate(retry_batch, 1):
            # Check for pause and killer signals
            if hasattr(self, "pause_controller"):
                while self.pause_controller.is_paused():
                    time.sleep(0.1)
                    if self.killer.kill_now:
                        break

            if self.killer.kill_now:
                break

            prompt = str(row.get("full_prompt_text", row.get("sentence", "")))
            pbar.write(f"🔄 Retry {retry_idx}/{len(retry_batch)}: {prompt[:50]}...")

            # Make API request for retry
            response_data = self.make_api_request(prompt)

            if response_data:
                surprisal_score = self.calculate_surprisal_score(response_data)
                response_time = response_data.get("response_time", 0)

                result = {
                    "sentence": prompt,
                    "name_category": str(row.get("name_category", "")),
                    "trait_category": str(row.get("trait_category", "")),
                    "profession": str(row.get("profession", "")),
                    "name": str(row.get("name", "")),
                    "trait": str(row.get("trait", "")),
                    "comparison_type": str(row.get("comparison_type", "")),
                    "template_id": str(row.get("template_id", "")),
                    "surprisal_score": surprisal_score,
                    "model_response": str(response_data.get("response", ""))[:200],
                    "eval_duration": response_data.get("eval_duration", 0),
                    "eval_count": response_data.get("eval_count", 0),
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time,
                }

                results.append({k: v for k, v in result.items()})
                self.progress.completed_tests += 1
                self.update_response_time(response_time)
                retry_successes += 1

                # Reset failure counter when retry succeeds
                if self.progress.failed_tests > 0:
                    self.progress.failed_tests -= 1

                pbar.write(
                    f"✅ Retry success! Score: {surprisal_score:.2f}, Time: {response_time:.1f}s (Failed count reduced)"
                )
            else:
                retry_failures += 1
                # Don't increment failed_tests here as it was already counted
                pbar.write("❌ Retry failed (will try again later)")

        pbar.write("─" * 60)
        pbar.write(
            f"✅ Retry batch completed: {retry_successes} successes, {retry_failures} failures\n"
        )
        if retry_successes > 0:
            pbar.write(f"📉 Total failed count reduced by {retry_successes}\n")

    def _verify_and_complete_missing_tests(self, df, results: list, pbar) -> None:
        """Verify completion and process any missing tests"""
        processed_indices = set()

        # Track which tests we've actually processed
        for result in results:
            sentence = result.get("sentence", "")
            for idx, row in df.iterrows():
                row_sentence = str(row.get("full_prompt_text", row.get("sentence", "")))
                if sentence == row_sentence:
                    processed_indices.add(idx)
                    break

        # Find missing tests
        all_indices = set(range(len(df)))
        missing_indices = all_indices - processed_indices

        if missing_indices:
            missing_count = len(missing_indices)
            pbar.write(f"\n⚠️  Found {missing_count} unprocessed tests!")
            pbar.write("🔍 Processing missing tests to ensure completeness...")
            pbar.write("─" * 60)

            for missing_idx in sorted(missing_indices):
                if self.killer.kill_now:
                    break

                row = df.iloc[missing_idx]
                prompt = str(row.get("full_prompt_text", row.get("sentence", "")))
                pbar.write(f"🔄 Missing test {missing_idx}: {prompt[:50]}...")

                response_data = self.make_api_request(prompt)

                if response_data:
                    surprisal_score = self.calculate_surprisal_score(response_data)
                    response_time = response_data.get("response_time", 0)

                    result: dict[str, Any] = {
                        "sentence": prompt,
                        "name_category": str(row.get("name_category", "")),
                        "trait_category": str(row.get("trait_category", "")),
                        "profession": str(row.get("profession", "")),
                        "name": str(row.get("name", "")),
                        "trait": str(row.get("trait", "")),
                        "comparison_type": str(row.get("comparison_type", "")),
                        "template_id": str(row.get("template_id", "")),
                        "surprisal_score": surprisal_score,
                        "model_response": str(response_data.get("response", ""))[:200],
                        "eval_duration": response_data.get("eval_duration", 0),
                        "eval_count": response_data.get("eval_count", 0),
                        "timestamp": datetime.now().isoformat(),
                        "response_time": response_time,
                    }

                    results.append(result)
                    self.progress.completed_tests += 1
                    self.update_response_time(response_time)

                    pbar.write(f"✅ Completed! Score: {surprisal_score:.2f}, Time: {response_time:.1f}s")
                else:
                    self.progress.failed_tests += 1
                    pbar.write("❌ Failed")

            pbar.write("─" * 60)
            pbar.write(f"✅ Missing tests completed: {missing_count} tests processed\n")
        else:
            pbar.write("\n✅ All tests completed - no missing tests found!\n")

    from collections.abc import Hashable

    def _process_test_batch_concurrent(
        self, test_batch: list, pbar
    ) -> list[dict[str, Any]]:
        """Process a batch of tests concurrently"""
        batch_results = []

        def process_single_test(test_data):
            idx, row = test_data
            prompt = str(row.get("full_prompt_text", row.get("sentence", "")))

            response_data = self.make_api_request(prompt)

            if response_data:
                surprisal_score = self.calculate_surprisal_score(response_data)
                response_time = response_data.get("response_time", 0)

                result: dict[str, Any] = {
                    "sentence": prompt,
                    "name_category": str(row.get("name_category", "")),
                    "trait_category": str(row.get("trait_category", "")),
                    "profession": str(row.get("profession", "")),
                    "name": str(row.get("name", "")),
                    "trait": str(row.get("trait", "")),
                    "comparison_type": str(row.get("comparison_type", "")),
                    "template_id": str(row.get("template_id", "")),
                    "surprisal_score": surprisal_score,
                    "model_response": str(response_data.get("response", ""))[:200],
                    "eval_duration": response_data.get("eval_duration", 0),
                    "eval_count": response_data.get("eval_count", 0),
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time,
                }
                return idx, result, True, response_time
            else:
                return idx, row, False, 0

        # Process batch concurrently
        with ThreadPoolExecutor(max_workers=self.current_workers) as executor:
            future_to_test = {executor.submit(process_single_test, test): test for test in test_batch}

            for future in as_completed(future_to_test):
                idx, result_or_row, success, response_time = future.result()

                if success:
                    batch_results.append(result_or_row)
                    self.progress.completed_tests += 1
                    self.update_response_time(response_time)

                    # Handle dynamic scaling
                    self._handle_request_result(True, pbar)

                    # Calculate success rate
                    total_processed = self.progress.completed_tests + self.progress.failed_tests
                    success_rate = (self.progress.completed_tests / total_processed * 100) if total_processed > 0 else 100.0

                    pbar.set_postfix({
                        'Workers': self.current_workers,
                        'Success Rate': f"{success_rate:.1f}%",
                        'Failed': f"{self.progress.failed_tests}",
                        'Score': f"{result_or_row['surprisal_score']:.2f}"
                    })
                else:
                    self.add_failed_test(idx, result_or_row)
                    self.progress.failed_tests += 1

                    # Handle dynamic scaling
                    self._handle_request_result(False, pbar)

                    # Calculate success rate
                    total_processed = self.progress.completed_tests + self.progress.failed_tests
                    success_rate = (self.progress.completed_tests / total_processed * 100) if total_processed > 0 else 100.0

                    pbar.set_postfix({
                        'Workers': self.current_workers,
                        'Success Rate': f"{success_rate:.1f}%",
                        'Failed': f"{self.progress.failed_tests}",
                        'Status': 'Failed'
                    })
                    self.progress.failed_tests += 1
                    pbar.write(f"❌ Test {idx}: Failed")

                pbar.update(1)

        return batch_results

    def _handle_request_result(self, success: bool, pbar) -> None:
        """Handle request results and manage dynamic concurrency scaling"""
        if success:
            self.consecutive_errors = 0
            self.consecutive_successes += 1

            # Gradually increase workers if we've had enough consecutive successes
            if (self.consecutive_successes >= self.recovery_threshold and
                self.current_workers < self.original_max_workers):
                self.current_workers = min(self.current_workers + 1, self.original_max_workers)
                self.consecutive_successes = 0  # Reset counter
                if self.current_workers == self.original_max_workers:
                    self.error_fallback_active = False
                pbar.write(f"� Scaling up: Increased to {self.current_workers} workers (success streak)")
                pbar.write("─" * 60)
        else:
            self.consecutive_successes = 0
            self.consecutive_errors += 1

            # Gradually reduce workers on repeated failures
            if (self.consecutive_errors >= self.max_consecutive_errors and
                self.current_workers > 1):
                self.current_workers = max(1, self.current_workers - 1)
                self.error_fallback_active = True
                self.consecutive_errors = 0  # Reset counter
                pbar.write(f"⚠️  {self.max_consecutive_errors} consecutive errors detected")
                pbar.write(f"� Scaling down: Reduced to {self.current_workers} workers for stability")
                pbar.write("─" * 60)

    def _check_system_load(self) -> dict:
        """Check system load and resource usage"""
        import psutil

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            load_info = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "high_load": cpu_percent > 80 or memory.percent > 85
            }

            return load_info
        except ImportError:
            # psutil not available, return safe defaults
            return {
                "cpu_usage": 0,
                "memory_usage": 0,
                "memory_available_gb": 8,
                "high_load": False
            }

    def exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = delay * 0.1 * (0.5 - abs(hash(str(time.time())) % 1000) / 1000)
        return delay + jitter

    def make_api_request(self, prompt: str) -> dict | None:
        """Make API request with exponential backoff retry and timing"""
        request_start = time.time()

        for attempt in range(self.max_retries):
            try:
                if self.killer.kill_now:
                    return None

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False},
                    timeout=300,  # Increased to 5 minutes for very slow models like llama2:latest
                )

                if response.status_code == 200:
                    result = response.json()
                    # Add timing information
                    result["response_time"] = time.time() - request_start
                    return result
                elif response.status_code == 500:
                    # Server error - likely model timeout or crash
                    self.error_counts["server_error_500"] += 1
                    if attempt == self.max_retries - 1:
                        logger.debug(
                            f"Server error 500 after {self.max_retries} attempts for prompt: {prompt[:50]}"
                        )
                else:
                    # Other HTTP errors
                    self.error_counts["other_error"] += 1
                    if attempt == self.max_retries - 1:
                        logger.debug(
                            f"HTTP {response.status_code} error for prompt: {prompt[:50]}"
                        )

            except requests.exceptions.Timeout:
                # Timeout - wait longer and retry
                self.error_counts["timeout"] += 1
                if attempt == self.max_retries - 1:
                    logger.debug(
                        f"Request timeout after {self.max_retries} attempts for prompt: {prompt[:50]}"
                    )
            except requests.exceptions.RequestException as e:
                # Network or connection errors
                self.error_counts["connection_error"] += 1
                if attempt == self.max_retries - 1:
                    logger.debug(
                        f"Request exception after {self.max_retries} attempts: {e}"
                    )
            except Exception as e:
                # Unexpected errors
                self.error_counts["other_error"] += 1
                if attempt == self.max_retries - 1:
                    logger.debug(
                        f"Unexpected error after {self.max_retries} attempts: {e}"
                    )

            if attempt < self.max_retries - 1:
                delay = self.exponential_backoff(attempt)
                time.sleep(delay)

        # Only log final failure (not individual retry attempts)
        return None

    def calculate_surprisal_score(self, response_data: dict) -> float:
        """Calculate surprisal score from response data"""
        try:
            eval_duration = response_data.get("eval_duration", 0)
            eval_count = response_data.get("eval_count", 1)

            if eval_count > 0:
                return eval_duration / eval_count
            else:
                return float("inf")

        except (ZeroDivisionError, TypeError):
            return float("inf")

    def calculate_relative_bias_score(
        self, male_response: dict, female_response: dict
    ) -> float:
        """Calculate bias as relative difference between gender responses"""
        try:
            male_time = male_response.get("eval_duration", 0) / male_response.get(
                "eval_count", 1
            )
            female_time = female_response.get("eval_duration", 0) / female_response.get(
                "eval_count", 1
            )

            if male_time > 0 and female_time > 0:
                # Return ratio - values > 1 indicate male took longer (potential bias)
                return male_time / female_time
            return 1.0  # No bias detected
        except (ZeroDivisionError, TypeError):
            return 1.0

    def calculate_normalized_surprisal_score(
        self, response_data: dict, baseline_time: float
    ) -> float:
        """Calculate surprisal normalized against model's baseline performance"""
        try:
            eval_duration = response_data.get("eval_duration", 0)
            eval_count = response_data.get("eval_count", 1)

            if eval_count > 0 and baseline_time > 0:
                raw_time_per_token = eval_duration / eval_count
                # Normalize against baseline to remove hardware effects
                return raw_time_per_token / baseline_time
            else:
                return float("inf")
        except (ZeroDivisionError, TypeError):
            return float("inf")

    def save_progress(self):
        """Save current progress to file"""
        try:
            self.progress.last_checkpoint = datetime.now().isoformat()
            with open(self.progress_file, "w") as f:
                json.dump(asdict(self.progress), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def save_progress_and_results(self, results: list[dict] | None) -> None:
        """
        Save current audit progress and results to disk in a cross-platform, robust way.
        This method ensures compatibility with Windows, macOS, and Linux.
        Results are saved as CSV and progress as JSON, using UTF-8 encoding.

        Args:
            results (list[dict], optional): List of result dictionaries to save. If None, only progress is saved.
        """
        try:
            self.save_progress()
            if results is not None:
                import os

                import pandas as pd

                # Ensure parent directory exists (cross-platform)
                os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

                # Avoid storing raw model responses for privacy/storage reasons.
                # Create a shallow copy of results and drop 'model_response' if present.
                sanitized = []
                for r in results:
                    if isinstance(r, dict) and "model_response" in r:
                        r_copy = dict(r)
                        r_copy.pop("model_response", None)
                        sanitized.append(r_copy)
                    else:
                        sanitized.append(r)

                df_results = pd.DataFrame(sanitized)
                # Save as UTF-8 CSV for universal compatibility
                df_results.to_csv(self.results_file, index=False, encoding="utf-8")
                logger.info(
                    f"💾 Results saved to {self.results_file} (responses removed)"
                )
                # Additionally, save full model responses to a separate CSV file
                try:
                    responses_file = (
                        str(self.results_file).rstrip(".csv")
                        + f"_responses_{self.session_id}.csv"
                    )

                    # Build response records with normalized single-line text
                    response_records = []
                    for idx, r in enumerate(results):
                        if not isinstance(r, dict):
                            continue
                        raw = r.get("model_response", "")
                        if raw is None:
                            raw = ""
                        # Normalize whitespace and newlines to a single space
                        one_line = " ".join(str(raw).split())

                        response_records.append(
                            {
                                "session_id": self.session_id,
                                "test_idx": r.get("test_idx", idx),
                                "name": r.get("name", ""),
                                "profession": r.get("profession", ""),
                                "trait": r.get("trait", ""),
                                "comparison_type": r.get("comparison_type", ""),
                                "template_id": r.get("template_id", ""),
                                "response_text": one_line,
                                "timestamp": r.get(
                                    "timestamp", datetime.now().isoformat()
                                ),
                            }
                        )

                    if response_records:
                        df_resp = pd.DataFrame(response_records)
                        df_resp.to_csv(responses_file, index=False, encoding="utf-8")
                        logger.info(f"💾 Full responses saved to {responses_file}")
                except Exception as e:
                    logger.warning(f"Failed to save full responses CSV: {e}")
        except Exception as e:
            logger.error(f"Failed to save progress and results: {e}")

    def calculate_eta(self, current_index: int, total_tests: int) -> tuple[str, float]:
        """Calculate ETA based on user input or actual performance"""
        remaining_tests = total_tests - current_index

        if remaining_tests <= 0:
            return "Completed", 0.0

        # Use user-provided ETA if available
        if self.user_eta_per_test is not None:
            eta_seconds = remaining_tests * self.user_eta_per_test
            eta_str = self.format_duration(eta_seconds)
            return f"ETA: {eta_str} (UE)", eta_seconds

        # Use actual response times if we have enough data
        if len(self.actual_response_times) >= 3:
            # Use the average of recent response times for better accuracy
            recent_times = self.actual_response_times[-10:]  # Last 10 responses
            avg_time = sum(recent_times) / len(recent_times)
            eta_seconds = remaining_tests * avg_time
            eta_str = self.format_duration(eta_seconds)
            return f"ETA: {eta_str} (adaptive)", eta_seconds

        # Fallback to session-based calculation if we have timing data
        if self.session_start_time and current_index > 0:
            elapsed = time.time() - self.session_start_time
            avg_time_per_test = elapsed / current_index
            eta_seconds = remaining_tests * avg_time_per_test
            eta_str = self.format_duration(eta_seconds)
            return f"ETA: {eta_str} (session avg)", eta_seconds

        return "ETA: Calculating...", 0.0

    def format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def update_response_time(self, response_time: float):
        """Update the response time tracking for ETA calculation"""
        self.actual_response_times.append(response_time)
        # Keep only the last 50 response times to avoid memory issues
        if len(self.actual_response_times) > 50:
            self.actual_response_times = self.actual_response_times[-50:]

    def load_progress(self, progress_file: str) -> bool:
        """Load progress from file for resumption"""
        try:
            # Validate and sanitize the progress file path to prevent path traversal
            progress_path = Path(progress_file).resolve()

            # Ensure the file exists and is within allowed directories
            if not progress_path.exists():
                logger.error(f"Progress file does not exist: {progress_path}")
                return False

            # Check if the file is within the current working directory or output directory
            cwd = Path.cwd().resolve()
            output_dir = Path(self.output_dir).resolve()

            if not (
                progress_path.is_relative_to(cwd)
                or progress_path.is_relative_to(output_dir)
            ):
                logger.error(f"Progress file path not allowed: {progress_path}")
                return False

            with open(progress_path) as f:
                data = json.load(f)

            # Filter data to only include fields that exist in AuditProgress
            from dataclasses import fields

            valid_fields = {field.name for field in fields(AuditProgress)}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}

            self.progress = AuditProgress(**filtered_data)
            logger.info(
                f"📂 Loaded progress: {self.progress.completed_tests}/{self.progress.total_tests} completed"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return False

    def run_audit(self, resume_file: str | None = None) -> bool:
        """Run the complete audit with error handling and resumption"""
        try:
            # Load existing progress if resuming
            if resume_file and os.path.exists(resume_file):
                if self.load_progress(resume_file):
                    logger.info("🔄 Resuming previous audit session")
                else:
                    logger.warning("Failed to load progress, starting fresh")

            # Check/start Ollama service
            if not self.check_ollama_service():
                logger.info("🚀 Starting Ollama service...")
                if not self.start_ollama_service():
                    logger.error("❌ Failed to start Ollama service")
                    return False

            # Ensure model is available
            if not self.ensure_model_available():
                logger.error("❌ Failed to ensure model availability")
                return False

            # Load corpus
            logger.info(f"📂 Loading corpus from {self.corpus_file}")
            try:
                df = pd.read_csv(self.corpus_file)
            except Exception as e:
                logger.error(f"❌ Failed to load corpus: {e}")
                return False

            # Initialize progress if starting fresh
            if self.progress.total_tests == 0:
                self.progress.total_tests = len(df)
                self.save_progress()

            logger.info(f"🎯 Starting audit: {self.progress.total_tests} total tests")
            logger.info(f"📊 Model: {self.model_name}")
            logger.info(f"💾 Results will be saved to: {self.results_file}")

            # Build a boxed summary for better readability
            try:
                total = self.progress.total_tests
                completed = self.progress.completed_tests
                pct = (completed / total * 100) if total else 0.0
                eta_mode = (
                    f"User {self.user_eta_per_test:.1f}s/test"
                    if self.user_eta_per_test is not None
                    else "Dynamic"
                )
                lines = [
                    f" Session: {self.session_id}",
                    f" Model:   {self.model_name}",
                    f" Corpus:  {self.corpus_file}",
                    f" Progress:{completed}/{total} ({pct:.1f}%)",
                    f" Output:  {self.results_file}",
                    f" ETA Mode:{eta_mode}",
                ]
                width = max(len(line_text) for line_text in lines) + 2
                top = "┌" + "─" * width + "┐"
                bottom = "└" + "─" * width + "┘"
                box = [top]
                for line_text in lines:
                    box.append("│" + line_text.ljust(width) + "│")
                box.append(bottom)
                for bl in box:
                    logger.info(bl)
            except Exception:
                # Fallback silently if box fails
                pass

            # Initialize session start time for ETA calculation
            if self.session_start_time is None:
                self.session_start_time = time.time()

            # Display ETA settings
            if self.user_eta_per_test is not None:
                total_estimated_time = (
                    self.progress.total_tests * self.user_eta_per_test
                )
                logger.info(
                    f"⏱️  User ETA estimate: {self.user_eta_per_test:.1f}s per test"
                )
                logger.info(
                    f"📅 Total estimated time: {self.format_duration(total_estimated_time)}"
                )
            else:
                logger.info(
                    "📈 ETA will be calculated dynamically based on actual performance"
                )

            # Prepare results file
            results: list[dict[str, Any]] = []
            if os.path.exists(self.results_file):
                try:
                    existing_df = pd.read_csv(self.results_file)
                    # Convert to the correct type
                    raw_results = existing_df.to_dict("records")
                    results = [
                        {str(k): v for k, v in record.items()} for record in raw_results
                    ]
                    logger.info(f"📂 Loaded {len(results)} existing results")
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}")

            # Process corpus with appropriate concurrency mode
            return self._process_corpus(df, resume_file)

        except KeyboardInterrupt:
            logger.info("🛑 Audit interrupted by user")
            logger.info("💾 Saving progress and partial results...")

            # Save current progress
            self.save_progress()

            # Save any partial results if they exist in the local scope
            # This will be handled by the processing methods
            logger.info("✅ Progress saved successfully")
            logger.info(
                f"🔄 To resume: uv run equilens audit --model {self.model_name} --corpus {self.corpus_file} --resume {self.progress_file}"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Audit error: {e}")
            self.save_progress()
            return False

    def _process_corpus(self, df: pd.DataFrame, resume_file: str | None = None) -> bool:
        """Process the corpus with appropriate concurrency settings"""

        # Filter out already processed tests if resuming
        if resume_file and self.progress.current_index >= 0:
            # Detect and correct inconsistent state produced by previous regression
            # (where current_index was advanced without increasing completed_tests).
            if (
                self.progress.completed_tests < self.progress.total_tests
                and self.progress.current_index + 1 != self.progress.completed_tests
            ):
                logger.warning(
                    "⚠️ Inconsistent progress detected: current_index=%s completed_tests=%s. Adjusting resume point.",
                    self.progress.current_index,
                    self.progress.completed_tests,
                )

                # If results file exists, try to derive a safer completed count
                derived_completed = None
                if os.path.exists(self.results_file):
                    try:
                        derived_completed = len(pd.read_csv(self.results_file))
                        # Use derived if it matches completed_tests delta significantly
                        if (
                            derived_completed is not None
                            and derived_completed > 0
                            and abs(derived_completed - self.progress.completed_tests)
                            <= 5
                        ):
                            self.progress.completed_tests = derived_completed
                            logger.info(
                                "🔄 Adjusted completed_tests from results file count: %s",
                                derived_completed,
                            )
                    except Exception as e:
                        logger.debug(
                            f"Could not derive completed tests from results: {e}"
                        )

                # Reset current_index to last truly completed test
                self.progress.current_index = max(self.progress.completed_tests - 1, -1)
                # Persist corrected progress early
                self.save_progress()

            resume_from = self.progress.completed_tests  # Next test index (0-based)
            tests_to_process = []
            for real_idx, (_, row) in enumerate(df.iterrows()):
                if real_idx >= resume_from:
                    tests_to_process.append((real_idx, row))
            logger.info(
                f"📂 Resuming at test {resume_from}/{self.progress.total_tests} (0-based index)"
            )
        else:
            tests_to_process = [(i, row) for i, (_, row) in enumerate(df.iterrows())]
            logger.info(f"🚀 Starting fresh audit with {len(tests_to_process)} tests")

        if not tests_to_process:
            logger.info("✅ All tests already completed!")
            return True

        start_time = time.time()

        # Setup concurrent processing if requested
        if self.max_workers and self.max_workers > 1:
            return self._process_concurrent(tests_to_process, start_time)
        else:
            return self._process_sequential(tests_to_process, start_time)

    def _process_concurrent(self, tests_to_process: list, start_time: float) -> bool:
        """Process tests using concurrent workers with dynamic scaling"""
        try:
            # Initialize progress tracking
            self._initialize_progress_tracking()

            # System monitoring for load detection
            if self._check_system_load()["high_load"]:
                logger.info(f"🔧 System under load - starting with reduced concurrency ({self.current_workers} workers)")

            logger.info(f"� Starting concurrent processing with {self.current_workers} workers...")
            logger.info(f"📊 Processing {len(tests_to_process)} tests")

            # Prepare results storage
            results: list[dict[str, Any]] = []
            if os.path.exists(self.results_file):
                try:
                    existing_df = pd.read_csv(self.results_file)
                    raw_results = existing_df.to_dict("records")
                    results = [
                        {str(k): v for k, v in record.items()} for record in raw_results
                    ]
                    logger.info(f"📂 Loaded {len(results)} existing results")
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}")

            # Process tests concurrently with dynamic scaling
            logger.info("\n🔍 AI Bias Detection Audit Progress")
            logger.info("Testing AI model responses against bias detection corpus")
            logger.info(
                "Progress shows: Completed/Total | Elapsed Time | Remaining Time | Success Rate"
            )
            logger.info("💡 Press 'p' at any time to pause/resume the audit\n")

            # Start the pause controller
            self.pause_controller.start_listener()

            with tqdm(
                total=self.progress.total_tests,
                initial=self.progress.completed_tests,
                desc="🔍 Auditing",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            ) as pbar:
                progress_handler = TqdmLoggingHandler()
                progress_handler.setLevel(logging.INFO)
                progress_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                logger.addHandler(progress_handler)
                original_console_level = console_handler.level
                console_handler.setLevel(logging.CRITICAL + 1)

                # Process tests in smaller batches to allow dynamic scaling
                batch_size = max(1, min(50, len(tests_to_process) // 10))

                for i in range(0, len(tests_to_process), batch_size):
                    if self.killer.kill_now:
                        logger.info("🛑 Audit interrupted by user")
                        break

                    # Check for pause before processing each batch
                    self.pause_controller.wait_if_paused()

                    batch = tests_to_process[i:i + batch_size]

                    # Process batch concurrently
                    batch_results = self._process_test_batch_concurrent(batch, pbar)
                    results.extend(
                        [{str(k): v for k, v in d.items()} for d in batch_results]
                    )

                    # Save progress periodically
                    if i % (batch_size * 5) == 0:  # Every 5 batches
                        self.save_progress_and_results(results)

                # Save final results
                self.save_progress_and_results(results)
                # Restore handlers
                logger.removeHandler(progress_handler)
                console_handler.setLevel(original_console_level)

            # Stop the pause controller
            self.pause_controller.stop_listener_thread()

            logger.info("✅ Concurrent processing completed!")
            logger.info(f"🤖 Final worker count: {self.current_workers}")
            return True

        except Exception as e:
            logger.error(f"⚠️  Concurrent processing failed: {e}")
            logger.info("🔄 Falling back to sequential processing...")
            return self._process_sequential(tests_to_process, start_time)

    def _process_sequential(self, tests_to_process: list, start_time: float) -> bool:
        """Process tests sequentially (fallback mode)"""

        from collections.abc import Hashable

        results: list[dict[Hashable, Any]] = []
        if os.path.exists(self.results_file):
            try:
                existing_df = pd.read_csv(self.results_file)
                results = existing_df.to_dict("records")
                logger.info(f"📂 Loaded {len(results)} existing results")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

        logger.info("🔄 Using sequential processing mode...")
        logger.info("🔍 AI Bias Detection Audit Progress")
        logger.info("Testing AI model responses against bias detection corpus")
        logger.info(
            "Progress shows: Completed/Total | Elapsed Time | Remaining Time | Success Rate | Failed Count"
        )
        logger.info("💡 Press 'p' at any time to pause/resume the audit\n")

        # Start the pause controller
        self.pause_controller.start_listener()

        with tqdm(
            total=self.progress.total_tests,
            initial=self.progress.completed_tests,
            desc="🔍 Auditing",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        ) as pbar:
            progress_handler = TqdmLoggingHandler()
            progress_handler.setLevel(logging.INFO)
            progress_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(progress_handler)
            original_console_level = console_handler.level
            console_handler.setLevel(logging.CRITICAL + 1)
            try:
                # NOTE: A previous regression dedented the request/processing block
                # out of this loop, causing only the final test to be executed while
                # the progress bar showed 0% then completed prematurely. This
                # indentation restores per-test processing.
                for test_idx, row in tests_to_process:
                    if self.killer.kill_now:
                        logger.info("🛑 Audit interrupted by user")
                        break

                    # Check for pause before processing each test
                    self.pause_controller.wait_if_paused()

                    self.progress.current_index = test_idx
                    prompt = str(row.get("full_prompt_text", row.get("sentence", "")))

                    # Calculate ETA
                    eta_str, eta_seconds = self.calculate_eta(
                        test_idx + 1, self.progress.total_tests
                    )

                    # Update description with pause status
                    pause_status = (
                        " [⏸️ PAUSED]" if self.pause_controller.is_paused() else ""
                    )
                    # Shorter description so the bar itself is longer; move ETA to postfix
                    snippet = prompt[:32].replace("\n", " ")
                    pbar.set_description(f"🔍 {snippet}...{pause_status}")

                    # Make API request
                    response_data = self.make_api_request(prompt)

                    if response_data:
                        # Process successful response
                        surprisal_score = self.calculate_surprisal_score(response_data)
                        response_time = response_data.get("response_time", 0)

                        result = {
                            "sentence": prompt,
                            "name_category": str(row.get("name_category", "")),
                            "trait_category": str(row.get("trait_category", "")),
                            "profession": str(row.get("profession", "")),
                            "name": str(row.get("name", "")),
                            "trait": str(row.get("trait", "")),
                            "comparison_type": str(row.get("comparison_type", "")),
                            "template_id": str(row.get("template_id", "")),
                            "surprisal_score": surprisal_score,
                            "model_response": str(response_data.get("response", ""))[
                                :200
                            ],
                            "eval_duration": response_data.get("eval_duration", 0),
                            "eval_count": response_data.get("eval_count", 0),
                            "timestamp": datetime.now().isoformat(),
                            "response_time": response_time,
                        }

                        results.append(result)
                        self.progress.completed_tests += 1
                        self.update_response_time(response_time)

                        # Increment success counter for retry processing
                        self.success_count_since_last_retry += 1

                        # Calculate success rate
                        total_processed = (
                            self.progress.completed_tests + self.progress.failed_tests
                        )
                        success_rate = (
                            (self.progress.completed_tests / total_processed * 100)
                            if total_processed > 0
                            else 100.0
                        )

                        pbar.set_postfix(
                            {
                                "ETA": eta_str.replace("ETA: ", ""),
                                "Succ": f"{success_rate:.1f}%",
                                "Fail": f"{self.progress.failed_tests}",
                                "Score": f"{surprisal_score:.1f}",
                                "T": f"{response_time:.1f}s",
                            }
                        )

                        # Check if we should process retries after successful responses
                        if self.should_process_retries():
                            self.process_retry_batch(pbar, results)

                    else:
                        # Handle failed request
                        self.add_failed_test(test_idx, row)
                        self.progress.failed_tests += 1

                        # Calculate success rate
                        total_processed = (
                            self.progress.completed_tests + self.progress.failed_tests
                        )
                        success_rate = (
                            (self.progress.completed_tests / total_processed * 100)
                            if total_processed > 0
                            else 100.0
                        )

                        pbar.set_postfix(
                            {
                                "ETA": eta_str.replace("ETA: ", ""),
                                "Succ": f"{success_rate:.1f}%",
                                "Fail": f"{self.progress.failed_tests}",
                                "Status": "Failed",
                            }
                        )

                    pbar.update(1)

                    # Save progress every 10 tests
                    if (test_idx + 1) % 10 == 0:
                        self.save_progress_and_results(results)

            except KeyboardInterrupt:
                pbar.write("\n🛑 KeyboardInterrupt detected - saving progress...")
                self.save_progress_and_results(results)
                pbar.write("✅ Progress and results saved successfully")
                pbar.write(
                    f"🔄 To resume: uv run equilens audit --model {self.model_name} --corpus {self.corpus_file} --resume {self.progress_file}"
                )
                raise  # Re-raise to be caught by outer handler

            # Process any remaining retries at the end
            if len(self.retry_queue) > 0:
                pbar.write(
                    f"\n🔄 Processing {len(self.retry_queue)} remaining failed tests..."
                )
                self.process_retry_batch(pbar, results)

            # Save final results
            self.save_progress_and_results(results)
            logger.removeHandler(progress_handler)
            console_handler.setLevel(original_console_level)

        # Stop the pause controller
        self.pause_controller.stop_listener_thread()

        logger.info("✅ Sequential processing completed!")
        if len(self.retry_queue) > 0:
            logger.info(
                f"⚠️  {len(self.retry_queue)} tests still in retry queue - may need additional processing"
            )

        # Display error statistics
        total_errors = sum(self.error_counts.values())
        if total_errors > 0:
            logger.info("\n📊 Error Statistics:")
            logger.info(
                f"   🔴 Server errors (500): {self.error_counts['server_error_500']}"
            )
            logger.info(f"   ⏰ Timeouts: {self.error_counts['timeout']}")
            logger.info(
                f"   🔌 Connection errors: {self.error_counts['connection_error']}"
            )
            logger.info(f"   ❓ Other errors: {self.error_counts['other_error']}")
            logger.info(f"   📈 Total error attempts: {total_errors}")

            if self.error_counts["server_error_500"] > 20:
                logger.info("\n💡 High number of 500 errors detected:")
                logger.info("   - Model may be running out of memory")
                logger.info(
                    "   - Consider using a smaller model (phi3:mini, llama3.2:1b)"
                )
                logger.info("   - Check Ollama container memory allocation")
                logger.info("\n🔧 Model Performance Issues Detected:")
                logger.info("   - llama2:latest is very resource-intensive")
                logger.info(
                    "   - Each request taking 30+ seconds indicates CPU-only processing"
                )
                logger.info("   - Recommended alternatives for faster processing:")
                logger.info("     • phi3:mini (1.3GB) - 2-5 seconds per request")
                logger.info("     • llama3.2:1b (1.3GB) - 3-8 seconds per request")
                logger.info("     • gemma2:2b (1.6GB) - 5-12 seconds per request")
                logger.info(
                    "\n🚀 To switch models: docker exec ollama-container ollama pull phi3:mini"
                )

        return True

    def _initialize_progress_tracking(self) -> None:
        """Initialize progress tracking variables"""
        self.processed_tests = 0

    def _save_intermediate_results(self, results: list[dict]) -> None:
        """Save intermediate results to CSV file"""
        try:
            if results:
                df_results = pd.DataFrame(results)
                df_results.to_csv(self.results_file, index=False)
                logger.debug(f"Saved {len(results)} results to {self.results_file}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def _process_test_batch(self, test_batch: list, pbar) -> list:
        """Process a batch of tests concurrently"""
        batch_results = []
        return batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced EquiLens Model Auditor",
        epilog="Example: uv run equilens audit --model llama2:latest --corpus data.csv --eta-per-test 5.0 --max-workers 3",
    )
    parser.add_argument("--model", required=True, help="Model name to audit")
    parser.add_argument("--corpus", required=True, help="Path to corpus CSV file")
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument("--resume", help="Resume from progress file")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--eta-per-test",
        type=float,
        default=None,
        help="Estimated time per test in seconds for ETA calculation (auto-detected if not provided)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent threads for processing (1-8, default: 1)",
    )

    args = parser.parse_args()

    # Quick model listing
    if args.list_models:
        auditor = ModelAuditor("dummy", "dummy")
        if auditor.check_ollama_service():
            auditor.list_available_models()
        else:
            print("❌ Ollama service not available")
        return

    # Validate inputs
    if not os.path.exists(args.corpus):
        logger.error(f"❌ Corpus file not found: {args.corpus}")
        return

    # Validate max_workers
    if args.max_workers < 1 or args.max_workers > 8:
        logger.error(f"❌ Invalid max_workers: {args.max_workers}. Must be 1-8.")
        return

    # Interactive ETA prompt if not provided via flag
    eta_per_test = args.eta_per_test

    if eta_per_test is None:
        print("\n" + "=" * 60)
        print("⏱️  ETA CONFIGURATION")
        print("=" * 60)
        print("Configure time estimation for progress tracking:")
        print("• Enter a number: Use custom seconds per test")
        print("• Press Enter: Auto-detect timing (recommended)")
        print("=" * 60)

        try:
            while True:
                try:
                    eta_input = input(
                        "Enter estimated seconds per test (or press Enter to skip): "
                    ).strip()
                    if eta_input == "":
                        print("⏭️  Skipping - will use auto-detection")
                        break

                    eta_per_test = float(eta_input)
                    if eta_per_test <= 0:
                        print("❌ Please enter a positive number")
                        continue

                    print(f"✅ ETA set to {eta_per_test:.1f} seconds per test")
                    break

                except ValueError:
                    print("❌ Please enter a valid number")
                    continue

        except KeyboardInterrupt:
            print("\n\n⏹️  Cancelled by user")
            return
        except Exception as e:
            print(f"\n❌ Input error: {e}")
            print("⏭️  Continuing with auto-detection")

        print("=" * 60 + "\n")

    # Create and run auditor with concurrent processing support
    auditor = ModelAuditor(args.model, args.corpus, args.output_dir, eta_per_test, args.max_workers)
    success = auditor.run_audit(args.resume)

    if success:
        logger.info("🎉 Audit completed successfully!")
        logger.info("📁 Next steps:")
        logger.info(f"   1. Review results: {auditor.results_file}")
        logger.info(
            f"   2. Run analysis: python Phase3_Analysis/analyze_results.py --results_file {auditor.results_file}"
        )
    else:
        logger.error("❌ Audit failed or was interrupted")
        logger.info(
            f"🔄 To resume: python {sys.argv[0]} --model {args.model} --corpus {args.corpus} --resume {auditor.progress_file}"
        )


if __name__ == "__main__":
    main()
