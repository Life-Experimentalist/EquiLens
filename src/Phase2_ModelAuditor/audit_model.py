#!/usr/bin/env python3
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
import json
import logging
import os
import signal
import subprocess

# Configure logging with Unicode-safe console output
import sys
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
        self.retry_batch_size = 10  # Retry after this many successes

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

        # Add to retry queue if failed 5 times
        if self.failed_tests[idx][1] >= 5:
            if idx not in [test_idx for test_idx, _ in self.retry_queue]:
                self.retry_queue.append((idx, row))
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

        for retry_idx, (_, row) in enumerate(retry_batch, 1):
            if self.killer.kill_now:
                break

            prompt = str(row.get("full_prompt_text", row.get("sentence", "")))
            pbar.write(f"🔄 Retry {retry_idx}/{len(retry_batch)}: {prompt[:50]}...")

            # Make API request for retry
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

                pbar.write(
                    f"✅ Retry success! Score: {surprisal_score:.2f}, Time: {response_time:.1f}s"
                )
            else:
                self.progress.failed_tests += 1
                pbar.write("❌ Retry failed")

        pbar.write("─" * 60)
        pbar.write(f"✅ Retry batch completed: {len(retry_batch)} tests processed\n")

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

    def _process_test_batch_concurrent(self, test_batch: list, pbar) -> list[dict[str, Any]]:
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
                    self.success_count_since_last_retry += 1

                    pbar.write(f"✅ Test {idx}: Score {result_or_row['surprisal_score']:.2f}, Time {response_time:.1f}s")
                else:
                    self.add_failed_test(idx, result_or_row)
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
                    timeout=120,  # 2 minute timeout per request
                )

                if response.status_code == 200:
                    result = response.json()
                    # Add timing information
                    result["response_time"] = time.time() - request_start
                    return result
                else:
                    # Don't log warnings for retries to avoid cluttering tqdm
                    pass

            except requests.exceptions.Timeout:
                # Don't log timeout warnings for retries to avoid cluttering tqdm
                pass
            except requests.exceptions.RequestException:
                # Don't log request exceptions for retries to avoid cluttering tqdm
                pass
            except Exception:
                # Don't log unexpected errors for retries to avoid cluttering tqdm
                pass

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

    def save_progress(self):
        """Save current progress to file"""
        try:
            self.progress.last_checkpoint = datetime.now().isoformat()
            with open(self.progress_file, "w") as f:
                json.dump(asdict(self.progress), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

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
            with open(progress_file) as f:
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

            # Process each test with tqdm progress bar
            # Initialize tqdm with total processed tests (completed + failed)
            with tqdm(
                total=len(df),
                desc="🔍 Auditing Model",
                unit="tests",
                initial=self.progress.completed_tests + self.progress.failed_tests,
                position=0,
                leave=True,
                ncols=120,  # Fixed width to prevent display issues
                disable=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            ) as pbar:
                for idx_raw, row in df.iterrows():
                    # Handle pandas index which can be various types
                    if isinstance(idx_raw, int | float):
                        idx = int(idx_raw)
                    else:
                        idx = hash(idx_raw) % len(
                            df
                        )  # Fallback for non-numeric indices

                    if self.killer.kill_now:
                        logger.info("🛑 Audit interrupted by user")
                        break

                    # Skip if already processed (when resuming)
                    # current_index is the last processed dataframe index (0-based)
                    if idx <= self.progress.current_index:
                        continue

                    self.progress.current_index = idx
                    prompt = str(row.get("full_prompt_text", row.get("sentence", "")))

                    # Calculate ETA
                    eta_str, eta_seconds = self.calculate_eta(
                        idx + 1, self.progress.total_tests
                    )

                    # Check system load before processing
                    load_info = self._check_system_load()
                    if load_info["high_load"] and self.max_workers > 1:
                        # Temporarily reduce concurrency under high load
                        if not self.error_fallback_active:
                            pbar.write(f"⚠️  High system load detected (CPU: {load_info['cpu_usage']:.1f}%, Memory: {load_info['memory_usage']:.1f}%)")
                            pbar.write("🔄 Reducing concurrency temporarily")

                    # Update progress bar description with current prompt and ETA
                    pbar.set_description(f"🔍 Processing ({eta_str}): {prompt[:20]}...")

                    # Make API request
                    response_data = self.make_api_request(prompt)

                    # Handle the result and update error tracking
                    success = response_data is not None
                    self._handle_request_result(success, pbar)

                # Process test based on concurrency mode
                if self.max_workers == 1 or self.error_fallback_active:
                    # Single-threaded processing
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
                            "model_response": str(response_data.get("response", ""))[
                                :200
                            ],  # Truncate for storage
                            "eval_duration": response_data.get("eval_duration", 0),
                            "eval_count": response_data.get("eval_count", 0),
                            "timestamp": datetime.now().isoformat(),
                            "response_time": response_time,
                        }

                        results.append(result)
                        self.progress.completed_tests += 1

                        # Update response time tracking for ETA calculation
                        self.update_response_time(response_time)

                        # Track successful tests for retry processing
                        self.success_count_since_last_retry += 1

                        # Update progress bar postfix with latest results
                        worker_status = f"W:{self.max_workers}"
                        if self.error_fallback_active:
                            worker_status += " (FB)"  # Fallback mode indicator

                        pbar.set_postfix(
                            {
                                "Score": f"{surprisal_score:.2f}",
                                "Time": f"{response_time:.1f}s",
                                "Success": f"{self.progress.completed_tests}/{self.progress.total_tests}",
                                "Workers": worker_status,
                            }
                        )

                    else:
                        # Failed request - add to retry tracking
                        self.add_failed_test(idx, row)
                        self.progress.failed_tests += 1
                        # Update progress bar postfix with failure info
                        worker_status = f"W:{self.max_workers}"
                        if self.error_fallback_active:
                            worker_status += " (FB)"

                        pbar.set_postfix(
                            {
                                "Failed": f"{self.progress.failed_tests}",
                                "Success": f"{self.progress.completed_tests}/{self.progress.total_tests}",
                                "Retries": f"{len(self.retry_queue)}",
                                "Workers": worker_status,
                            }
                        )

                    # Check if we should process retries
                    if self.should_process_retries():
                        self.process_retry_batch(pbar, results)

                    # Update progress bar
                    pbar.update(1)

                    # Save progress every 10 tests
                    if (idx + 1) % 10 == 0:
                        self.save_progress()
                        self._save_intermediate_results(results)

            # Process any remaining retry tests at the end
            if self.retry_queue:
                pbar.write(
                    f"\n🔄 Processing final {len(self.retry_queue)} retry tests..."
                )
                self.process_retry_batch(pbar, results)

            # Verify completion and handle missing tests
            self._verify_and_complete_missing_tests(df, results, pbar)

            # Final save
            self.save_progress()
            self._save_final_results(results)

            # Calculate comprehensive metrics for completion summary
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.progress.start_time)
            total_duration = end_time - start_time

            # Calculate timing metrics
            total_seconds = total_duration.total_seconds()
            total_minutes = total_seconds / 60
            total_hours = total_minutes / 60

            # Calculate average response time from results
            response_times = [
                r.get("response_time", 0) for r in results if r.get("response_time")
            ]
            avg_response_time = (
                sum(response_times) / len(response_times) if response_times else 0
            )
            min_response_time = min(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0

            # Calculate throughput
            throughput_per_second = (
                self.progress.completed_tests / total_seconds
                if total_seconds > 0
                else 0
            )
            throughput_per_minute = throughput_per_second * 60

            # Success rate
            success_rate = (
                (self.progress.completed_tests / self.progress.total_tests * 100)
                if self.progress.total_tests > 0
                else 0
            )

            # Colorful completion summary with comprehensive metrics
            if COLORS_AVAILABLE:
                logger.info(
                    f"{COLOR_GREEN}{STYLE_BRIGHT}🎉 Audit completed!{STYLE_RESET}"
                )
                logger.info(f"{COLOR_CYAN}📊 {STYLE_BRIGHT}AUDIT SUMMARY{STYLE_RESET}")
                logger.info(f"{COLOR_CYAN}{'=' * 50}{STYLE_RESET}")

                # Basic results
                logger.info(
                    f"{COLOR_WHITE}Tests: {STYLE_RESET}"
                    f"{COLOR_GREEN}{self.progress.completed_tests} completed{STYLE_RESET}, "
                    f"{COLOR_RED}{self.progress.failed_tests} failed{STYLE_RESET} "
                    f"({COLOR_YELLOW}{success_rate:.1f}% success rate{STYLE_RESET})"
                )

                # Timing metrics
                if total_hours >= 1:
                    duration_str = f"{total_hours:.1f} hours ({total_minutes:.1f} min)"
                elif total_minutes >= 1:
                    duration_str = f"{total_minutes:.1f} minutes ({total_seconds:.1f}s)"
                else:
                    duration_str = f"{total_seconds:.1f} seconds"

                logger.info(
                    f"{COLOR_BLUE}⏱️  Total Time: {STYLE_BRIGHT}{duration_str}{STYLE_RESET}"
                )
                logger.info(
                    f"{COLOR_MAGENTA}🚀 Throughput: {STYLE_BRIGHT}{throughput_per_minute:.2f} tests/min{STYLE_RESET} ({throughput_per_second:.3f} tests/sec)"
                )

                # Response time metrics
                if response_times:
                    logger.info(f"{COLOR_YELLOW}📈 Response Times:{STYLE_RESET}")
                    logger.info(
                        f"   • Average: {COLOR_WHITE}{avg_response_time:.1f}s{STYLE_RESET}"
                    )
                    logger.info(
                        f"   • Fastest: {COLOR_GREEN}{min_response_time:.1f}s{STYLE_RESET}"
                    )
                    logger.info(
                        f"   • Slowest: {COLOR_RED}{max_response_time:.1f}s{STYLE_RESET}"
                    )

                logger.info(f"{COLOR_CYAN}{'=' * 50}{STYLE_RESET}")

            else:
                logger.info("🎉 Audit completed!")
                logger.info("📊 AUDIT SUMMARY")
                logger.info("=" * 50)
                logger.info(
                    f"Tests: {self.progress.completed_tests} completed, {self.progress.failed_tests} failed "
                    f"({success_rate:.1f}% success rate)"
                )

                # Timing metrics
                if total_hours >= 1:
                    duration_str = f"{total_hours:.1f} hours ({total_minutes:.1f} min)"
                elif total_minutes >= 1:
                    duration_str = f"{total_minutes:.1f} minutes ({total_seconds:.1f}s)"
                else:
                    duration_str = f"{total_seconds:.1f} seconds"

                logger.info(f"⏱️  Total Time: {duration_str}")
                logger.info(
                    f"🚀 Throughput: {throughput_per_minute:.2f} tests/min ({throughput_per_second:.3f} tests/sec)"
                )

                # Response time metrics
                if response_times:
                    logger.info("📈 Response Times:")
                    logger.info(f"   • Average: {avg_response_time:.1f}s")
                    logger.info(f"   • Fastest: {min_response_time:.1f}s")
                    logger.info(f"   • Slowest: {max_response_time:.1f}s")

                logger.info("=" * 50)

            logger.info(f"💾 Results saved to: {self.results_file}")

            return True

        except Exception as e:
            logger.error(f"❌ Audit failed with exception: {e}")
            self.save_progress()
            return False

    def _save_intermediate_results(self, results: list[dict]):
        """Save intermediate results"""
        try:
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.results_file, index=False)
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def _save_final_results(self, results: list[dict]):
        """Save final results with comprehensive summary"""
        try:
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.results_file, index=False)

            # Calculate comprehensive metrics
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.progress.start_time)
            total_duration = end_time - start_time
            total_seconds = total_duration.total_seconds()

            # Calculate response time metrics
            response_times = [
                r.get("response_time", 0) for r in results if r.get("response_time")
            ]
            avg_response_time = (
                sum(response_times) / len(response_times) if response_times else 0
            )
            min_response_time = min(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0

            # Calculate throughput
            throughput_per_second = (
                self.progress.completed_tests / total_seconds
                if total_seconds > 0
                else 0
            )
            throughput_per_minute = throughput_per_second * 60

            # Calculate success rate
            success_rate = (
                (self.progress.completed_tests / self.progress.total_tests * 100)
                if self.progress.total_tests > 0
                else 0
            )

            # Create comprehensive summary
            summary = {
                "session_info": {
                    "session_id": self.session_id,
                    "model_name": self.model_name,
                    "corpus_file": self.corpus_file,
                    "start_time": self.progress.start_time,
                    "end_time": end_time.isoformat(),
                    "results_file": str(self.results_file),
                },
                "test_results": {
                    "total_tests": self.progress.total_tests,
                    "completed_tests": self.progress.completed_tests,
                    "failed_tests": self.progress.failed_tests,
                    "success_rate_percent": (
                        self.progress.completed_tests / self.progress.total_tests * 100
                    )
                    if self.progress.total_tests > 0
                    else 0,
                },
                "timing_metrics": {
                    "total_duration_seconds": total_seconds,
                    "total_duration_minutes": total_seconds / 60,
                    "total_duration_hours": total_seconds / 3600,
                    "average_response_time_seconds": avg_response_time,
                    "min_response_time_seconds": min_response_time,
                    "max_response_time_seconds": max_response_time,
                    "throughput_tests_per_second": throughput_per_second,
                    "throughput_tests_per_minute": throughput_per_minute,
                },
                "performance_summary": {
                    "fastest_test_time": min_response_time,
                    "slowest_test_time": max_response_time,
                    "efficiency_score": success_rate * throughput_per_minute
                    if self.progress.total_tests > 0
                    else 0,
                    "total_api_calls": len(response_times),
                },
            }

            summary_file = self.model_session_dir / f"summary_{self.session_id}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"📋 Summary saved to: {summary_file}")

        except Exception as e:
            logger.error(f"Failed to save final results: {e}")


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
        print("🔮 ETA ESTIMATION SETUP")
        print("=" * 60)
        print("You can provide an estimated time per test for better ETA predictions.")
        print("This helps you plan your audit session duration.")
        print("\nOptions:")
        print("  • Provide estimate: Get accurate ETA throughout the session")
        print("  • Skip: System will auto-detect timing after first few tests")
        print("=" * 60)

        try:
            provide_eta = (
                input("Would you like to provide an ETA estimate? [y/N]: ")
                .strip()
                .lower()
            )

            if provide_eta in ["y", "yes"]:
                print("\n📋 Quick Reference - Typical times per test:")
                print("  • Fast models (phi3:mini, qwen2:0.5b):     1-3 seconds")
                print("  • Medium models (llama3.2:3b, gemma2:2b): 3-8 seconds")
                print("  • Large models (llama2:latest):           8-20 seconds")
                print("  • Very large models (llama3.1:70b):       20+ seconds")
                print(f"\n🎯 Estimating for model: {args.model}")

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
            else:
                print("⏭️  Using auto-detection - ETA will be calculated dynamically")

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
