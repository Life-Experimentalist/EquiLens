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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

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

    def __init__(self, model_name: str, corpus_file: str, output_dir: str = "results"):
        self.model_name = model_name
        self.corpus_file = corpus_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

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
                    logger.warning(
                        f"API request failed (attempt {attempt + 1}): {response.status_code}"
                    )

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request exception (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                delay = self.exponential_backoff(attempt)
                logger.info(f"⏳ Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

        logger.error(f"❌ Failed to get response after {self.max_retries} attempts")
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

    def load_progress(self, progress_file: str) -> bool:
        """Load progress from file for resumption"""
        try:
            with open(progress_file) as f:
                data = json.load(f)
                self.progress = AuditProgress(**data)
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

            # Process each test
            for idx_raw, row in df.iterrows():
                # Handle pandas index which can be various types
                if isinstance(idx_raw, int | float):
                    idx = int(idx_raw)
                else:
                    idx = hash(idx_raw) % len(df)  # Fallback for non-numeric indices

                if self.killer.kill_now:
                    logger.info("🛑 Audit interrupted by user")
                    break

                # Skip if already processed
                if idx < self.progress.current_index:
                    continue

                self.progress.current_index = idx
                prompt = str(row.get("full_prompt_text", row.get("sentence", "")))

                # Colorful progress indicator with timing
                progress_percent = ((idx + 1) / self.progress.total_tests) * 100
                if COLORS_AVAILABLE:
                    logger.info(
                        f"{COLOR_CYAN}🔍 Processing {STYLE_BRIGHT}{idx + 1}/{self.progress.total_tests}{STYLE_RESET} "
                        f"{COLOR_YELLOW}({progress_percent:.1f}%){STYLE_RESET}: {COLOR_WHITE}{prompt[:50]}...{STYLE_RESET}"
                    )
                else:
                    logger.info(
                        f"🔍 Processing {idx + 1}/{self.progress.total_tests} ({progress_percent:.1f}%): {prompt[:50]}..."
                    )

                # Make API request
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

                    # Colorful success message with timing
                    if COLORS_AVAILABLE:
                        logger.info(
                            f"{COLOR_GREEN}✅ Success!{STYLE_RESET} "
                            f"{COLOR_MAGENTA}Score: {surprisal_score:.2f}{STYLE_RESET} | "
                            f"{COLOR_BLUE}Time: {response_time:.1f}s{STYLE_RESET}"
                        )
                    else:
                        logger.info(
                            f"✅ Success! Score: {surprisal_score:.2f} | Time: {response_time:.1f}s"
                        )

                else:
                    # Colorful failure message
                    if COLORS_AVAILABLE:
                        logger.warning(
                            f"{COLOR_RED}❌ Failed{STYLE_RESET} to get response for test {COLOR_YELLOW}{idx + 1}{STYLE_RESET}"
                        )
                    else:
                        logger.warning(f"❌ Failed to get response for test {idx + 1}")
                    self.progress.failed_tests += 1

                # Save progress every 10 tests
                if (idx + 1) % 10 == 0:
                    self.save_progress()
                    self._save_intermediate_results(results)
                    completion_rate = (
                        self.progress.completed_tests / self.progress.total_tests
                    ) * 100

                    # Colorful progress reporting
                    if COLORS_AVAILABLE:
                        logger.info(
                            f"{COLOR_CYAN}📊 Progress: {STYLE_BRIGHT}{completion_rate:.1f}%{STYLE_RESET} "
                            f"({COLOR_GREEN}{self.progress.completed_tests}{STYLE_RESET}/"
                            f"{COLOR_BLUE}{self.progress.total_tests}{STYLE_RESET})"
                        )
                    else:
                        logger.info(
                            f"📊 Progress: {completion_rate:.1f}% ({self.progress.completed_tests}/{self.progress.total_tests})"
                        )

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
    parser = argparse.ArgumentParser(description="Enhanced EquiLens Model Auditor")
    parser.add_argument("--model", required=True, help="Model name to audit")
    parser.add_argument("--corpus", required=True, help="Path to corpus CSV file")
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument("--resume", help="Resume from progress file")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
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

    # Create and run auditor
    auditor = ModelAuditor(args.model, args.corpus, args.output_dir)
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
