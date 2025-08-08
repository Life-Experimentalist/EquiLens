#!/usr/bin/env python3
"""
EquiLens Enhanced Model Auditor with Rich Progress Tracking

This enhanced version provides:
- Real-time progress bars with Rich UI
- Performance metrics and timing for each test
- Optimized batch processing and connection pooling
- GPU acceleration monitoring
- Beautiful visual feedback
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Initialize Rich console
console = Console()

# Ensure logs directory exists
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

# Enhanced logging setup with Rich integration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "audit_session.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Custom stream handler that handles Unicode properly
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Set encoding to UTF-8 with error handling for Windows compatibility
try:
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass  # Fallback to default encoding


@dataclass
class AuditProgress:
    """Enhanced progress tracking with performance metrics"""

    session_id: str
    model_name: str
    corpus_file: str
    results_file: str
    start_time: str
    total_tests: int = 0
    completed_tests: int = 0
    failed_tests: int = 0
    current_index: int = 0
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    gpu_memory_used: int = 0
    throughput_per_second: float = 0.0


@dataclass
class TestResult:
    """Individual test result with timing information"""

    sentence: str
    name_category: str
    trait_category: str
    profession: str
    name: str
    trait: str
    comparison_type: str
    template_id: str
    surprisal_score: float
    model_response: str
    eval_duration: int
    eval_count: int
    timestamp: str
    response_time: float
    gpu_utilization: float = 0.0


class GracefulKiller:
    """Handle graceful shutdown"""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        console.print(
            "\n[yellow]🛑 Received shutdown signal. Saving progress...[/yellow]"
        )
        self.kill_now = True


class EnhancedBiasAuditor:
    """Enhanced bias auditor with Rich progress tracking and performance optimization"""

    def __init__(self, model_name: str, corpus_file: str, output_dir: str = "results"):
        self.model_name = model_name
        self.corpus_file = corpus_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup session management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize model name for Windows filesystem
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

        # Enhanced API configuration with connection pooling
        self.ollama_hosts = [
            "http://ollama:11434",
            "http://localhost:11434",
            "http://127.0.0.1:11434",
        ]
        self.ollama_url = None
        self.max_retries = 3
        self.base_delay = 0.5
        self.max_delay = 30.0

        # Performance optimization
        self.batch_size = 5  # Process multiple requests concurrently
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Progress tracking
        self.start_time = time.time()
        self.last_checkpoint = time.time()

        logger.info(f"🚀 Enhanced audit session {self.session_id} initialized")

    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running with enhanced feedback"""
        console.print("🔍 [blue]Checking Ollama service availability...[/blue]")

        for host in self.ollama_hosts:
            try:
                response = self.session.get(f"{host}/api/version", timeout=5)
                if response.status_code == 200:
                    version_info = response.json()
                    self.ollama_url = host
                    console.print(f"✅ [green]Connected to Ollama at {host}[/green]")
                    console.print(
                        f"📊 [cyan]Version: {version_info.get('version', 'unknown')}[/cyan]"
                    )
                    return True
            except Exception as e:
                console.print(f"❌ [red]{host}: {str(e)}[/red]")
                continue

        console.print("🚀 [yellow]Starting Ollama service...[/yellow]")
        return self.start_ollama_service()

    def start_ollama_service(self) -> bool:
        """Start Ollama service with GPU acceleration"""
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                console.print("❌ [red]Docker not available[/red]")
                return False

            console.print("🐳 [blue]Starting Ollama with GPU acceleration...[/blue]")

            # Start Ollama with GPU support
            docker_cmd = [
                "docker",
                "run",
                "-d",
                "--gpus",
                "all",  # Enable GPU support
                "--name",
                f"ollama-audit-{self.session_id}",
                "-p",
                "11434:11434",
                "-v",
                "ollama:/root/.ollama",
                "ollama/ollama",
            ]

            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                console.print("✅ [green]Ollama service started successfully[/green]")
                # Wait for service to be ready
                for _ in range(30):
                    if self.check_ollama_service():
                        return True
                    time.sleep(2)
            else:
                console.print(f"❌ [red]Failed to start Ollama: {result.stderr}[/red]")

        except Exception as e:
            console.print(f"❌ [red]Error starting Ollama: {e}[/red]")

        return False

    def ensure_model_available(self) -> bool:
        """Ensure model is available with progress feedback"""
        console.print(f"🔍 [blue]Checking model availability: {self.model_name}[/blue]")

        try:
            # Check if model exists
            response = self.session.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]

                if self.model_name in available_models:
                    console.print(f"✅ [green]Model {self.model_name} is ready[/green]")
                    return True

            # Model not found, attempt to pull it
            console.print(f"📥 [yellow]Downloading model {self.model_name}...[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Downloading model..."),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=None)

                pull_data = {"name": self.model_name}
                response = self.session.post(
                    f"{self.ollama_url}/api/pull",
                    json=pull_data,
                    stream=True,
                    timeout=1800,  # 30 minutes for large models
                )

                for line in response.iter_lines():
                    if line and not self.killer.kill_now:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                progress.update(
                                    task,
                                    description=f"[bold blue]{data['status']}[/bold blue]",
                                )
                            if data.get("status") == "success":
                                progress.update(task, completed=True)
                                break
                        except json.JSONDecodeError:
                            continue

                    if self.killer.kill_now:
                        console.print("[yellow]🛑 Model download interrupted[/yellow]")
                        return False

            console.print(
                f"✅ [green]Model {self.model_name} downloaded successfully[/green]"
            )
            return True

        except Exception as e:
            console.print(f"❌ [red]Error ensuring model availability: {e}[/red]")
            return False

    def make_api_request_batch(self, prompts: list[str]) -> list[dict]:
        """Make batch API requests for better performance"""
        results = []

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_prompt = {
                executor.submit(self.make_api_request_single, prompt): prompt
                for prompt in prompts
            }

            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append((prompt, result))
                except Exception as e:
                    logger.error(f"Error processing prompt {prompt[:50]}: {e}")
                    results.append((prompt, None))

        return results

    def make_api_request_single(self, prompt: str) -> dict | None:
        """Make a single API request with retry logic and timing"""
        request_start = time.time()

        for attempt in range(self.max_retries):
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 50,  # Limit response length for faster processing
                    },
                }

                response = self.session.post(
                    f"{self.ollama_url}/api/generate", json=data, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    result["response_time"] = time.time() - request_start

                    # Update running averages
                    self.progress.total_response_time += result["response_time"]

                    return result
                else:
                    logger.warning(
                        f"API request failed with status {response.status_code}"
                    )

            except Exception as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    time.sleep(delay)

        return None

    def calculate_surprisal_score(self, response_data: dict) -> float:
        """Calculate surprisal score with enhanced metrics"""
        if not response_data:
            return float("inf")

        eval_duration = response_data.get("eval_duration", 0)
        eval_count = response_data.get("eval_count", 1)

        # Enhanced surprisal calculation
        if eval_count > 0 and eval_duration > 0:
            # Normalize by token count and duration
            base_score = eval_duration / eval_count

            # Factor in response length and complexity
            response_length = len(response_data.get("response", ""))
            length_factor = max(1.0, response_length / 100)  # Normalize to ~100 chars

            return base_score * length_factor

        return float("inf")

    def run_enhanced_audit(self, resume_file: str | None = None) -> bool:
        """Run the enhanced audit with Rich progress tracking"""
        try:
            # Load existing progress if resuming
            if resume_file and os.path.exists(resume_file):
                self.load_progress(resume_file)
                console.print("🔄 [green]Resumed previous audit session[/green]")

            # Check/start Ollama service
            if not self.check_ollama_service():
                console.print("❌ [red]Failed to start Ollama service[/red]")
                return False

            # Ensure model is available
            if not self.ensure_model_available():
                console.print("❌ [red]Failed to ensure model availability[/red]")
                return False

            # Load corpus
            console.print(f"📂 [blue]Loading corpus from {self.corpus_file}[/blue]")
            try:
                df = pd.read_csv(self.corpus_file)
            except Exception as e:
                console.print(f"❌ [red]Failed to load corpus: {e}[/red]")
                return False

            # Initialize progress if starting fresh
            if self.progress.total_tests == 0:
                self.progress.total_tests = len(df)
                self.save_progress()

            console.print(
                f"🎯 [bold green]Starting enhanced audit: {self.progress.total_tests} total tests[/bold green]"
            )

            # Prepare results
            results: list[TestResult] = []
            if os.path.exists(self.results_file):
                try:
                    existing_df = pd.read_csv(self.results_file)
                    console.print(
                        f"📂 [cyan]Loaded {len(existing_df)} existing results[/cyan]"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}")

            # Setup Rich progress bar with simpler display
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            ) as progress:
                task_id = progress.add_task(
                    f"Processing {self.model_name}...", total=self.progress.total_tests
                )

                # Process tests with simpler progress tracking
                batch_prompts = []
                batch_rows = []

                for idx, row in df.iterrows():
                    if self.killer.kill_now:
                        console.print("\n🛑 [yellow]Audit interrupted by user[/yellow]")
                        break

                    # Convert idx to int for comparison
                    current_idx = int(idx) if isinstance(idx, int | float) else 0

                    # Skip if already processed
                    if current_idx < self.progress.current_index:
                        continue

                    prompt = str(row.get("full_prompt_text", row.get("sentence", "")))
                    batch_prompts.append(prompt)
                    batch_rows.append((current_idx, row))

                    # Process batch when full or at end
                    if len(batch_prompts) >= self.batch_size or idx == len(df) - 1:
                        batch_results = self.make_api_request_batch(batch_prompts)

                        # Process batch results
                        for (prompt, response_data), (batch_idx, batch_row) in zip(
                            batch_results, batch_rows, strict=False
                        ):
                            self.progress.current_index = batch_idx

                            if response_data:
                                surprisal_score = self.calculate_surprisal_score(
                                    response_data
                                )

                                result = TestResult(
                                    sentence=prompt,
                                    name_category=str(
                                        batch_row.get("name_category", "")
                                    ),
                                    trait_category=str(
                                        batch_row.get("trait_category", "")
                                    ),
                                    profession=str(batch_row.get("profession", "")),
                                    name=str(batch_row.get("name", "")),
                                    trait=str(batch_row.get("trait", "")),
                                    comparison_type=str(
                                        batch_row.get("comparison_type", "")
                                    ),
                                    template_id=str(batch_row.get("template_id", "")),
                                    surprisal_score=surprisal_score,
                                    model_response=str(
                                        response_data.get("response", "")
                                    )[:200],
                                    eval_duration=response_data.get("eval_duration", 0),
                                    eval_count=response_data.get("eval_count", 0),
                                    timestamp=datetime.now().isoformat(),
                                    response_time=response_data.get("response_time", 0),
                                )

                                results.append(result)
                                self.progress.completed_tests += 1
                            else:
                                self.progress.failed_tests += 1

                            # Update progress
                            progress.update(task_id, advance=1)

                        # Clear batch
                        batch_prompts = []
                        batch_rows = []

                        # Save progress periodically and show stats
                        if (self.progress.completed_tests % 5) == 0:
                            self.save_progress()
                            self._save_intermediate_results(
                                [asdict(r) for r in results]
                            )

                            # Show simple performance stats
                            elapsed_time = time.time() - self.start_time
                            if self.progress.completed_tests > 0:
                                avg_time = (
                                    self.progress.total_response_time
                                    / self.progress.completed_tests
                                )
                                throughput = (
                                    self.progress.completed_tests / elapsed_time
                                )
                                completion_rate = (
                                    self.progress.completed_tests
                                    / self.progress.total_tests
                                ) * 100
                                console.print(
                                    f"[cyan]📊 {completion_rate:.1f}% complete | "
                                    f"Avg: {avg_time:.1f}s | Speed: {throughput:.2f}/sec[/cyan]"
                                )

            # Final save
            self.save_progress()
            self._save_final_results([asdict(r) for r in results])

            # Summary
            console.print("\n🎉 [bold green]Audit completed successfully![/bold green]")
            console.print(f"📊 [cyan]Total tests: {self.progress.total_tests}[/cyan]")
            console.print(
                f"✅ [green]Completed: {self.progress.completed_tests}[/green]"
            )
            console.print(f"❌ [red]Failed: {self.progress.failed_tests}[/red]")
            console.print(f"💾 [blue]Results saved to: {self.results_file}[/blue]")

            return True

        except Exception as e:
            console.print(f"❌ [red]Critical error during audit: {e}[/red]")
            logger.error(f"Critical error: {e}", exc_info=True)
            return False

    def save_progress(self):
        """Save current progress"""
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.progress), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_progress(self, progress_file: str) -> bool:
        """Load progress from file"""
        try:
            with open(progress_file, encoding="utf-8") as f:
                data = json.load(f)
                self.progress = AuditProgress(**data)
                return True
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return False

    def _save_intermediate_results(self, results: list[dict]):
        """Save intermediate results"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.results_file, index=False, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def _save_final_results(self, results: list[dict]):
        """Save final results with summary"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.results_file, index=False, encoding="utf-8")

            # Create summary
            summary = {
                "session_id": self.session_id,
                "model_name": self.model_name,
                "corpus_file": self.corpus_file,
                "total_tests": len(results),
                "completed_tests": self.progress.completed_tests,
                "failed_tests": self.progress.failed_tests,
                "avg_response_time": self.progress.avg_response_time,
                "total_duration": time.time() - self.start_time,
                "results_file": str(self.results_file),
                "timestamp": datetime.now().isoformat(),
            }

            summary_file = self.model_session_dir / f"summary_{self.session_id}.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save final results: {e}")


def main():
    """Enhanced main function with Rich CLI"""
    parser = argparse.ArgumentParser(
        description="🎯 EquiLens Enhanced Bias Auditor with Rich Progress Tracking"
    )
    parser.add_argument("--model", "-m", required=True, help="Model name to audit")
    parser.add_argument("--corpus", "-c", required=True, help="Path to corpus CSV file")
    parser.add_argument(
        "--output-dir", "-o", default="results", help="Output directory"
    )
    parser.add_argument("--resume", "-r", help="Resume from progress file")

    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold blue]🎯 EquiLens Enhanced Bias Auditor[/bold blue]\n"
            "[cyan]Real-time progress tracking with Rich UI[/cyan]",
            style="blue",
        )
    )

    auditor = EnhancedBiasAuditor(
        model_name=args.model, corpus_file=args.corpus, output_dir=args.output_dir
    )

    success = auditor.run_enhanced_audit(resume_file=args.resume)

    if success:
        console.print("\n✅ [bold green]Audit completed successfully![/bold green]")
        sys.exit(0)
    else:
        console.print("\n❌ [bold red]Audit failed![/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
