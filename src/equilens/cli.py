"""
Modern CLI interface for EquiLens using Typer and Rich

A comprehensive command-line interface for the EquiLens AI bias detection platform.
Features interactive commands, beautiful output formatting, and comprehensive help.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import requests
import typer
from rich.console import Console
from rich.panel import Panel

try:
    from equilens.core.manager import EquiLensManager
except ImportError:
    # Fallback for relative imports when run as script
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from equilens.core.manager import EquiLensManager

console = Console()


def measure_single_request_time(model_name: str, prompt: str) -> dict:
    """Measure actual time for a single model request with specific prompt"""
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=60,  # Increased timeout for multiple tests
        )
        end_time = time.time()
        actual_time = end_time - start_time

        if response.status_code == 200:
            return {"success": True, "time": actual_time, "error": None}
        else:
            return {
                "success": False,
                "time": actual_time,
                "error": f"API error {response.status_code}",
            }

    except requests.exceptions.Timeout:
        end_time = time.time()
        actual_time = end_time - start_time
        return {"success": False, "time": actual_time, "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        # For connection errors, we don't have a meaningful time measurement
        return {
            "success": False,
            "time": 0.0,
            "error": "Connection error - is Ollama running?",
        }
    except Exception as e:
        end_time = time.time()
        actual_time = end_time - start_time if "start_time" in locals() else 0.0
        return {"success": False, "time": actual_time, "error": str(e)}


def measure_average_request_time(model_name: str, num_tests: int = 5) -> dict:
    """Measure average time across multiple test prompts, accounting for all time spent"""
    test_prompts = [
        "Hello",
        "What is AI?",
        "Explain machine learning briefly.",
        "Define bias in simple terms.",
        "How does neural network work?",
    ]

    # Use the specified number of prompts (3 or 5)
    prompts_to_test = test_prompts[:num_tests]

    console.print(
        f"[dim]📊 Testing {num_tests} prompts to measure average response time for {model_name}...[/dim]"
    )

    results = []
    successful_times = []
    total_time_spent = 0.0

    for i, prompt in enumerate(prompts_to_test, 1):
        prompt_display = prompt[:20] + ("..." if len(prompt) > 20 else "")
        console.print(f'[dim]  Test {i}/{num_tests}: "{prompt_display}"[/dim]')

        result = measure_single_request_time(model_name, prompt)
        results.append(result)
        total_time_spent += result["time"]

        if result["success"]:
            successful_times.append(result["time"])
            console.print(f"[dim]    ✓ {result['time']:.1f}s (success)[/dim]")
        else:
            console.print(
                f"[dim]    ✗ {result['time']:.1f}s (failed: {result['error']})[/dim]"
            )

    # Calculate statistics
    successful_count = len(successful_times)
    failed_count = num_tests - successful_count

    if successful_count > 0:
        average_successful_time = sum(successful_times) / len(successful_times)
        console.print(
            f"[dim]📈 Average successful response time: {average_successful_time:.1f}s (from {successful_count}/{num_tests} successful tests)[/dim]"
        )
        if failed_count > 0:
            console.print(
                f"[dim]⚠️ {failed_count} tests failed but their time ({total_time_spent - sum(successful_times):.1f}s) is included in ETA[/dim]"
            )

        return {
            "average_time": average_successful_time,
            "total_time_spent": total_time_spent,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "success": True,
        }
    else:
        console.print(
            f"[dim]⚠️ All {num_tests} tests failed. Total time spent: {total_time_spent:.1f}s[/dim]"
        )
        return {
            "average_time": None,
            "total_time_spent": total_time_spent,
            "successful_count": 0,
            "failed_count": num_tests,
            "success": False,
        }


def estimate_corpus_eta(
    corpus_path: str,
    model_name: str | None = None,
    timing_data: dict | None = None,
    user_eta_preference: float | None = None,
) -> dict:
    """Estimate ETA based on timing data from multiple test prompts or user preference"""
    try:
        # Count rows in corpus
        df = pd.read_csv(corpus_path)
        test_count = len(df)

        # Prioritize user preference over timing data
        if user_eta_preference is not None:
            # Use user-provided ETA estimate
            buffered_time_per_test = user_eta_preference
            total_time_seconds = test_count * buffered_time_per_test

            return {
                "test_count": test_count,
                "single_request_time": round(user_eta_preference, 2),
                "buffered_time_per_test": round(buffered_time_per_test, 2),
                "total_seconds": round(total_time_seconds),
                "formatted": format_duration(total_time_seconds),
                "timing_stats": {
                    "source": "user_preference",
                    "user_eta": user_eta_preference,
                },
            }
        # Use timing data if provided and no user preference
        elif timing_data and timing_data.get("success"):
            average_time = timing_data["average_time"]
            # Apply 1.4x buffer as requested
            buffered_time_per_test = average_time * 1.4
            total_time_seconds = test_count * buffered_time_per_test

            return {
                "test_count": test_count,
                "single_request_time": round(average_time, 2),
                "buffered_time_per_test": round(buffered_time_per_test, 2),
                "total_seconds": round(total_time_seconds),
                "formatted": format_duration(total_time_seconds),
                "timing_stats": {
                    "successful_tests": timing_data["successful_count"],
                    "failed_tests": timing_data["failed_count"],
                    "total_measurement_time": round(timing_data["total_time_spent"], 2),
                },
            }
        elif timing_data and not timing_data.get("success"):
            # All tests failed, but we still have timing data
            return {
                "test_count": test_count,
                "single_request_time": None,
                "buffered_time_per_test": None,
                "total_seconds": None,
                "formatted": "Unable to estimate (all tests failed)",
                "timing_stats": {
                    "successful_tests": 0,
                    "failed_tests": timing_data["failed_count"],
                    "total_measurement_time": round(timing_data["total_time_spent"], 2),
                },
                "error": "All timing tests failed",
            }
        else:
            # No timing data provided
            return {
                "test_count": test_count,
                "single_request_time": None,
                "buffered_time_per_test": None,
                "total_seconds": None,
                "formatted": "No timing data available",
                "error": "No timing measurement performed",
            }

    except Exception as e:
        # Fallback estimates if file analysis fails
        return {
            "test_count": "unknown",
            "single_request_time": None,
            "buffered_time_per_test": None,
            "total_seconds": None,
            "formatted": "Error loading corpus",
            "error": str(e),
        }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format with days and years support"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:  # Less than 1 hour
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        else:
            return f"{minutes}m"
    elif seconds < 86400:  # Less than 1 day
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if minutes > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{hours}h"
    elif seconds < 31536000:  # Less than 1 year (365 days)
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        if hours > 0:
            return f"{days}d {hours}h"
        else:
            return f"{days}d"
    else:  # 1 year or more
        years = int(seconds // 31536000)
        days = int((seconds % 31536000) // 86400)
        if days > 0:
            return f"{years}y {days}d"
        else:
            return f"{years}y"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_kb = size_bytes / 1024
        return f"{size_kb:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.1f} MB"
    elif size_bytes < 1024 * 1024 * 1024 * 1024:
        size_gb = size_bytes / (1024 * 1024 * 1024)
        return f"{size_gb:.1f} GB"
    else:
        size_tb = size_bytes / (1024 * 1024 * 1024 * 1024)
        return f"{size_tb:.1f} TB"


def get_available_models() -> list[str]:
    """Get available models from Ollama API"""
    try:
        # Try direct API call first (Docker-based Ollama)
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [model["name"] for model in models]
    except Exception:
        pass

    # Fallback to direct ollama command (if available)
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
    except Exception:
        pass

    return []


app = typer.Typer(
    name="equilens",
    help="🔍 EquiLens - AI Bias Detection Platform",
    rich_markup_mode="rich",
    no_args_is_help=False,
)

# Global manager instance
manager = None


def get_manager() -> EquiLensManager:
    """Get or create the EquiLens manager instance"""
    global manager
    if manager is None:
        manager = EquiLensManager()
    return manager


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None, typer.Option("--version", "-V", help="Show version and exit")
    ] = None,
):
    """
    🔍 **EquiLens** - AI Bias Detection Platform

    A comprehensive platform for detecting and analyzing bias in AI language models.
    This unified CLI provides all functionality needed to run bias audits, analyze results,
    and manage the underlying services.

    **Quick Start:**

    1. `uv run equilens gpu-check` - Check GPU acceleration
    2. `uv run equilens start` - Start Ollama services
    3. `uv run equilens audit config.json` - Run bias audit
    4. `uv run equilens analyze results.csv` - Analyze results

    **Interactive Mode:**

    • `uv run equilens tui` - Launch interactive terminal UI
    • `uv run equilens web` - Start web interface (future)
    """
    if version:
        from equilens import __version__

        console.print(f"EquiLens version {__version__}")
        raise typer.Exit()

    # When no subcommand is provided, show introduction and help
    if ctx.invoked_subcommand is None:
        # Show introduction
        console.print(Panel.fit(
            "[bold blue]🔍 EquiLens - AI Bias Detection Platform[/bold blue]\n\n"
            "[cyan]Welcome to EquiLens![/cyan] 🎯\n\n"
            "A comprehensive platform for detecting and analyzing bias in AI language models.\n"
            "This unified CLI provides all functionality needed to run bias audits, analyze\n"
            "results, and manage the underlying services.\n\n"
            "[yellow]✨ Key Features:[/yellow]\n"
            "• Comprehensive bias detection across multiple dimensions\n"
            "• Interactive model selection and configuration\n"
            "• Real-time progress tracking with resume capability\n"
            "• Advanced session management and backup system\n"
            "• Rich visual reporting and analysis tools\n"
            "• Docker-based service orchestration\n\n"
            "[green]Ready to detect bias in AI models? Explore the commands below![/green]",
            border_style="blue",
            title="🎯 AI Bias Detection Platform"
        ))

        # Show the help content below
        console.print("\n[bold]📖 Available Commands and Options:[/bold]\n")
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def status():
    """📊 Show comprehensive service status"""
    manager = get_manager()
    manager.display_status()


@app.command()
def start():
    """🚀 Start EquiLens services"""
    manager = get_manager()

    with console.status("Starting services..."):
        success = manager.start_services()

    if success:
        console.print("✅ [green]Services started successfully![/green]")
        console.print("\n💡 Next steps:")
        console.print(
            "  • [cyan]uv run equilens models list[/cyan] - List available models"
        )
        console.print(
            "  • [cyan]uv run equilens models pull llama2[/cyan] - Download a model"
        )
        console.print(
            "  • [cyan]uv run equilens audit config.json[/cyan] - Run bias audit"
        )
    else:
        console.print("[red]❌ Failed to start services[/red]")
        raise typer.Exit(1)


@app.command()
def stop():
    """🛑 Stop EquiLens services"""
    manager = get_manager()

    with console.status("Stopping services..."):
        success = manager.stop_services()

    if success:
        console.print("✅ [green]Services stopped successfully[/green]")
    else:
        console.print("[red]❌ Failed to stop services[/red]")
        raise typer.Exit(1)


@app.command("gpu-check")
def gpu_check():
    """🎮 Check GPU support and CUDA installation"""
    manager = get_manager()
    manager.gpu_manager.display_gpu_status()


# Models subcommand group
models_app = typer.Typer(help="🎯 Manage Ollama models")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list():
    """📋 List available models"""
    manager = get_manager()
    manager.list_models()


@models_app.command("pull")
def models_pull(
    model: Annotated[
        str, typer.Argument(help="Model name to download (e.g., llama2, phi3)")
    ],
):
    """📥 Download a model"""
    manager = get_manager()

    success = manager.pull_model(model)
    if not success:
        raise typer.Exit(1)


def find_interrupted_sessions(model: str | None = None) -> list[tuple[str, dict]]:
    """Find interrupted audit sessions that can be resumed"""
    interrupted_sessions = []
    results_dir = Path("results")

    if not results_dir.exists():
        return interrupted_sessions

    # Look for progress files in session directories
    for session_dir in results_dir.iterdir():
        if session_dir.is_dir():
            # Look for progress files
            for progress_file in session_dir.glob("progress_*.json"):
                try:
                    with open(progress_file, encoding="utf-8") as f:
                        progress_data = json.load(f)

                    # Check if it's an incomplete session
                    total_tests = progress_data.get("total_tests", 0)
                    completed_tests = progress_data.get("completed_tests", 0)
                    session_model = progress_data.get("model_name", "")

                    # Only include if incomplete and matches model (if specified)
                    if completed_tests < total_tests and (
                        not model or session_model == model
                    ):
                        interrupted_sessions.append((str(progress_file), progress_data))

                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    continue

    # Sort by start time (most recent first)
    interrupted_sessions.sort(key=lambda x: x[1].get("start_time", ""), reverse=True)
    return interrupted_sessions


def prompt_for_resume(model: str | None = None) -> str | None:
    """Check for interrupted sessions and prompt user to resume"""
    interrupted_sessions = find_interrupted_sessions(model)

    if not interrupted_sessions:
        return None

    console.print("\n[yellow]🔄 Found interrupted audit sessions:[/yellow]")

    for i, (_progress_file, progress_data) in enumerate(interrupted_sessions, 1):
        session_model = progress_data.get("model_name", "Unknown")
        completed = progress_data.get("completed_tests", 0)
        total = progress_data.get("total_tests", 0)
        completion_percent = (completed / total * 100) if total > 0 else 0
        start_time = progress_data.get("start_time", "Unknown")

        # Parse start time for better display
        try:
            from datetime import datetime

            start_dt = datetime.fromisoformat(start_time)
            time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = start_time

        # Extract folder ID from progress file path for easier identification
        folder_id = "Unknown"
        try:
            progress_path = Path(_progress_file)
            folder_id = progress_path.parent.name
        except Exception:
            pass

        console.print(
            f"  {i}. [cyan]{session_model}[/cyan] - {completed}/{total} tests ({completion_percent:.1f}% complete)"
        )
        console.print(f"     [dim]Started: {time_str} | Folder: {folder_id}[/dim]")

    console.print("\n[bold]Options:[/bold]")
    console.print(
        f"  [green]1-{len(interrupted_sessions)}[/green]: Resume interrupted session"
    )
    console.print("  [yellow]n[/yellow]: Start new audit session")
    console.print("  [red]q[/red]: Quit")

    while True:
        choice = typer.prompt("\nSelect option").lower().strip()

        if choice == "q":
            console.print("[yellow]Audit cancelled.[/yellow]")
            raise typer.Exit(0)
        elif choice == "n":
            return None
        elif choice.isdigit() and 1 <= int(choice) <= len(interrupted_sessions):
            return interrupted_sessions[int(choice) - 1][0]
        else:
            console.print(
                f"[red]Invalid choice. Please enter 1-{len(interrupted_sessions)}, 'n', or 'q'[/red]"
            )


@app.command()
def audit(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model name to audit")
    ] = None,
    corpus: Annotated[
        str | None, typer.Option("--corpus", "-c", help="Path to corpus CSV file")
    ] = None,
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory for results")
    ] = "results",
    enhanced: Annotated[
        bool,
        typer.Option(
            "--enhanced",
            "-e",
            help="[BETA] Use experimental enhanced auditor (may have reliability issues)",
        ),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", "-b", help="Number of concurrent requests for enhanced mode"
        ),
    ] = 5,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            "-r",
            help="Resume from a previous interrupted audit session (provide progress file path)",
        ),
    ] = None,
    silent: Annotated[
        bool,
        typer.Option(
            "--silent",
            "-s",
            help="Suppress subprocess output to avoid emoji encoding errors",
        ),
    ] = False,
    help_cmd: Annotated[
        bool,
        typer.Option(
            "--help",
            help="Show this help message and exit"
        )
    ] = False,
):
    """🔍 Run bias audit with interactive prompts and enhanced visual design"""

    # Show introduction and help only when --help is explicitly requested
    if help_cmd:
        console.print(Panel.fit(
            "[bold blue]🔍 EquiLens Bias Audit System[/bold blue]\n\n"
            "[cyan]Welcome to the EquiLens AI Bias Detection Platform![/cyan]\n\n"
            "This tool performs comprehensive bias audits on AI language models\n"
            "by testing them against carefully crafted test corpora and analyzing\n"
            "the responses for various forms of bias including gender, racial,\n"
            "cultural, and socioeconomic biases.\n\n"
            "[yellow]✨ Features:[/yellow]\n"
            "• Interactive model selection from available Ollama models\n"
            "• Automatic corpus detection and ETA estimation\n"
            "• Resume functionality for interrupted audits\n"
            "• Real-time progress tracking with dynamic concurrency\n"
            "• Comprehensive bias analysis and reporting\n"
            "• Automatic backup system (every 100 tests)\n\n"
            "[green]Ready to start your bias audit? Use the options below![/green]",
            border_style="blue",
            title="🎯 AI Bias Detection"
        ))

        # Show help content
        console.print("\n[bold]📖 Command Options:[/bold]\n")
        console.print("[cyan]--model, -m[/cyan]        Model name to audit")
        console.print("[cyan]--corpus, -c[/cyan]       Path to corpus CSV file")
        console.print("[cyan]--output-dir, -o[/cyan]   Output directory for results (default: results)")
        console.print("[cyan]--enhanced, -e[/cyan]     Use experimental enhanced auditor")
        console.print("[cyan]--batch-size, -b[/cyan]   Number of concurrent requests (default: 5)")
        console.print("[cyan]--resume, -r[/cyan]       Resume from previous session")
        console.print("[cyan]--silent, -s[/cyan]       Suppress subprocess output")
        console.print("[cyan]--help[/cyan]             Show this help message")

        console.print("\n[bold green]🚀 Quick Start Examples:[/bold green]")
        console.print("[dim]# Interactive mode (recommended for beginners)[/dim]")
        console.print("[yellow]uv run equilens audit[/yellow]")
        console.print("\n[dim]# Specify model and corpus directly[/dim]")
        console.print("[yellow]uv run equilens audit --model llama2:latest --corpus corpus.csv[/yellow]")
        console.print("\n[dim]# Resume a previous audit session[/dim]")
        console.print("[yellow]uv run equilens audit --resume path/to/progress.json[/yellow]")

        console.print("\n[bold]💡 Tip:[/bold] Run without options for interactive setup!")
        return

    # Auto-resume detection (if not explicitly resuming and no model specified)
    if resume is None and model is None:
        auto_resume_file = prompt_for_resume()
        if auto_resume_file:
            resume = auto_resume_file

    # If resuming, extract model and corpus from progress file
    if resume:
        try:
            with open(resume, encoding="utf-8") as f:
                progress_data = json.load(f)

            resume_model = progress_data.get("model_name")
            resume_corpus = progress_data.get("corpus_file")

            if resume_model and resume_corpus:
                model = resume_model
                corpus = resume_corpus
                console.print("🔄 [green]Resuming audit session...[/green]")
                console.print(f"📊 Model: [cyan]{model}[/cyan]")
                console.print(f"📂 Corpus: [cyan]{corpus}[/cyan]")
                console.print(
                    f"📋 Progress: {progress_data.get('completed_tests', 0)}/{progress_data.get('total_tests', 0)} tests completed"
                )

                # Initialize ETA variables for resume flow
                timing_data = None
                show_eta = False
                user_eta_preference = None

                # Always ask for concurrency configuration during resume
                console.print("\n[bold]🔧 Performance Configuration[/bold]")
                console.print(
                    "[dim]Configure concurrency for resumed audit session[/dim]"
                )
                console.print("  • [cyan]Higher values[/cyan]: Faster processing but more load")
                console.print("  • [cyan]Lower values[/cyan]: Safer for system stability")
                console.print("  • [cyan]1[/cyan]: Sequential processing (safest)")

                workers_input = typer.prompt(
                    "Enter number of concurrent workers (1-10)",
                    default="3",
                    show_default=True
                ).strip()

                try:
                    max_workers = int(workers_input)
                    max_workers = max(1, min(max_workers, 10))  # Clamp between 1-10
                    if max_workers > 1:
                        console.print(f"[green]✓ Configured for {max_workers} concurrent workers with dynamic scaling[/green]")
                        console.print("[dim]Workers will automatically scale down on errors and back up on success[/dim]")
                    else:
                        console.print("[yellow]⚡ Using sequential processing mode[/yellow]")
                except ValueError:
                    console.print("[yellow]⚠️ Invalid input, using default of 3 workers[/yellow]")
                    max_workers = 3
            else:
                console.print(f"[red]❌ Invalid progress file: {resume}[/red]")
                raise typer.Exit(1)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            console.print(f"[red]❌ Error reading progress file: {e}[/red]")
            raise typer.Exit(1) from e

    # Step 1: Model Selection
    if model is None:
        console.print(
            Panel.fit(
                "[bold blue]Step 1: Model Selection[/bold blue]\n"
                "Select or specify the language model to audit for bias.",
                border_style="blue",
            )
        )

        # Try to detect available models
        models = get_available_models()

        if models:
            console.print("\n[green]✓ Found available models:[/green]")
            for i, m in enumerate(models, 1):
                console.print(f"  {i}. [cyan]{m}[/cyan]")

            while True:
                console.print(
                    "\n[bold]Select model number or enter custom name:[/bold]"
                )
                choice = typer.prompt("")
                if choice.isdigit() and 1 <= int(choice) <= len(models):
                    model = models[int(choice) - 1]
                    break
                elif choice in models:
                    model = choice
                    break
                else:
                    console.print(
                        f"[red]❌ Invalid choice. Please select 1-{len(models)} or enter a valid model name.[/red]"
                    )
        else:
            console.print("\n[yellow]⚠ Could not detect models automatically[/yellow]")
            console.print("[bold]Enter model name:[/bold]")
            model = typer.prompt("")

    # Initialize variables that might be needed later
    timing_data = None
    show_eta = False
    user_eta_preference = None
    max_workers = 1  # Default concurrent workers

    # Step 2: Corpus Selection
    if corpus is None:
        console.print(
            Panel.fit(
                "[bold blue]Step 2: Corpus Selection[/bold blue]\n"
                "Choose a test corpus file containing bias evaluation prompts.",
                border_style="blue",
            )
        )

        # Configure ETA preference BEFORE any timing tests
        if model:
            console.print("\n[bold]� ETA Configuration[/bold]")
            console.print(
                "[dim]Would you like to see time estimates for available corpuses?[/dim]"
            )
            console.print(
                "  • [cyan]Number (e.g., 5.2)[/cyan]: Use custom seconds per test"
            )
            console.print("  • [cyan]y[/cyan]: Auto-detect with 5 test prompts")
            console.print("  • [cyan]n[/cyan]: Skip ETA estimates")

            eta_input = (
                typer.prompt("Enter your choice", default="y", show_default=True)
                .strip()
                .lower()
            )

            # Parse user input
            if eta_input == "n":
                show_eta = False
                user_eta_preference = None
                console.print("[yellow]⏭️ Skipping ETA estimates[/yellow]")
            elif eta_input == "y":
                show_eta = True
                user_eta_preference = None  # Will auto-detect with timing test
            else:
                # Try to parse as number
                try:
                    custom_eta = float(eta_input)
                    if custom_eta <= 0:
                        console.print(
                            "[red]❌ Invalid number, using auto-detection instead[/red]"
                        )
                        show_eta = True
                        user_eta_preference = None
                    else:
                        show_eta = True
                        user_eta_preference = custom_eta
                        console.print(
                            f"[green]✓ Using custom ETA: {custom_eta:.1f}s per test[/green]"
                        )
                except ValueError:
                    console.print(
                        "[red]❌ Invalid input, using auto-detection instead[/red]"
                    )
                    show_eta = True
                    user_eta_preference = None

        # Ask about concurrent processing for faster performance
        console.print("\n[bold]🚀 Performance Configuration[/bold]")
        console.print("[dim]Would you like to enable concurrent processing?[/dim]")
        console.print("  • [cyan]1[/cyan]: Single threaded (stable, recommended)")
        console.print("  • [cyan]2-5[/cyan]: Multiple threads (faster but may stress Ollama)")
        console.print("  • [cyan]n[/cyan]: Use default (single threaded)")

        worker_input = typer.prompt("Number of concurrent workers", default="1", show_default=True).strip().lower()

        if worker_input == "n" or worker_input == "":
            max_workers = 1
        else:
            try:
                workers = int(worker_input)
                if 1 <= workers <= 8:  # Reasonable limit
                    max_workers = workers
                    if workers > 1:
                        console.print(f"[yellow]⚡ Using {workers} concurrent threads[/yellow]")
                        console.print("[dim]Note: This may stress Ollama - monitor for connection errors[/dim]")
                    else:
                        console.print("[green]✓ Using single threaded processing[/green]")
                else:
                    console.print("[red]❌ Invalid number (1-8), using single threaded[/red]")
                    max_workers = 1
            except ValueError:
                console.print("[red]❌ Invalid input, using single threaded[/red]")
                max_workers = 1

            # Only run timing tests if user chose auto-detection (y) and no custom ETA provided
            if show_eta and user_eta_preference is None and model:
                console.print(
                    f"\n[yellow]🔬 Measuring average response time for {model}...[/yellow]"
                )
                timing_data = measure_average_request_time(model, num_tests=5)
                if timing_data["success"]:
                    console.print(
                        f"[green]✓ Average timing measurement complete: {timing_data['average_time']:.1f}s[/green]"
                    )
                    console.print(
                        f"[green]  ({timing_data['successful_count']}/{timing_data['successful_count'] + timing_data['failed_count']} tests successful)[/green]\n"
                    )
                else:
                    console.print(
                        f"[red]⚠️ Timing measurement failed. Total time spent: {timing_data['total_time_spent']:.1f}s[/red]"
                    )
                    console.print(
                        "[yellow]⚠️ Will try to estimate ETA based on corpus size only[/yellow]"
                    )

        # Look for common corpus files
        common_paths = [
            "src/Phase1_CorpusGenerator/corpus/quick_test_corpus.csv",
            "src/Phase1_CorpusGenerator/corpus/test_corpus.csv",
            "src/Phase1_CorpusGenerator/corpus/audit_corpus_gender_bias.csv",
            "quick_test_corpus.csv",
            "test_corpus.csv",
        ]

        found_files = []
        for file_path in common_paths:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                found_files.append((file_path, file_size))

        if found_files:
            console.print("[green]✓ Found corpus files:[/green]")

            if show_eta:
                console.print(
                    "\n[bold cyan]📊 ETA Estimates for All Available Corpuses:[/bold cyan]"
                )

            for i, (path, size) in enumerate(found_files, 1):
                file_size_formatted = format_file_size(size)

                # Get ETA estimates for this corpus using user preference or timing data
                eta_display = ""
                if show_eta:
                    eta_info = estimate_corpus_eta(
                        path, model, timing_data, user_eta_preference
                    )
                    if (
                        "error" not in eta_info
                        and eta_info.get("single_request_time") is not None
                    ):
                        test_count = eta_info["test_count"]
                        eta_time = eta_info["formatted"]
                        per_test_time = eta_info["buffered_time_per_test"]

                        # Check if this is user preference or timing data
                        timing_stats = eta_info.get("timing_stats", {})
                        if timing_stats.get("source") == "user_preference":
                            eta_display = f" | [dim]Tests: {test_count} | ETA: {eta_time} ({per_test_time}s/test) [cyan]((user estimate(UE)))[/cyan][/dim]"
                        else:
                            eta_display = f" | [dim]Tests: {test_count} | ETA: {eta_time} ({per_test_time}s/test)[/dim]"
                            if timing_data and timing_data.get("failed_count", 0) > 0:
                                eta_display += f" | [yellow]⚠️ {timing_data['failed_count']} timing tests failed[/yellow]"
                    else:
                        eta_display = " | [red]ETA: Calculation failed[/red]"
                        if "timing_stats" in eta_info:
                            stats = eta_info["timing_stats"]
                            eta_display += f" | [dim]Failed tests: {stats.get('failed_tests', 0)}[/dim]"

                console.print(
                    f"  {i}. [cyan]{path}[/cyan] ([dim]{file_size_formatted}[/dim]){eta_display}"
                )

            # Show timing statistics if we have timing data
            if show_eta and timing_data:
                timing_stats = timing_data
                console.print(
                    f"\n[dim]📈 Timing Statistics: {timing_stats['successful_count']} successful, {timing_stats['failed_count']} failed tests (Total: {timing_stats['total_time_spent']:.1f}s)[/dim]"
                )
            elif show_eta and user_eta_preference is not None:
                console.print(
                    f"\n[dim]📈 Using custom ETA estimate: {user_eta_preference:.1f}s per test[/dim]"
                )

                console.print(
                    "\n[bold]Do you want to proceed with corpus selection?[/bold]"
                )
                if not typer.confirm("Continue?", default=True):
                    console.print(
                        "[yellow]Corpus selection cancelled by user.[/yellow]"
                    )
                    raise typer.Exit(0)

            while True:
                console.print(
                    "\n[bold]Select corpus file number or enter custom path:[/bold]"
                )
                choice = typer.prompt("")
                if choice.isdigit() and 1 <= int(choice) <= len(found_files):
                    corpus = found_files[int(choice) - 1][0]
                    break
                elif Path(choice).exists():
                    corpus = choice
                    break
                else:
                    console.print(f"[red]❌ File not found: {choice}[/red]")
        else:
            console.print(
                "\n[yellow]⚠ No corpus files found in common locations[/yellow]"
            )
            console.print("[bold]Enter corpus file path:[/bold]")
            corpus_path = typer.prompt("")
            while not Path(corpus_path).exists():
                console.print(f"[red]❌ File not found: {corpus_path}[/red]")
                console.print("[bold]Enter corpus file path:[/bold]")
                corpus_path = typer.prompt("")
            corpus = corpus_path

            # Show ETA for custom path if ETA is enabled
            if show_eta:
                console.print(
                    "\n[bold cyan]📊 ETA Estimate for Custom Corpus:[/bold cyan]"
                )
                eta_info = estimate_corpus_eta(
                    corpus_path, model, timing_data, user_eta_preference
                )
                if (
                    "error" not in eta_info
                    and eta_info.get("single_request_time") is not None
                ):
                    test_count = eta_info["test_count"]
                    eta_time = eta_info["formatted"]
                    per_test_time = eta_info["buffered_time_per_test"]

                    # Check if this is user preference or timing data
                    timing_stats = eta_info.get("timing_stats", {})
                    if timing_stats.get("source") == "user_preference":
                        console.print(
                            f"  📊 Tests: [cyan]{test_count}[/cyan] | ETA: [yellow]{eta_time}[/yellow] ({per_test_time}s/test) [cyan](user estimate)[/cyan]"
                        )
                    else:
                        console.print(
                            f"  📊 Tests: [cyan]{test_count}[/cyan] | ETA: [yellow]{eta_time}[/yellow] ({per_test_time}s/test)"
                        )
                        if timing_data and timing_data.get("failed_count", 0) > 0:
                            console.print(
                                f"  ⚠️ [yellow]{timing_data['failed_count']} timing tests failed during measurement[/yellow]"
                            )
                else:
                    console.print("  [red]❌ ETA calculation failed[/red]")
                    if "timing_stats" in eta_info:
                        stats = eta_info["timing_stats"]
                        console.print(f"  Failed tests: {stats.get('failed_tests', 0)}")

                console.print("\n[bold]Do you want to proceed with this corpus?[/bold]")
                if not typer.confirm("Continue?", default=True):
                    console.print("[yellow]Audit cancelled by user.[/yellow]")
                    raise typer.Exit(0)

    # Validate inputs
    assert model is not None, "Model should not be None at this point"
    assert corpus is not None, "Corpus should not be None at this point"

    if not Path(corpus).exists():
        console.print(f"[red]❌ Corpus file not found: {corpus}[/red]")
        raise typer.Exit(1)

    # Step 3: Configuration Review
    corpus_size = format_file_size(Path(corpus).stat().st_size)

    # Determine if this is a custom path (not in the found_files list)
    is_custom_path = True
    if "found_files" in locals() and found_files:
        for path, _ in found_files:
            if Path(corpus).resolve() == Path(path).resolve():
                is_custom_path = False
                break

    corpus_display = f"[cyan]{corpus}[/cyan]"
    if is_custom_path:
        corpus_display += " [yellow](Custom Path)[/yellow]"

    # Get detailed ETA estimates for final review using pre-measured timing data
    eta_info = estimate_corpus_eta(corpus, model, timing_data, user_eta_preference)
    eta_text = ""
    if "error" not in eta_info and eta_info.get("single_request_time") is not None:
        test_count = eta_info["test_count"]
        eta_time = eta_info["formatted"]
        single_time = eta_info["single_request_time"]
        buffered_time = eta_info["buffered_time_per_test"]
        timing_stats = eta_info.get("timing_stats", {})
        successful_tests = timing_stats.get("successful_tests", 0)
        failed_tests = timing_stats.get("failed_tests", 0)
        total_measurement_time = timing_stats.get("total_measurement_time", 0)

        eta_text = f"""[bold]Test Count:[/bold] [cyan]{test_count}[/cyan]
[bold]Average Request Time:[/bold] [dim]{single_time}s (from {successful_tests} successful tests)[/dim]
[bold]Failed Tests:[/bold] [dim]{failed_tests} (total measurement time: {total_measurement_time}s)[/dim]
[bold]Buffered Time per Test:[/bold] [dim]{buffered_time}s (1.4x safety margin)[/dim]
[bold]Estimated Total Time:[/bold] [yellow]{eta_time}[/yellow]"""
    else:
        eta_text = f"[bold]Estimated Time:[/bold] [dim]{eta_info.get('formatted', 'Unable to calculate')}"
        if "timing_stats" in eta_info:
            stats = eta_info["timing_stats"]
            eta_text += f" | Failed tests: {stats.get('failed_tests', 0)} (time: {stats.get('total_measurement_time', 0)}s)[/dim]"
        else:
            eta_text += "[/dim]"

    console.print(
        Panel.fit(
            f"[bold green]Step 3: Configuration Review[/bold green]\n\n"
            f"[bold]Model:[/bold] [cyan]{model}[/cyan]\n"
            f"[bold]Corpus:[/bold] {corpus_display} ([dim]{corpus_size}[/dim])\n"
            f"[bold]Output Directory:[/bold] [cyan]{output_dir}[/cyan]\n"
            f"[bold]Silent Mode:[/bold] [cyan]{'Enabled' if silent else 'Disabled'}[/cyan]\n\n"
            f"{eta_text}",
            border_style="green",
        )
    )

    console.print("\n[bold]Proceed with bias audit?[/bold]")
    if not typer.confirm(""):
        console.print("[yellow]Audit cancelled by user.[/yellow]")
        raise typer.Exit(0)

    # Step 4: Execute Audit
    console.print(
        Panel.fit(
            "[bold yellow]Step 4: Executing Bias Audit[/bold yellow]\n"
            "Running model evaluation against the test corpus...\n"
            "[dim]This may take several minutes depending on model size and corpus length.[/dim]",
            border_style="yellow",
        )
    )

    try:
        if enhanced:
            # Warning for beta enhanced auditor
            console.print(
                "⚠️ [bold yellow]BETA WARNING: Enhanced auditor is experimental and may have reliability issues[/bold yellow]"
            )
            console.print(
                "💡 [dim]For stable results, use the standard auditor (without --enhanced flag)[/dim]\n"
            )

            # Use enhanced auditor with Rich progress bars
            console.print(
                "🚀 [bold cyan]Starting enhanced audit with real-time progress...[/bold cyan]"
            )

            # Import the enhanced auditor with better error handling
            import sys

            enhanced_path = Path("src/Phase2_ModelAuditor").resolve()
            if str(enhanced_path) not in sys.path:
                sys.path.insert(0, str(enhanced_path))

            try:
                from Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor

                # Create and run enhanced auditor
                auditor = EnhancedBiasAuditor(
                    model_name=model,
                    corpus_file=corpus,
                    output_dir=output_dir,
                    eta_per_test=user_eta_preference,
                )

                # Set batch size if specified
                auditor.batch_size = batch_size

                success = auditor.run_enhanced_audit(resume_file=resume)

                if success:
                    console.print(
                        "\n✅ [bold green]Enhanced audit completed successfully![/bold green]"
                    )
                else:
                    console.print("\n❌ [bold red]Enhanced audit failed![/bold red]")
                    raise typer.Exit(1)

            except ImportError as e:
                console.print(f"[red]❌ Failed to import enhanced auditor: {e}[/red]")
                console.print("[yellow]Falling back to enhanced auditor via subprocess...[/yellow]")
                enhanced = False

        if not enhanced:
            # Use enhanced auditor via subprocess wrapper
            console.print("🚀 [green]Using enhanced auditor with dynamic concurrency...[/green]")

            # Configure subprocess parameters based on silent mode
            cmd = [
                "python",
                "src/Phase2_ModelAuditor/audit_model_fixed.py",
                "--model",
                model,
                "--corpus",
                corpus,
                "--output-dir",
                output_dir,
                "--max-workers",
                str(max_workers),
            ]

            # Add ETA parameter if user specified one
            if user_eta_preference is not None:
                cmd.extend(["--eta-per-test", str(user_eta_preference)])

            # Add resume parameter if specified
            if resume:
                cmd.extend(["--resume", resume])

            if silent:
                # Redirect both stdout and stderr to suppress emoji encoding errors
                subprocess.run(
                    cmd,
                    check=True,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                console.print(
                    "[green]✓[/green] Audit process completed successfully (output suppressed)"
                )
            else:
                # Normal execution with full output but with proper encoding handling
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                except UnicodeDecodeError:
                    # Fallback for Unicode issues - run in silent mode automatically
                    console.print(
                        "[yellow]⚠ Unicode display issues detected, switching to silent mode...[/yellow]"
                    )
                    fallback_cmd = [
                        "python",
                        "src/Phase2_ModelAuditor/audit_model.py",
                        "--model",
                        model,
                        "--corpus",
                        corpus,
                        "--output-dir",
                        output_dir,
                    ]

                    # Add ETA parameter if user specified one
                    if user_eta_preference is not None:
                        fallback_cmd.extend(
                            ["--eta-per-test", str(user_eta_preference)]
                        )

                    subprocess.run(
                        fallback_cmd,
                        check=True,
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        text=True,
                    )
                    console.print(
                        "[green]✓[/green] Audit completed (Unicode issues bypassed)"
                    )

        # Success Panel with enhanced information
        safe_model = model.replace(":", "_")

        # Try to find the actual results file in both old and new directory structures
        safe_model = model.replace(":", "_")

        # Check for files in session directories (new structure)
        session_dirs = list(Path(output_dir).glob(f"{safe_model}_*"))
        session_results = []
        for session_dir in session_dirs:
            if session_dir.is_dir():
                session_files = list(session_dir.glob(f"results_{safe_model}_*.csv"))
                session_results.extend(session_files)

        # Check for files directly in output_dir (old structure)
        direct_results = list(Path(output_dir).glob(f"results_{safe_model}_*.csv"))

        # Combine all results and find the latest
        all_results = session_results + direct_results
        latest_results = (
            max(all_results, key=lambda x: x.stat().st_mtime) if all_results else None
        )

        success_message = (
            "[bold green]✅ Audit Completed Successfully![/bold green]\n\n"
        )
        success_message += f"[bold]Model Evaluated:[/bold] [cyan]{model}[/cyan]\n"
        success_message += (
            f"[bold]Corpus Processed:[/bold] [cyan]{Path(corpus).name}[/cyan]\n"
        )

        if latest_results:
            # Show the session directory if using new structure
            if latest_results.parent.name != output_dir:
                success_message += f"[bold]Session Directory:[/bold] [cyan]{latest_results.parent}/[/cyan]\n"
            else:
                success_message += (
                    f"[bold]Results Location:[/bold] [cyan]{output_dir}/[/cyan]\n"
                )
        else:
            success_message += (
                f"[bold]Results Location:[/bold] [cyan]{output_dir}/[/cyan]\n"
            )

        if latest_results:
            file_size = latest_results.stat().st_size / 1024
            success_message += f"[bold]Results File:[/bold] [cyan]{latest_results.name}[/cyan] ([dim]{file_size:.1f} KB[/dim])\n"

        success_message += "\n[bold]Next Steps:[/bold]\n"
        success_message += f"  • Review output files in [cyan]{output_dir}/[/cyan]\n"
        if latest_results:
            success_message += f"  • Run analysis: [cyan]uv run equilens analyze --results {latest_results}[/cyan]\n"
        else:
            success_message += "  • Run analysis: [cyan]uv run equilens analyze[/cyan] (auto-detect results)\n"
        success_message += (
            "  • Use [cyan]--silent[/cyan] flag if you see Unicode encoding errors"
        )

        console.print(Panel.fit(success_message, border_style="green"))

    except subprocess.CalledProcessError as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Audit Failed[/bold red]\n\n"
                f"[bold]Exit Code:[/bold] [red]{e.returncode}[/red]\n"
                f"[bold]Troubleshooting:[/bold]\n"
                f"  • Verify model [cyan]{model}[/cyan] is available\n"
                f"  • Check corpus file [cyan]{corpus}[/cyan] is valid\n"
                f"  • Ensure you're in the EquiLens project directory\n"
                f"  • Try running with [cyan]--silent[/cyan] flag to suppress output errors",
                border_style="red",
            )
        )
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        console.print(
            Panel.fit(
                "[bold red]❌ Script Not Found[/bold red]\n\n"
                "[bold]Issue:[/bold] Could not find audit script\n"
                "[bold]Solution:[/bold] Make sure you're in the EquiLens project directory\n"
                "[bold]Expected Location:[/bold] [cyan]src/Phase2_ModelAuditor/audit_model.py[/cyan]",
                border_style="red",
            )
        )
        raise typer.Exit(1) from e


@app.command()
def generate(
    config: Annotated[
        str, typer.Argument(help="Configuration file path for corpus generation")
    ],
):
    """📝 Generate test corpus using configuration file"""
    manager = get_manager()

    success = manager.generate_corpus(config)
    if success:
        console.print("✅ [green]Corpus generated successfully![/green]")
    else:
        console.print("[red]❌ Corpus generation failed[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    results: Annotated[
        str | None, typer.Option("--results", "-r", help="Results file path")
    ] = None,
    silent: Annotated[
        bool,
        typer.Option(
            "--silent",
            "-s",
            help="Suppress subprocess output to avoid emoji encoding errors",
        ),
    ] = False,
):
    """📊 Analyze bias audit results with enhanced visual interface"""

    # Step 1: Results File Selection
    if results is None:
        console.print(
            Panel.fit(
                "[bold blue]Step 1: Results File Selection[/bold blue]\n"
                "Choose a bias audit results file to analyze and visualize.",
                border_style="blue",
            )
        )

        # Look for results files
        results_dir = Path("results")
        if results_dir.exists():
            result_files = list(results_dir.glob("results_*.csv"))
            if result_files:
                # Sort by modification time (newest first)
                result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                console.print("\n[green]✓ Found results files:[/green]")
                for i, path in enumerate(result_files, 1):
                    file_size = path.stat().st_size / 1024
                    mod_time = path.stat().st_mtime
                    from datetime import datetime

                    mod_str = datetime.fromtimestamp(mod_time).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    console.print(
                        f"  {i}. [cyan]{path}[/cyan] ([dim]{file_size:.1f} KB, {mod_str}[/dim])"
                    )

                while True:
                    console.print(
                        "\n[bold]Select results file number or enter custom path:[/bold]"
                    )
                    choice = typer.prompt("")
                    if choice.isdigit() and 1 <= int(choice) <= len(result_files):
                        results = str(result_files[int(choice) - 1])
                        break
                    elif Path(choice).exists():
                        results = choice
                        break
                    else:
                        console.print(f"[red]❌ File not found: {choice}[/red]")
            else:
                console.print(
                    "\n[yellow]⚠ No results files found in results directory[/yellow]"
                )
                console.print("[bold]Enter results file path:[/bold]")
                results_path = typer.prompt("")
                while not Path(results_path).exists():
                    console.print(f"[red]❌ File not found: {results_path}[/red]")
                    console.print("[bold]Enter results file path:[/bold]")
                    results_path = typer.prompt("")
                results = results_path
        else:
            console.print("\n[yellow]⚠ Results directory not found[/yellow]")
            console.print("[bold]Enter results file path:[/bold]")
            results_path = typer.prompt("")
            while not Path(results_path).exists():
                console.print(f"[red]❌ File not found: {results_path}[/red]")
                console.print("[bold]Enter results file path:[/bold]")
                results_path = typer.prompt("")
            results = results_path

    # Validate inputs
    assert results is not None, "Results should not be None at this point"

    if not Path(results).exists():
        console.print(f"[red]❌ Results file not found: {results}[/red]")
        raise typer.Exit(1)

    # Step 2: Configuration Review
    results_size = Path(results).stat().st_size / 1024
    console.print(
        Panel.fit(
            f"[bold green]Step 2: Analysis Configuration[/bold green]\n\n"
            f"[bold]Results File:[/bold] [cyan]{results}[/cyan] ([dim]{results_size:.1f} KB[/dim])\n"
            f"[bold]Silent Mode:[/bold] [cyan]{'Enabled' if silent else 'Disabled'}[/cyan]\n"
            f"[bold]Output:[/bold] [cyan]bias_report.png + console summary[/cyan]",
            border_style="green",
        )
    )

    # Step 3: Execute Analysis
    console.print(
        Panel.fit(
            "[bold yellow]Step 3: Executing Bias Analysis[/bold yellow]\n"
            "Generating statistical summary and visualization chart...\n"
            "[dim]This will create bias_report.png and display metrics in the console.[/dim]",
            border_style="yellow",
        )
    )

    try:
        # Configure subprocess parameters based on silent mode
        if silent:
            # Redirect both stdout and stderr to suppress emoji encoding errors
            subprocess.run(
                [
                    "python",
                    "src/Phase3_Analysis/analyze_results.py",
                    "--results_file",
                    results,
                ],
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # Show clean progress feedback
            console.print(
                "[green]✓[/green] Analysis process completed successfully (output suppressed)"
            )
        else:
            # Normal execution with full output but with proper encoding handling
            try:
                subprocess.run(
                    [
                        "python",
                        "src/Phase3_Analysis/analyze_results.py",
                        "--results_file",
                        results,
                    ],
                    check=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except UnicodeDecodeError:
                # Fallback for Unicode issues - run in silent mode automatically
                console.print(
                    "[yellow]⚠ Unicode display issues detected, switching to silent mode...[/yellow]"
                )
                subprocess.run(
                    [
                        "python",
                        "src/Phase3_Analysis/analyze_results.py",
                        "--results_file",
                        results,
                    ],
                    check=True,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    text=True,
                )
                console.print(
                    "[green]✓[/green] Analysis completed (Unicode issues bypassed)"
                )

        # Check if bias_report.png was generated (check multiple possible locations)
        report_paths = [
            Path("bias_report.png"),  # Current directory
            Path("results/bias_report.png"),  # Results directory
        ]

        # Check if results file is in a session directory (new structure)
        if results:
            results_path = Path(results)
            results_dir = results_path.parent

            # If the results file is in a session directory, check there first
            if results_dir.name != "results":
                report_paths.insert(0, results_dir / "bias_report.png")

            # Also check legacy model-specific directory structure
            model_name = results_path.stem.replace("results_", "")
            # Remove session timestamp if present
            model_name = (
                "_".join(model_name.split("_")[:-1])
                if "_" in model_name
                else model_name
            )
            model_dir_path = Path(f"results/{model_name}")
            if model_dir_path.exists():
                report_paths.append(model_dir_path / "bias_report.png")

        report_path = None
        for path in report_paths:
            if path.exists():
                report_path = path
                break

        report_exists = report_path is not None

        # Enhanced success panel with file information
        success_message = (
            "[bold green]✅ Analysis Completed Successfully![/bold green]\n\n"
        )
        success_message += (
            f"[bold]Analyzed File:[/bold] [cyan]{Path(results).name}[/cyan]\n"
        )

        if report_exists:
            report_size = report_path.stat().st_size / 1024
            success_message += f"[bold]Generated Report:[/bold] [cyan]{report_path}[/cyan] ([dim]{report_size:.1f} KB[/dim])\n"
        else:
            success_message += "[bold]Generated Report:[/bold] [yellow]bias_report.png (check output above)[/yellow]\n"

        success_message += "\n[bold]Generated Files:[/bold]\n"
        if report_exists:
            success_message += f"  • [green]✓[/green] [cyan]{report_path}[/cyan] - Visualization chart\n"
        else:
            success_message += "  • [yellow]?[/yellow] [cyan]bias_report.png[/cyan] - Check console output\n"
        success_message += (
            "  • [green]✓[/green] [cyan]Console statistics[/cyan] - Detailed analysis\n"
        )
        success_message += "\n[bold]Next Steps:[/bold]\n"
        if report_exists:
            success_message += (
                f"  • Open [cyan]{report_path}[/cyan] to view bias metrics\n"
            )
        success_message += "  • Review console statistics for detailed analysis\n"
        success_message += (
            "  • Use [cyan]--silent[/cyan] flag if you see Unicode encoding errors"
        )

        console.print(Panel.fit(success_message, border_style="green"))

    except subprocess.CalledProcessError as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Analysis Failed[/bold red]\n\n"
                f"[bold]Exit Code:[/bold] [red]{e.returncode}[/red]\n"
                f"[bold]Troubleshooting:[/bold]\n"
                f"  • Verify results file [cyan]{results}[/cyan] is valid CSV\n"
                f"  • Check file contains expected bias audit data\n"
                f"  • Ensure you're in the EquiLens project directory\n"
                f"  • Try running with [cyan]--silent[/cyan] flag to suppress output errors",
                border_style="red",
            )
        )
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        console.print(
            Panel.fit(
                "[bold red]❌ Script Not Found[/bold red]\n\n"
                "[bold]Issue:[/bold] Could not find analysis script\n"
                "[bold]Solution:[/bold] Make sure you're in the EquiLens project directory\n"
                "[bold]Expected Location:[/bold] [cyan]src/Phase3_Analysis/analyze_results.py[/cyan]",
                border_style="red",
            )
        )
        raise typer.Exit(1) from e


@app.command()
def web():
    """🌐 Start web interface (coming soon)"""
    console.print(
        Panel.fit(
            "🚧 [yellow]Web interface coming soon![/yellow]\n\n"
            "For now, use CLI commands or the interactive TUI:\n"
            "• [cyan]uv run equilens tui[/cyan] - Terminal UI\n"
            "• [cyan]uv run equilens --help[/cyan] - All commands",
            title="Web Interface",
            border_style="yellow",
        )
    )


@app.command()
def tui():
    """🖥️ Launch interactive terminal UI"""
    try:
        from equilens.tui import EquiLensTUI

        app_instance = EquiLensTUI()
        app_instance.run()
    except ImportError:
        console.print("[red]❌ TUI dependencies not available[/red]")
        console.print("Install with: [cyan]uv add textual[/cyan]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Failed to start TUI: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("resume-list")
def resume_list(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Filter by model name")
    ] = None,
):
    """List all available resume sessions with detailed information"""
    try:
        interrupted_sessions = find_interrupted_sessions(model)

        if not interrupted_sessions:
            console.print("[yellow]📭 No interrupted audit sessions found.[/yellow]")
            return

        console.print(f"\n[bold cyan]📋 Available Resume Sessions ({len(interrupted_sessions)} total):[/bold cyan]\n")

        for i, (progress_file, progress_data) in enumerate(interrupted_sessions, 1):
            session_model = progress_data.get("model_name", "Unknown")
            completed = progress_data.get("completed_tests", 0)
            total = progress_data.get("total_tests", 0)
            failed = progress_data.get("failed_tests", 0)
            completion_percent = (completed / total * 100) if total > 0 else 0
            start_time = progress_data.get("start_time", "Unknown")
            last_checkpoint = progress_data.get("last_checkpoint", "Unknown")
            session_id = progress_data.get("session_id", "Unknown")
            avg_response_time = progress_data.get("avg_response_time", 0.0)
            throughput = progress_data.get("throughput_per_second", 0.0)

            # Parse start time for better display
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time)
                time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = start_time

            # Parse last checkpoint time
            try:
                checkpoint_dt = datetime.fromisoformat(last_checkpoint)
                checkpoint_str = checkpoint_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                checkpoint_str = last_checkpoint

            # Extract folder information
            folder_id = "Unknown"
            folder_path = "Unknown"
            folder_size = "Unknown"
            try:
                progress_path = Path(progress_file)
                folder_id = progress_path.parent.name
                folder_path = str(progress_path.parent)

                # Calculate folder size
                total_size = 0
                for file in progress_path.parent.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

                if total_size > 1024 * 1024:  # MB
                    folder_size = f"{total_size / (1024 * 1024):.1f} MB"
                elif total_size > 1024:  # KB
                    folder_size = f"{total_size / 1024:.1f} KB"
                else:
                    folder_size = f"{total_size} bytes"
            except Exception:
                pass

            # Calculate estimated time remaining
            eta_str = "Unknown"
            if completed > 0 and avg_response_time > 0:
                remaining_tests = total - completed
                eta_seconds = remaining_tests * avg_response_time
                if eta_seconds > 3600:  # Hours
                    eta_str = f"{eta_seconds / 3600:.1f} hours"
                elif eta_seconds > 60:  # Minutes
                    eta_str = f"{eta_seconds / 60:.1f} minutes"
                else:
                    eta_str = f"{eta_seconds:.0f} seconds"

            # Display session information
            console.print(f"[bold green]{i:2d}.[/bold green] [cyan]{session_model}[/cyan] - {completed:,}/{total:,} tests ({completion_percent:.1f}% complete)")
            console.print(f"     [dim]Session ID: {session_id}[/dim]")
            console.print(f"     [dim]Started: {time_str}[/dim]")
            console.print(f"     [dim]Last Save: {checkpoint_str}[/dim]")
            console.print(f"     [dim]Folder: {folder_id} ({folder_size})[/dim]")

            # Progress details
            if failed > 0:
                success_rate = ((completed - failed) / completed * 100) if completed > 0 else 0
                console.print(f"     [dim]Progress: {completed:,} completed, {failed:,} failed ({success_rate:.1f}% success)[/dim]")
            else:
                console.print(f"     [dim]Progress: {completed:,} completed, 0 failed (100% success)[/dim]")

            # Performance metrics
            if avg_response_time > 0:
                console.print(f"     [dim]Performance: {avg_response_time:.1f}s avg, {throughput:.2f} tests/sec[/dim]")
                console.print(f"     [dim]Est. Time Remaining: {eta_str}[/dim]")

            # Show backup information if available
            try:
                backup_dir = Path(folder_path) / f"{session_id}_backups"
                if backup_dir.exists():
                    backup_files = list(backup_dir.glob("progress_backup_*.json"))
                    if backup_files:
                        # Get latest backup info
                        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                        backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
                        console.print(f"     [dim]Backups: {len(backup_files)} available (latest: {backup_time.strftime('%H:%M:%S')})[/dim]")
            except Exception:
                pass

            console.print(f"     [dim]Path: {folder_path}[/dim]")
            console.print()

    except Exception as e:
        console.print(f"[red]❌ Error listing resume sessions: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("resume-remove")
def resume_remove(
    identifiers: Annotated[
        list[str], typer.Argument(help="Session indices (1,2,3...), session IDs, or folder names to remove (space-separated)")
    ],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompts")
    ] = False,
):
    """Remove specific resume sessions by index number, session ID, or folder name

    Examples:
        uv run equilens resume-remove 1 3 5          # Remove sessions 1, 3, and 5 from the list
        uv run equilens resume-remove folder_name    # Remove by folder name
        uv run equilens resume-remove session_id     # Remove by session ID (if unique)
    """
    try:
        interrupted_sessions = find_interrupted_sessions()

        if not interrupted_sessions:
            console.print("[yellow]📭 No interrupted audit sessions found.[/yellow]")
            return

        # Build mappings for different identifier types
        index_map = {}           # index -> progress_file
        session_map = {}         # session_id -> [progress_files] (can be multiple)
        folder_map = {}          # folder_name -> progress_file
        session_details = {}     # progress_file -> progress_data

        for i, (progress_file, progress_data) in enumerate(interrupted_sessions, 1):
            session_id = progress_data.get("session_id", "")
            index_map[str(i)] = progress_file
            session_details[progress_file] = progress_data

            # Handle multiple sessions with same ID
            if session_id in session_map:
                session_map[session_id].append(progress_file)
            else:
                session_map[session_id] = [progress_file]

            try:
                folder_name = Path(progress_file).parent.name
                folder_map[folder_name] = progress_file
            except Exception:
                continue

        sessions_to_remove = []
        removed_count = 0
        total_size_freed = 0

        for identifier in identifiers:
            found_sessions = []

            # Try to match by index first (most reliable)
            if identifier in index_map:
                found_sessions = [index_map[identifier]]
                console.print(f"[cyan]📍 Found session by index #{identifier}[/cyan]")

            # Try to match by folder name (second most reliable)
            elif identifier in folder_map:
                found_sessions = [folder_map[identifier]]
                console.print(f"[cyan]📂 Found session by folder name: {identifier}[/cyan]")

            # Try to match by session ID (least reliable due to duplicates)
            elif identifier in session_map:
                found_sessions = session_map[identifier]
                if len(found_sessions) > 1:
                    console.print(f"[yellow]⚠️ Session ID '{identifier}' matches {len(found_sessions)} sessions:[/yellow]")
                    for j, progress_file in enumerate(found_sessions, 1):
                        folder_name = Path(progress_file).parent.name
                        progress_data = session_details[progress_file]
                        completed = progress_data.get("completed_tests", 0)
                        console.print(f"  {j}. {folder_name} ({completed:,} tests)")

                    console.print(f"[yellow]Use folder names instead: {' '.join([Path(pf).parent.name for pf in found_sessions])}[/yellow]")
                    continue
                else:
                    console.print(f"[cyan]🆔 Found session by session ID: {identifier}[/cyan]")

            else:
                console.print(f"[yellow]⚠️ Identifier '{identifier}' not found. Use 'resume-list' to see available sessions.[/yellow]")
                continue

            # Process found sessions
            for progress_file in found_sessions:
                progress_data = session_details[progress_file]

                try:
                    folder_path = Path(progress_file).parent

                    # Get detailed session information
                    session_model = progress_data.get("model_name", "Unknown")
                    completed = progress_data.get("completed_tests", 0)
                    total = progress_data.get("total_tests", 0)
                    failed = progress_data.get("failed_tests", 0)
                    completion_percent = (completed / total * 100) if total > 0 else 0
                    start_time = progress_data.get("start_time", "Unknown")
                    last_checkpoint = progress_data.get("last_checkpoint", "Unknown")

                    # Parse timestamps
                    try:
                        from datetime import datetime
                        start_dt = datetime.fromisoformat(start_time)
                        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        start_str = start_time

                    try:
                        checkpoint_dt = datetime.fromisoformat(last_checkpoint)
                        checkpoint_str = checkpoint_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        checkpoint_str = last_checkpoint

                    # Calculate folder size and file count
                    folder_size = 0
                    file_count = 0
                    backup_count = 0

                    for file in folder_path.rglob("*"):
                        if file.is_file():
                            folder_size += file.stat().st_size
                            file_count += 1
                            if "backup" in file.name:
                                backup_count += 1

                    if folder_size > 1024 * 1024:  # MB
                        size_str = f"{folder_size / (1024 * 1024):.1f} MB"
                    else:
                        size_str = f"{folder_size / 1024:.1f} KB"

                    # Show detailed information about what will be removed
                    console.print(f"\n[yellow]🗑️ Session to remove:[/yellow]")
                    console.print(f"     [bold]{session_model}[/bold] - {completed:,}/{total:,} tests ({completion_percent:.1f}% complete)")
                    console.print(f"     [dim]Matched by: {identifier}[/dim]")
                    console.print(f"     [dim]Started: {start_str}[/dim]")
                    console.print(f"     [dim]Last Save: {checkpoint_str}[/dim]")
                    console.print(f"     [dim]Failed Tests: {failed}[/dim]")
                    console.print(f"     [dim]Files: {file_count} ({backup_count} backups)[/dim]")
                    console.print(f"     [dim]Size: {size_str}[/dim]")
                    console.print(f"     [dim]Folder: {folder_path}[/dim]")

                    # Confirm unless force flag is used
                    if not force:
                        confirm = typer.confirm("Remove this session?")
                        if not confirm:
                            console.print("[yellow]Skipped.[/yellow]")
                            continue

                    # Remove the entire session folder
                    import shutil
                    if folder_path.exists():
                        shutil.rmtree(folder_path)
                        console.print(f"[green]✅ Removed session: {folder_path.name}[/green]")
                        console.print(f"[green]💾 Freed {size_str} of disk space[/green]")
                        removed_count += 1
                        total_size_freed += folder_size
                    else:
                        console.print(f"[yellow]⚠️ Folder not found: {folder_path}[/yellow]")

                except Exception as e:
                    console.print(f"[red]❌ Failed to remove session for '{identifier}': {e}[/red]")

        # Show final summary
        if removed_count > 0:
            if total_size_freed > 1024 * 1024 * 1024:  # GB
                total_freed_str = f"{total_size_freed / (1024 * 1024 * 1024):.1f} GB"
            elif total_size_freed > 1024 * 1024:  # MB
                total_freed_str = f"{total_size_freed / (1024 * 1024):.1f} MB"
            else:
                total_freed_str = f"{total_size_freed / 1024:.1f} KB"

            console.print(f"\n[green]✅ Successfully removed {removed_count} session(s).[/green]")
            console.print(f"[green]💾 Total space freed: {total_freed_str}[/green]")
        else:
            console.print(f"\n[yellow]📭 No sessions were removed.[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error removing resume sessions: {e}[/red]")
        raise typer.Exit(1) from e
@app.command("resume-remove-range")
def resume_remove_range(
    range_spec: Annotated[
        str, typer.Argument(help="Range specification like '1-5', '1,3,5-8', or 'all'")
    ],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompts")
    ] = False,
):
    """Remove multiple resume sessions by range specification

    Examples:
        uv run equilens resume-remove-range "1-5"       # Remove sessions 1 through 5
        uv run equilens resume-remove-range "1,3,5"     # Remove sessions 1, 3, and 5
        uv run equilens resume-remove-range "1-3,7-9"   # Remove sessions 1-3 and 7-9
        uv run equilens resume-remove-range "all"       # Remove all sessions
    """
    try:
        interrupted_sessions = find_interrupted_sessions()

        if not interrupted_sessions:
            console.print("[yellow]📭 No interrupted audit sessions found.[/yellow]")
            return

        # Parse range specification
        indices_to_remove = set()
        total_sessions = len(interrupted_sessions)

        if range_spec.lower() == "all":
            indices_to_remove = set(range(1, total_sessions + 1))
        else:
            # Parse comma-separated ranges and individual numbers
            parts = range_spec.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range like "1-5"
                    try:
                        start, end = map(int, part.split('-', 1))
                        if start < 1 or end > total_sessions:
                            console.print(f"[red]❌ Range {start}-{end} is out of bounds (1-{total_sessions})[/red]")
                            return
                        indices_to_remove.update(range(start, end + 1))
                    except ValueError:
                        console.print(f"[red]❌ Invalid range format: {part}[/red]")
                        return
                else:
                    # Handle individual number
                    try:
                        index = int(part)
                        if index < 1 or index > total_sessions:
                            console.print(f"[red]❌ Index {index} is out of bounds (1-{total_sessions})[/red]")
                            return
                        indices_to_remove.add(index)
                    except ValueError:
                        console.print(f"[red]❌ Invalid number: {part}[/red]")
                        return

        if not indices_to_remove:
            console.print("[yellow]📭 No valid indices specified.[/yellow]")
            return

        # Show what will be removed
        console.print(f"[yellow]🗑️ Will remove {len(indices_to_remove)} session(s):[/yellow]\n")

        total_size_to_remove = 0
        sessions_to_remove = []

        for index in sorted(indices_to_remove):
            progress_file, progress_data = interrupted_sessions[index - 1]  # Convert to 0-based
            sessions_to_remove.append((progress_file, progress_data))

            session_model = progress_data.get("model_name", "Unknown")
            completed = progress_data.get("completed_tests", 0)
            total = progress_data.get("total_tests", 0)
            completion_percent = (completed / total * 100) if total > 0 else 0

            try:
                folder_path = Path(progress_file).parent
                folder_size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
                total_size_to_remove += folder_size

                if folder_size > 1024 * 1024:
                    size_str = f"{folder_size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{folder_size / 1024:.1f} KB"
            except Exception:
                size_str = "Unknown"

            console.print(f"  {index:2d}. [red]{session_model}[/red] - {completed:,}/{total:,} ({completion_percent:.1f}%) - {folder_path.name} ({size_str})")

        # Show total space to be freed
        if total_size_to_remove > 1024 * 1024 * 1024:  # GB
            total_size_str = f"{total_size_to_remove / (1024 * 1024 * 1024):.1f} GB"
        elif total_size_to_remove > 1024 * 1024:  # MB
            total_size_str = f"{total_size_to_remove / (1024 * 1024):.1f} MB"
        else:
            total_size_str = f"{total_size_to_remove / 1024:.1f} KB"

        console.print(f"\n[yellow]💾 Total disk space to be freed: {total_size_str}[/yellow]")

        if not force:
            console.print()
            confirm = typer.confirm("Proceed with removal?")
            if not confirm:
                console.print("[yellow]Removal cancelled.[/yellow]")
                return

        # Remove sessions
        removed_count = 0
        freed_space = 0

        for progress_file, progress_data in sessions_to_remove:
            try:
                folder_path = Path(progress_file).parent

                # Calculate size before removal
                folder_size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())

                import shutil
                if folder_path.exists():
                    shutil.rmtree(folder_path)
                    removed_count += 1
                    freed_space += folder_size
                    console.print(f"[green]✅ Removed: {folder_path.name}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to remove {folder_path}: {e}[/red]")

        # Show final summary
        if freed_space > 1024 * 1024 * 1024:  # GB
            freed_str = f"{freed_space / (1024 * 1024 * 1024):.1f} GB"
        elif freed_space > 1024 * 1024:  # MB
            freed_str = f"{freed_space / (1024 * 1024):.1f} MB"
        else:
            freed_str = f"{freed_space / 1024:.1f} KB"

        console.print(f"\n[green]✅ Successfully removed {removed_count} session(s).[/green]")
        console.print(f"[green]💾 Freed {freed_str} of disk space.[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error removing resume sessions: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("resume-clean")
def resume_clean(
    keep: Annotated[
        int, typer.Option("--keep", "-k", help="Number of most recent sessions to keep")
    ] = 5,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt")
    ] = False,
):
    """Clean up old resume sessions, keeping only the most recent ones"""
    try:
        interrupted_sessions = find_interrupted_sessions()

        if not interrupted_sessions:
            console.print("[yellow]📭 No interrupted audit sessions found.[/yellow]")
            return

        if len(interrupted_sessions) <= keep:
            console.print(f"[green]✅ Only {len(interrupted_sessions)} sessions found. Nothing to clean.[/green]")
            return

        # Sort sessions by start time (most recent first)
        try:
            interrupted_sessions.sort(
                key=lambda x: x[1].get("start_time", ""),
                reverse=True
            )
        except Exception:
            console.print("[yellow]⚠️ Could not sort sessions by time. Using current order.[/yellow]")

        sessions_to_keep = interrupted_sessions[:keep]
        sessions_to_remove = interrupted_sessions[keep:]

        console.print(f"\n[yellow]🧹 Found {len(interrupted_sessions)} sessions. Will keep {keep} most recent.[/yellow]")

        # Show sessions that will be kept
        console.print(f"\n[green]✅ Sessions to KEEP ({len(sessions_to_keep)}):[/green]")
        for i, (progress_file, progress_data) in enumerate(sessions_to_keep, 1):
            session_model = progress_data.get("model_name", "Unknown")
            completed = progress_data.get("completed_tests", 0)
            total = progress_data.get("total_tests", 0)
            completion_percent = (completed / total * 100) if total > 0 else 0
            start_time = progress_data.get("start_time", "Unknown")

            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time)
                time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = start_time

            try:
                folder_name = Path(progress_file).parent.name

                # Calculate folder size
                total_size = 0
                for file in Path(progress_file).parent.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

                if total_size > 1024 * 1024:  # MB
                    size_str = f"{total_size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{total_size / 1024:.1f} KB"
            except Exception:
                folder_name = "Unknown"
                size_str = "Unknown"

            console.print(f"  {i}. [green]{session_model}[/green] - {completed:,}/{total:,} ({completion_percent:.1f}%) - {time_str} - {folder_name} ({size_str})")

        # Show sessions that will be removed
        console.print(f"\n[red]🗑️ Sessions to REMOVE ({len(sessions_to_remove)}):[/red]")
        total_size_to_remove = 0

        for i, (progress_file, progress_data) in enumerate(sessions_to_remove, 1):
            session_model = progress_data.get("model_name", "Unknown")
            completed = progress_data.get("completed_tests", 0)
            total = progress_data.get("total_tests", 0)
            failed = progress_data.get("failed_tests", 0)
            completion_percent = (completed / total * 100) if total > 0 else 0
            start_time = progress_data.get("start_time", "Unknown")
            last_checkpoint = progress_data.get("last_checkpoint", "Unknown")

            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time)
                time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = start_time

            try:
                checkpoint_dt = datetime.fromisoformat(last_checkpoint)
                checkpoint_str = checkpoint_dt.strftime("%H:%M:%S")
            except Exception:
                checkpoint_str = "Unknown"

            try:
                folder_path = Path(progress_file).parent
                folder_name = folder_path.name

                # Calculate folder size
                folder_size = 0
                file_count = 0
                for file in folder_path.rglob("*"):
                    if file.is_file():
                        folder_size += file.stat().st_size
                        file_count += 1

                total_size_to_remove += folder_size

                if folder_size > 1024 * 1024:  # MB
                    size_str = f"{folder_size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{folder_size / 1024:.1f} KB"
            except Exception:
                folder_name = "Unknown"
                size_str = "Unknown"
                file_count = 0

            console.print(f"  {i}. [red]{session_model}[/red] - {completed:,}/{total:,} ({completion_percent:.1f}%) - Started: {time_str}")
            console.print(f"     [dim]Last Save: {checkpoint_str} | Failed: {failed} | Files: {file_count} | Size: {size_str}[/dim]")
            console.print(f"     [dim]Folder: {folder_name}[/dim]")

        # Show total space that will be freed
        if total_size_to_remove > 1024 * 1024 * 1024:  # GB
            total_size_str = f"{total_size_to_remove / (1024 * 1024 * 1024):.1f} GB"
        elif total_size_to_remove > 1024 * 1024:  # MB
            total_size_str = f"{total_size_to_remove / (1024 * 1024):.1f} MB"
        else:
            total_size_str = f"{total_size_to_remove / 1024:.1f} KB"

        console.print(f"\n[yellow]💾 Total disk space to be freed: {total_size_str}[/yellow]")

        if not force:
            console.print()
            confirm = typer.confirm("Proceed with cleanup?")
            if not confirm:
                console.print("[yellow]Cleanup cancelled.[/yellow]")
                return

        # Remove old sessions
        removed_count = 0
        freed_space = 0
        for progress_file, _ in sessions_to_remove:
            try:
                folder_path = Path(progress_file).parent

                # Calculate size before removal
                folder_size = 0
                for file in folder_path.rglob("*"):
                    if file.is_file():
                        folder_size += file.stat().st_size

                import shutil
                if folder_path.exists():
                    shutil.rmtree(folder_path)
                    removed_count += 1
                    freed_space += folder_size
                    console.print(f"[green]✅ Removed: {folder_path.name}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to remove {folder_path}: {e}[/red]")

        # Show final summary
        if freed_space > 1024 * 1024 * 1024:  # GB
            freed_str = f"{freed_space / (1024 * 1024 * 1024):.1f} GB"
        elif freed_space > 1024 * 1024:  # MB
            freed_str = f"{freed_space / (1024 * 1024):.1f} MB"
        else:
            freed_str = f"{freed_space / 1024:.1f} KB"

        console.print(f"\n[green]✅ Successfully cleaned up {removed_count} old sessions.[/green]")
        console.print(f"[green]💾 Freed {freed_str} of disk space.[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error cleaning resume sessions: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
