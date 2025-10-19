"""
Modern CLI interface for EquiLens using Typer and Rich

A comprehensive command-line interface for the EquiLens AI bias detection platform.
Features interactive commands, beautiful output formatting, and comprehensive help.
"""

import json
import subprocess
from pathlib import Path
from typing import Annotated

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


# Corpus discovery paths - includes Phase1_CorpusGenerator/corpus
DEFAULT_CORPUS_SEARCH_PATHS = [
    Path.cwd(),
    Path.cwd() / "corpus",
    Path.cwd() / "src" / "Phase1_CorpusGenerator" / "corpus",
    Path(__file__).parent.parent / "Phase1_CorpusGenerator" / "corpus",
    Path(__file__).parent.parent.parent / "src" / "Phase1_CorpusGenerator" / "corpus",
]


def find_corpus_files(
    user_path: str | None = None, pattern: str = "*.csv"
) -> list[Path]:
    """
    Robust corpus discovery:
      1) use explicit user_path (if provided)
      2) fallback to search paths in DEFAULT_CORPUS_SEARCH_PATHS
      3) return absolute Path list (may be empty)
    """
    # 1) explicit path
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted([f.resolve() for f in p.glob(pattern)])
    # 2) try defaults
    found = []
    for sp in DEFAULT_CORPUS_SEARCH_PATHS:
        try:
            sp = Path(sp).expanduser().resolve()
            if sp.exists():
                if sp.is_dir():
                    found.extend([f.resolve() for f in sp.glob(pattern)])
                elif sp.is_file() and sp.match(pattern):
                    found.append(sp)
        except (PermissionError, OSError):
            continue  # Skip paths we can't access
    # de-duplicate and sort
    unique = sorted(dict.fromkeys(found))
    return unique


def interactive_corpus_selection(user_path: str | None = None) -> Path | None:
    """
    Interactive corpus file selection with auto-discovery.
    Returns selected Path or None if cancelled/no files found.
    """
    corpus_files = find_corpus_files(user_path)

    if not corpus_files:
        console.print("\n[yellow]⚠ No corpus files found in common locations:[/yellow]")
        for sp in DEFAULT_CORPUS_SEARCH_PATHS:
            console.print(f"  [dim]• {sp}[/dim]")
        console.print(
            "\n[bold]Enter corpus file path (or press Enter to cancel):[/bold]"
        )
        manual_path = typer.prompt("", default="").strip()
        if not manual_path:
            return None
        manual_corpus = find_corpus_files(manual_path)
        if not manual_corpus:
            console.print(
                f"[red]❌ No valid corpus files found at: {manual_path}[/red]"
            )
            return None
        corpus_files = manual_corpus

    # Single file found - auto-select with confirmation
    if len(corpus_files) == 1:
        selected = corpus_files[0]
        console.print(f"\n[green]✓ Found corpus:[/green] [cyan]{selected.name}[/cyan]")
        console.print(f"  [dim]Location: {selected.parent}[/dim]")
        console.print("\n[bold]Use this corpus?[/bold]")
        if typer.confirm("", default=True):
            return selected
        return None

    # Multiple files - show interactive menu
    console.print(f"\n[cyan]📊 Found {len(corpus_files)} corpus files:[/cyan]\n")
    for idx, corpus in enumerate(corpus_files, 1):
        try:
            file_size = corpus.stat().st_size / 1024  # KB
            console.print(
                f"  [bold cyan][{idx}][/bold cyan] [white]{corpus.name}[/white]"
            )
            console.print(f"      [dim]📁 {corpus.parent}[/dim]")
            console.print(f"      [dim]💾 {file_size:.1f} KB[/dim]")
            console.print()
        except (PermissionError, OSError):
            console.print(
                f"  [bold cyan][{idx}][/bold cyan] [white]{corpus.name}[/white] [yellow](⚠ Cannot read file info)[/yellow]"
            )
            console.print()

    while True:
        try:
            console.print(
                f"[bold]Select corpus [1-{len(corpus_files)}] (or 'q' to quit):[/bold]"
            )
            choice = typer.prompt("").strip()
            if choice.lower() in ("q", "quit", "exit", ""):
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(corpus_files):
                selected = corpus_files[idx]
                console.print(
                    f"\n[green]✓ Selected:[/green] [cyan]{selected.name}[/cyan]"
                )
                return selected
            else:
                console.print(
                    f"[yellow]⚠ Please enter a number between 1 and {len(corpus_files)}[/yellow]"
                )
        except ValueError:
            console.print(
                "[yellow]⚠ Please enter a valid number or 'q' to quit[/yellow]"
            )
        except KeyboardInterrupt:
            console.print("\n\n[yellow]⚠ Selection cancelled[/yellow]")
            return None


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
    invoke_without_command=True,
)

# Global manager instance
manager = None


def get_manager() -> EquiLensManager:
    """Get or create the EquiLens manager instance"""
    global manager
    if manager is None:
        manager = EquiLensManager()
    return manager


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None, typer.Option("--version", "-V", help="Show version and exit")
    ] = None,
):
    """Main callback to handle version flag and show help when no command provided"""
    if version:
        from equilens import __version__

        console.print(f"EquiLens version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        # Show help and exit cleanly without error
        console.print(ctx.get_help())
        raise typer.Exit(0)


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
        console.print("  • [cyan]uv run equilens audit[/cyan] - Run bias audit")
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
models_app = typer.Typer(
    help="🎯 Manage Ollama models",
    invoke_without_command=True,
)


@models_app.callback()
def models_callback(ctx: typer.Context):
    """Handle models group to show help when no subcommand provided"""
    if ctx.invoked_subcommand is None:
        # Show help and exit cleanly without error
        console.print(ctx.get_help())
        raise typer.Exit(0)


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
    """⬇️ Download a model"""
    manager = get_manager()
    success = manager.pull_model(model)
    if success:
        console.print(f"✅ [green]Model {model} downloaded successfully![/green]")
    else:
        console.print(f"[red]❌ Failed to download model {model}[/red]")
        raise typer.Exit(1)


# Audit subcommand group for session management
audit_app = typer.Typer(
    help="🔍 Bias audit operations and session management",
    invoke_without_command=True,
)


@audit_app.callback()
def audit_callback(ctx: typer.Context):
    """Handle audit group to show help when no subcommand provided"""
    if ctx.invoked_subcommand is None:
        # Show help and exit cleanly without error
        console.print(ctx.get_help())
        raise typer.Exit(0)


app.add_typer(audit_app, name="audit")


@audit_app.command("run")
def audit_run(
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
    retry_immediate: Annotated[
        bool,
        typer.Option(
            "--retry-immediate",
            help="Attempt immediate retries for failed tuples before queuing",
        ),
    ] = False,
    retry_batch_size: Annotated[
        int,
        typer.Option(
            "--retry-batch-size",
            help="Number of successes between processing the retry queue",
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
        bool, typer.Option("--help", help="Show this help message and exit")
    ] = False,
):
    """🔍 Run bias audit with interactive prompts and enhanced visual design"""
    # This will contain the main audit logic (moved from the standalone audit command)
    # For now, redirect to the main audit function
    from equilens.cli import audit

    audit(
        model=model,
        corpus=corpus,
        output_dir=output_dir,
        enhanced=enhanced,
        batch_size=batch_size,
        retry_immediate=retry_immediate,
        retry_batch_size=retry_batch_size,
        resume=resume,
        silent=silent,
        help_cmd=help_cmd,
    )


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
            help="Use enhanced auditor with dynamic concurrency and better progress tracking (default, auto-fallback to standard if issues occur)",
        ),
    ] = True,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", "-b", help="Number of concurrent requests for enhanced mode"
        ),
    ] = 5,
    retry_immediate: Annotated[
        bool,
        typer.Option(
            "--retry-immediate",
            help="Attempt immediate retries for failed tuples before queuing",
        ),
    ] = False,
    retry_batch_size: Annotated[
        int,
        typer.Option(
            "--retry-batch-size",
            help="Number of successes between processing the retry queue",
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
        bool, typer.Option("--help", help="Show this help message and exit")
    ] = False,
):
    """🔍 Run bias audit with interactive prompts and enhanced visual design"""

    # Show introduction and help only when --help is explicitly requested
    if help_cmd:
        console.print(
            Panel.fit(
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
                title="🎯 AI Bias Detection",
            )
        )

        # Show help content
        console.print("\n[bold]📖 Command Options:[/bold]\n")
        console.print("[cyan]--model, -m[/cyan]        Model name to audit")
        console.print("[cyan]--corpus, -c[/cyan]       Path to corpus CSV file")
        console.print(
            "[cyan]--output-dir, -o[/cyan]   Output directory for results (default: results)"
        )
        console.print(
            "[cyan]--enhanced[/cyan]         Use enhanced auditor (default: true, auto-fallback enabled)"
        )
        console.print(
            "[cyan]--no-enhanced[/cyan]      Disable enhanced auditor, use standard mode"
        )
        console.print(
            "[cyan]--batch-size, -b[/cyan]   Number of concurrent requests (default: 5)"
        )
        console.print(
            "[cyan]--retry-immediate[/cyan]  Attempt immediate retries for failed tuples"
        )
        console.print(
            "[cyan]--retry-batch-size[/cyan] Number of successes between retry batches (default: 5)"
        )
        console.print("[cyan]--resume, -r[/cyan]       Resume from previous session")
        console.print("[cyan]--silent, -s[/cyan]       Suppress subprocess output")
        console.print("[cyan]--help[/cyan]             Show this help message")

        console.print("\n[bold green]🚀 Quick Start Examples:[/bold green]")
        console.print("[dim]# Interactive mode (recommended for beginners)[/dim]")
        console.print("[yellow]uv run equilens audit[/yellow]")
        console.print("\n[dim]# Specify model and corpus directly[/dim]")
        console.print(
            "[yellow]uv run equilens audit --model llama2:latest --corpus corpus.csv[/yellow]"
        )
        console.print("\n[dim]# Resume a previous audit session[/dim]")
        console.print(
            "[yellow]uv run equilens audit --resume path/to/progress.json[/yellow]"
        )

        console.print(
            "\n[bold]💡 Tip:[/bold] Run without options for interactive setup!"
        )
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

                # Always ask for concurrency configuration during resume
                console.print("\n[bold]🔧 Performance Configuration[/bold]")
                console.print(
                    "[dim]Configure concurrency for resumed audit session[/dim]"
                )
                console.print(
                    "  • [cyan]Higher values[/cyan]: Faster processing but more load"
                )
                console.print(
                    "  • [cyan]Lower values[/cyan]: Safer for system stability"
                )
                console.print("  • [cyan]1[/cyan]: Sequential processing (safest)")

                workers_input = typer.prompt(
                    "Enter number of concurrent workers (1-10)",
                    default="3",
                    show_default=True,
                ).strip()

                try:
                    max_workers = int(workers_input)
                    max_workers = max(1, min(max_workers, 10))  # Clamp between 1-10
                    if max_workers > 1:
                        console.print(
                            f"[green]✓ Configured for {max_workers} concurrent workers with dynamic scaling[/green]"
                        )
                        console.print(
                            "[dim]Workers will automatically scale down on errors and back up on success[/dim]"
                        )
                    else:
                        console.print(
                            "[yellow]⚡ Using sequential processing mode[/yellow]"
                        )
                except ValueError:
                    console.print(
                        "[yellow]⚠️ Invalid input, using default of 3 workers[/yellow]"
                    )
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

        # Ask about concurrent processing for faster performance
        console.print("\n[bold]🚀 Performance Configuration[/bold]")
        console.print("[dim]Would you like to enable concurrent processing?[/dim]")
        console.print("  • [cyan]1[/cyan]: Single threaded (stable, recommended)")
        console.print(
            "  • [cyan]2-5[/cyan]: Multiple threads (faster but may stress Ollama)"
        )
        console.print("  • [cyan]n[/cyan]: Use default (single threaded)")

        worker_input = (
            typer.prompt("Number of concurrent workers", default="1", show_default=True)
            .strip()
            .lower()
        )

        if worker_input == "n" or worker_input == "":
            max_workers = 1
        else:
            try:
                workers = int(worker_input)
                if 1 <= workers <= 8:  # Reasonable limit
                    max_workers = workers
                    if workers > 1:
                        console.print(
                            f"[yellow]⚡ Using {workers} concurrent threads[/yellow]"
                        )
                        console.print(
                            "[dim]Note: This may stress Ollama - monitor for connection errors[/dim]"
                        )
                    else:
                        console.print(
                            "[green]✓ Using single threaded processing[/green]"
                        )
                else:
                    console.print(
                        "[red]❌ Invalid number (1-8), using single threaded[/red]"
                    )
                    max_workers = 1
            except ValueError:
                console.print("[red]❌ Invalid input, using single threaded[/red]")
                max_workers = 1

        # Auto-discover all corpus files in the corpus directory
        corpus_dir = Path("src/Phase1_CorpusGenerator/corpus")
        found_files = []

        if corpus_dir.exists() and corpus_dir.is_dir():
            # Find all CSV files in the corpus directory
            for csv_file in sorted(corpus_dir.glob("*.csv")):
                if csv_file.is_file():
                    file_size = csv_file.stat().st_size
                    found_files.append((str(csv_file), file_size))

        # Also check current directory for corpus files as fallback
        for csv_file in sorted(Path.cwd().glob("*.csv")):
            if csv_file.is_file() and "corpus" in csv_file.name.lower():
                file_path = str(csv_file)
                # Avoid duplicates
                if not any(file_path == existing[0] for existing in found_files):
                    file_size = csv_file.stat().st_size
                    found_files.append((file_path, file_size))

        if found_files:
            console.print("[green]✓ Found corpus files:[/green]")

            for i, (path, size) in enumerate(found_files, 1):
                file_size_formatted = format_file_size(size)
                console.print(
                    f"  {i}. [cyan]{path}[/cyan] ([dim]{file_size_formatted}[/dim])"
                )

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
            # Use interactive corpus selection
            selected_corpus = interactive_corpus_selection()
            if selected_corpus is None:
                console.print("[yellow]Corpus selection cancelled by user.[/yellow]")
                raise typer.Exit(0)
            corpus = str(selected_corpus)

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

    console.print(
        Panel.fit(
            f"[bold green]Step 3: Configuration Review[/bold green]\n\n"
            f"[bold]Model:[/bold] [cyan]{model}[/cyan]\n"
            f"[bold]Corpus:[/bold] {corpus_display} ([dim]{corpus_size}[/dim])\n"
            f"[bold]Output Directory:[/bold] [cyan]{output_dir}[/cyan]\n"
            f"[bold]Silent Mode:[/bold] [cyan]{'Enabled' if silent else 'Disabled'}[/cyan]",
            border_style="green",
        )
    )

    # Step 3.5: Analytics Preference (only for new audits, not resume)
    analytics_preference = "none"  # Default: no auto-analysis

    if not resume:
        console.print(
            Panel.fit(
                "[bold magenta]Step 3.5: Post-Audit Analytics Preference[/bold magenta]\n\n"
                "Would you like to automatically run analytics after the audit completes?\n\n"
                "[cyan]Options:[/cyan]\n"
                "  1. [green]None[/green] - Skip automatic analysis (run manually later)\n"
                "  2. [yellow]Standard[/yellow] - Quick analysis with basic visualizations (~5 sec)\n"
                "  3. [blue]Advanced[/blue] - Comprehensive analysis with 8+ charts + statistical report (~15 sec)\n\n"
                "[dim]You can always run analysis manually later using:[/dim]\n"
                "[dim]  uv run equilens analyze[/dim]",
                border_style="magenta",
            )
        )

        console.print("\n[bold]Select analytics preference (1/2/3):[/bold]")
        while True:
            choice = typer.prompt("", default="1")
            if choice in ["1", "2", "3"]:
                if choice == "1":
                    analytics_preference = "none"
                    console.print("[green]✓[/green] No automatic analysis selected")
                elif choice == "2":
                    analytics_preference = "standard"
                    console.print("[green]✓[/green] Standard analytics will run after audit")
                elif choice == "3":
                    analytics_preference = "advanced"
                    console.print("[green]✓[/green] Advanced analytics will run after audit")
                break
            else:
                console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")


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
            # Use enhanced auditor with Rich progress bars
            console.print(
                "🚀 [bold cyan]Starting enhanced audit with dynamic concurrency...[/bold cyan]"
            )

            # Import the enhanced auditor with better error handling
            import sys

            enhanced_path = Path("src/Phase2_ModelAuditor").resolve()
            if str(enhanced_path) not in sys.path:
                sys.path.insert(0, str(enhanced_path))

            try:
                from Phase2_ModelAuditor.enhanced_audit_model import EnhancedBiasAuditor

                # Create and run enhanced auditor (auditor will handle ETA configuration)
                auditor = EnhancedBiasAuditor(
                    model_name=model,
                    corpus_file=corpus,
                    output_dir=output_dir,
                )

                # Set batch size if specified
                auditor.batch_size = batch_size

                success = auditor.run_enhanced_audit(resume_file=resume)

                if success:
                    console.print(
                        "\n✅ [bold green]Enhanced audit completed successfully![/bold green]"
                    )
                else:
                    console.print(
                        f"\n⚠️ [yellow]Enhanced audit encountered issues: {success}. Falling back to standard auditor...[/yellow]"
                    )
                    enhanced = False

            except ImportError as e:
                console.print(f"[yellow]⚠️ Enhanced auditor import failed: {e}[/yellow]")
                console.print(
                    "[yellow]Falling back to standard auditor...[/yellow]"
                )
                enhanced = False
            except Exception as e:
                console.print(f"[yellow]⚠️ Enhanced auditor error: {e}[/yellow]")
                console.print(
                    "[yellow]Falling back to standard auditor...[/yellow]"
                )
                enhanced = False

        if not enhanced:
            # Use enhanced auditor via subprocess wrapper
            console.print(
                "🚀 [green]Using enhanced auditor with dynamic concurrency...[/green]"
            )

            # Configure subprocess parameters with robust fallback and error handling
            import logging
            import traceback

            # Ensure logs directory exists
            logs_dir = Path("logs")
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If we can't create logs dir, continue without file logging
                pass

            log_file = logs_dir / "audit_errors.log"
            logging.basicConfig(
                filename=str(log_file),
                level=logging.DEBUG,
                format="%(asctime)s %(levelname)s: %(message)s",
            )

            # Find repository root by looking for pyproject.toml or .git
            def find_repo_root(start_path: Path = Path.cwd()) -> Path | None:
                """Find repository root by looking for pyproject.toml or .git"""
                current = start_path.resolve()
                for parent in [current] + list(current.parents):
                    if (parent / "pyproject.toml").exists() or (
                        parent / ".git"
                    ).exists():
                        return parent
                return None

            repo_root = find_repo_root()
            if repo_root is None:
                repo_root = Path.cwd()  # Fallback to current directory

            # Candidate auditor scripts in preferred order (absolute paths from repo root)
            candidates = [
                repo_root / "src" / "Phase2_ModelAuditor" / "audit_model.py",
                repo_root / "src" / "Phase2_ModelAuditor" / "enhanced_audit_model.py",
                Path(
                    "src/Phase2_ModelAuditor/audit_model.py"
                ),  # Fallback relative path
                Path("src/Phase2_ModelAuditor/enhanced_audit_model.py"),
            ]

            auditor_script = None
            for p in candidates:
                if p.exists():
                    auditor_script = p
                    console.print(f"[dim]✓ Found auditor: {p}[/dim]")
                    break

            if auditor_script is None:
                msg = (
                    "No auditor script found. Expected one of: "
                    "src/Phase2_ModelAuditor/audit_model.py, audit_model.py, "
                    "or enhanced_audit_model.py."
                )
                console.print(f"[red]❌ {msg}[/red]")
                console.print(f"[dim]Searched in repository root: {repo_root}[/dim]")
                console.print(f"[dim]Current working directory: {Path.cwd()}[/dim]")
                logging.error(msg)
                raise typer.Exit(2)

            base_cmd = [
                "python",
                str(auditor_script),
                "--model",
                model,
                "--corpus",
                corpus,
                "--output-dir",
                output_dir,
            ]

            # Add concurrency option only if the target script supports it
            base_cmd.extend(["--max-workers", str(max_workers)])

            if resume:
                base_cmd.extend(["--resume", resume])
            # Forward retry options
            if retry_immediate:
                base_cmd.append("--retry-immediate")
            if retry_batch_size is not None:
                base_cmd.extend(["--retry-batch-size", str(retry_batch_size)])

            def run_audit(cmd, silent_mode=False):
                cwd = Path.cwd()
                try:
                    if silent_mode:
                        subprocess.run(
                            cmd,
                            check=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            cwd=str(cwd),
                        )
                    else:
                        subprocess.run(
                            cmd,
                            check=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            cwd=str(cwd),
                        )
                    return True
                except FileNotFoundError as fnf:
                    console.print(
                        f"[red]❌ Unable to start auditor: {fnf}. Is Python on the PATH and file present?[/red]"
                    )
                    logging.exception("FileNotFoundError while starting auditor")
                    return False
                except UnicodeDecodeError:
                    # Try again in silent mode to avoid console encoding problems
                    console.print(
                        "[yellow]⚠ Unicode decode error detected; retrying in silent mode...[/yellow]"
                    )
                    logging.warning("UnicodeDecodeError, retrying in silent mode")
                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            cwd=str(cwd),
                        )
                        return True
                    except Exception:
                        logging.exception("Retry in silent mode failed")
                        return False
                except subprocess.CalledProcessError as cpe:
                    console.print(
                        f"[red]❌ Auditor exited with non-zero status {cpe.returncode}. Check logs for details.[/red]"
                    )
                    logging.error("Auditor failed: %s", cpe)
                    logging.debug(traceback.format_exc())
                    return False
                except KeyboardInterrupt:
                    console.print(
                        "[yellow]⏹️ Audit cancelled by user (KeyboardInterrupt).[/yellow]"
                    )
                    logging.info("Audit cancelled by user")
                    raise
                except Exception as exc:
                    console.print(
                        "[red]❌ Unexpected error while running auditor. See logs/audit_errors.log for details.[/red]"
                    )
                    logging.exception("Unexpected error while running auditor: %s", exc)
                    return False

            # Attempt to run normally first, then with fallbacks if needed
            try:
                success = run_audit(base_cmd, silent_mode=silent)

                if not success and not silent:
                    # If initial run failed, try a silent run to bypass console issues
                    console.print(
                        "[yellow]⚠ Attempting fallback: running auditor in silent mode...[/yellow]"
                    )
                    logging.info("Attempting fallback silent run for auditor")
                    success = run_audit(base_cmd, silent_mode=True)

                if not success:
                    # If still failing, and we used a non-primary script, try other candidates
                    for alt in candidates:
                        if str(alt) == str(auditor_script):
                            continue
                        if not alt.exists():
                            continue
                        alt_cmd = [
                            "python",
                            str(alt),
                            "--model",
                            model,
                            "--corpus",
                            corpus,
                            "--output-dir",
                            output_dir,
                        ]
                        if resume:
                            alt_cmd.extend(["--resume", resume])
                        # Forward retry options
                        if retry_immediate:
                            alt_cmd.append("--retry-immediate")
                        if retry_batch_size is not None:
                            alt_cmd.extend(
                                ["--retry-batch-size", str(retry_batch_size)]
                            )
                        console.print(
                            f"[yellow]⚠ Trying alternative auditor script: {alt}[/yellow]"
                        )
                        logging.info("Trying alternative auditor: %s", alt)
                        if run_audit(alt_cmd, silent_mode=silent):
                            success = True
                            break

                if success:
                    console.print(
                        "[green]✓[/green] Audit process completed successfully"
                    )
                else:
                    console.print(
                        "[red]❌ Audit failed after retries. Check logs/audit_errors.log for details.[/red]"
                    )
                    raise typer.Exit(1) from None
            except KeyboardInterrupt:
                raise typer.Exit(1) from None

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

        # Auto-run analytics if preference was set (only for new audits)
        if not resume and analytics_preference != "none" and latest_results:
            _run_auto_analytics(analytics_preference, latest_results)

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
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Configuration file path for corpus generation (optional - uses interactive mode if not provided)",
        ),
    ] = None,
):
    """📝 Generate test corpus with interactive mode or configuration file"""

    if config is None:
        # Interactive mode - run the generate_corpus.py script directly without arguments
        console.print("🎯 [bold]Starting interactive corpus generation...[/bold]")
        console.print(
            "[dim]This will use the word_lists.json configuration and guide you through the process.[/dim]"
        )

        try:
            # Run the corpus generator in interactive mode
            subprocess.run(
                ["python", "src/Phase1_CorpusGenerator/generate_corpus.py"],
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            console.print("✅ [green]Corpus generated successfully![/green]")
            console.print(
                "📁 [cyan]Check the src/Phase1_CorpusGenerator/corpus/ directory for the generated files.[/cyan]"
            )
        except subprocess.CalledProcessError as e:
            console.print(
                f"[red]❌ Corpus generation failed with exit code: {e.returncode}[/red]"
            )
            raise typer.Exit(1) from e
        except FileNotFoundError as e:
            console.print("[red]❌ Could not find corpus generator script[/red]")
            console.print(
                "[yellow]💡 Make sure you're in the EquiLens project root directory[/yellow]"
            )
            raise typer.Exit(1) from e
    else:
        # Config file mode - use the manager as before
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
    advanced: Annotated[
        bool,
        typer.Option(
            "--advanced",
            "-a",
            help="Use advanced analytics with comprehensive visualizations and statistics",
        ),
    ] = False,
):
    """📊 Analyze bias audit results with enhanced visual interface

    Use --advanced for comprehensive statistical analysis with:
    - Violin plots, box plots, heatmaps, scatter plots
    - Effect sizes (Cohen's d), confidence intervals
    - Time-series progression analysis
    - Statistical significance testing (t-tests)
    - Professional presentation-ready dashboard
    """

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

    # Step 1.5: Analytics Mode Selection (if not specified via flag)
    if not advanced:
        # Check if user explicitly used --advanced=False or just didn't use the flag
        # We'll ask interactively
        console.print(
            Panel.fit(
                "[bold magenta]Step 1.5: Analytics Mode Selection[/bold magenta]\n\n"
                "Choose the type of analysis you want to perform:\n\n"
                "[cyan]Options:[/cyan]\n"
                "  1. [yellow]Standard[/yellow] - Quick analysis with basic bar chart (~5 sec)\n"
                "     • Single bias_report.png visualization\n"
                "     • Console statistics summary\n"
                "     • Perfect for quick checks\n\n"
                "  2. [blue]Advanced[/blue] - Comprehensive analysis with 8+ charts + statistical report (~15 sec)\n"
                "     • comprehensive_dashboard.png (multi-panel overview)\n"
                "     • 7 additional professional charts\n"
                "     • statistical_report.md (full statistical analysis)\n"
                "     • Effect sizes, t-tests, confidence intervals\n"
                "     • Perfect for presentations and research\n\n"
                "[dim]Tip: Use --advanced flag to skip this prompt next time[/dim]",
                border_style="magenta",
            )
        )

        console.print("\n[bold]Select analysis mode (1 or 2):[/bold]")
        while True:
            choice = typer.prompt("", default="1")
            if choice in ["1", "2"]:
                if choice == "2":
                    advanced = True
                    console.print("[green]✓[/green] Advanced analytics selected")
                else:
                    console.print("[green]✓[/green] Standard analytics selected")
                break
            else:
                console.print("[red]Invalid choice. Please enter 1 or 2.[/red]")

    # Step 2: Configuration Review
    results_size = Path(results).stat().st_size / 1024

    analysis_type = "Advanced (Comprehensive)" if advanced else "Standard (Quick)"
    output_desc = (
        "8+ charts + statistical report + dashboard"
        if advanced
        else "bias_report.png + console summary"
    )

    console.print(
        Panel.fit(
            f"[bold green]Step 2: Analysis Configuration[/bold green]\n\n"
            f"[bold]Results File:[/bold] [cyan]{results}[/cyan] ([dim]{results_size:.1f} KB[/dim])\n"
            f"[bold]Analysis Type:[/bold] [cyan]{analysis_type}[/cyan]\n"
            f"[bold]Silent Mode:[/bold] [cyan]{'Enabled' if silent else 'Disabled'}[/cyan]\n"
            f"[bold]Output:[/bold] [cyan]{output_desc}[/cyan]",
            border_style="green",
        )
    )

    # Step 3: Execute Analysis
    analysis_desc = (
        "Generating comprehensive statistical analysis with multiple visualizations..."
        if advanced
        else "Generating statistical summary and visualization chart..."
    )
    output_note = (
        "[dim]This will create 8+ charts, statistical report, and comprehensive dashboard.[/dim]"
        if advanced
        else "[dim]This will create bias_report.png and display metrics in the console.[/dim]"
    )

    console.print(
        Panel.fit(
            f"[bold yellow]Step 3: Executing Bias Analysis[/bold yellow]\n"
            f"{analysis_desc}\n"
            f"{output_note}",
            border_style="yellow",
        )
    )

    try:
        # Import the unified analytics module
        # Add project root to path if not already there
        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from Phase3_Analysis.analytics import BiasAnalytics

        # Run analysis using the unified analytics class
        analytics = BiasAnalytics(str(results))

        if advanced:
            # Generate comprehensive analysis with HTML report and AI insights
            analytics.run_complete_analysis(
                generate_html=True,
                generate_ai_insights=True
            )
        else:
            # Generate basic analysis only
            analytics.run_complete_analysis(
                generate_html=False,
                generate_ai_insights=False
            )

        # Show success message
        if not silent:
            console.print(
                "[green]✓[/green] Analysis completed successfully"
            )

        # Check for generated files based on mode
        if results:
            results_path = Path(results)
            results_dir = results_path.parent
        else:
            results_dir = Path(".")

        # List of expected output files
        if advanced:
            expected_files = [
                "comprehensive_dashboard.png",
                "violin_plot_distribution.png",
                "box_plot_profession.png",
                "heatmap_bias_matrix.png",
                "scatter_correlations.png",
                "effect_sizes_cohens_d.png",
                "time_series_progression.png",
                "statistical_report.md",
            ]
        else:
            expected_files = ["bias_report.png"]

        # Find generated files
        generated_files = []
        for filename in expected_files:
            filepath = results_dir / filename
            if filepath.exists():
                generated_files.append(filepath)

        # Enhanced success panel with file information
        success_message = (
            "[bold green]✅ Analysis Completed Successfully![/bold green]\n\n"
        )
        success_message += (
            f"[bold]Analyzed File:[/bold] [cyan]{Path(results).name}[/cyan]\n"
        )
        success_message += (
            f"[bold]Analysis Type:[/bold] [cyan]{'Advanced' if advanced else 'Standard'}[/cyan]\n"
        )
        success_message += (
            f"[bold]Output Directory:[/bold] [cyan]{results_dir}[/cyan]\n"
        )

        success_message += "\n[bold]Generated Files:[/bold]\n"

        if generated_files:
            for filepath in generated_files:
                file_size = filepath.stat().st_size / 1024
                success_message += f"  • [green]✓[/green] [cyan]{filepath.name}[/cyan] ([dim]{file_size:.1f} KB[/dim])\n"
        else:
            success_message += "  • [yellow]?[/yellow] Files generated (check output directory)\n"

        if not advanced:
            success_message += (
                "  • [green]✓[/green] [cyan]Console statistics[/cyan] - Detailed analysis\n"
            )

        success_message += "\n[bold]Next Steps:[/bold]\n"

        if advanced:
            success_message += "  • Open [cyan]comprehensive_dashboard.png[/cyan] for overview\n"
            success_message += "  • Read [cyan]statistical_report.md[/cyan] for detailed findings\n"
            success_message += "  • Explore individual charts for specific insights\n"
        else:
            if generated_files:
                success_message += f"  • Open [cyan]{generated_files[0].name}[/cyan] to view bias metrics\n"
            success_message += "  • Review console statistics for detailed analysis\n"
            success_message += "  • Use [cyan]--advanced[/cyan] flag for comprehensive analysis\n"

        success_message += "  • Share results with your team or faculty\n"
        success_message += "  • Use [cyan]--silent[/cyan] flag if you see Unicode encoding errors"

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
def gui():
    """🖥️ Launch web-based Gradio GUI"""
    try:
        from equilens.web_ui import main as web_ui_main

        console.print("🚀 [green]Starting EquiLens Web Interface...[/green]")
        console.print("🌐 Opening web interface in your browser...")
        web_ui_main()
    except ImportError:
        console.print("[red]❌ Gradio dependencies not available[/red]")
        console.print("Install with: [cyan]uv add gradio[/cyan]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Failed to start Web GUI: {e}[/red]")
        raise typer.Exit(1) from e


@audit_app.command("list")
def resume_list(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Filter by model name")
    ] = None,
):
    """📋 List all available resume sessions with detailed information"""
    try:
        interrupted_sessions = find_interrupted_sessions(model)

        if not interrupted_sessions:
            console.print("[yellow]📭 No interrupted audit sessions found.[/yellow]")
            return

        console.print(
            f"\n[bold cyan]📋 Available Resume Sessions ({len(interrupted_sessions)} total):[/bold cyan]\n"
        )

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
            console.print(
                f"[bold green]{i:2d}.[/bold green] [cyan]{session_model}[/cyan] - {completed:,}/{total:,} tests ({completion_percent:.1f}% complete)"
            )
            console.print(f"     [dim]Session ID: {session_id}[/dim]")
            console.print(f"     [dim]Started: {time_str}[/dim]")
            console.print(f"     [dim]Last Save: {checkpoint_str}[/dim]")
            console.print(f"     [dim]Folder: {folder_id} ({folder_size})[/dim]")

            # Progress details
            if failed > 0:
                success_rate = (
                    ((completed - failed) / completed * 100) if completed > 0 else 0
                )
                console.print(
                    f"     [dim]Progress: {completed:,} completed, {failed:,} failed ({success_rate:.1f}% success)[/dim]"
                )
            else:
                console.print(
                    f"     [dim]Progress: {completed:,} completed, 0 failed (100% success)[/dim]"
                )

            # Performance metrics
            if avg_response_time > 0:
                console.print(
                    f"     [dim]Performance: {avg_response_time:.1f}s avg, {throughput:.2f} tests/sec[/dim]"
                )
                console.print(f"     [dim]Est. Time Remaining: {eta_str}[/dim]")

            # Show backup information if available
            try:
                backup_dir = Path(folder_path) / f"{session_id}_backups"
                if backup_dir.exists():
                    backup_files = list(backup_dir.glob("progress_backup_*.json"))
                    if backup_files:
                        # Get latest backup info
                        latest_backup = max(
                            backup_files, key=lambda x: x.stat().st_mtime
                        )
                        backup_time = datetime.fromtimestamp(
                            latest_backup.stat().st_mtime
                        )
                        console.print(
                            f"     [dim]Backups: {len(backup_files)} available (latest: {backup_time.strftime('%H:%M:%S')})[/dim]"
                        )
            except Exception:
                pass

            console.print(f"     [dim]Path: {folder_path}[/dim]")
            console.print()

    except Exception as e:
        console.print(f"[red]❌ Error listing resume sessions: {e}[/red]")
        raise typer.Exit(1) from e


@audit_app.command("remove")
def resume_remove(
    identifiers: Annotated[
        list[str],
        typer.Argument(
            help="Session indices (1,2,3...), session IDs, or folder names to remove (space-separated)"
        ),
    ],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompts")
    ] = False,
):
    """🗑️ Remove specific resume sessions by index number, session ID, or folder name

    Examples:
        uv run equilens audit remove 1 3 5          # Remove sessions 1, 3, and 5 from the list
        uv run equilens audit remove folder_name    # Remove by folder name
        uv run equilens audit remove session_id     # Remove by session ID (if unique)
    """
    try:
        interrupted_sessions = find_interrupted_sessions()

        if not interrupted_sessions:
            console.print("[yellow]📭 No interrupted audit sessions found.[/yellow]")
            return

        # Build mappings for different identifier types
        index_map = {}  # index -> progress_file
        session_map = {}  # session_id -> [progress_files] (can be multiple)
        folder_map = {}  # folder_name -> progress_file
        session_details = {}  # progress_file -> progress_data

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
                console.print(
                    f"[yellow]⚠️ Identifier '{identifier}' not found. Use 'audit list' to see available sessions.[/yellow]"
                )
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
                    console.print("\n[yellow]🗑️ Session to remove:[/yellow]")
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
            console.print("\n[yellow]📭 No sessions were removed.[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error removing resume sessions: {e}[/red]")
        raise typer.Exit(1) from e
@audit_app.command("remove-range")
def resume_remove_range(
    range_spec: Annotated[
        str, typer.Argument(help="Range specification like '1-5', '1,3,5-8', or 'all'")
    ],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompts")
    ] = False,
):
    """🗑️ Remove multiple resume sessions by range specification

    Examples:
        uv run equilens audit remove-range "1-5"       # Remove sessions 1 through 5
        uv run equilens audit remove-range "1,3,5"     # Remove sessions 1, 3, and 5
        uv run equilens audit remove-range "1-3,7-9"   # Remove sessions 1-3 and 7-9
        uv run equilens audit remove-range "all"       # Remove all sessions
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

        for progress_file, _progress_data in sessions_to_remove:
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


@audit_app.command("clean")
def resume_clean(
    keep: Annotated[
        int, typer.Option("--keep", "-k", help="Number of most recent sessions to keep")
    ] = 5,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Skip confirmation prompt")
    ] = False,
):
    """🧹 Clean up old resume sessions, keeping only the most recent ones"""
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
def _run_auto_analytics(analytics_preference: str, latest_results: Path):
    """
    Run automatic analytics after audit completes, based on user preference.
    """
    console.print("\n" + "=" * 70)
    console.print(
        Panel.fit(
            f"[bold cyan]🚀 Auto-Running {analytics_preference.title()} Analytics[/bold cyan]\n\n"
            f"[dim]As requested during setup, running {analytics_preference} analysis...[/dim]",
            border_style="cyan",
        )
    )

    try:
        # Import the unified analytics module
        # Add project root to path if not already there
        import sys
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from Phase3_Analysis.analytics import BiasAnalytics

        # Run analysis using the unified analytics class
        analytics = BiasAnalytics(str(latest_results))

        if analytics_preference == "advanced":
            # Generate comprehensive analysis with HTML report
            analytics.run_complete_analysis(
                generate_html=True,
                generate_ai_insights=True
            )
        else:
            # Generate basic analysis only
            analytics.run_complete_analysis(
                generate_html=False,
                generate_ai_insights=False
            )

        console.print(
            Panel.fit(
                f"[bold green]✅ {analytics_preference.title()} Analysis Complete![/bold green]\n\n"
                f"[bold]Analytics files saved to:[/bold] [cyan]{latest_results.parent}/[/cyan]\n\n"
                f"{'[bold]Generated:[/bold] 8+ charts + statistical_report.md' if analytics_preference == 'advanced' else '[bold]Generated:[/bold] bias_report.png'}",
                border_style="green",
            )
        )
    except subprocess.CalledProcessError as e:
        console.print(
            f"[yellow]⚠️  Auto-analysis failed (exit code: {e.returncode})[/yellow]"
        )
        console.print(
            "[dim]You can run analysis manually later using: uv run equilens analyze[/dim]"
        )
    except Exception as e:
        console.print(f"[yellow]⚠️  Auto-analysis error: {e}[/yellow]")
        console.print(
            "[dim]You can run analysis manually later using: uv run equilens analyze[/dim]"
        )

if __name__ == "__main__":
if __name__ == "__main__":
    app()
