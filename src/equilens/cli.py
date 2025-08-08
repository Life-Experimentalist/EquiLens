"""
Modern CLI interface for EquiLens using Typer and Rich

A comprehensive command-line interface for the EquiLens AI bias detection platform.
Features interactive commands, beautiful output formatting, and comprehensive help.
"""

import subprocess
from pathlib import Path
from typing import Annotated

import requests
import typer
from rich.console import Console
from rich.panel import Panel

from equilens.core.manager import EquiLensManager

console = Console()


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
    no_args_is_help=True,
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
    • `uv run equilens-web` - Start web interface (future)
    """
    if version:
        from equilens import __version__

        console.print(f"EquiLens version {__version__}")
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
    silent: Annotated[
        bool,
        typer.Option(
            "--silent",
            "-s",
            help="Suppress subprocess output to avoid emoji encoding errors",
        ),
    ] = False,
):
    """🔍 Run bias audit with interactive prompts and enhanced visual design"""

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

    # Step 2: Corpus Selection
    if corpus is None:
        console.print(
            Panel.fit(
                "[bold blue]Step 2: Corpus Selection[/bold blue]\n"
                "Choose a test corpus file containing bias evaluation prompts.",
                border_style="blue",
            )
        )

        # Look for common corpus files
        common_paths = [
            "quick_test_corpus.csv",
            "src/Phase1_CorpusGenerator/corpus/test_corpus.csv",
            "test_corpus.csv",
        ]

        found_files = []
        for path in common_paths:
            if Path(path).exists():
                file_size = Path(path).stat().st_size
                found_files.append((path, file_size))

        if found_files:
            console.print("\n[green]✓ Found corpus files:[/green]")
            for i, (path, size) in enumerate(found_files, 1):
                size_kb = size / 1024
                console.print(
                    f"  {i}. [cyan]{path}[/cyan] ([dim]{size_kb:.1f} KB[/dim])"
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

    # Validate inputs
    assert model is not None, "Model should not be None at this point"
    assert corpus is not None, "Corpus should not be None at this point"

    if not Path(corpus).exists():
        console.print(f"[red]❌ Corpus file not found: {corpus}[/red]")
        raise typer.Exit(1)

    # Step 3: Configuration Review
    corpus_size = Path(corpus).stat().st_size / 1024
    console.print(
        Panel.fit(
            f"[bold green]Step 3: Configuration Review[/bold green]\n\n"
            f"[bold]Model:[/bold] [cyan]{model}[/cyan]\n"
            f"[bold]Corpus:[/bold] [cyan]{corpus}[/cyan] ([dim]{corpus_size:.1f} KB[/dim])\n"
            f"[bold]Output Directory:[/bold] [cyan]{output_dir}[/cyan]\n"
            f"[bold]Silent Mode:[/bold] [cyan]{'Enabled' if silent else 'Disabled'}[/cyan]",
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
        # Configure subprocess parameters based on silent mode
        if silent:
            # Redirect both stdout and stderr to suppress emoji encoding errors
            # This prevents the Unicode errors from cluttering output
            subprocess.run(
                [
                    "python",
                    "src/Phase2_ModelAuditor/audit_model.py",
                    "--model",
                    model,
                    "--corpus",
                    corpus,
                    "--output-dir",
                    output_dir,
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
                "[green]✓[/green] Audit process completed successfully (output suppressed)"
            )
        else:
            # Normal execution with full output but with proper encoding handling
            try:
                subprocess.run(
                    [
                        "python",
                        "src/Phase2_ModelAuditor/audit_model.py",
                        "--model",
                        model,
                        "--corpus",
                        corpus,
                        "--output-dir",
                        output_dir,
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
                        "src/Phase2_ModelAuditor/audit_model.py",
                        "--model",
                        model,
                        "--corpus",
                        corpus,
                        "--output-dir",
                        output_dir,
                    ],
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


if __name__ == "__main__":
    app()
