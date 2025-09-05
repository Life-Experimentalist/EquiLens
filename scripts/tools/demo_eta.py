"""
Demo script to show the new ETA estimation system in action
"""

import sys
from pathlib import Path

# Add src to path to import equilens modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from equilens.cli import estimate_corpus_eta

console = Console()


def demo_eta_display():
    """Demo the ETA display for different corpus files"""
    console.print("\n[bold green]🕒 EquiLens ETA Estimation Demo[/bold green]")
    console.print("Real-time estimation using 1.4x buffer of actual request timing\n")

    corpus_files = [
        ("Quick Test Corpus", "../Phase1_CorpusGenerator/corpus/quick_test_corpus.csv"),
        ("Standard Test Corpus", "../Phase1_CorpusGenerator/corpus/test_corpus.csv"),
    ]

    model = "llama2:latest"
    console.print(f"[bold]Model:[/bold] [cyan]{model}[/cyan]\n")

    for name, path in corpus_files:
        try:
            eta = estimate_corpus_eta(path, model)
            if "error" not in eta:
                console.print(f"[bold]{name}:[/bold]")
                console.print(f"  📊 Test count: [cyan]{eta['test_count']}[/cyan]")
                console.print(
                    f"  ⏱️  Single request: [dim]{eta['single_request_time']}s[/dim]"
                )
                console.print(
                    f"  🛡️  Buffered per test: [yellow]{eta['buffered_time_per_test']}s[/yellow]"
                )
                console.print(
                    f"  🎯 [bold]Total ETA: [green]{eta['formatted']}[/green][/bold]"
                )
                console.print()
            else:
                console.print(f"[bold]{name}:[/bold] [red]Error loading[/red]")
                console.print()
        except Exception as e:
            console.print(f"[bold]{name}:[/bold] [red]Error: {e}[/red]")
            console.print()


if __name__ == "__main__":
    demo_eta_display()
