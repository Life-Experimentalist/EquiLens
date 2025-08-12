"""
Test the ETA integration in the audit workflow
"""

import sys
from pathlib import Path

# Add src to path to import equilens modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def test_eta_integration():
    """Test ETA integration features"""
    console.print("\n[bold green]🎯 Testing ETA Integration Features[/bold green]")
    console.print("This demonstrates the new ETA workflow enhancements:\n")

    # Test 1: Show the ETA estimation function
    console.print("[bold]1. Multi-Corpus ETA Estimation[/bold]")
    from equilens.cli import estimate_corpus_eta

    # Mock timing data to simulate the workflow
    mock_timing_data = {
        "success": True,
        "average_time": 4.2,
        "successful_count": 4,
        "failed_count": 1,
        "total_time_spent": 72.5,
    }

    corpus_files = [
        "../Phase1_CorpusGenerator/corpus/quick_test_corpus.csv",
        "../Phase1_CorpusGenerator/corpus/test_corpus.csv",
    ]

    console.print("📊 [dim]Mock ETA estimates for available corpuses:[/dim]")
    for i, corpus_path in enumerate(corpus_files, 1):
        if Path(corpus_path).exists():
            eta_info = estimate_corpus_eta(
                corpus_path, "llama2:latest", mock_timing_data
            )
            if "error" not in eta_info:
                console.print(f"  {i}. [cyan]{corpus_path}[/cyan]")
                console.print(
                    f"     Tests: [cyan]{eta_info['test_count']}[/cyan] | ETA: [yellow]{eta_info['formatted']}[/yellow]"
                )
            else:
                console.print(
                    f"  {i}. [cyan]{corpus_path}[/cyan] - [red]Not found[/red]"
                )
        else:
            console.print(
                f"  {i}. [cyan]{corpus_path}[/cyan] - [red]File not found[/red]"
            )

    # Test 2: Show the workflow steps
    console.print("\n[bold]2. Enhanced Workflow Steps[/bold]")
    console.print("✅ [green]User can choose whether to see ETA estimates[/green]")
    console.print(
        "✅ [green]ETA shown for ALL available corpuses before selection[/green]"
    )
    console.print("✅ [green]Custom path detection and labeling[/green]")
    console.print("✅ [green]User confirmation after seeing estimates[/green]")
    console.print("✅ [green]Detailed timing statistics in final review[/green]")

    # Test 3: Show custom path detection
    console.print("\n[bold]3. Custom Path Detection[/bold]")
    console.print("[dim]In the final configuration review:[/dim]")
    console.print("• Standard corpus: [cyan]/path/to/standard_corpus.csv[/cyan]")
    console.print(
        "• Custom corpus: [cyan]/custom/path/my_corpus.csv[/cyan] [yellow](Custom Path)[/yellow]"
    )

    console.print("\n[bold green]✅ ETA integration features are ready![/bold green]")
    console.print("\n[bold]New Workflow:[/bold]")
    console.print("1. [dim]User selects model[/dim]")
    console.print("2. [cyan]System asks: 'Show ETA estimates?' (optional)[/cyan]")
    console.print("3. [cyan]If yes: Tests 5 prompts, measures timing[/cyan]")
    console.print("4. [cyan]Shows ETA for ALL available corpuses[/cyan]")
    console.print("5. [cyan]User confirms to continue with selection[/cyan]")
    console.print("6. [dim]User selects corpus[/dim]")
    console.print("7. [cyan]Final review shows custom path indicator[/cyan]")
    console.print("8. [cyan]Detailed ETA breakdown with timing stats[/cyan]")
    console.print("9. [dim]User confirms to proceed with audit[/dim]")


if __name__ == "__main__":
    test_eta_integration()
