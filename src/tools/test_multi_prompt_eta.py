"""
Test the new multi-prompt ETA estimation system
"""

import sys
from pathlib import Path

# Add src to path to import equilens modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from equilens.cli import estimate_corpus_eta, measure_average_request_time

console = Console()


def test_multi_prompt_eta():
    """Test the new multi-prompt ETA estimation system"""
    console.print(
        "\n[bold green]🎯 Testing Multi-Prompt ETA Estimation System[/bold green]"
    )
    console.print(
        "Tests 5 different prompts, tracks all time spent, averages successful requests\n"
    )

    model = "llama2:latest"

    # Test the multi-prompt timing measurement
    console.print(f"[bold]🔬 Testing average timing measurement for {model}:[/bold]")
    timing_data = measure_average_request_time(model, num_tests=5)

    console.print("\n[bold]📊 Results Summary:[/bold]")
    if timing_data["success"]:
        console.print(
            f"  Average Response Time: [cyan]{timing_data['average_time']:.1f}s[/cyan]"
        )
        console.print(
            f"  Successful Tests: [green]{timing_data['successful_count']}/{timing_data['successful_count'] + timing_data['failed_count']}[/green]"
        )
        console.print(
            f"  Total Time Spent: [yellow]{timing_data['total_time_spent']:.1f}s[/yellow]"
        )
        console.print(
            f"  Buffered Time (1.4x): [yellow]{timing_data['average_time'] * 1.4:.1f}s[/yellow]"
        )
    else:
        console.print("  [red]All tests failed[/red]")
        console.print(
            f"  Total Time Spent: [yellow]{timing_data['total_time_spent']:.1f}s[/yellow]"
        )

    # Test ETA estimation with the timing data
    console.print("\n[bold]🎯 Testing ETA estimation with timing data:[/bold]")

    corpus_files = [
        ("Quick Test Corpus", "../Phase1_CorpusGenerator/corpus/quick_test_corpus.csv")
    ]

    for name, path in corpus_files:
        try:
            eta = estimate_corpus_eta(path, model, timing_data)
            if "error" not in eta and eta.get("single_request_time") is not None:
                console.print(f"\n[bold]{name}:[/bold]")
                console.print(f"  📊 Test count: [cyan]{eta['test_count']}[/cyan]")
                console.print(
                    f"  ⏱️  Average request time: [dim]{eta['single_request_time']}s[/dim]"
                )
                console.print(
                    f"  🛡️  Buffered per test: [yellow]{eta['buffered_time_per_test']}s[/yellow]"
                )
                console.print(
                    f"  🎯 [bold]Total ETA: [green]{eta['formatted']}[/green][/bold]"
                )

                stats = eta.get("timing_stats", {})
                console.print(
                    f"  📈 Timing: {stats.get('successful_tests', 0)} successful, {stats.get('failed_tests', 0)} failed"
                )
            else:
                console.print(
                    f"\n[bold]{name}:[/bold] [red]{eta.get('formatted', 'Error loading')}[/red]"
                )
                if "timing_stats" in eta:
                    stats = eta["timing_stats"]
                    console.print(
                        f"  Failed tests: {stats.get('failed_tests', 0)}, Time spent: {stats.get('total_measurement_time', 0)}s"
                    )
        except Exception as e:
            console.print(f"\n[bold]{name}:[/bold] [red]Error: {e}[/red]")

    console.print("\n[bold green]✅ Multi-prompt testing completed![/bold green]")


if __name__ == "__main__":
    test_multi_prompt_eta()
