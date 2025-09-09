"""Main EquiLens manager that orchestrates all components"""

import json
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .docker import DockerManager
from .gpu import GPUManager

console = Console()


class EquiLensManager:
    """Main EquiLens manager that coordinates all operations"""

    def __init__(self, project_root: Path | None = None):
        if project_root is None:
            # Try to find project root by looking for pyproject.toml
            current = Path.cwd()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()

        self.project_root = project_root
        self.gpu_manager = GPUManager()
        self.docker_manager = DockerManager(project_root)

    def check_system_status(self) -> dict[str, Any]:
        """Check overall system status"""
        console.print("🔍 [bold]Checking EquiLens system status...[/bold]")

        # Check GPU support
        gpu_info = self.gpu_manager.check_gpu_support()

        # Check Docker services
        docker_status = self.docker_manager.get_service_status()

        return {
            "gpu": gpu_info,
            "docker": docker_status,
            "recommendation": self.gpu_manager.get_performance_recommendation(),
        }

    def display_status(self) -> None:
        """Display comprehensive system status"""
        status = self.check_system_status()

        # Create main status panel
        console.print(
            Panel.fit(
                "📊 EquiLens Service Status", style="bold blue", border_style="blue"
            )
        )

        # GPU Status
        self.gpu_manager.display_gpu_status()

        # Docker Status
        self._display_docker_status(status["docker"])

        # Models Status (if Ollama is accessible)
        if status["docker"]["ollama_accessible"]:
            self._display_models_status()

        # Performance Recommendation
        console.print(f"\n💡 {status['recommendation']}")

        # Quick Commands
        self._display_quick_commands(status["docker"]["ollama_accessible"])

    def _display_docker_status(self, docker_status: dict) -> None:
        """Display Docker service status"""
        table = Table(title="🐳 Docker Services")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")

        # Ollama API status
        ollama_status = (
            "🟢 Ready" if docker_status["ollama_accessible"] else "🔴 Not accessible"
        )
        table.add_row("Ollama API", ollama_status, "http://localhost:11434")

        # Container status
        if docker_status["containers"]:
            for container in docker_status["containers"]:
                table.add_row(
                    f"Container: {container['name']}",
                    "🟢 Running",
                    container["status"],
                )
        else:
            table.add_row(
                "Containers", "🔴 None running", "Start with: uv run equilens start"
            )

        # Storage status
        if docker_status["volumes"]:
            table.add_row("Storage", "🟢 Ready", "Model volume exists")
        else:
            table.add_row("Storage", "🟡 Pending", "Will be created on first start")

        console.print(table)

    def _display_models_status(self) -> None:
        """Display available models status"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                table = Table(title="🎯 Available Models")
                table.add_column("Model", style="cyan")
                table.add_column("Size", style="magenta")

                if models:
                    for model in models[:5]:  # Show first 5 models
                        size_gb = model.get("size", 0) / (1024**3)
                        table.add_row(model["name"], f"{size_gb:.1f}GB")

                    if len(models) > 5:
                        table.add_row("...", f"and {len(models) - 5} more")
                else:
                    table.add_row(
                        "No models", "Install with: uv run equilens models pull <model>"
                    )

                console.print(table)
            else:
                console.print("[yellow]🎯 Models: Unable to retrieve list[/yellow]")
        except Exception:
            console.print("[red]🎯 Models: Connection failed[/red]")

    def _display_quick_commands(self, ollama_accessible: bool) -> None:
        """Display quick command suggestions"""
        console.print("\n💡 [bold]Quick Commands:[/bold]")
        if not ollama_accessible:
            console.print(
                "  [cyan]uv run equilens start[/cyan]          # Start services"
            )
        else:
            console.print("  [cyan]uv run equilens models list[/cyan]    # List models")
            console.print("  [cyan]uv run equilens audit[/cyan]   # Run audit")
            console.print(
                "  [cyan]uv run equilens gui[/cyan]            # Interactive Web GUI"
            )
        console.print(
            "  [cyan]uv run equilens --help[/cyan]         # Show all commands"
        )

    def start_services(self) -> bool:
        """Start all EquiLens services"""
        gpu_info = self.gpu_manager.check_gpu_support()
        return self.docker_manager.start_services(gpu_info["gpu_available"])

    def stop_services(self) -> bool:
        """Stop all EquiLens services"""
        return self.docker_manager.stop_services()

    def list_models(self) -> None:
        """List available Ollama models"""
        console.print("📋 [bold]Available Ollama models:[/bold]")

        # Try direct API call first
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                if models:
                    table = Table()
                    table.add_column("Model", style="cyan")
                    table.add_column("Size", style="magenta")
                    table.add_column("Modified", style="green")

                    for model in models:
                        size_gb = model.get("size", 0) / (1024**3)
                        modified = model.get("modified_at", "Unknown")
                        table.add_row(model["name"], f"{size_gb:.1f}GB", modified)

                    console.print(table)
                else:
                    console.print("  [yellow]No models installed[/yellow]")
                return
        except Exception as e:
            console.print(f"[red]Failed to connect to Ollama: {e}[/red]")

        # Fallback to container execution
        try:
            result = self.docker_manager.run_in_container(
                ["curl", "-s", "http://ollama:11434/api/tags"]
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                models = data.get("models", [])
                if models:
                    for model in models:
                        size_gb = model.get("size", 0) / (1024**3)
                        console.print(f"  • {model['name']} ({size_gb:.1f}GB)")
                else:
                    console.print("  [yellow]No models installed[/yellow]")
            else:
                console.print("  [red]❌ Failed to connect to Ollama[/red]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def pull_model(self, model_name: str) -> bool:
        """Download an Ollama model"""
        console.print(f"📥 [bold]Downloading model: {model_name}[/bold]")
        console.print("   [yellow]This may take several minutes...[/yellow]")

        # Try direct API call first
        try:
            response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name},
                timeout=300,  # 5 minutes timeout
            )
            if response.status_code == 200:
                console.print(
                    f"✅ [green]Model {model_name} downloaded successfully![/green]"
                )
                return True
            else:
                console.print(
                    f"[red]❌ Failed to download {model_name}: {response.text}[/red]"
                )
                return False
        except Exception as e:
            console.print(f"[red]❌ Error downloading {model_name}: {e}[/red]")
            return False

    def run_audit(self, config_file: str) -> bool:
        """Run bias audit"""
        config_path = Path(config_file)
        if not config_path.exists():
            console.print(f"[red]❌ Config file not found: {config_file}[/red]")
            return False

        console.print(f"🔍 [bold]Running bias audit with config: {config_file}[/bold]")

        try:
            result = self.docker_manager.run_in_container(
                ["python", "Phase2_ModelAuditor/audit_model.py", config_file]
            )
            return result.returncode == 0
        except Exception as e:
            console.print(f"[red]❌ Error running audit: {e}[/red]")
            return False

    def generate_corpus(self, config_file: str) -> bool:
        """Generate test corpus"""
        config_path = Path(config_file)
        if not config_path.exists():
            console.print(f"[red]❌ Config file not found: {config_file}[/red]")
            return False

        console.print(f"📝 [bold]Generating corpus with config: {config_file}[/bold]")

        try:
            result = self.docker_manager.run_in_container(
                ["python", "src/Phase1_CorpusGenerator/generate_corpus.py", config_file]
            )
            return result.returncode == 0
        except Exception as e:
            console.print(f"[red]❌ Error generating corpus: {e}[/red]")
            return False

    def analyze_results(self, results_file: str) -> bool:
        """Analyze audit results"""
        results_path = Path(results_file)
        if not results_path.exists():
            console.print(f"[red]❌ Results file not found: {results_file}[/red]")
            return False

        console.print(f"📊 [bold]Analyzing results: {results_file}[/bold]")

        try:
            result = self.docker_manager.run_in_container(
                ["python", "Phase3_Analysis/analyze_results.py", results_file]
            )
            return result.returncode == 0
        except Exception as e:
            console.print(f"[red]❌ Error analyzing results: {e}[/red]")
            return False
