"""Docker management and container orchestration"""

import subprocess
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .ollama_config import get_ollama_url

console = Console()


class DockerManager:
    """Manages Docker containers and services for EquiLens"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.compose_file = project_root / "docker-compose.yml"
        self.ollama_container = "equilens-ollama"
        self.app_container = "equilens-app"
        self.ollama_url = get_ollama_url()

    def _run_command(
        self, cmd: list[str], capture_output: bool = False, silent_fail: bool = False
    ) -> subprocess.CompletedProcess:
        """Run command with proper error handling"""
        try:
            return subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=False,
                cwd=self.project_root,
            )
        except FileNotFoundError:
            if silent_fail:
                return subprocess.CompletedProcess(
                    cmd, 1, "", f"Command not found: {cmd[0]}"
                )
            console.print(f"[red]❌ Command not found: {cmd[0]}[/red]")
            console.print("Please ensure Docker Desktop is installed and running.")
            raise

    def check_docker(self) -> bool:
        """Check if Docker is available and running"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Checking Docker availability...", total=None)

            # Check Docker installation
            result = self._run_command(["docker", "--version"], capture_output=True)
            if result.returncode != 0:
                console.print("[red]❌ Docker not found.[/red]")
                console.print(
                    "   Windows: https://docs.docker.com/desktop/install/windows-install/"
                )
                return False

            # Check Docker Compose
            result = self._run_command(
                ["docker", "compose", "version"], capture_output=True
            )
            if result.returncode != 0:
                console.print("[red]❌ Docker Compose not found.[/red]")
                return False

            # Check if Docker daemon is running
            result = self._run_command(["docker", "ps"], capture_output=True)
            if result.returncode != 0:
                console.print("[red]❌ Docker daemon not running.[/red]")
                return False

            progress.update(task, description="✅ Docker is ready")

        return True

    def detect_existing_ollama(self) -> str | None:
        """Detect existing Ollama containers and prefer running ones"""
        console.print("🔍 Checking for existing Ollama containers...")

        # First check for running containers
        result = self._run_command(
            [
                "docker",
                "ps",
                "--filter",
                "ancestor=ollama/ollama",
                "--format",
                "{{.Names}}\t{{.Ports}}\t{{.Status}}",
            ],
            capture_output=True,
        )

        running_containers = []
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if line and "11434" in line:
                    parts = line.split("\t")
                    name = parts[0]
                    ports = parts[1] if len(parts) > 1 else ""

                    # Test connectivity to ensure it's actually working
                    if self._test_ollama_connection(self.ollama_url):
                        console.print(f"✅ Found working Ollama container: {name}")
                        console.print(f"   Ports: {ports}")
                        return name
                    else:
                        running_containers.append(name)

        # Check for stopped containers we could restart
        result = self._run_command(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "ancestor=ollama/ollama",
                "--filter",
                "status=exited",
                "--format",
                "{{.Names}}\t{{.Status}}",
            ],
            capture_output=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    name = parts[0]
                    console.print(f"🔄 Found stopped Ollama container: {name}")

                    # Try to restart it
                    restart_result = self._run_command(
                        ["docker", "restart", name], capture_output=True
                    )

                    if restart_result.returncode == 0:
                        console.print(f"✅ Restarted Ollama container: {name}")
                        # Wait a moment for it to start
                        time.sleep(3)
                        if self._test_ollama_connection(self.ollama_url):
                            console.print(f"✅ Container {name} is now accessible")
                            return name
                        else:
                            console.print(
                                f"⚠️  Container {name} restarted but not accessible yet"
                            )

        if running_containers:
            console.print(
                f"⚠️  Found running containers but port 11434 not accessible: {', '.join(running_containers)}"
            )

        console.print(
            "ℹ️  No existing accessible Ollama containers found - will create new one"
        )
        return None

    def _test_ollama_connection(self, url: str, timeout: int = 5) -> bool:
        """Test Ollama API connectivity"""
        try:
            response = requests.get(f"{url}/api/version", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def start_services(self, gpu_available: bool = False) -> bool:
        """Start EquiLens services with GPU detection"""
        console.print("🚀 [bold]Starting EquiLens services...[/bold]")

        if not self.check_docker():
            return False

        if gpu_available:
            console.print("🎮 GPU acceleration detected - using GPU mode!")
        else:
            console.print("⚡ GPU not available - using CPU mode")

        # TODO: Future enhancement - EquiLens app container support
        # Check if our services are already running
        # if self._is_service_running(self.app_container):
        #     console.print("ℹ️  EquiLens app already running")
        #     return True

        # Check for existing external Ollama
        existing_ollama = self.detect_existing_ollama()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if existing_ollama and existing_ollama != self.ollama_container:
                console.print(f"✅ Using existing Ollama container: {existing_ollama}")
                console.print(
                    "🎯 No additional services needed - Ollama is already accessible!"
                )

                # TODO: Future enhancement - EquiLens app container integration
                # Check if the app container exists
                # if not self._is_service_running(self.app_container):
                #     # Only start our app container if needed (future enhancement)
                console.print(
                    "ℹ️  EquiLens app container not needed - direct Ollama access available"
                )

                return True
            else:
                task = progress.add_task("Starting Ollama service...", total=None)
                # TODO: Future enhancement - full docker-compose with app container
                # For now, only start Ollama container
                result = self._run_command(["docker", "compose", "up", "-d", "ollama"])

            if result.returncode == 0:
                progress.update(task, description="✅ Services started successfully!")
                return self._wait_for_services()
            else:
                progress.update(task, description="❌ Failed to start services")
                console.print(f"Error: {getattr(result, 'stderr', 'Unknown error')}")
                return False

    def _is_service_running(self, container_name: str) -> bool:
        """Check if a specific container is running"""
        result = self._run_command(
            [
                "docker",
                "ps",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
        )
        return container_name in result.stdout

    def _wait_for_services(self, max_wait: int = 60) -> bool:
        """Wait for services to be ready"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Waiting for services to initialize...", total=None
            )

            for i in range(max_wait):
                if self._test_ollama_connection(self.ollama_url, timeout=2):
                    progress.update(task, description="✅ Ollama is ready")
                    time.sleep(2)  # Give app container time to start
                    return True

                if i % 5 == 0:  # Update progress every 5 seconds
                    progress.update(task, description=f"Waiting... ({i}/{max_wait}s)")
                time.sleep(1)

            progress.update(task, description="❌ Services failed to start")
            return False

    def stop_services(self) -> bool:
        """Stop EquiLens services"""
        console.print("🛑 [bold]Stopping EquiLens services...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Stopping services...", total=None)
            result = self._run_command(["docker", "compose", "down"])

            if result.returncode == 0:
                progress.update(task, description="✅ Services stopped successfully")
                return True
            else:
                progress.update(task, description="❌ Failed to stop services")
                return False

    def run_in_container(
        self, command: list[str], interactive: bool = False
    ) -> subprocess.CompletedProcess:
        """Execute command inside the EquiLens app container"""
        # TODO: Future enhancement - app container support
        console.print("[yellow]⚠️  EquiLens app container not implemented yet.[/yellow]")
        console.print("[cyan]💡 Commands run directly on host for now.[/cyan]")
        raise RuntimeError("App container not implemented yet")

        # Future implementation:
        # if not self._is_service_running(self.app_container):
        #     console.print(
        #         "[red]❌ EquiLens app container not running.[/red] Start with: [cyan]uv run equilens start[/cyan]"
        #     )
        #     raise RuntimeError("Container not running")
        #
        # docker_cmd = ["docker", "exec"]
        # if interactive:
        #     docker_cmd.extend(["-it"])
        #
        # docker_cmd.extend([self.app_container] + command)
        # return self._run_command(docker_cmd)

    def get_service_status(self) -> dict:
        """Get comprehensive service status"""
        status = {
            "docker_available": self.check_docker(),
            "ollama_accessible": False,
            "containers": [],
            "volumes": [],
        }

        if status["docker_available"]:
            # Check Ollama API
            status["ollama_accessible"] = self._test_ollama_connection(self.ollama_url)

            # Check containers (commenting out app container for now)
            result = self._run_command(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=ollama",  # Only check Ollama containers
                    "--format",
                    "{{.Names}}\t{{.Status}}\t{{.Ports}}",
                ],
                capture_output=True,
            )

            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            status["containers"].append(
                                {"name": parts[0], "status": parts[1]}
                            )

            # Check volumes
            result = self._run_command(
                [
                    "docker",
                    "volume",
                    "ls",
                    "--filter",
                    "name=ollama_data",
                    "--format",
                    "{{.Name}}",
                ],
                capture_output=True,
            )

            if "ollama_data" in result.stdout:
                status["volumes"].append("ollama_data")

        return status
