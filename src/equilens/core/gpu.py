"""GPU detection and management functionality"""

import re
import subprocess
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


class GPUManager:
    """Manages GPU detection and CUDA support checking"""

    def __init__(self):
        self.gpu_info: dict[str, Any] = {}

    def _run_command(
        self, cmd: list[str], silent_fail: bool = True
    ) -> subprocess.CompletedProcess:
        """Run command with proper error handling"""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            if silent_fail:
                return subprocess.CompletedProcess(
                    cmd, 1, "", f"Command not found: {cmd[0]}"
                )
            raise

    def check_gpu_support(self) -> dict[str, Any]:
        """Check GPU and CUDA availability"""
        gpu_info = {
            "nvidia_driver": False,
            "cuda_runtime": False,
            "docker_gpu": False,
            "gpu_available": False,
            "gpu_details": {},
        }

        try:
            # Check NVIDIA driver and CUDA runtime from nvidia-smi
            result = self._run_command(["nvidia-smi"], silent_fail=True)
            if result.returncode == 0:
                gpu_info["nvidia_driver"] = True
                gpu_details = self._parse_nvidia_smi(result.stdout)
                gpu_info["gpu_details"] = gpu_details

                # Check for CUDA version in nvidia-smi output
                if (
                    "cuda_version" in gpu_details
                    and gpu_details["cuda_version"] != "N/A"
                ):
                    gpu_info["cuda_runtime"] = True
                    gpu_info["cuda_version"] = gpu_details["cuda_version"]

            # Fallback: Check CUDA development toolkit if available
            if not gpu_info["cuda_runtime"]:
                result = self._run_command(["nvcc", "--version"], silent_fail=True)
                if result.returncode == 0:
                    gpu_info["cuda_runtime"] = True
                    gpu_info["cuda_version"] = self._parse_cuda_version(result.stdout)

            # Check Docker GPU support
            gpu_info["docker_gpu"] = self._check_docker_gpu()

            gpu_info["gpu_available"] = all(
                [
                    gpu_info["nvidia_driver"],
                    gpu_info["cuda_runtime"],
                    gpu_info["docker_gpu"],
                ]
            )

        except Exception as e:
            console.print(f"[red]Error checking GPU support: {e}[/red]")

        self.gpu_info = gpu_info
        return gpu_info

    def _parse_nvidia_smi(self, output: str) -> dict[str, str]:
        """Parse nvidia-smi output for GPU details including CUDA version"""
        details = {}
        lines = output.split("\n")
        for line in lines:
            if "NVIDIA-SMI" in line and "CUDA Version:" in line:
                # Extract driver version and CUDA version
                parts = line.split()
                if len(parts) >= 2:
                    details["driver_version"] = parts[2]

                # Extract CUDA version
                cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", line)
                if cuda_match:
                    details["cuda_version"] = cuda_match.group(1)
                else:
                    details["cuda_version"] = "N/A"

            elif "GeForce" in line or "RTX" in line or "GTX" in line:
                # Extract GPU model
                if "|" in line:
                    gpu_part = line.split("|")[1].strip()
                    gpu_words = gpu_part.split()
                    if len(gpu_words) >= 3:
                        details["gpu_model"] = " ".join(
                            gpu_words[0:4]
                        )  # Get more of the model name

        return details

    def _parse_cuda_version(self, output: str) -> str:
        """Parse CUDA version from nvcc output"""
        lines = output.split("\n")
        for line in lines:
            if "release" in line.lower():
                match = re.search(r"release (\d+\.\d+)", line)
                if match:
                    return match.group(1)
        return "Unknown"

    def _check_docker_gpu(self) -> bool:
        """Check if Docker GPU support is available"""
        try:
            # Check if nvidia runtime is available
            result = self._run_command(["docker", "info"], silent_fail=True)
            if result.returncode == 0 and "nvidia" in result.stdout.lower():
                # Test with NVIDIA CUDA sample
                result = self._run_command(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--gpus=all",
                        "nvcr.io/nvidia/k8s/cuda-sample:nbody",
                        "nbody",
                        "-gpu",
                        "-benchmark",
                    ],
                    silent_fail=True,
                )
                return result.returncode == 0
        except Exception:
            pass
        return False

    def display_gpu_status(self) -> None:
        """Display GPU status in a rich table"""
        if not self.gpu_info:
            self.check_gpu_support()

        table = Table(title="🎮 GPU Support Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")

        # Status icons
        def status_icon(status: bool) -> str:
            return "✅" if status else "❌"

        table.add_row(
            "NVIDIA Driver",
            status_icon(self.gpu_info["nvidia_driver"]),
            self.gpu_info.get("gpu_details", {}).get("driver_version", "N/A"),
        )

        table.add_row(
            "CUDA Runtime",
            status_icon(self.gpu_info["cuda_runtime"]),
            self.gpu_info.get("cuda_version", "N/A"),
        )

        table.add_row(
            "Docker GPU Support",
            status_icon(self.gpu_info["docker_gpu"]),
            "Ready" if self.gpu_info["docker_gpu"] else "Not available",
        )

        console.print(table)

        # Overall status
        if self.gpu_info["gpu_available"]:
            console.print("\n🚀 [bold green]GPU acceleration is READY![/bold green]")
            console.print("   Ollama will automatically use GPU for faster inference.")
        else:
            console.print("\n⚡ [yellow]Using CPU mode[/yellow]")
            self._display_gpu_guidance()

    def _display_gpu_guidance(self) -> None:
        """Display GPU setup guidance"""
        if not self.gpu_info["nvidia_driver"]:
            console.print("   [red]Missing: NVIDIA GPU driver[/red]")
            console.print("\n🎯 [bold]GPU Driver Setup[/bold]")
            console.print("Install NVIDIA GPU drivers:")
            console.print("📥 [blue]Download:[/blue] https://www.nvidia.com/drivers")

        elif not self.gpu_info["cuda_runtime"]:
            console.print("   [yellow]Missing: CUDA Runtime[/yellow]")
            console.print("\n🎯 [bold]GPU Acceleration Setup[/bold]")
            console.print("For optimal performance, install NVIDIA CUDA Toolkit:")
            console.print()
            console.print("📥 [blue]Download Links:[/blue]")
            console.print(
                "   • CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit"
            )
            console.print()
            console.print(
                "💡 [dim]Note: For containerized apps like EquiLens, CUDA Runtime in Docker is sufficient[/dim]"
            )

        elif not self.gpu_info["docker_gpu"]:
            console.print("   [yellow]Missing: Docker GPU support[/yellow]")
            console.print(
                "   Enable GPU support in Docker Desktop or install NVIDIA Container Toolkit"
            )

    def get_performance_recommendation(self) -> str:
        """Get performance recommendation based on GPU availability"""
        if not self.gpu_info:
            self.check_gpu_support()

        if self.gpu_info["gpu_available"]:
            return "🎮 GPU acceleration enabled - expect 5-10x faster performance"
        else:
            return "⚡ CPU mode - consider installing CUDA for faster processing"
