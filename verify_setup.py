#!/usr/bin/env python3
"""
EquiLens Setup Verification Script
==================================

Verifies that the EquiLens platform is properly configured and ready for use.
This script checks dependencies, configurations, and system requirements.

Usage:
    python verify_setup.py

    Or from Docker:
    docker-compose exec equilens-app python verify_setup.py
"""

import platform
import subprocess
import sys
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(message: str, status: str = "INFO") -> None:
    """Print formatted status message."""
    colors = {
        "PASS": Colors.GREEN,
        "FAIL": Colors.RED,
        "WARN": Colors.YELLOW,
        "INFO": Colors.BLUE,
    }
    color = colors.get(status, Colors.BLUE)
    print(f"{color}[{status}]{Colors.END} {message}")


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    print_status("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_status(
            f"Python {version.major}.{version.minor}.{version.micro} ✓", "PASS"
        )
        return True
    else:
        print_status(
            f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.11+",
            "FAIL",
        )
        return False


def check_required_packages() -> bool:
    """Check if required Python packages are installed."""
    print_status("Checking required packages...")
    required_packages = ["typer", "rich", "textual", "requests", "attrs"]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"  {package} ✓", "PASS")
        except ImportError:
            print_status(f"  {package} ✗", "FAIL")
            missing.append(package)

    if missing:
        print_status(f"Missing packages: {', '.join(missing)}", "FAIL")
        print_status("Run: pip install -e . or uv sync", "INFO")
        return False

    return True


def check_directory_structure() -> bool:
    """Check if directory structure is correct."""
    print_status("Checking directory structure...")
    required_dirs = [
        "src/equilens",
        "src/equilens/core",
        "src/Phase1_CorpusGenerator",
        "src/Phase2_ModelAuditor",
        "src/Phase3_Analysis",
        "docs",
        "results",
        "logs",
    ]

    project_root = Path(__file__).parent
    missing_dirs = []

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_status(f"  {dir_path} ✓", "PASS")
        else:
            print_status(f"  {dir_path} ✗", "FAIL")
            missing_dirs.append(dir_path)

    if missing_dirs:
        print_status(f"Missing directories: {', '.join(missing_dirs)}", "FAIL")
        return False

    return True


def check_configuration_files() -> bool:
    """Check if configuration files exist."""
    print_status("Checking configuration files...")
    config_files = [
        "pyproject.toml",
        "docker-compose.yml",
        "Dockerfile",
        "src/Phase1_CorpusGenerator/word_lists.json",
        "src/Phase1_CorpusGenerator/word_lists_schema.json",
    ]

    project_root = Path(__file__).parent
    missing_files = []

    for file_path in config_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_status(f"  {file_path} ✓", "PASS")
        else:
            print_status(f"  {file_path} ✗", "FAIL")
            missing_files.append(file_path)

    if missing_files:
        print_status(f"Missing files: {', '.join(missing_files)}", "FAIL")
        return False

    return True


def check_docker_availability() -> bool:
    """Check if Docker is available."""
    print_status("Checking Docker availability...")
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"  {version} ✓", "PASS")
            return True
        else:
            print_status("Docker not found or not working", "FAIL")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Docker not found or not working", "FAIL")
        return False


def check_ollama_connection() -> bool:
    """Check if Ollama is accessible."""
    print_status("Checking Ollama connection...")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print_status(f"  Ollama {version_info.get('version', 'unknown')} ✓", "PASS")
            return True
        else:
            print_status("Ollama API not responding", "WARN")
            return False
    except Exception as e:
        print_status(f"Ollama not accessible: {str(e)}", "WARN")
        print_status("Start with: docker-compose up ollama", "INFO")
        return False


def check_gpu_support() -> bool:
    """Check for GPU support."""
    print_status("Checking GPU support...")

    # Check NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print_status("NVIDIA GPU detected ✓", "PASS")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for other GPU indicators
    if platform.system() == "Darwin":  # macOS
        print_status("macOS detected - Metal GPU support available", "INFO")
        return True

    print_status("No GPU acceleration detected (CPU mode)", "WARN")
    return False


def check_system_resources() -> bool:
    """Check system resources."""
    print_status("Checking system resources...")

    try:
        import psutil

        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 8:
            print_status(f"  Memory: {memory_gb:.1f} GB ✓", "PASS")
        else:
            print_status(f"  Memory: {memory_gb:.1f} GB (8GB+ recommended)", "WARN")

        # Disk space check
        disk = psutil.disk_usage(".")
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 10:
            print_status(f"  Free disk space: {disk_gb:.1f} GB ✓", "PASS")
        else:
            print_status(
                f"  Free disk space: {disk_gb:.1f} GB (10GB+ recommended)", "WARN"
            )

        return True

    except ImportError:
        print_status("psutil not available - skipping resource check", "WARN")
        return True


def main():
    """Run comprehensive setup verification."""
    print(f"\n{Colors.BOLD}EquiLens Setup Verification{Colors.END}")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("Docker Availability", check_docker_availability),
        ("Ollama Connection", check_ollama_connection),
        ("GPU Support", check_gpu_support),
        ("System Resources", check_system_resources),
    ]

    results = []

    print(f"\n{Colors.BLUE}Running verification checks...{Colors.END}\n")

    for check_name, check_func in checks:
        print(f"\n{Colors.BOLD}🔍 {check_name}{Colors.END}")
        print("-" * 30)
        result = check_func()
        results.append((check_name, result))

    # Summary
    print(f"\n{Colors.BOLD}📋 Verification Summary{Colors.END}")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print_status(f"{check_name:<25} {'✓' if result else '✗'}", status)

    print(f"\n{Colors.BOLD}Result: {passed}/{total} checks passed{Colors.END}")

    if passed == total:
        print_status("\n🎉 EquiLens is ready to use!", "PASS")
        print("\nNext steps:")
        print("  • Run: python -m equilens.cli --help")
        print("  • Or: docker-compose up")
        return 0
    else:
        print_status(f"\n⚠️  {total - passed} issues found", "WARN")
        print("\nRefer to SETUP.md for troubleshooting")
        return 1


if __name__ == "__main__":
    sys.exit(main())
