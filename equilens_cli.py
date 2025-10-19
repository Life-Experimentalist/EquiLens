#!/usr/bin/env python3
"""
EquiLens CLI entry point.
This script ensures the CLI works from any directory in Docker or host.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Now import and run the CLI
from src.equilens.cli import app

if __name__ == "__main__":
    app()
