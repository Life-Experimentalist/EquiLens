#!/usr/bin/env python3
"""
Extract version from pyproject.toml for use in deployment scripts.

This ensures a single source of truth for version numbering.
"""

import sys
from pathlib import Path


def extract_version():
    """Extract version from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        # Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            print("Error: tomli package required for Python < 3.11", file=sys.stderr)
            print("Install with: pip install tomli", file=sys.stderr)
            sys.exit(1)

    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print("Error: pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        version = data["project"]["version"]
        print(version)
        return version

    except Exception as e:
        print(f"Error extracting version: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    extract_version()
