"""
Pytest configuration and shared fixtures for EquiLens tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_corpus():
    """Return path to sample corpus file for testing."""
    # This would point to a small test corpus
    return Path(__file__).parent / "data" / "sample_corpus.csv"
