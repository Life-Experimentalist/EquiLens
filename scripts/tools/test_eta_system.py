"""
Simple test for the new ETA estimation system
"""

import sys
from pathlib import Path

# Add src to path to import equilens modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from equilens.cli import (
        estimate_corpus_eta,  # type: ignore[attr-defined]
        measure_single_request_time,  # type: ignore[attr-defined]
    )
except ImportError:

    def estimate_corpus_eta(*args, **kwargs):  # type: ignore[misc]
        return {"error": "estimate_corpus_eta is not available in this version"}

    def measure_single_request_time(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError(
            "measure_single_request_time is not available in this version"
        )


def test_eta_system():
    """Test the ETA estimation with real timing"""
    print("Testing Real-Time ETA Estimation System")
    print("=" * 50)

    # Test single request timing (requires Ollama to be running)
    print("\n1. Testing single request timing:")
    try:
        timing = measure_single_request_time("llama2:latest", "Hello, world!")
        timing_value = (
            timing if isinstance(timing, int | float) else timing.get("time", 0)
        )
        print(f"   Single request time: {timing_value:.2f}s")
        print(f"   Buffered time (1.4x): {timing_value * 1.4:.2f}s")
    except Exception as e:
        print(f"   Error measuring timing: {e}")

    # Test ETA for quick corpus
    print("\n2. Testing ETA for quick test corpus:")
    try:
        eta = estimate_corpus_eta(
            "../Phase1_CorpusGenerator/corpus/quick_test_corpus.csv", "llama2:latest"
        )
        print(f"   Test count: {eta['test_count']}")
        print(f"   Single request: {eta['single_request_time']}s")
        print(f"   Buffered per test: {eta['buffered_time_per_test']}s")
        print(f"   Total ETA: {eta['formatted']}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\nTest completed!")


if __name__ == "__main__":
    test_eta_system()
