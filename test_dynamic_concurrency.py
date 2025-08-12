#!/usr/bin/env python3
"""
Test script for dynamic concurrency scaling in EquiLens
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.Phase2_ModelAuditor.audit_model import ModelAuditor


def test_dynamic_scaling():
    """Test the dynamic concurrency scaling functionality"""
    print("🧪 Testing Dynamic Concurrency Scaling")
    print("=" * 50)

    # Create a ModelAuditor instance with concurrency
    auditor = ModelAuditor(
        model_name="test-model",
        corpus_file="test.csv",
        output_dir="test_output",
        max_workers=5
    )

    # Test initial state
    print(f"📊 Initial Configuration:")
    print(f"   Original Max Workers: {auditor.original_max_workers}")
    print(f"   Current Workers: {auditor.current_workers}")
    print(f"   Recovery Threshold: {auditor.recovery_threshold}")
    print(f"   Max Consecutive Errors: {auditor.max_consecutive_errors}")
    print()

    # Mock a progress bar for testing
    class MockProgressBar:
        def write(self, msg):
            print(f"   {msg}")

    pbar = MockProgressBar()

    # Test scaling down on consecutive errors
    print("🔴 Testing Error Handling (Scale Down):")
    for i in range(1, 6):
        auditor._handle_request_result(success=False, pbar=pbar)
        print(f"   Error {i}: Workers={auditor.current_workers}, Consecutive Errors={auditor.consecutive_errors}")
        if auditor.current_workers == 1:
            break
    print()

    # Test scaling up on consecutive successes
    print("🟢 Testing Success Recovery (Scale Up):")
    for i in range(1, 25):
        auditor._handle_request_result(success=True, pbar=pbar)
        if i % 5 == 0:  # Show progress every 5 successes
            print(f"   Success {i}: Workers={auditor.current_workers}, Consecutive Successes={auditor.consecutive_successes}")
        if auditor.current_workers == auditor.original_max_workers:
            print(f"   ✅ Fully recovered to {auditor.current_workers} workers after {i} successes")
            break

    print()
    print("✅ Dynamic scaling test completed!")


if __name__ == "__main__":
    test_dynamic_scaling()
