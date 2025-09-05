#!/usr/bin/env python3
"""
Test script to verify ETA functionality in audit_model.py
"""

from src.Phase2_ModelAuditor.audit_model import ModelAuditor


def test_eta_calculation():
    """Test the ETA calculation functionality"""

    # Create a test auditor instance
    auditor = ModelAuditor(
        "test_model", "test_corpus.csv", "test_results", eta_per_test=5.0
    )

    print("Testing ETA functionality...")

    # Test 1: User-provided ETA
    print("\n1. Testing user-provided ETA (5.0s per test):")
    eta_str, eta_seconds = auditor.calculate_eta(10, 100)
    print("   Current: 10/100 tests completed")
    print(f"   {eta_str}")
    print(f"   ETA seconds: {eta_seconds}")

    # Test 2: Dynamic ETA with actual response times
    print("\n2. Testing dynamic ETA with actual response times:")
    auditor.user_eta_per_test = None  # Disable user ETA
    # Simulate some response times
    for time in [3.2, 4.1, 3.8, 4.5, 3.9, 4.2, 3.6, 4.0]:
        auditor.update_response_time(time)

    eta_str, eta_seconds = auditor.calculate_eta(20, 100)
    print("   Current: 20/100 tests completed")
    print(
        f"   Average response time: {sum(auditor.actual_response_times) / len(auditor.actual_response_times):.2f}s"
    )
    print(f"   {eta_str}")
    print(f"   ETA seconds: {eta_seconds}")

    # Test 3: Duration formatting
    print("\n3. Testing duration formatting:")
    test_durations = [30, 90, 300, 3600, 7200, 86400]
    for duration in test_durations:
        formatted = auditor.format_duration(duration)
        print(f"   {duration}s → {formatted}")

    print("\n✅ All ETA tests completed successfully!")


if __name__ == "__main__":
    test_eta_calculation()
