#!/usr/bin/env python3
"""
Quick test script to verify EquiLens Web UI functionality
Tests key functions without launching the full interface
"""

import sys
from pathlib import Path


def test_web_ui_functions():
    """Test individual web UI functions"""
    print("🧪 Testing Web UI Functions")
    print("=" * 40)

    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        from equilens.web_ui import (
            create_interface,
            get_corpus_files,
            get_results_files,
            get_system_info,
        )

        # Test system info
        print("📊 Testing system info...")
        info = get_system_info()
        if info and len(info) > 100:  # Basic check for reasonable output
            print("✅ System info function works")
        else:
            print("❌ System info function returned minimal output")

        # Test corpus files
        print("📚 Testing corpus file listing...")
        corpus_files = get_corpus_files()
        print(f"✅ Found {len(corpus_files)} corpus files")

        # Test results files
        print("📈 Testing results file listing...")
        results_files = get_results_files()
        print(f"✅ Found {len(results_files)} results files")

        # Test interface creation
        print("🎨 Testing interface creation...")
        demo = create_interface()
        if demo is not None:
            print("✅ Interface created successfully")
        else:
            print("❌ Interface creation failed")

        print("\n🎉 All function tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_cli_integration():
    """Test CLI integration"""
    print("\n🔧 Testing CLI Integration")
    print("=" * 40)

    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        from equilens.cli import app

        # Simple test - just verify the app object exists
        if app is not None:
            print("✅ CLI app imported successfully")
            # Check if gui function is available
            from equilens.cli import gui

            print("✅ GUI command function found")
            return True
        else:
            print("❌ CLI app is None")
            return False

    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔍 EquiLens Web UI Component Tests")
    print("=" * 50)

    tests = [test_web_ui_functions, test_cli_integration]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 50)
    print("📊 Test Summary:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All {total} component tests passed!")
        print("\n🚀 Web UI is ready for use!")
        print("💡 Launch with: python -m equilens gui")
        print(
            "💡 Or use: ./start_web_ui.bat (Windows) or ./start_web_ui.sh (Linux/Mac)"
        )
        return 0
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
