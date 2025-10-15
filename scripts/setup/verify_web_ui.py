"""
Verification script for EquiLens Web UI
Tests that all imports work correctly and the interface can be created
"""

import sys


def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")

    try:
        import gradio as gr

        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Gradio: {e}")
        return False

    try:
        import requests

        print("✅ Requests imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import requests: {e}")
        return False

    try:
        from equilens.core.manager import EquiLensManager

        print("✅ EquiLensManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import EquiLensManager: {e}")
        return False

    try:
        from equilens.web_ui import create_interface, main

        print("✅ Web UI functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import web UI functions: {e}")
        return False

    return True


def test_interface_creation():
    """Test that the Gradio interface can be created"""
    print("\n🎨 Testing interface creation...")

    try:
        from equilens.web_ui import create_interface

        # Create the interface (without launching)
        demo = create_interface()

        if demo is not None:
            print("✅ Interface created successfully")
            return True
        else:
            print("❌ Interface creation returned None")
            return False

    except Exception as e:
        print(f"❌ Failed to create interface: {e}")
        return False


def test_manager_initialization():
    """Test that the EquiLens manager can be initialized"""
    print("\n🔧 Testing manager initialization...")

    try:
        from equilens.core.manager import EquiLensManager

        manager = EquiLensManager()

        if manager is not None:
            print("✅ Manager initialized successfully")
            return True
        else:
            print("❌ Manager initialization returned None")
            return False

    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return False


def verify_web_ui():
    """Run all verification tests"""
    print("🔍 EquiLens Web UI Verification")
    print("=" * 40)

    tests = [test_imports, test_interface_creation, test_manager_initialization]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 40)
    print("📊 Test Summary:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\n🚀 Web UI is ready to launch!")
        print("💡 Run: uv run equilens gui")
        return True
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        print("\n🔧 Please fix the issues above before launching the web UI")
        return False


if __name__ == "__main__":
    success = verify_web_ui()
    sys.exit(0 if success else 1)
