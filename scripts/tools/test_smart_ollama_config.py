#!/usr/bin/env python3
"""
Smart Ollama Configuration Test Script

This script tests the intelligent Ollama configuration system that
automatically detects your deployment environment and connects appropriately.
"""

import os
import sys

import requests


def test_smart_configuration():
    """Test smart configuration system"""
    print("🧠 Testing Smart Ollama Configuration System")
    print("=" * 60)

    try:
        from equilens.core.ollama_config import get_environment_info, get_ollama_url

        # Get environment information
        env_info = get_environment_info()

        print("\n📊 Environment Detection:")
        print(f"  Scenario: {env_info['scenario']}")
        print(f"  Description: {env_info['description']}")
        print(f"  EquiLens in container: {env_info['equilens_in_container']}")
        print(f"  Ollama in container: {env_info['ollama_in_container']}")
        print(f"  Environment override: {env_info['env_override']}")
        print(f"  Detected URL: {env_info['ollama_url']}")

        ollama_url = get_ollama_url()

    except ImportError:
        print("⚠️  Smart config module not available, using fallback")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        print(f"\n📡 Using URL: {ollama_url}")

    return ollama_url


def test_ollama_connection(ollama_url: str):
    """Test Ollama connection using detected URL"""
    print("\n🔍 Testing Ollama connectivity...")
    print(f"📡 URL: {ollama_url}")
    print("-" * 60)

    try:
        # Test API version endpoint
        print("1️⃣  Testing /api/version endpoint...")
        response = requests.get(f"{ollama_url}/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print("   ✅ Version endpoint: OK")
            print(f"   📊 Version: {version_info.get('version', 'unknown')}")
        else:
            print(f"   ❌ Version endpoint failed: HTTP {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Error: Cannot reach {ollama_url}")
        print("   💡 Tip: Check if Ollama is running")
        return False
    except requests.exceptions.Timeout:
        print("   ❌ Timeout: Ollama did not respond within 5 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
        return False

    try:
        # Test tags/models endpoint
        print("\n2️⃣  Testing /api/tags endpoint...")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print("   ✅ Tags endpoint: OK")
            print(f"   📦 Available models: {len(models)}")
            if models:
                print("   🎯 Models:")
                for model in models[:5]:  # Show first 5
                    size_gb = model.get("size", 0) / (1024**3)
                    print(f"      - {model['name']} ({size_gb:.1f} GB)")
                if len(models) > 5:
                    print(f"      ... and {len(models) - 5} more")
            else:
                print("   ℹ️  No models installed yet")
                print(
                    "   💡 Install a model with: uv run equilens models pull <model-name>"
                )
        else:
            print(f"   ❌ Tags endpoint failed: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ Error testing tags endpoint: {type(e).__name__}")
        return False

    print("\n" + "=" * 60)
    print("✅ SUCCESS: Ollama is accessible and working correctly!")
    print("=" * 60)
    return True


def test_alternative_urls():
    """Test alternative connection URLs for debugging"""
    alternative_urls = [
        "http://ollama:11434",
        "http://host.docker.internal:11434",
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]

    print("\n🔄 Testing all possible URLs (for debugging)...")
    print("-" * 60)

    working_urls = []
    for url in alternative_urls:
        try:
            response = requests.get(f"{url}/api/version", timeout=2)
            if response.status_code == 200:
                print(f"✅ {url} - ACCESSIBLE")
                working_urls.append(url)
            else:
                print(f"❌ {url} - HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ {url} - CONNECTION REFUSED")
        except requests.exceptions.Timeout:
            print(f"⏱️  {url} - TIMEOUT")
        except Exception as e:
            print(f"❌ {url} - ERROR: {type(e).__name__}")

    if working_urls:
        print(f"\n💡 Working URLs found: {', '.join(working_urls)}")
        print("   Set OLLAMA_BASE_URL to use a specific URL")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🐳 EquiLens Smart Ollama Configuration Test")
    print("=" * 60 + "\n")

    # Test smart configuration
    ollama_url = test_smart_configuration()

    # Test primary connection
    success = test_ollama_connection(ollama_url)

    # If primary fails, test alternatives
    if not success:
        test_alternative_urls()
        sys.exit(1)

    print("\n💡 Next steps:")
    print("   - Build Docker image: docker build -t equilens:latest .")
    print("   - Start services: docker compose up -d")
    print("   - Access web UI: http://localhost:7860")

    sys.exit(0)
