#!/usr/bin/env python3
"""
Simple helper to check if Ollama is available at localhost:11434
"""
import sys
import requests
from typing import Tuple


def check_ollama(host: str = "localhost", port: int = 11434) -> Tuple[bool, str]:
    """
    Check if Ollama is running and accessible.

    Args:
        host: Ollama host (default: localhost)
        port: Ollama port (default: 11434)

    Returns:
        Tuple of (is_available, message)
    """
    url = f"http://{host}:{port}/api/tags"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_count = len(models)

            if model_count > 0:
                model_names = [m.get("name", "unknown") for m in models]
                return True, f"✅ Ollama is running with {model_count} model(s): {', '.join(model_names)}"
            else:
                return True, "⚠️  Ollama is running but no models are installed. Run: ollama pull llama3.2"
        else:
            return False, f"❌ Ollama responded with status code: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, f"❌ Cannot connect to Ollama at {host}:{port}. Is Ollama running?"

    except requests.exceptions.Timeout:
        return False, f"❌ Connection to Ollama at {host}:{port} timed out"

    except Exception as e:
        return False, f"❌ Error checking Ollama: {str(e)}"


def main():
    """Main function to check Ollama and print status."""
    print("Checking Ollama availability...")
    print()

    is_available, message = check_ollama()
    print(message)
    print()

    if not is_available:
        print("📖 To install and run Ollama:")
        print("   • Download: https://ollama.ai/download")
        print("   • Install and run the application")
        print("   • Pull a model: ollama pull llama3.2")
        print()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
