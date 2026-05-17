"""
Backend launcher script for EquiLens API.
"""

import uvicorn

from equilens.backend.api import app


def main():
    """Launch the EquiLens backend API."""
    from equilens.core.ports import get_backend_port

    # Get available port
    port = get_backend_port()

    print("\n" + "=" * 70)
    print("🚀 EquiLens Backend API")
    print("=" * 70)
    print()
    print(f"📡 Starting API server on port {port}...")
    print(f"   API URL:      http://localhost:{port}")
    print(f"   Health Check: http://localhost:{port}/api/health")
    print(f"   API Docs:     http://localhost:{port}/docs")
    print()
    print("💡 Tip: Set custom port with environment variable:")
    print(f"   $env:BACKEND_PORT = {port + 1}")
    print("=" * 70)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
