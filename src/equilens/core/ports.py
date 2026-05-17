"""
Port management utilities for EquiLens services.

Handles automatic port detection and configuration for backend and frontend services.
"""

import os
import socket


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """
    Find an available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found within max_attempts
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port

    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}"
    )


def is_port_available(port: int) -> bool:
    """
    Check if a port is available for binding.

    Args:
        port: Port number to check

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            return True
    except OSError:
        return False


def get_backend_port() -> int:
    """
    Get the backend port from environment or find available port.

    Environment variable: BACKEND_PORT
    Default: 8000 (or next available)

    Returns:
        Backend port number
    """
    env_port = os.getenv("BACKEND_PORT")
    if env_port:
        try:
            port = int(env_port)
            if is_port_available(port):
                return port
            else:
                print(f"⚠️  Port {port} (from BACKEND_PORT) is already in use")
                print("   Searching for next available port...")
                return find_available_port(port + 1)
        except ValueError:
            print(f"⚠️  Invalid BACKEND_PORT value: {env_port}")

    # Try default port
    default_port = 8000
    if is_port_available(default_port):
        return default_port

    # Find next available
    print(f"⚠️  Default backend port {default_port} is already in use")
    print("   Searching for next available port...")
    return find_available_port(default_port + 1)


def get_frontend_port() -> int:
    """
    Get the frontend port from environment or find available port.

    Environment variable: FRONTEND_PORT or GRADIO_PORT
    Default: 7860 (or next available)

    Returns:
        Frontend port number
    """
    env_port = os.getenv("FRONTEND_PORT") or os.getenv("GRADIO_PORT")
    if env_port:
        try:
            port = int(env_port)
            if is_port_available(port):
                return port
            else:
                print(f"⚠️  Port {port} (from environment) is already in use")
                print("   Searching for next available port...")
                return find_available_port(port + 1)
        except ValueError:
            print(f"⚠️  Invalid port value in environment: {env_port}")

    # Try default port
    default_port = 7860
    if is_port_available(default_port):
        return default_port

    # Find next available
    print(f"⚠️  Default frontend port {default_port} is already in use")
    print("   Searching for next available port...")
    return find_available_port(default_port + 1)


def get_backend_url(port: int | None = None) -> str:
    """
    Get the backend URL, auto-detecting environment.

    Args:
        port: Backend port number (optional, will auto-detect if not provided)

    Returns:
        Backend URL (e.g., http://localhost:8000)
    """
    from pathlib import Path

    # Check environment variable first
    env_url = os.getenv("BACKEND_URL")
    if env_url:
        return env_url

    # Detect Docker environment
    in_docker = Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV") == "true"

    if in_docker:
        # In Docker, use service name
        backend_host = os.getenv("BACKEND_HOST", "backend")
        backend_port = port or 8000
    else:
        # Local environment
        backend_host = os.getenv("BACKEND_HOST", "localhost")
        backend_port = port or get_backend_port()

    return f"http://{backend_host}:{backend_port}"


def get_service_ports() -> tuple[int, int]:
    """
    Get both backend and frontend ports.

    Returns:
        Tuple of (backend_port, frontend_port)
    """
    backend_port = get_backend_port()
    frontend_port = get_frontend_port()

    return backend_port, frontend_port


import logging


def print_service_info(backend_port: int, frontend_port: int):
    """
    Print service information banner.

    Args:
        backend_port: Backend API port
        frontend_port: Frontend/Gradio port
    """
    logger = logging.getLogger("equilens.core.ports")
    logger.info("")
    logger.info("=" * 70)
    logger.info("🔍 EquiLens Services")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"📡 Backend API:     http://localhost:{backend_port}")
    logger.info(f"   API Docs:        http://localhost:{backend_port}/docs")
    logger.info(f"   Health Check:    http://localhost:{backend_port}/api/health")
    logger.info("")
    logger.info(f"🌐 Web Interface:   http://localhost:{frontend_port}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")
    logger.info("💡 Tip: Use environment variables to set custom ports:")
    logger.info(f"   $env:BACKEND_PORT = {backend_port + 1}")
    logger.info(f"   $env:FRONTEND_PORT = {frontend_port + 1}")
    logger.info("")
