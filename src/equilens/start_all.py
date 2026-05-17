"""
Start both backend and frontend services.

This script launches the backend API and Gradio frontend in separate processes.
"""

import multiprocessing
import sys
import time


def start_backend():
    """Start the backend API server."""
    from equilens.backend_server import main as backend_main

    print("🔧 Starting Backend API...")
    backend_main()


def start_frontend():
    """Start the Gradio frontend."""
    from equilens.gradio_app import main as gradio_main

    # Wait a bit for backend to start
    time.sleep(3)
    print("🌐 Starting Gradio Frontend...")
    gradio_main()


def main():
    """Launch both services."""
    from equilens.core.ports import get_service_ports, print_service_info

    # Get available ports
    backend_port, frontend_port = get_service_ports()

    print_service_info(backend_port, frontend_port)
    print("Press Ctrl+C to stop all services")
    print()

    # Start processes
    backend_process = multiprocessing.Process(target=start_backend)
    frontend_process = multiprocessing.Process(target=start_frontend)

    try:
        backend_process.start()
        frontend_process.start()

        # Keep main process alive
        backend_process.join()
        frontend_process.join()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.join()
        frontend_process.join()
        print("✅ All services stopped")
        sys.exit(0)


if __name__ == "__main__":
    # Required for Windows
    multiprocessing.freeze_support()
    main()
