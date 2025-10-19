"""
Smart Ollama Configuration with Runtime Environment Detection

This module intelligently detects the runtime environment and determines
the correct Ollama endpoint URL for different deployment scenarios.

Key Features:
- Configurable Ollama port via OLLAMA_PORT env var (default: 11434)
- Environment variable priority: OLLAMA_BASE_URL > auto-detection
- Smart fallback logic when env vars are not set or don't work
- Container detection via multiple methods (.dockerenv, cgroup, env var)

Deployment Scenarios:
1. EquiLens in Docker → Use host.docker.internal:PORT
   - Works for both: Ollama in container OR Ollama on host
   - Docker Desktop routes host.docker.internal to host network

2. EquiLens local → Use localhost:PORT
   - Works for both: Ollama in container (exposed port) OR Ollama on host
   - Containerized Ollama exposes port to host network

Environment Variables:
- OLLAMA_BASE_URL: Explicit URL override (highest priority, validated)
- OLLAMA_HOST: Alternative URL override (validated)
- OLLAMA_PORT: Custom Ollama port (default: 11434)
- EQUILENS_IN_CONTAINER: Force container detection ("true"/"1"/"yes")

Detection Logic:
1. If OLLAMA_BASE_URL exists AND works → Use it (local install)
2. If OLLAMA_BASE_URL doesn't exist → EquiLens is likely running locally
3. If OLLAMA_BASE_URL exists but doesn't work → EquiLens is in container
4. Check container markers (.dockerenv, cgroup) to confirm
5. Try container URLs first (host.docker.internal), then local URLs
"""

import os
import subprocess
from pathlib import Path

import requests


class OllamaConfig:
    """Smart Ollama configuration with environment detection"""

    def __init__(self):
        self._cached_url: str | None = None
        self._is_container_cached: bool | None = None

    def is_running_in_container(self) -> bool:
        """
        Detect if EquiLens is running inside a Docker container.

        Returns:
            True if running in container, False otherwise
        """
        # Detection Strategy:
        # 1. Check for explicit env var override (EQUILENS_IN_CONTAINER)
        # 2. Check if OLLAMA_BASE_URL is NOT set (indicates local install)
        # 3. Check for container markers (.dockerenv, cgroup)
        # Note: Absence of OLLAMA_BASE_URL env var is a strong indicator
        # that EquiLens is running locally (not in our Docker container).
        if self._is_container_cached is not None:
            return self._is_container_cached

        # Method 1: Explicit environment variable override
        if os.getenv("EQUILENS_IN_CONTAINER", "").lower() in ("true", "1", "yes"):
            self._is_container_cached = True
            return True

        # Method 2: Check if OLLAMA_BASE_URL is not set (indicates local install)
        # Our docker-compose.yml always sets OLLAMA_BASE_URL.
        # If it's not set, we're running locally.
        # In EquiLens deployments, docker-compose always sets OLLAMA_BASE_URL for containerized runs,
        # so its absence reliably indicates a local (non-container) installation.
        if not os.getenv("OLLAMA_BASE_URL") and not os.getenv("OLLAMA_HOST"):
            self._is_container_cached = False
            return False

        # Method 3: Check for .dockerenv file
        if Path("/.dockerenv").exists():
            self._is_container_cached = True
            return True

        # Method 4: Check cgroup file for docker
        try:
            with open("/proc/1/cgroup") as f:
                content = f.read()
                if "docker" in content or "containerd" in content:
                    self._is_container_cached = True
                    return True
        except (FileNotFoundError, PermissionError):
            pass

        # If OLLAMA_BASE_URL is set but no container markers found,
        # assume local install with explicit configuration
        self._is_container_cached = False
        return False

    def _test_connection(self, url: str, timeout: int = 2) -> bool:
        """
        Test if Ollama is accessible at given URL.

        Args:
            url: Ollama endpoint URL
            timeout: Connection timeout in seconds

        Returns:
            True if accessible, False otherwise
        """
        try:
            response = requests.get(f"{url}/api/version", timeout=timeout)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False
        except Exception:
            return False

    def _check_ollama_container_exists(self) -> bool:
        """
        Check if Ollama is running as a Docker container.

        Returns:
            True if Ollama container exists, False otherwise
        """
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "ancestor=ollama/ollama",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return False

    def get_ollama_url(self, force_refresh: bool = False) -> str:
        """
        Intelligently determine the correct Ollama URL based on environment.

        This method:
        1. Checks for explicit environment variable override (if valid, uses it directly)
        2. Detects if running in container (checks for env vars first, then auto-detect)
        3. Uses configurable port from OLLAMA_PORT env var (default: 11434)
        4. Tests available endpoints in priority order
        5. Caches the working URL for performance

        Args:
            force_refresh: If True, bypass cache and re-detect

        Returns:
            Working Ollama URL

        Environment Variables:
            OLLAMA_BASE_URL: Explicit override (highest priority, if working)
            OLLAMA_HOST: Alternative override (e.g., "host.docker.internal:11434")
            OLLAMA_PORT: Custom port for Ollama (default: 11434)
            EQUILENS_IN_CONTAINER: Force container detection ("true"/"1"/"yes")
        """
        # Return cached URL if available
        if not force_refresh and self._cached_url:
            return self._cached_url

        # Get configurable port (default: 11434)
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        env_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
        if env_url:
            # Ensure it has http:// prefix
            if not env_url.startswith(("http://", "https://")):
                env_url = f"http://{env_url}"
            # Verify it works - if it does, use it; otherwise fall through to auto-detection
            if self._test_connection(env_url):
                self._cached_url = env_url
                return env_url
            else:
            if self._test_connection(env_url):
                self._cached_url = env_url
                return env_url
            else:
                import warnings

                warnings.warn(
                    f"Ollama URL override '{env_url}' is set but not reachable. Falling back to auto-detection.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # Do not cache a non-working URL; proceed to auto-detection
        in_container = self.is_running_in_container()
        # Note: We check if Ollama is containerized for diagnostic purposes,
        # but it doesn't affect URL selection (host.docker.internal works for both)

        # Priority 3: Build intelligent URL list based on environment
        candidate_urls = []

        if in_container:
            # EquiLens is in Docker container
            # ALWAYS try host.docker.internal FIRST for container-to-anything communication
            # This works for both: Ollama in container OR Ollama on host
            candidate_urls.extend(
                [
                    f"http://host.docker.internal:{ollama_port}",  # Primary: container-to-host gateway
                    f"http://localhost:{ollama_port}",  # Fallback (usually won't work from container)
                ]
            )
        else:
            # EquiLens running locally (not in container)
            # ALWAYS use localhost - works for both containerized and local Ollama
            # (containerized Ollama exposes port to host)
            candidate_urls.extend(
                [
                    f"http://localhost:{ollama_port}",  # Primary: works for container OR local Ollama
                    f"http://127.0.0.1:{ollama_port}",  # Fallback: same as localhost
                ]
            )

        # Priority 4: Test each candidate and return first working URL
        for url in candidate_urls:
            if self._test_connection(url):
                self._cached_url = url
                return url

        # Priority 5: Fallback to default (will fail but provide clear error)
        default_url = f"http://localhost:{ollama_port}"
        self._cached_url = default_url
        return default_url

    def get_environment_info(self) -> dict:
        """
        Get detailed information about the detected environment.

        Returns:
            Dictionary with environment details including:
            - equilens_in_container: Whether EquiLens is running in Docker
            - ollama_in_container: Whether Ollama container exists
            - ollama_url: The detected Ollama endpoint URL
            - scenario: Human-readable deployment scenario
            - description: Detailed explanation
            - ollama_port: The port Ollama is listening on
        """
        in_container = self.is_running_in_container()
        ollama_containerized = self._check_ollama_container_exists()
        ollama_url = self.get_ollama_url()
        ollama_port = os.getenv("OLLAMA_PORT", "11434")

        # Determine scenario - simplified logic
        if in_container:
            scenario = "Container → host.docker.internal"
            description = f"EquiLens in Docker container, using host.docker.internal:{ollama_port} (works for both containerized and host Ollama)"
        else:
            scenario = "Local → localhost"
            description = f"EquiLens running locally, using localhost:{ollama_port} (works for both containerized and host Ollama)"

        return {
            "equilens_in_container": in_container,
            "ollama_in_container": ollama_containerized,
            "ollama_port": ollama_port,
            "ollama_url": ollama_url,
            "scenario": scenario,
            "description": description,
            "env_override": bool(
                os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
            ),
        }

    def clear_cache(self):
        """Clear cached URL to force re-detection"""
        self._cached_url = None
        self._is_container_cached = None


# Global singleton instance
_ollama_config = OllamaConfig()


def get_ollama_url(force_refresh: bool = False) -> str:
    """
    Convenience function to get Ollama URL.

    Args:
        force_refresh: If True, re-detect environment

    Returns:
        Ollama endpoint URL
    """
    return _ollama_config.get_ollama_url(force_refresh=force_refresh)


def get_environment_info() -> dict:
    """
    Convenience function to get environment information.

    Returns:
        Environment details dictionary
    """
    return _ollama_config.get_environment_info()


def is_running_in_container() -> bool:
    """
    Convenience function to check if running in container.

    Returns:
        True if in container, False otherwise
    """
    return _ollama_config.is_running_in_container()
