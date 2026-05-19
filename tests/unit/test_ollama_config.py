"""
Unit tests for src/equilens/core/ollama_config.py

All tests use fresh OllamaConfig() instances to avoid cache pollution.
No real network requests, Docker, or Ollama required.
"""

import warnings
from unittest.mock import MagicMock, patch

import requests

from equilens.core.ollama_config import (
    OllamaConfig,
    get_environment_info,
    get_ollama_url,
    is_running_in_container,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config() -> OllamaConfig:
    """Return a fresh OllamaConfig instance with no cached state."""
    return OllamaConfig()


def _mock_get(status_code=200, raises=None):
    """Return a patch target for requests.get with the given behaviour."""
    if raises is not None:
        return patch("equilens.core.ollama_config.requests.get", side_effect=raises)
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    return patch("equilens.core.ollama_config.requests.get", return_value=mock_resp)


# ---------------------------------------------------------------------------
# is_running_in_container()
# ---------------------------------------------------------------------------


class TestIsRunningInContainer:
    def test_explicit_env_true(self, monkeypatch):
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "true")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert _config().is_running_in_container() is True

    def test_explicit_env_one(self, monkeypatch):
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "1")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert _config().is_running_in_container() is True

    def test_explicit_env_yes(self, monkeypatch):
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "yes")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert _config().is_running_in_container() is True

    def test_no_ollama_env_vars_returns_false(self, monkeypatch):
        """Absence of OLLAMA_BASE_URL and OLLAMA_HOST means local install."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert _config().is_running_in_container() is False

    def test_dockerenv_file_exists(self, monkeypatch):
        """When /.dockerenv exists and OLLAMA_BASE_URL is set, detect as container."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://some-host:11434")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with patch("equilens.core.ollama_config.Path.exists", return_value=True):
            assert _config().is_running_in_container() is True

    def test_dockerenv_file_absent_with_ollama_url(self, monkeypatch):
        """OLLAMA_BASE_URL set, no .dockerenv, no cgroup → False."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://some-host:11434")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        with (
            patch("equilens.core.ollama_config.Path.exists", return_value=False),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            assert _config().is_running_in_container() is False

    def test_cgroup_contains_docker(self, monkeypatch):
        """If /proc/1/cgroup contains 'docker', detect as container."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://some-host:11434")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        cgroup_content = "12:cpu:/docker/abc123\n"
        with (
            patch("equilens.core.ollama_config.Path.exists", return_value=False),
            patch("builtins.open", mock_open_content(cgroup_content)),
        ):
            assert _config().is_running_in_container() is True

    def test_cgroup_contains_containerd(self, monkeypatch):
        """If /proc/1/cgroup contains 'containerd', detect as container."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://some-host:11434")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        cgroup_content = "12:cpu:/containerd/abc123\n"
        with (
            patch("equilens.core.ollama_config.Path.exists", return_value=False),
            patch("builtins.open", mock_open_content(cgroup_content)),
        ):
            assert _config().is_running_in_container() is True

    def test_result_is_cached(self, monkeypatch):
        """Second call does not re-read env vars; cached result returned."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        cfg = _config()
        first = cfg.is_running_in_container()
        # Change env var after first call — should not affect cached result
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "true")
        second = cfg.is_running_in_container()
        assert first == second

    def test_clear_cache_resets_container_cache(self, monkeypatch):
        """clear_cache() resets _is_container_cached."""
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        cfg = _config()
        assert cfg.is_running_in_container() is False
        # Now inject container env var and clear cache
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "true")
        cfg.clear_cache()
        assert cfg.is_running_in_container() is True


# ---------------------------------------------------------------------------
# _test_connection()
# ---------------------------------------------------------------------------


class TestTestConnection:
    def test_returns_true_on_200(self):
        cfg = _config()
        with _mock_get(status_code=200):
            assert cfg._test_connection("http://localhost:11434") is True

    def test_returns_false_on_non_200(self):
        cfg = _config()
        with _mock_get(status_code=500):
            assert cfg._test_connection("http://localhost:11434") is False

    def test_returns_false_on_connection_error(self):
        cfg = _config()
        with _mock_get(raises=requests.exceptions.ConnectionError("refused")):
            assert cfg._test_connection("http://localhost:11434") is False

    def test_returns_false_on_timeout(self):
        cfg = _config()
        with _mock_get(raises=requests.exceptions.Timeout("timed out")):
            assert cfg._test_connection("http://localhost:11434") is False

    def test_returns_false_on_generic_exception(self):
        cfg = _config()
        with _mock_get(raises=RuntimeError("unexpected")):
            assert cfg._test_connection("http://localhost:11434") is False

    def test_calls_correct_endpoint(self):
        cfg = _config()
        with _mock_get(status_code=200) as mock:
            cfg._test_connection("http://localhost:11434")
            mock.assert_called_once_with(
                "http://localhost:11434/api/version", timeout=2
            )

    def test_custom_timeout_passed_through(self):
        cfg = _config()
        with _mock_get(status_code=200) as mock:
            cfg._test_connection("http://localhost:11434", timeout=5)
            mock.assert_called_once_with(
                "http://localhost:11434/api/version", timeout=5
            )


# ---------------------------------------------------------------------------
# get_ollama_url()
# ---------------------------------------------------------------------------


class TestGetOllamaUrl:
    # --- caching ---

    def test_returns_cached_url_on_second_call(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with patch.object(cfg, "_test_connection", return_value=True) as mock_test:
            first = cfg.get_ollama_url()
            second = cfg.get_ollama_url()
            assert first == second
            # _test_connection called once only (cache hit on second call)
            assert mock_test.call_count == 1

    # --- env var override ---

    def test_uses_ollama_base_url_when_reachable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-host:9999")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        cfg = _config()
        with patch.object(cfg, "_test_connection", return_value=True):
            url = cfg.get_ollama_url()
        assert url == "http://custom-host:9999"

    def test_adds_http_prefix_to_bare_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "custom-host:9999")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        cfg = _config()
        with patch.object(cfg, "_test_connection", return_value=True):
            url = cfg.get_ollama_url()
        assert url == "http://custom-host:9999"

    def test_falls_back_to_autodetect_when_env_url_unreachable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://bad-host:9999")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        call_count = [0]

        def fake_test(url, timeout=2):
            call_count[0] += 1
            if "bad-host" in url:
                return False
            if "localhost:11434" in url:
                return True
            return False

        with patch.object(cfg, "_test_connection", side_effect=fake_test):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                url = cfg.get_ollama_url()
                assert any(
                    issubclass(warning.category, RuntimeWarning) for warning in w
                )
        assert url == "http://localhost:11434"

    # --- local mode ---

    def test_local_mode_returns_localhost(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "is_running_in_container", return_value=False),
            patch.object(cfg, "_test_connection", return_value=True),
        ):
            url = cfg.get_ollama_url()
        assert url == "http://localhost:11434"

    def test_local_mode_tries_127_when_localhost_fails(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()

        def fake_test(url, timeout=2):
            return "127.0.0.1" in url

        with (
            patch.object(cfg, "is_running_in_container", return_value=False),
            patch.object(cfg, "_test_connection", side_effect=fake_test),
        ):
            url = cfg.get_ollama_url()
        assert url == "http://127.0.0.1:11434"

    # --- container mode ---

    def test_container_mode_returns_host_docker_internal(self, monkeypatch):
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "true")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()

        def fake_test(url, timeout=2):
            return "host.docker.internal" in url

        with patch.object(cfg, "_test_connection", side_effect=fake_test):
            url = cfg.get_ollama_url()
        assert url == "http://host.docker.internal:11434"

    # --- custom port ---

    def test_custom_port_via_env(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_PORT", "12345")
        cfg = _config()
        with (
            patch.object(cfg, "is_running_in_container", return_value=False),
            patch.object(cfg, "_test_connection", return_value=True),
        ):
            url = cfg.get_ollama_url()
        assert url == "http://localhost:12345"

    # --- force_refresh ---

    def test_force_refresh_re_detects(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with patch.object(cfg, "_test_connection", return_value=True) as mock_test:
            cfg.get_ollama_url()
            cfg.get_ollama_url(force_refresh=True)
            # _test_connection should have been called at least twice (once per detection)
            assert mock_test.call_count >= 2

    # --- fallback ---

    def test_fallback_to_default_when_nothing_reachable(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "is_running_in_container", return_value=False),
            patch.object(cfg, "_test_connection", return_value=False),
        ):
            url = cfg.get_ollama_url()
        assert url == "http://localhost:11434"

    def test_fallback_uses_custom_port(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_PORT", "9999")
        cfg = _config()
        with (
            patch.object(cfg, "is_running_in_container", return_value=False),
            patch.object(cfg, "_test_connection", return_value=False),
        ):
            url = cfg.get_ollama_url()
        assert url == "http://localhost:9999"


# ---------------------------------------------------------------------------
# get_environment_info()
# ---------------------------------------------------------------------------


class TestGetEnvironmentInfo:
    REQUIRED_KEYS = {
        "equilens_in_container",
        "ollama_in_container",
        "ollama_port",
        "ollama_url",
        "scenario",
        "description",
        "env_override",
    }

    def test_returns_all_required_keys(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "_test_connection", return_value=False),
            patch.object(cfg, "_check_ollama_container_exists", return_value=False),
        ):
            info = cfg.get_environment_info()
        assert self.REQUIRED_KEYS.issubset(set(info.keys()))

    def test_env_override_true_when_ollama_base_url_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-host:11434")
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "_test_connection", return_value=True),
            patch.object(cfg, "_check_ollama_container_exists", return_value=False),
        ):
            info = cfg.get_environment_info()
        assert info["env_override"] is True

    def test_env_override_true_when_ollama_host_set(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://my-host:11434")
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "_test_connection", return_value=True),
            patch.object(cfg, "_check_ollama_container_exists", return_value=False),
        ):
            info = cfg.get_environment_info()
        assert info["env_override"] is True

    def test_env_override_false_when_no_url_set(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "_test_connection", return_value=False),
            patch.object(cfg, "_check_ollama_container_exists", return_value=False),
        ):
            info = cfg.get_environment_info()
        assert info["env_override"] is False

    def test_ollama_port_reflects_env_var(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.setenv("OLLAMA_PORT", "9876")
        cfg = _config()
        with (
            patch.object(cfg, "_test_connection", return_value=False),
            patch.object(cfg, "_check_ollama_container_exists", return_value=False),
        ):
            info = cfg.get_environment_info()
        assert info["ollama_port"] == "9876"

    def test_equilens_in_container_field(self, monkeypatch):
        monkeypatch.setenv("EQUILENS_IN_CONTAINER", "true")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with (
            patch.object(cfg, "_test_connection", return_value=False),
            patch.object(cfg, "_check_ollama_container_exists", return_value=False),
        ):
            info = cfg.get_environment_info()
        assert info["equilens_in_container"] is True


# ---------------------------------------------------------------------------
# clear_cache()
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clears_cached_url(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with patch.object(cfg, "_test_connection", return_value=True):
            cfg.get_ollama_url()
        assert cfg._cached_url is not None
        cfg.clear_cache()
        assert cfg._cached_url is None

    def test_clears_container_cache(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        cfg = _config()
        cfg.is_running_in_container()
        assert cfg._is_container_cached is not None
        cfg.clear_cache()
        assert cfg._is_container_cached is None

    def test_clears_both_at_once(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        cfg = _config()
        with patch.object(cfg, "_test_connection", return_value=True):
            cfg.get_ollama_url()  # sets _cached_url
        cfg.is_running_in_container()  # sets _is_container_cached
        cfg.clear_cache()
        assert cfg._cached_url is None
        assert cfg._is_container_cached is None


# ---------------------------------------------------------------------------
# Module-level convenience functions (exercise the global singleton)
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_get_ollama_url_returns_string(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        with patch(
            "equilens.core.ollama_config.requests.get",
            side_effect=requests.exceptions.ConnectionError,
        ):
            # Force a fresh detection on the singleton
            import equilens.core.ollama_config as mod

            mod._ollama_config.clear_cache()
            url = get_ollama_url()
        assert isinstance(url, str)
        assert url.startswith("http")

    def test_get_environment_info_returns_dict_with_keys(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        monkeypatch.delenv("OLLAMA_PORT", raising=False)
        with (
            patch(
                "equilens.core.ollama_config.requests.get",
                side_effect=requests.exceptions.ConnectionError,
            ),
            patch(
                "equilens.core.ollama_config.subprocess.run",
                return_value=MagicMock(returncode=1, stdout=""),
            ),
        ):
            import equilens.core.ollama_config as mod

            mod._ollama_config.clear_cache()
            info = get_environment_info()
        assert isinstance(info, dict)
        assert "ollama_url" in info

    def test_is_running_in_container_returns_bool(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("EQUILENS_IN_CONTAINER", raising=False)
        import equilens.core.ollama_config as mod

        mod._ollama_config.clear_cache()
        result = is_running_in_container()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def mock_open_content(content: str):
    """Return a mock for builtins.open that yields the given string content."""
    from unittest.mock import mock_open

    return mock_open(read_data=content)
