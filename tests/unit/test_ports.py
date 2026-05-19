"""Unit tests for equilens.core.ports"""

from unittest.mock import MagicMock, patch

import pytest

from equilens.core.ports import (
    find_available_port,
    get_backend_port,
    get_backend_url,
    get_frontend_port,
    get_service_ports,
    is_port_available,
)

# ---------------------------------------------------------------------------
# is_port_available
# ---------------------------------------------------------------------------


def test_is_port_available_returns_true_when_bind_succeeds():
    mock_sock = MagicMock()
    mock_sock.__enter__ = lambda s: s
    mock_sock.__exit__ = MagicMock(return_value=False)

    with patch("equilens.core.ports.socket.socket", return_value=mock_sock):
        assert is_port_available(12345) is True

    mock_sock.bind.assert_called_once_with(("0.0.0.0", 12345))


def test_is_port_available_returns_false_on_oserror():
    mock_sock = MagicMock()
    mock_sock.__enter__ = lambda s: s
    mock_sock.__exit__ = MagicMock(return_value=False)
    mock_sock.bind.side_effect = OSError("address already in use")

    with patch("equilens.core.ports.socket.socket", return_value=mock_sock):
        assert is_port_available(12345) is False


# ---------------------------------------------------------------------------
# find_available_port
# ---------------------------------------------------------------------------


def test_find_available_port_returns_first_available():
    # First port is available immediately
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert find_available_port(9000) == 9000


def test_find_available_port_skips_busy_ports():
    # First two ports busy, third available
    side_effects = [False, False, True]
    with patch("equilens.core.ports.is_port_available", side_effect=side_effects):
        assert find_available_port(9000, max_attempts=5) == 9002


def test_find_available_port_raises_when_all_busy():
    with patch("equilens.core.ports.is_port_available", return_value=False):
        with pytest.raises(RuntimeError, match="Could not find available port"):
            find_available_port(9000, max_attempts=3)


# ---------------------------------------------------------------------------
# get_backend_port
# ---------------------------------------------------------------------------


def test_get_backend_port_default_when_available(monkeypatch):
    monkeypatch.delenv("BACKEND_PORT", raising=False)
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert get_backend_port() == 8000


def test_get_backend_port_env_var_available(monkeypatch):
    monkeypatch.setenv("BACKEND_PORT", "9100")
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert get_backend_port() == 9100


def test_get_backend_port_env_var_busy_finds_next(monkeypatch):
    monkeypatch.setenv("BACKEND_PORT", "9200")

    def available(port):
        # 9200 is busy, 9201 is free
        return port != 9200

    with patch("equilens.core.ports.is_port_available", side_effect=available):
        assert get_backend_port() == 9201


def test_get_backend_port_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BACKEND_PORT", "not_a_number")
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert get_backend_port() == 8000


def test_get_backend_port_default_busy_finds_next(monkeypatch):
    monkeypatch.delenv("BACKEND_PORT", raising=False)

    def available(port):
        return port != 8000

    with patch("equilens.core.ports.is_port_available", side_effect=available):
        assert get_backend_port() == 8001


# ---------------------------------------------------------------------------
# get_frontend_port
# ---------------------------------------------------------------------------


def test_get_frontend_port_default_when_available(monkeypatch):
    monkeypatch.delenv("FRONTEND_PORT", raising=False)
    monkeypatch.delenv("GRADIO_PORT", raising=False)
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert get_frontend_port() == 7860


def test_get_frontend_port_env_var(monkeypatch):
    monkeypatch.setenv("FRONTEND_PORT", "7900")
    monkeypatch.delenv("GRADIO_PORT", raising=False)
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert get_frontend_port() == 7900


def test_get_frontend_port_gradio_port_env_var(monkeypatch):
    monkeypatch.delenv("FRONTEND_PORT", raising=False)
    monkeypatch.setenv("GRADIO_PORT", "7950")
    with patch("equilens.core.ports.is_port_available", return_value=True):
        assert get_frontend_port() == 7950


# ---------------------------------------------------------------------------
# get_backend_url
# ---------------------------------------------------------------------------


def test_get_backend_url_env_var_takes_priority(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://custom-host:1234")
    # No mocking needed — env var short-circuits everything
    assert get_backend_url() == "http://custom-host:1234"


def test_get_backend_url_docker_mode(monkeypatch):
    monkeypatch.delenv("BACKEND_URL", raising=False)
    # DOCKER_ENV=true is sufficient to trigger docker branch without /.dockerenv
    monkeypatch.setenv("DOCKER_ENV", "true")
    monkeypatch.delenv("BACKEND_HOST", raising=False)

    result = get_backend_url(port=8000)

    assert result == "http://backend:8000"


def test_get_backend_url_local_mode(monkeypatch):
    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.delenv("DOCKER_ENV", raising=False)
    monkeypatch.delenv("BACKEND_HOST", raising=False)

    # Patch pathlib.Path so /.dockerenv appears absent
    with patch("pathlib.Path.exists", return_value=False):
        result = get_backend_url(port=8000)

    assert result == "http://localhost:8000"


# ---------------------------------------------------------------------------
# get_service_ports
# ---------------------------------------------------------------------------


def test_get_service_ports_returns_tuple_of_two_ints(monkeypatch):
    monkeypatch.delenv("BACKEND_PORT", raising=False)
    monkeypatch.delenv("FRONTEND_PORT", raising=False)
    monkeypatch.delenv("GRADIO_PORT", raising=False)

    with patch("equilens.core.ports.is_port_available", return_value=True):
        result = get_service_ports()

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(p, int) for p in result)
