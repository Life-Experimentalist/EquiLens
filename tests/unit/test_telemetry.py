"""Tests for equilens.telemetry module."""

import json
from unittest.mock import MagicMock, patch

from equilens.telemetry import _DEFAULTS, fmt, load, stats_html, stats_markdown

# ---------------------------------------------------------------------------
# fmt()
# ---------------------------------------------------------------------------


def test_fmt_zero():
    assert fmt(0) == "0"


def test_fmt_thousands():
    assert fmt(1000) == "1,000"


def test_fmt_large():
    assert fmt(94200) == "94,200"


def test_fmt_default_audits():
    assert fmt(1847) == "1,847"


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_returns_all_required_keys():
    result = load()
    for key in _DEFAULTS:
        assert key in result


def test_load_returns_defaults_when_files_raises():
    with patch("equilens.telemetry.files", side_effect=Exception("no resource")):
        result = load()
    assert result == _DEFAULTS


def test_load_merges_file_data_overrides_defaults():
    override = {"audits_completed": 9999, "models_evaluated": 99}
    mock_resource = MagicMock()
    mock_resource.joinpath.return_value.read_text.return_value = json.dumps(override)

    with patch("equilens.telemetry.files", return_value=mock_resource):
        result = load()

    assert result["audits_completed"] == 9999
    assert result["models_evaluated"] == 99
    # Keys not in the file still come from defaults
    assert result["prompts_processed"] == _DEFAULTS["prompts_processed"]


def test_load_falls_back_to_defaults_on_invalid_json():
    mock_resource = MagicMock()
    mock_resource.joinpath.return_value.read_text.return_value = "not valid json {{{"

    with patch("equilens.telemetry.files", return_value=mock_resource):
        result = load()

    assert result == _DEFAULTS


def test_load_returns_defaults_when_read_text_raises():
    mock_resource = MagicMock()
    mock_resource.joinpath.return_value.read_text.side_effect = FileNotFoundError

    with patch("equilens.telemetry.files", return_value=mock_resource):
        result = load()

    assert result == _DEFAULTS


# ---------------------------------------------------------------------------
# stats_markdown()
# ---------------------------------------------------------------------------


def test_stats_markdown_returns_string():
    assert isinstance(stats_markdown(), str)


def test_stats_markdown_contains_formatted_numbers():
    result = stats_markdown()
    assert "1,847" in result
    assert "94,200" in result


def test_stats_markdown_contains_expected_labels():
    result = stats_markdown()
    assert "bias audits completed" in result
    assert "models evaluated" in result
    assert "prompts processed" in result


# ---------------------------------------------------------------------------
# stats_html()
# ---------------------------------------------------------------------------


def test_stats_html_returns_string():
    assert isinstance(stats_html(), str)


def test_stats_html_contains_div_elements():
    result = stats_html()
    assert "<div" in result
    assert "</div>" in result


def test_stats_html_contains_expected_labels():
    result = stats_html()
    assert "Bias Audits" in result
    assert "Models Tested" in result
    assert "Prompts Run" in result


def test_stats_html_contains_formatted_number():
    result = stats_html()
    assert "94,200" in result
