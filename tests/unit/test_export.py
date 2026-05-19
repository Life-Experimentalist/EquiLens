"""Tests for equilens.backend.export module."""

import zipfile
from pathlib import Path

from equilens.backend.export import create_results_export

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results_dir(tmp_path: Path, name: str = "run_001") -> Path:
    """Return an empty results directory inside tmp_path."""
    d = tmp_path / name
    d.mkdir()
    return d


def _zip_names(zip_path: str) -> list[str]:
    """Return all member names inside a zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


# ---------------------------------------------------------------------------
# Return value / file existence
# ---------------------------------------------------------------------------


def test_returns_string_path_ending_in_zip(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    result = create_results_export(results_dir)
    assert isinstance(result, str)
    assert result.endswith(".zip")


def test_returned_zip_exists_on_disk(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    result = create_results_export(results_dir)
    assert result is not None
    assert Path(result).exists()


# ---------------------------------------------------------------------------
# README.md always present
# ---------------------------------------------------------------------------


def test_zip_contains_readme(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any("README.md" in n for n in names)


# ---------------------------------------------------------------------------
# File-type inclusion
# ---------------------------------------------------------------------------


def test_zip_contains_csv_when_present(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    (results_dir / "audit_results.csv").write_text("col_a,col_b\n1,2")
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any(n.endswith(".csv") for n in names)


def test_zip_contains_png_when_present(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    (results_dir / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any(n.endswith(".png") for n in names)


def test_zip_contains_html_report_when_present(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    (results_dir / "bias_analysis_report.html").write_text("<html></html>")
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any("bias_analysis_report.html" in n for n in names)


def test_zip_contains_md_report_when_present(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    (results_dir / "bias_analysis_report.md").write_text("# Report")
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any("bias_analysis_report.md" in n for n in names)


def test_zip_contains_json_files_when_present(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    (results_dir / "summary.json").write_text('{"key": "value"}')
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any(n.endswith(".json") for n in names)


# ---------------------------------------------------------------------------
# Failure / edge cases
# ---------------------------------------------------------------------------


def test_returns_none_when_export_genuinely_fails(tmp_path, monkeypatch):
    """Force an internal error (e.g. bad archive call) and confirm None is returned."""
    results_dir = _make_results_dir(tmp_path)
    import shutil as _shutil

    def _bad_archive(*_):
        raise OSError("simulated archive failure")

    monkeypatch.setattr(_shutil, "make_archive", _bad_archive)
    # Also patch the reference used inside the module
    import equilens.backend.export as _export_mod

    monkeypatch.setattr(_export_mod.shutil, "make_archive", _bad_archive)
    result = create_results_export(results_dir)
    assert result is None


def test_works_with_empty_results_dir(tmp_path):
    """Empty dir — no source files — README.md should still be in zip."""
    results_dir = _make_results_dir(tmp_path)
    zip_path = create_results_export(results_dir)
    assert zip_path is not None
    names = _zip_names(zip_path)
    assert any("README.md" in n for n in names)
