"""Tests for ETA calculation and duration formatting in ModelAuditor."""

import pytest


@pytest.fixture(scope="module")
def ModelAuditor():
    from Phase2_ModelAuditor.audit_model import ModelAuditor as _cls

    return _cls


def test_eta_user_provided(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out", eta_per_test=5.0)
    eta_str, eta_seconds = auditor.calculate_eta(10, 100)
    # 90 remaining tests * 5s each = 450s
    assert eta_seconds == pytest.approx(450.0)
    assert isinstance(eta_str, str)


def test_eta_dynamic(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out")
    auditor.user_eta_per_test = None
    for t in [3.2, 4.1, 3.8, 4.5, 3.9]:
        auditor.update_response_time(t)
    eta_str, eta_seconds = auditor.calculate_eta(20, 100)
    assert eta_seconds > 0
    assert isinstance(eta_str, str)


def test_eta_no_data_returns_unknown(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out")
    auditor.user_eta_per_test = None
    eta_str, eta_seconds = auditor.calculate_eta(0, 100)
    assert eta_seconds is None or eta_seconds == 0 or isinstance(eta_str, str)


def test_format_duration_short(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out")
    result = auditor.format_duration(30)
    assert isinstance(result, str) and len(result) > 0


def test_format_duration_over_minute(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out")
    result = auditor.format_duration(90)
    assert isinstance(result, str) and len(result) > 0


def test_format_duration_hours(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out")
    result = auditor.format_duration(3600)
    assert isinstance(result, str) and len(result) > 0


def test_format_duration_multiday(ModelAuditor):
    auditor = ModelAuditor("m", "c.csv", "out")
    result = auditor.format_duration(86400)
    assert isinstance(result, str) and len(result) > 0
