"""Unit tests for Phase3_Analysis.analytics.BiasAnalytics."""

from unittest.mock import patch

import pandas as pd

from Phase3_Analysis.analytics import BiasAnalytics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DATA = pd.DataFrame(
    {
        "surprisal_score": [1.5, 2.3, 1.8, 2.9, 1.2, 3.1],
        "name_category": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "trait_category": [
            "Competence",
            "Social",
            "Competence",
            "Social",
            "Competence",
            "Social",
        ],
        "profession": ["Engineer", "Nurse", "Engineer", "Nurse", "Doctor", "Doctor"],
        "name": ["John", "Jane", "James", "Mary", "Bob", "Alice"],
    }
)


def _make_analytics(results_file: str) -> BiasAnalytics:
    """Create a BiasAnalytics instance with Ollama network call suppressed."""
    with patch("requests.get", side_effect=Exception("no network in tests")):
        return BiasAnalytics(results_file, ollama_url=None)


def _write_sample_csv(tmp_path, filename="results_llama3.2.csv", df=None) -> str:
    path = tmp_path / filename
    (df if df is not None else SAMPLE_DATA).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# _extract_model_name()
# ---------------------------------------------------------------------------


def test_extract_model_name_basic(tmp_path):
    path = _write_sample_csv(tmp_path, "results_llama3.2.csv")
    ba = _make_analytics(path)
    assert ba.model_name == "llama3.2"


def test_extract_model_name_strips_responses_suffix(tmp_path):
    path = _write_sample_csv(tmp_path, "results_mistral_responses.csv")
    ba = _make_analytics(path)
    assert ba.model_name == "mistral"


def test_extract_model_name_strips_timestamp(tmp_path):
    path = _write_sample_csv(tmp_path, "results_llama2_20240101_120000.csv")
    ba = _make_analytics(path)
    assert ba.model_name == "llama2"


def test_extract_model_name_empty_string(tmp_path):
    # Just ensure no crash for a weird filename
    path = _write_sample_csv(tmp_path, "results_.csv")
    ba = _make_analytics(path)
    # Should not raise; value may be empty or a string
    assert isinstance(ba.model_name, str)


# ---------------------------------------------------------------------------
# load_and_validate_data()
# ---------------------------------------------------------------------------


def test_load_valid_csv_returns_true(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    assert ba.load_and_validate_data() is True


def test_load_populates_df(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.load_and_validate_data()
    assert len(ba.df) == len(SAMPLE_DATA)


def test_load_file_not_found_returns_false(tmp_path):
    ba = _make_analytics(str(tmp_path / "nonexistent.csv"))
    assert ba.load_and_validate_data() is False


def test_load_missing_required_columns_returns_false(tmp_path):
    bad_df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    path = _write_sample_csv(tmp_path, df=bad_df)
    ba = _make_analytics(path)
    assert ba.load_and_validate_data() is False


def test_load_drops_nan_surprisal_rows(tmp_path):
    df_with_nan = SAMPLE_DATA.copy()
    df_with_nan.loc[0, "surprisal_score"] = float("nan")
    path = _write_sample_csv(tmp_path, df=df_with_nan)
    ba = _make_analytics(path)
    ba.load_and_validate_data()
    assert len(ba.df) == len(SAMPLE_DATA) - 1


# ---------------------------------------------------------------------------
# _detect_score_method()
# ---------------------------------------------------------------------------


def test_detect_score_method_no_column_returns_empty(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.df = SAMPLE_DATA.copy()
    assert ba._detect_score_method() == ""


def test_detect_score_method_all_logprobs(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.df = SAMPLE_DATA.copy()
    ba.df["logprobs_used"] = True
    result = ba._detect_score_method()
    assert "100%" in result


def test_detect_score_method_no_logprobs(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.df = SAMPLE_DATA.copy()
    ba.df["logprobs_used"] = False
    result = ba._detect_score_method()
    assert "0%" in result


def test_detect_score_method_mixed(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.df = SAMPLE_DATA.copy()
    ba.df["logprobs_used"] = [True, True, False, False, False, False]
    result = ba._detect_score_method()
    # Mixed: neither 0% nor 100%; should mention both logprobs and timing
    assert "logprobs" in result.lower()
    assert "timing" in result.lower() or "fallback" in result.lower()


# ---------------------------------------------------------------------------
# _set_score_labels()
# ---------------------------------------------------------------------------


def test_set_score_labels_logprobs_majority(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.df = SAMPLE_DATA.copy()
    ba.df["logprobs_used"] = True  # 100% logprobs >= 50%
    ba._set_score_labels()
    assert ba.score_label == "Bias Score"


def test_set_score_labels_timing_fallback(tmp_path):
    path = _write_sample_csv(tmp_path)
    ba = _make_analytics(path)
    ba.df = SAMPLE_DATA.copy()
    # No logprobs_used column → pct = 0 < 50
    if "logprobs_used" in ba.df.columns:
        ba.df = ba.df.drop(columns=["logprobs_used"])
    ba._set_score_labels()
    assert ba.score_label == "Surprisal Score"
