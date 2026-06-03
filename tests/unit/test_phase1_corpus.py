"""Unit tests for Phase1_CorpusGenerator."""

import json
from pathlib import Path

import pandas as pd
import pytest

import Phase1_CorpusGenerator.test_config as tc
from Phase1_CorpusGenerator.generate_corpus import generate_single_corpus

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "active_comparison": "gender_bias",
    "comparisons": {
        "gender_bias": {
            "description": "Test gender bias comparison",
            "name_categories": [
                {"category": "Male", "items": ["John", "James"]},
                {"category": "Female", "items": ["Jane", "Mary"]},
            ],
            "professions": ["Engineer", "Nurse"],
            "trait_categories": [
                {"category": "Competence", "items": ["Logical"]},
                {"category": "Social", "items": ["Caring"]},
            ],
            "templates": ["{name} the {profession} is {trait}."],
        }
    },
}


def _write_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg), encoding="utf-8")


def _patch_module_file(monkeypatch, tmp_path: Path) -> None:
    """Make validate() think its __file__ lives in tmp_path."""
    monkeypatch.setattr(tc, "__file__", str(tmp_path / "test_config.py"))


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------


def test_valid_config_returns_zero(tmp_path, monkeypatch):
    _write_config(tmp_path / "word_lists.json", MINIMAL_CONFIG)
    _patch_module_file(monkeypatch, tmp_path)
    assert tc.validate() == 0


def test_missing_active_comparison_key(tmp_path, monkeypatch):
    cfg = {k: v for k, v in MINIMAL_CONFIG.items() if k != "active_comparison"}
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


def test_missing_comparisons_key(tmp_path, monkeypatch):
    cfg = {"active_comparison": "gender_bias"}
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


def test_active_comparison_not_in_comparisons(tmp_path, monkeypatch):
    cfg = {
        "active_comparison": "nonexistent",
        "comparisons": MINIMAL_CONFIG["comparisons"],
    }
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


def test_missing_professions_key(tmp_path, monkeypatch):
    import copy

    cfg = copy.deepcopy(MINIMAL_CONFIG)
    del cfg["comparisons"]["gender_bias"]["professions"]
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


def test_empty_professions_list(tmp_path, monkeypatch):
    import copy

    cfg = copy.deepcopy(MINIMAL_CONFIG)
    cfg["comparisons"]["gender_bias"]["professions"] = []
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


def test_duplicate_names_in_category(tmp_path, monkeypatch):
    import copy

    cfg = copy.deepcopy(MINIMAL_CONFIG)
    cfg["comparisons"]["gender_bias"]["name_categories"][0]["items"] = [
        "John",
        "John",
    ]
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


def test_empty_items_in_trait_category(tmp_path, monkeypatch):
    import copy

    cfg = copy.deepcopy(MINIMAL_CONFIG)
    cfg["comparisons"]["gender_bias"]["trait_categories"][0]["items"] = []
    _write_config(tmp_path / "word_lists.json", cfg)
    _patch_module_file(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        tc.validate()


# ---------------------------------------------------------------------------
# generate_single_corpus() tests
# ---------------------------------------------------------------------------


def test_generate_single_corpus_returns_true(tmp_path):
    result = generate_single_corpus(MINIMAL_CONFIG, "gender_bias", tmp_path)
    assert result is True


def test_generate_single_corpus_creates_csv(tmp_path):
    generate_single_corpus(MINIMAL_CONFIG, "gender_bias", tmp_path)
    expected = tmp_path / "corpus" / "audit_corpus_gender_bias.csv"
    assert expected.exists()


def test_generate_single_corpus_expected_columns(tmp_path):
    generate_single_corpus(MINIMAL_CONFIG, "gender_bias", tmp_path)
    df = pd.read_csv(tmp_path / "corpus" / "audit_corpus_gender_bias.csv")
    expected_cols = {
        "comparison_type",
        "name",
        "name_category",
        "profession",
        "trait",
        "trait_category",
        "template_id",
        "full_prompt_text",
    }
    assert expected_cols.issubset(set(df.columns))


def test_generate_single_corpus_row_count(tmp_path):
    # 4 names × 2 professions × 2 traits × 1 template = 16
    generate_single_corpus(MINIMAL_CONFIG, "gender_bias", tmp_path)
    df = pd.read_csv(tmp_path / "corpus" / "audit_corpus_gender_bias.csv")
    assert len(df) == 16


def test_generate_single_corpus_returns_false_for_unknown_comparison(tmp_path):
    result = generate_single_corpus(MINIMAL_CONFIG, "nonexistent_bias", tmp_path)
    assert result is False


def test_generate_single_corpus_prompt_substitution(tmp_path):
    generate_single_corpus(MINIMAL_CONFIG, "gender_bias", tmp_path)
    df = pd.read_csv(tmp_path / "corpus" / "audit_corpus_gender_bias.csv")
    # Each row should have the name, profession, and trait substituted into the template
    for _, row in df.iterrows():
        assert row["name"] in row["full_prompt_text"]
        assert row["profession"] in row["full_prompt_text"]
        assert row["trait"] in row["full_prompt_text"]
