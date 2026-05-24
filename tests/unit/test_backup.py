"""Tests for the periodic backup module."""

import time
import zipfile
from pathlib import Path

import pytest


@pytest.fixture()
def work_dir(tmp_path, monkeypatch):
    """Switch cwd to tmp_path and patch BACKUP_DIR and _PROJECT_ROOT."""
    monkeypatch.chdir(tmp_path)
    import equilens.backup as bk

    monkeypatch.setattr(bk, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(bk, "_PROJECT_ROOT", tmp_path)
    return tmp_path


def _seed(work_dir: Path) -> None:
    """Create minimal results/ and DB for backup tests."""
    (work_dir / "results" / "run1").mkdir(parents=True)
    (work_dir / "results" / "run1" / "audit.csv").write_text("col_a,col_b\n1,2")
    (work_dir / "data" / "jobs").mkdir(parents=True)
    (work_dir / "data" / "jobs" / "equilens_jobs.db").write_bytes(b"SQLite format 3")


def test_create_backup_returns_zip_path(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    assert path.exists()
    assert path.suffix == ".zip"


def test_create_backup_zip_contains_csv(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert any("audit.csv" in n for n in names)


def test_create_backup_zip_contains_db(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert any("equilens_jobs.db" in n for n in names)


def test_create_backup_name_has_timestamp(work_dir):
    _seed(work_dir)
    from equilens.backup import create_backup

    path = create_backup()
    assert path.name.startswith("backup_")
    assert path.name.endswith(".zip")


def test_create_backup_cleans_up_on_error(work_dir, monkeypatch):
    """If zipping fails mid-way, the partial zip is removed."""
    import equilens.backup as bk

    original_zip = zipfile.ZipFile

    call_count = {"n": 0}

    class FailingZip:
        def __init__(self, *a, **kw):
            self._z = original_zip(*a, **kw)

        def __enter__(self):
            self._z.__enter__()
            return self

        def __exit__(self, *a):
            self._z.__exit__(*a)

        def write(self, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise OSError("disk full")
            return self._z.write(*a, **kw)

    _seed(work_dir)
    monkeypatch.setattr(zipfile, "ZipFile", FailingZip)
    with pytest.raises(RuntimeError, match="Backup failed"):
        bk.create_backup()
    # The partial zip must not exist
    backups = list((work_dir / "backups").glob("*.zip"))
    assert backups == []


def test_prune_backups_keeps_retention(work_dir):
    from equilens.backup import BACKUP_DIR, _prune_backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        p = BACKUP_DIR / f"backup_{i:04d}.zip"
        p.write_bytes(b"x")
        time.sleep(0.01)
    _prune_backups(retention=10)
    remaining = list(BACKUP_DIR.glob("backup_*.zip"))
    assert len(remaining) == 10


def test_prune_backups_removes_oldest(work_dir):
    from equilens.backup import BACKUP_DIR, _prune_backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        p = BACKUP_DIR / f"backup_{i:04d}.zip"
        p.write_bytes(b"x")
        time.sleep(0.01)
    _prune_backups(retention=3)
    remaining = sorted(BACKUP_DIR.glob("backup_*.zip"))
    names = [p.name for p in remaining]
    assert "backup_0000.zip" not in names
    assert "backup_0001.zip" not in names
    assert "backup_0004.zip" in names


def test_list_backups_returns_empty_when_no_dir(work_dir):
    from equilens.backup import list_backups

    result = list_backups()
    assert result == []


def test_list_backups_returns_sorted_newest_first(work_dir):
    from equilens.backup import BACKUP_DIR, list_backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["backup_0001.zip", "backup_0002.zip", "backup_0003.zip"]:
        (BACKUP_DIR / name).write_bytes(b"x")
        time.sleep(0.02)
    result = list_backups()
    names = [b["name"] for b in result]
    assert names[0] == "backup_0003.zip"


def test_list_backups_has_required_keys(work_dir):
    from equilens.backup import BACKUP_DIR, list_backups

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    (BACKUP_DIR / "backup_test.zip").write_bytes(b"x")
    result = list_backups()
    assert len(result) == 1
    b = result[0]
    assert "name" in b
    assert "size" in b
    assert "created_at" in b


def test_get_scheduler_status_not_running(work_dir):
    from equilens.backup import get_scheduler_status

    status = get_scheduler_status()
    assert status["running"] is False
    assert status["next_run"] is None


def test_start_and_stop_scheduler(work_dir):
    from equilens.backup import get_scheduler_status, start_scheduler, stop_scheduler

    try:
        start_scheduler(interval_minutes=60)
        status = get_scheduler_status()
        assert status["running"] is True
        assert status["next_run"] is not None
    finally:
        stop_scheduler()
    stopped = get_scheduler_status()
    assert stopped["running"] is False
