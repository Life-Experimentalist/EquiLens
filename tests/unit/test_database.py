"""Unit tests for src/equilens/backend/database.py."""

import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect get_db_path() to tmp_path and reset the thread-local connection."""
    import equilens.backend.database as db_mod

    db_file = tmp_path / "test_jobs.db"
    monkeypatch.setattr(db_mod, "get_db_path", lambda: db_file)

    # Clear any cached thread-local connection so the test gets a fresh DB.
    if hasattr(db_mod._thread_local, "connection"):
        try:
            db_mod._thread_local.connection.close()
        except Exception:
            pass
        del db_mod._thread_local.connection

    db_mod.init_db()

    yield

    # Teardown: close connection so tmp_path can be cleaned up on Windows.
    if hasattr(db_mod._thread_local, "connection"):
        try:
            db_mod._thread_local.connection.close()
        except Exception:
            pass
        del db_mod._thread_local.connection


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


class TestCreateJob:
    def test_returns_true_on_success(self):
        from equilens.backend.database import JobDatabase

        result = JobDatabase.create_job("job-1", "audit")
        assert result is True

    def test_returns_false_on_duplicate_job_id(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-dup", "audit")
        result = JobDatabase.create_job("job-dup", "audit")
        assert result is False

    def test_initial_status_is_queued(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-status", "audit")
        job = JobDatabase.get_job("job-status")
        assert job["status"] == "queued"

    def test_config_stored_when_provided(self):
        from equilens.backend.database import JobDatabase

        cfg = {"model": "llama3", "bias_types": ["gender"]}
        JobDatabase.create_job("job-cfg", "audit", config=cfg)
        job = JobDatabase.get_job("job-cfg")
        assert job["config"] == cfg

    def test_config_none_when_not_provided(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-nocfg", "audit")
        job = JobDatabase.get_job("job-nocfg")
        assert job["config"] is None

    def test_different_job_ids_both_succeed(self):
        from equilens.backend.database import JobDatabase

        assert JobDatabase.create_job("job-a", "audit") is True
        assert JobDatabase.create_job("job-b", "analyze") is True


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_returns_none_for_unknown_id(self):
        from equilens.backend.database import JobDatabase

        assert JobDatabase.get_job("nonexistent") is None

    def test_returns_dict_for_known_id(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-get", "audit")
        result = JobDatabase.get_job("job-get")
        assert isinstance(result, dict)

    def test_correct_job_id_returned(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-id-check", "audit")
        job = JobDatabase.get_job("job-id-check")
        assert job["job_id"] == "job-id-check"

    def test_correct_job_type_returned(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-type-check", "analyze")
        job = JobDatabase.get_job("job-type-check")
        assert job["job_type"] == "analyze"

    def test_config_deserialized_from_json(self):
        from equilens.backend.database import JobDatabase

        cfg = {"key": "value", "nested": {"a": 1}}
        JobDatabase.create_job("job-json", "audit", config=cfg)
        job = JobDatabase.get_job("job-json")
        assert job["config"] == cfg
        assert isinstance(job["config"], dict)

    def test_created_at_field_present(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-ts", "audit")
        job = JobDatabase.get_job("job-ts")
        assert job["created_at"] is not None
        assert len(job["created_at"]) > 0


# ---------------------------------------------------------------------------
# update_job
# ---------------------------------------------------------------------------


class TestUpdateJob:
    def test_returns_false_with_no_updates(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-noupdate", "audit")
        result = JobDatabase.update_job("job-noupdate")
        assert result is False

    def test_update_status(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-upd-status", "audit")
        JobDatabase.update_job("job-upd-status", status="running")
        job = JobDatabase.get_job("job-upd-status")
        assert job["status"] == "running"

    def test_sets_started_at_on_first_running_transition(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-start", "audit")
        assert JobDatabase.get_job("job-start")["started_at"] is None
        JobDatabase.update_job("job-start", status="running")
        job = JobDatabase.get_job("job-start")
        assert job["started_at"] is not None

    def test_does_not_overwrite_started_at_on_second_running_call(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-start2", "audit")
        JobDatabase.update_job("job-start2", status="running")
        first_started_at = JobDatabase.get_job("job-start2")["started_at"]
        # Simulate a second "running" update (e.g. re-queued edge case)
        JobDatabase.update_job("job-start2", status="queued")
        JobDatabase.update_job("job-start2", status="running")
        second_started_at = JobDatabase.get_job("job-start2")["started_at"]
        assert first_started_at == second_started_at

    def test_sets_completed_at_on_completed(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-comp", "audit")
        JobDatabase.update_job("job-comp", status="running")
        JobDatabase.update_job("job-comp", status="completed")
        job = JobDatabase.get_job("job-comp")
        assert job["completed_at"] is not None

    def test_sets_completed_at_on_failed(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-fail", "audit")
        JobDatabase.update_job("job-fail", status="failed")
        job = JobDatabase.get_job("job-fail")
        assert job["completed_at"] is not None

    def test_sets_completed_at_on_cancelled(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-cancel", "audit")
        JobDatabase.update_job("job-cancel", status="cancelled")
        job = JobDatabase.get_job("job-cancel")
        assert job["completed_at"] is not None

    def test_update_progress(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-prog", "audit")
        JobDatabase.update_job("job-prog", progress=42)
        assert JobDatabase.get_job("job-prog")["progress"] == 42

    def test_update_total(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-total", "audit")
        JobDatabase.update_job("job-total", total=200)
        assert JobDatabase.get_job("job-total")["total"] == 200

    def test_update_pid(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-pid", "audit")
        JobDatabase.update_job("job-pid", pid=12345)
        assert JobDatabase.get_job("job-pid")["pid"] == 12345

    def test_update_result_path(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-rpath", "audit")
        JobDatabase.update_job("job-rpath", result_path="/results/audit.csv")
        assert JobDatabase.get_job("job-rpath")["result_path"] == "/results/audit.csv"

    def test_update_error_message(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-err", "audit")
        JobDatabase.update_job("job-err", error_message="Something went wrong")
        assert JobDatabase.get_job("job-err")["error_message"] == "Something went wrong"

    def test_update_progress_zero_is_valid(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-prog0", "audit")
        JobDatabase.update_job("job-prog0", progress=50)
        JobDatabase.update_job("job-prog0", progress=0)
        assert JobDatabase.get_job("job-prog0")["progress"] == 0

    def test_update_pid_zero_is_valid(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("job-pid0", "audit")
        JobDatabase.update_job("job-pid0", pid=0)
        assert JobDatabase.get_job("job-pid0")["pid"] == 0


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_empty_list_when_no_jobs(self):
        from equilens.backend.database import JobDatabase

        assert JobDatabase.list_jobs() == []

    def test_returns_all_jobs(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("list-a", "audit")
        JobDatabase.create_job("list-b", "analyze")
        result = JobDatabase.list_jobs()
        assert len(result) == 2

    def test_ordered_newest_first(self):
        import time

        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("list-old", "audit")
        time.sleep(0.01)
        JobDatabase.create_job("list-new", "audit")
        result = JobDatabase.list_jobs()
        assert result[0]["job_id"] == "list-new"
        assert result[1]["job_id"] == "list-old"

    def test_filters_by_status(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("list-q", "audit")
        JobDatabase.create_job("list-r", "audit")
        JobDatabase.update_job("list-r", status="running")
        queued = JobDatabase.list_jobs(status="queued")
        assert all(j["status"] == "queued" for j in queued)
        assert len(queued) == 1
        assert queued[0]["job_id"] == "list-q"

    def test_status_filter_returns_empty_for_no_match(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("list-only-queued", "audit")
        result = JobDatabase.list_jobs(status="completed")
        assert result == []

    def test_respects_limit(self):
        from equilens.backend.database import JobDatabase

        for i in range(10):
            JobDatabase.create_job(f"limit-{i}", "audit")
        result = JobDatabase.list_jobs(limit=3)
        assert len(result) == 3

    def test_config_deserialized_in_list(self):
        from equilens.backend.database import JobDatabase

        cfg = {"model": "llama3"}
        JobDatabase.create_job("list-cfg", "audit", config=cfg)
        result = JobDatabase.list_jobs()
        job = next(j for j in result if j["job_id"] == "list-cfg")
        assert job["config"] == cfg


# ---------------------------------------------------------------------------
# add_log + get_logs
# ---------------------------------------------------------------------------


class TestLogs:
    def test_log_persists(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("log-job", "audit")
        JobDatabase.add_log("log-job", "INFO", "Started")
        logs = JobDatabase.get_logs("log-job")
        assert len(logs) == 1
        assert logs[0]["message"] == "Started"
        assert logs[0]["level"] == "INFO"

    def test_multiple_logs_for_same_job(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("log-multi", "audit")
        for i in range(5):
            JobDatabase.add_log("log-multi", "INFO", f"Step {i}")
        logs = JobDatabase.get_logs("log-multi")
        assert len(logs) == 5

    def test_get_logs_returns_required_keys(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("log-keys", "audit")
        JobDatabase.add_log("log-keys", "ERROR", "Oops")
        log = JobDatabase.get_logs("log-keys")[0]
        assert "timestamp" in log
        assert "level" in log
        assert "message" in log

    def test_get_logs_respects_limit(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("log-limit", "audit")
        for i in range(20):
            JobDatabase.add_log("log-limit", "DEBUG", f"msg {i}")
        logs = JobDatabase.get_logs("log-limit", limit=5)
        assert len(logs) == 5

    def test_get_logs_returns_empty_for_unknown_job(self):
        from equilens.backend.database import JobDatabase

        assert JobDatabase.get_logs("no-such-job") == []

    def test_logs_isolated_per_job(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("log-iso-a", "audit")
        JobDatabase.create_job("log-iso-b", "audit")
        JobDatabase.add_log("log-iso-a", "INFO", "A message")
        logs_b = JobDatabase.get_logs("log-iso-b")
        assert logs_b == []


# ---------------------------------------------------------------------------
# delete_job
# ---------------------------------------------------------------------------


class TestDeleteJob:
    def test_delete_existing_job_returns_true(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("del-job", "audit")
        assert JobDatabase.delete_job("del-job") is True

    def test_deleted_job_no_longer_retrievable(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("del-gone", "audit")
        JobDatabase.delete_job("del-gone")
        assert JobDatabase.get_job("del-gone") is None

    def test_delete_removes_associated_logs(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("del-logs", "audit")
        JobDatabase.add_log("del-logs", "INFO", "log entry")
        JobDatabase.delete_job("del-logs")
        assert JobDatabase.get_logs("del-logs") == []

    def test_delete_nonexistent_job_returns_false(self):
        from equilens.backend.database import JobDatabase

        assert JobDatabase.delete_job("does-not-exist") is False

    def test_delete_does_not_affect_other_jobs(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("del-target", "audit")
        JobDatabase.create_job("del-keep", "audit")
        JobDatabase.delete_job("del-target")
        assert JobDatabase.get_job("del-keep") is not None

    def test_delete_removes_from_list(self):
        from equilens.backend.database import JobDatabase

        JobDatabase.create_job("del-list", "audit")
        JobDatabase.delete_job("del-list")
        ids = [j["job_id"] for j in JobDatabase.list_jobs()]
        assert "del-list" not in ids
