from __future__ import annotations

from io import StringIO
import logging

import pytest

from airquality.run_logging import RunLogging


def test_run_logging_writes_status_and_restores_logger(tmp_path) -> None:
    package_logger = logging.getLogger("airquality")
    previous_handlers = list(package_logger.handlers)
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    previous_disabled = package_logger.disabled
    stream = StringIO()

    with RunLogging(tmp_path, "test_run", stream=stream):
        logging.getLogger("airquality.test").info("work item completed")

    logged = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert "Run started: test_run" in logged
    assert "work item completed" in logged
    assert "Run completed: test_run" in logged
    assert "work item completed" in stream.getvalue()
    assert package_logger.handlers == previous_handlers
    assert package_logger.level == previous_level
    assert package_logger.propagate == previous_propagate
    assert package_logger.disabled == previous_disabled


def test_run_logging_preserves_failure_and_interruption(tmp_path) -> None:
    failed_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="broken"):
        with RunLogging(failed_dir, "failed_run"):
            raise RuntimeError("broken")

    failed_log = (failed_dir / "benchmark.log").read_text(encoding="utf-8")
    assert "Run failed: failed_run" in failed_log
    assert "RuntimeError: broken" in failed_log

    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        with RunLogging(interrupted_dir, "interrupted_run"):
            raise KeyboardInterrupt

    interrupted_log = (interrupted_dir / "benchmark.log").read_text(
        encoding="utf-8"
    )
    assert "Run interrupted: interrupted_run" in interrupted_log


def test_consecutive_runs_do_not_duplicate_handlers(tmp_path) -> None:
    for run_number in (1, 2):
        with RunLogging(tmp_path, f"run_{run_number}"):
            logging.getLogger("airquality.test").info("message_%d", run_number)

    logged = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert logged.count("message_1") == 1
    assert logged.count("message_2") == 1
