"""Checks for concise forecasting progress and heartbeat logging."""

from __future__ import annotations

from io import StringIO

import pytest

from airquality.forecasting.progress import BenchmarkProgress, get_progress_logger


def test_progress_writes_structured_records_and_heartbeat(tmp_path) -> None:
    stream = StringIO()
    progress = BenchmarkProgress(
        tmp_path,
        heartbeat_seconds=60,
        stream=stream,
    ).start()
    try:
        get_progress_logger().info("[backtest task=2/10][ST0][raw][TiDE] start")
        progress.update(
            stage="backtest",
            detail="station=ST0 model=TiDE",
            completed=2,
            total=10,
            pending=3,
        )
        progress.emit_heartbeat()
    finally:
        progress.close()

    logged = (tmp_path / "benchmark.log").read_text(encoding="utf-8")
    assert "[backtest task=2/10][ST0][raw][TiDE] start" in logged
    assert "[heartbeat] stage=backtest completed=2 total=10 progress=2/10 pending_gpu=3" in logged
    assert "detail=station=ST0 model=TiDE" in logged
    assert "[heartbeat]" in stream.getvalue()


def test_progress_close_is_idempotent(tmp_path) -> None:
    progress = BenchmarkProgress(tmp_path, heartbeat_seconds=60).start()
    progress.close()
    progress.close()


def test_progress_rejects_nonpositive_heartbeat(tmp_path) -> None:
    with pytest.raises(ValueError, match="heartbeat_seconds"):
        BenchmarkProgress(tmp_path, heartbeat_seconds=0)
