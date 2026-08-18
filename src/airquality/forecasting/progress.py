"""Structured, process-safe progress logging for the forecasting benchmark."""

from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp
from pathlib import Path
import sys
import threading
import time
from typing import Any


LOGGER_NAME = "airquality.forecasting.progress"


def get_progress_logger() -> logging.Logger:
    """Return the dedicated logger unaffected by third-party root log levels."""
    return logging.getLogger(LOGGER_NAME)


def configure_worker_progress_logging(log_queue: Any) -> None:
    """Route one spawned worker's progress records through the parent queue."""
    logger = get_progress_logger()
    logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class BenchmarkProgress:
    """Serialize progress records and emit periodic state heartbeats."""

    def __init__(
        self,
        output_dir: Path,
        *,
        heartbeat_seconds: float = 60.0,
        stream: Any = None,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds debe ser positivo")

        context = mp.get_context("spawn")
        self.log_queue = context.Queue()
        self.log_path = output_dir / "benchmark.log"
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._stream = sys.stderr if stream is None else stream
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = {
            "stage": "initializing",
            "detail": "",
            "completed": 0,
            "total": 0,
            "pending": 0,
        }
        self._started_at = time.perf_counter()
        self._thread: threading.Thread | None = None
        self._listener: QueueListener | None = None
        self._handlers: list[logging.Handler] = []
        self._previous_logger_state: tuple[list[logging.Handler], int, bool] | None = None
        self._closed = False

    def start(self) -> "BenchmarkProgress":
        """Start queue consumption and the heartbeat thread."""
        if self._listener is not None:
            return self

        formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        stream_handler = logging.StreamHandler(self._stream)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        for handler in (stream_handler, file_handler):
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
        self._handlers = [stream_handler, file_handler]
        self._listener = QueueListener(
            self.log_queue,
            *self._handlers,
            respect_handler_level=True,
        )
        self._listener.start()

        logger = get_progress_logger()
        self._previous_logger_state = (list(logger.handlers), logger.level, logger.propagate)
        configure_worker_progress_logging(self.log_queue)

        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="forecasting-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def update(
        self,
        *,
        stage: str | None = None,
        detail: str | None = None,
        completed: int | None = None,
        total: int | None = None,
        pending: int | None = None,
    ) -> None:
        """Update the state shown by future heartbeat records."""
        with self._lock:
            if stage is not None:
                self._state["stage"] = stage
            if detail is not None:
                self._state["detail"] = detail
            if completed is not None:
                self._state["completed"] = int(completed)
            if total is not None:
                self._state["total"] = int(total)
            if pending is not None:
                self._state["pending"] = int(pending)

    def emit_heartbeat(self) -> None:
        """Emit the current state immediately; public for deterministic tests."""
        with self._lock:
            state = dict(self._state)
        progress = (
            f"{state['completed']}/{state['total']}"
            if state["total"]
            else str(state["completed"])
        )
        detail = f" detail={state['detail']}" if state["detail"] else ""
        get_progress_logger().info(
            "[heartbeat] stage=%s completed=%d total=%d progress=%s pending_gpu=%d "
            "elapsed=%s%s",
            state["stage"],
            state["completed"],
            state["total"],
            progress,
            state["pending"],
            _format_elapsed(time.perf_counter() - self._started_at),
            detail,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_seconds):
            self.emit_heartbeat()

    def close(self) -> None:
        """Stop background resources after flushing all queued records."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._heartbeat_seconds))

        logger = get_progress_logger()
        logger.handlers.clear()
        if self._previous_logger_state is not None:
            handlers, level, propagate = self._previous_logger_state
            logger.handlers.extend(handlers)
            logger.setLevel(level)
            logger.propagate = propagate

        if self._listener is not None:
            self._listener.stop()
        for handler in self._handlers:
            handler.close()
        self.log_queue.close()
        self.log_queue.join_thread()

    def __enter__(self) -> "BenchmarkProgress":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "BenchmarkProgress",
    "LOGGER_NAME",
    "configure_worker_progress_logging",
    "get_progress_logger",
]
