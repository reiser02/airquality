"""Persistent logging for long-running benchmark commands."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from types import TracebackType
from typing import IO


class RunLogging:
    """Write one run's package logs to the terminal and ``benchmark.log``."""

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str,
        *,
        stream: IO[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "benchmark.log"
        self.run_name = run_name
        self._stream = sys.stderr if stream is None else stream
        self._logger = logging.getLogger("airquality")
        self._previous_state: tuple[list[logging.Handler], int, bool, bool] | None = None
        self._handlers: list[logging.Handler] = []
        self._started_at = 0.0
        self._active = False

    def __enter__(self) -> "RunLogging":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(self._stream)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        for handler in (stream_handler, file_handler):
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)

        self._previous_state = (
            list(self._logger.handlers),
            self._logger.level,
            self._logger.propagate,
            self._logger.disabled,
        )
        self._handlers = [stream_handler, file_handler]
        self._logger.handlers.clear()
        self._logger.handlers.extend(self._handlers)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.disabled = False
        self._started_at = time.monotonic()
        self._active = True
        self._logger.info("Run started: %s", self.run_name)
        self._logger.info("Run directory: %s", self.output_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        elapsed = time.monotonic() - self._started_at
        try:
            if exc_type is None:
                self._logger.info(
                    "Run completed: %s (elapsed=%.1fs)", self.run_name, elapsed
                )
            elif issubclass(exc_type, KeyboardInterrupt):
                self._logger.error(
                    "Run interrupted: %s (elapsed=%.1fs)", self.run_name, elapsed
                )
            else:
                self._logger.error(
                    "Run failed: %s (elapsed=%.1fs): %s",
                    self.run_name,
                    elapsed,
                    exc,
                    exc_info=(exc_type, exc, traceback),
                )
        finally:
            self.close()

    def close(self) -> None:
        """Flush the run log and restore the logger's previous configuration."""
        if not self._active:
            return
        self._active = False
        for handler in self._handlers:
            handler.flush()
        self._logger.handlers.clear()
        if self._previous_state is not None:
            handlers, level, propagate, disabled = self._previous_state
            self._logger.handlers.extend(handlers)
            self._logger.setLevel(level)
            self._logger.propagate = propagate
            self._logger.disabled = disabled
        for handler in self._handlers:
            handler.close()


__all__ = ["RunLogging"]
