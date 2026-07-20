import sys
from pathlib import Path

from airquality.data import fetch


class _Response:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, list[object]]:
        return {"rows": []}


class _Session:
    verify = True

    def __init__(self) -> None:
        self.closed = False
        self.posts = 0

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def post(self, *args: object, **kwargs: object) -> _Response:
        self.posts += 1
        return _Response()


def test_scraper_closes_session_and_uses_output_dir(tmp_path, monkeypatch) -> None:
    session = _Session()
    monkeypatch.setattr(fetch.requests, "Session", lambda: session)
    monkeypatch.setattr(fetch.time, "sleep", lambda _seconds: None)

    fetch.ejecutar_scraper(contaminantes=["NO2"], output_dir=tmp_path)

    assert session.verify is False
    assert session.closed
    assert session.posts > 0
    assert (tmp_path / "Aquatec - Calle Jorge Juan").is_dir()


def test_parse_args_accepts_output_dir(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fetch", "--output-dir", "custom"])

    assert fetch._parse_args().output_dir == Path("custom")
