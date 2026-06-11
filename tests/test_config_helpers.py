"""Tests config loading and the typed helper accessors."""

from __future__ import annotations

from configparser import ConfigParser

from airquality import config


def test_load_config_uses_last_loaded_value_when_candidates_overlap(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.cfg"
    second = tmp_path / "second.cfg"
    first.write_text("[data]\nkey_word = NO2\n", encoding="utf-8")
    second.write_text("[data]\nkey_word = CO\n", encoding="utf-8")

    monkeypatch.setattr(config, "_candidate_config_paths", lambda: [first, second])

    cfg = config.load_config()

    assert cfg.get("data", "key_word") == "CO"


def test_cfg_get_helpers_and_csv_list_use_defaults_and_trim(monkeypatch) -> None:
    cfg = ConfigParser()
    cfg.read_dict(
        {
            "section": {
                "text": " value ",
                "count": "7",
                "ratio": "2.5",
                "enabled": "yes",
                "names": " A, B ,, C ",
                "empty_names": " , , ",
            }
        }
    )
    monkeypatch.setattr(config, "CONFIG", cfg)

    assert config.cfg_get_str("section", "text", "x") == " value "
    assert config.cfg_get_int("section", "count", 0) == 7
    assert config.cfg_get_float("section", "ratio", 0.0) == 2.5
    assert config.cfg_get_bool("section", "enabled", False) is True
    assert config.cfg_get_csv_list("section", "names", ("fallback",)) == ("A", "B", "C")
    assert config.cfg_get_csv_list("section", "empty_names", ("fallback",)) == ("fallback",)
    assert config.cfg_get_csv_list("section", "missing", ("fallback",)) == ("fallback",)


def test_cfg_get_helpers_accept_explicit_config_without_mutating_global() -> None:
    global_cfg = ConfigParser()
    global_cfg.read_dict({"section": {"value": "global"}})

    local_cfg = ConfigParser()
    local_cfg.read_dict({"section": {"value": "local", "count": "9", "names": "x, y"}})

    original = config.CONFIG
    config.CONFIG = global_cfg
    try:
        assert config.get_config() is global_cfg
        assert config.get_config(local_cfg) is local_cfg
        assert config.cfg_get_str("section", "value", "fallback") == "global"
        assert config.cfg_get_str("section", "value", "fallback", cfg=local_cfg) == "local"
        assert config.cfg_get_int("section", "count", 0, cfg=local_cfg) == 9
        assert config.cfg_get_csv_list("section", "names", ("fallback",), cfg=local_cfg) == (
            "x",
            "y",
        )
    finally:
        config.CONFIG = original
