import pytest

from airquality.train import _select_trainable_methods, main


def test_main_trains_from_config(monkeypatch) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr("airquality.train.cfg_get_str", lambda *args, **kwargs: "h")

    def fake_cfg_get_int(section, option, default, cfg=None):
        del default, cfg
        values = {
            ("benchmark", "size_k"): 5,
            ("benchmark", "val_size"): 48,
            ("benchmark", "val_context_len"): 72,
            ("benchmark", "min_train_len_base"): 72,
        }
        return values[(section, option)]

    monkeypatch.setattr("airquality.train.cfg_get_int", fake_cfg_get_int)

    def fake_cfg_get_csv_list(section, option, default):
        del default
        calls["model_config_key"] = (section, option)
        return ("TiDE", "NHiTS")

    monkeypatch.setattr("airquality.train.cfg_get_csv_list", fake_cfg_get_csv_list)
    monkeypatch.setattr(
        "airquality.train.build_model_configs",
        lambda: {"TiDE": object(), "NHiTS": object()},
    )

    def fake_load_and_normalize_series(**kwargs):
        calls["series_kwargs"] = kwargs
        return ["series"]

    monkeypatch.setattr(
        "airquality.train.load_and_normalize_series",
        fake_load_and_normalize_series,
    )

    class DummySegment:
        empty = False

    segment = DummySegment()

    def fake_get_longest_segment(series_dfs, verbose=False):
        calls["segment_kwargs"] = {"series_dfs": series_dfs, "verbose": verbose}
        return segment

    monkeypatch.setattr(
        "airquality.train.get_longest_segment",
        fake_get_longest_segment,
    )

    def fake_build_training_dataset_bundle(**kwargs):
        calls["bundle_kwargs"] = kwargs
        return "bundle"

    monkeypatch.setattr(
        "airquality.train.build_training_dataset_bundle",
        fake_build_training_dataset_bundle,
    )

    def fake_train_global_methods(**kwargs):
        calls["train_kwargs"] = kwargs
        return {"TiDE": object()}

    monkeypatch.setattr(
        "airquality.train.train_global_methods",
        fake_train_global_methods,
    )

    main()

    assert calls["series_kwargs"] == {"freq": "h", "name_from_path": True}
    assert calls["segment_kwargs"] == {"series_dfs": ["series"], "verbose": False}
    assert calls["bundle_kwargs"] == {
        "series_dfs": ["series"],
        "longest_segment": segment,
        "val_size": 48,
        "min_train_len": 77,
        "val_context_len": 72,
    }
    assert calls["model_config_key"] == ("training", "model_names")
    assert calls["train_kwargs"] == {
        "dataset_bundle": "bundle",
        "size_k": 5,
        "method_names": ["TiDE", "NHiTS"],
    }


def test_select_trainable_methods_rejects_non_darts_names(monkeypatch) -> None:
    monkeypatch.setattr(
        "airquality.train.build_model_configs",
        lambda: {"TiDE": object()},
    )

    with pytest.raises(ValueError, match=r"\[training\] model_names.*TSPulse"):
        _select_trainable_methods(["TiDE", "TSPulse"])
