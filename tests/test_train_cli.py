import pandas as pd
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
            ("benchmark", "holdout_target_points"): 192,
            ("benchmark", "holdout_context_points"): 72,
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

    series_dfs = [pd.DataFrame({"S": [1.0]})]

    def fake_load_and_normalize_series(**kwargs):
        calls["series_kwargs"] = kwargs
        return series_dfs

    monkeypatch.setattr(
        "airquality.train.load_and_normalize_series",
        fake_load_and_normalize_series,
    )

    holdouts = {"S": object()}
    holdout_metadata = pd.DataFrame(
        {
            "Serie": ["S"],
            "Test_Start": ["2024-01-01"],
            "Test_End": ["2024-01-08"],
            "Test_Block_Points": [192],
        }
    )
    monkeypatch.setattr(
        "airquality.train.select_retrospective_holdouts_with_exclusions",
        lambda series_dfs, **kwargs: calls.setdefault(
            "holdout_kwargs", {"series_dfs": series_dfs, **kwargs}
        )
        and (holdouts, holdout_metadata, pd.DataFrame()),
    )
    monkeypatch.setattr(
        "airquality.train.build_holdout_manifest",
        lambda metadata, **kwargs: {"split_id": "split", "darts_models": ["TiDE", "NHiTS"]},
    )
    monkeypatch.setattr(
        "airquality.train.write_holdout_manifest",
        lambda manifest, path: path,
    )
    monkeypatch.setattr(
        "pandas.DataFrame.to_csv",
        lambda self, path, index=False: None,
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

    assert calls["series_kwargs"] == {"freq": "h"}
    assert calls["holdout_kwargs"]["series_dfs"] is series_dfs
    assert {key: value for key, value in calls["holdout_kwargs"].items() if key != "series_dfs"} == {
        "target_points": 192,
        "context_points": 72,
        "min_train_points": 125,
    }
    assert calls["bundle_kwargs"] == {
        "series_dfs": calls["holdout_kwargs"]["series_dfs"],
        "holdouts_by_series": holdouts,
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
