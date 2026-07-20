from airquality.paths import create_run_dir


def test_create_run_dir_adds_suffix_on_collision(tmp_path) -> None:
    first = create_run_dir(tmp_path, "run")
    second = create_run_dir(tmp_path, "run")

    assert first.name == "run"
    assert second.name == "run_1"
