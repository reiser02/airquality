## Setup

- Use `uv`, not plain `pip`. Install with `uv sync`.
- Python version is `3.11` (`.python-version` and `pyproject.toml` both require it).
- `pyproject.toml` pins `torch==2.7.1+cu118` through a custom `uv` index; do not replace install commands with generic `pip install torch`.

## Verification

- Full test suite: `uv run pytest`
- Single file: `uv run pytest tests/test_tspulse_helpers.py`
- Single test: `uv run pytest tests/test_tspulse_helpers.py -k split_long_train_valid`
- Tests import code from `src/` via `tests/conftest.py`; they do not depend on an editable install.

## Code Layout

- Main package is `src/airquality`.
- `airquality.data` handles file discovery, CSV/JSON loading, datetime normalization, and device/warning helpers.
- `airquality.modeling.training` and `airquality.modeling.training_config` build dataset bundles, define Darts model configs, train models, and save artifacts.
- Three separated subsystems: `airquality.anomaly` (detector **benchmark**), `airquality.imputation` (imputation **benchmark**), and `airquality.forecasting` (the raw-vs-preprocessed forecasting **pipeline** that consumes the other two — `cleaning.py` + `fill.py` + `backtest.py` + `pipeline.py`).
- `airquality.imputation.benchmark` and `airquality.imputation.run_benchmark` run the imputation evaluation pipeline on top of trained model artifacts.
- `airquality.anomaly.benchmark` has two modes (`--mode` / `[anomaly] mode`). **`unlabeled`** (default, matches production): detectors score the real series, scores are binarized with a median + k·MAD threshold, and detectors whose detection rate exceeds `max_detection_rate` (default 7%) are discarded; survivors form the consensus ensemble. **`synthetic`**: anomalies (`combined` shape mix) are injected **directly into the real series** (the STL synthetic base was removed — see `docs/estudio_inyeccion_stl_2026-07-03.md`) twice per station (selection + held-out eval seeds) and detectors are scored with VUS-PR et al.; the ensemble is the top-k by selection VUS-PR. It persists `results.json` only — plots come from the separate `airquality.anomaly.plot_benchmark_results` script (figure set follows the mode). Label-free rationale: `docs/seleccion_detectores_sin_etiquetas.md`.
- Top-level zero-argument CLI entrypoints live in `airquality.train` and `airquality.benchmark`; `airquality.imputation.tspulse_finetune` is the argparse-based fine-tuning CLI. `run_benchmark.py` exposes functions, not an argparse CLI.

## Config And Data

- Runtime config candidates are read in this order: `config/pipeline.cfg`, then `pipeline.cfg` at repo root, then `src/airquality/pipeline.cfg`.
- `ConfigParser.read()` uses last-file-wins semantics, so the package-local config currently has the highest precedence. Keep `_candidate_config_paths()` ordered from lowest precedence to highest precedence.
- Current defaults assume processed hourly CSV data under `data/processed/Datos-post-COUTA/*/` with keyword `NO2`.
- `load_to_df()` only supports `.json` and `.csv`; many loaders silently skip files with unsupported formats or invalid shapes.

## Artifacts And Entry Points

- Trained Darts models are saved to `models/{ModelName}_k{size_k}.pt`; benchmark loading expects that naming convention.
- `train_global_methods()` also appends CSV metrics to `reports/metrics/training_curves_and_times.csv` unless disabled.
- Scraper entrypoint: `uv run python -m airquality.data.fetch`
- TSPulse fine-tuning entrypoint: `uv run python -m airquality.imputation.tspulse_finetune`
- Anomaly benchmark: `uv run python -m airquality.anomaly.run`; plots: `uv run python -m airquality.anomaly.plot_benchmark_results <run>/results.json`.
- Forecasting comparison pipeline (config `[forecasting]`): `uv run python -m airquality.forecasting.pipeline`.

## Important Quirks

- Optional Hugging Face / `tsfm_public` imports in TSPulse modules are intentionally guarded; keep new imports optional there or tests and CPU-only workflows will break.
- Darts model configs default to GPU/mixed precision in `training_config.py`, but `resolve_device()` falls back to CPU when CUDA is unavailable.
- Repo contains committed runtime artifacts like `models/` and `__pycache__/`; avoid treating them as source when searching or editing.
