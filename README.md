## Air Quality Time-Series Benchmark

This repository trains forecasting models on air-quality sensor series, evaluates gap-imputation performance, and supports fine-tuning IBM Granite TSPulse on the same data.

The installable distribution is named `airquality-benchmark`; its Python package remains `airquality`.

The code is organized as a Python package under `src/airquality`, with config-driven entrypoints for:

- training Darts global forecasting models
- running Monte Carlo imputation benchmarks
- scraping raw station measurements
- fine-tuning TSPulse for reconstruction/imputation

## Requirements

- Python `3.11`
- [`uv`](https://github.com/astral-sh/uv)

`pyproject.toml` pins `torch==2.7.1+cu118` through a custom `uv` index. Use `uv sync`; do not replace it with a generic `pip install` flow unless you also manage Torch manually.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/reiser02/airquality.git
cd airquality
```

### 2. Install dependencies

```bash
uv sync
```

## Repository Layout

### Top-level directories

- `src/airquality/`: main package code
- `config/`: shared runtime configuration, especially `config/pipeline.cfg`
- `data/`: processed datasets used by training and benchmarking
- `models/`: saved model checkpoints and fine-tuned artifacts
- `reports/`: benchmark outputs and training metrics
- `tests/`: pytest coverage for config, loaders, training, benchmark helpers, and CLIs
- `notebooks/`: exploratory notebooks
- `docs/`: extra project documentation if present

### Main Python modules

- `src/airquality/config.py`: loads config files and exposes typed helpers such as `cfg_get_int()` and `cfg_get_csv_list()`.
- `src/airquality/train.py`: zero-argument training entrypoint. Reads config, loads data, builds a training bundle, and trains configured Darts models.
- `src/airquality/benchmark.py`: zero-argument Monte Carlo benchmark entrypoint. Runs imputation evaluation and saves CSV summaries and plot images.
- `src/airquality/data/fetch.py`: scraper for Cartagena air-quality API data. Saves per-station pollutant CSV files.
- `src/airquality/data/io.py`: shared data loading, normalization, warning control, and device helpers.
- `src/airquality/data/loaders.py`: file discovery and low-level CSV/JSON loading.
- `src/airquality/data/segments.py`: utilities for finding continuous segments, including the held-out longest segment used in evaluation.
- `src/airquality/modeling/training.py`: dataset preparation and model training helpers.
- `src/airquality/modeling/training_config.py`: model configuration and trainer settings for Darts models.
- `src/airquality/imputation/benchmark.py`: imputation logic and TSPulse integration helpers.
- `src/airquality/imputation/run_benchmark.py`: high-level orchestration for loading trained models and executing the benchmark pipeline.
- `src/airquality/imputation/tspulse_finetune.py`: argparse-based CLI for fine-tuning TSPulse on the repository dataset.
- `src/airquality/visualization/plotting.py`: plotting helpers used by visualization tests and analysis.

## Configuration

The runtime config is read from these locations in order:

1. `config/pipeline.cfg`
2. `pipeline.cfg`
3. `src/airquality/pipeline.cfg`

Because `ConfigParser.read()` uses last-file-wins semantics, later files override earlier ones. That means the package-local config has the highest precedence if it exists.

The main shared config is `config/pipeline.cfg`.

### Important config sections

#### `[data]`

Controls where series are loaded from and their shared time-series schema.

- `data_root`: base dataset directory
- `base_path_glob`: folders searched for files
- `key_word`: pollutant keyword such as `NO2`
- `file_extension`: expected file type, currently `csv`
- `freq`: sampling frequency, default `h`
- `timestamp_column`: datetime column name

#### `[benchmark]`

Controls shared split sizes and direct imputation benchmark behavior.

- `size_k`: forecast/imputation horizon used by the benchmark helpers
- `gap_sizes`: synthetic missing-gap sizes
- `num_gaps`: number of gaps injected per series
- `gap_strategy`: gap placement strategy
- `metrics`: evaluation metrics such as `mae`, `rmse`, `mase`
- `random_seed`: benchmark seed
- `seasonality_m`: seasonality used by MASE
- `val_size`, `val_context_len`, `min_train_len_base`: shared split settings

#### `[imputation]`

Controls the direct imputation benchmark independently from training.

- `model_names`: explicit methods to evaluate; the default is eight Darts models plus `TSPulse`
- `strict_artifacts`: abort when a selected Darts `.pt` or Torch checkpoint is absent; set to `false` to omit incomplete artifacts
- `max_workers`: process limit for model-level parallelism; the default is `1`

#### `[tspulse]`

Controls the TSPulse base checkpoint, optional fine-tuned checkpoint, and fine-tuning.

- `target_column_index`: value column chosen from each file
- `min_non_nan_ratio`: minimum observed ratio required
- `min_series_points`: minimum series length required
- `model_id` and `revision`: Hugging Face base checkpoint used by `TSPulse`
- `finetuned_model_path`: local checkpoint used only by an explicitly selected `TSPulse_FineTuned`
- context length and masking settings
- epochs, batch sizes, learning rate, weight decay
- validation fraction and early stopping
- output directory for fine-tuned artifacts

#### `[training]` and `[models]`

`[training] model_names` explicitly selects the Darts artifacts to train. The
remaining keys control common trainer hyperparameters, while `[models]` controls
per-model architecture parameters.

## Expected Data Layout

Training and imputation use processed hourly CSV data under:

```text
data/processed/Datos-post-COUTA/*/
```

with filenames or paths matching the configured pollutant keyword, currently `NO2`.
Anomaly detection, forecasting, and the block reports instead read raw 5-minute
stations under `data/raw/datos_estaciones_5m/` and apply the shared hourly
preprocessing.

Key assumptions in the loaders:

- only `.csv` and `.json` are supported
- many loaders silently skip unsupported files or invalid shapes
- the default timestamp column is `fecha`
- each workflow expects a loaded station to become a single-column time series

## Main Workflows

### Generate data-block reports

Three diagnostics can be generated under `reports/data_blocks/`. They all read
the raw 5-minute station files, apply the shared hourly preprocessing, and
preserve gaps instead of concatenating observations across missing timestamps.
Use `--output-dir` with any command to select a fixed destination.

#### Available forecasting blocks

```bash
uv run python -m airquality.data.block_analysis
```

This is the fast, model-free pre-study for NO2 and CO. It reserves the fixed
holdout, measures contiguous observed blocks before it, and reports whether
each block can support the configured short/long forecasting and validation
geometry. It does not run anomaly detectors or forecasting models, so the
reported support is an upper bound for detector-aware runs.

The default output is `reports/data_blocks/YYYYMMDD_HHMMSS/` and contains:

- `summary.csv`: aggregate block counts, usable hours, and retention by pollutant
- `series_summary.csv`: holdout, validation, and retained-support details per station and pollutant
- `blocks.csv`: start, end, length, eligibility, and usage of every contiguous block
- `excluded_series.csv`: series that could not provide the required holdout or training history
- `retention_overview.png`, `block_length_distribution.png`, `retained_hours_by_series.png`, and `usable_blocks_by_series.png`

Useful options include `--pollutants`, `--context`, short/long horizon, stride,
and validation lengths, `--holdout`, and `--forecast-models`. Defaults are read
from the forecasting configuration where applicable.

#### Support recovered by bridging short gaps

```bash
uv run python -m airquality.data.gap_bridge_analysis
```

This report estimates how much usable support would be gained by joining
observed runs across short gaps. By default it analyzes NO2, requires a final
component of at least 80 hourly points, and bridges gaps of at most 5 hours.
It distinguishes newly eligible components from extensions of blocks that were
already eligible.

The default output is `reports/data_blocks/gap_bridge_YYYYMMDD_HHMMSS/` and contains:

- `components.csv`: composition, observed/imputed points, age, and recovery type for each joined component
- `summary.csv`: aggregate baseline support, recovered real points, imputation cost, and usable gain
- `gap_bridge_overview.png`: support gain, recovery type, and recency overview

Use `--pollutant`, `--minimum`, and `--max-gap` to change the analysis.

#### Blocks after anomaly detection

```bash
uv run python -m airquality.data.detected_block_analysis
```

This is the detector-aware NO2 audit. It compares raw blocks with every
registered detector, the rate-filtered `unlabeled` consensus, and the
injection-ranked `inject-vote` strategy. Unlike the other two reports, this
command fits and scores detectors and can therefore take substantially longer.
Detection results are cached in `reports/data_blocks/detection_cache/` by default.

The default output is `reports/data_blocks/detected_NO2_YYYYMMDD_HHMMSS/` and contains:

- `summary.csv` and `series_summary.csv`: aggregate and per-station impact of each detector/strategy
- `blocks.csv` and `block_distribution.csv`: resulting blocks and their length distribution
- `detector_coverage.csv`: coverage, detection rates, unlabeled filtering, and unscored segments
- `selection.csv`: detector ranking and selection used by `inject-vote` for each block
- `manifest.json`: effective parameters and detector list
- `README.md`: protocol and principal results for that run
- `retention_overview.png`, `block_length_distribution.png`, `block_impact_by_detector.png`, and `detection_rate_distribution.png`

Use `--cache-dir` to change the detection cache or `--output-dir` to select the
report directory.

### Generate the imputation-support report

This audit measures how many train-eligible hourly points are recovered by each
configured anomaly-detection strategy after imputation. It uses the shared
holdout and the strictest configured training-length requirement, but does not
run forecasting models.

First generate the persisted data from the `[forecasting]` configuration:

```bash
uv run python -m airquality.forecasting.imputation_support_analysis
```

The command reuses the forecasting detection/imputation cache and writes
`training_support.csv` under a new timestamped directory:

```text
reports/forecasting_support/YYYYMMDD_HHMMSS/
```

Then render the figures from that CSV without rerunning detection or imputation:

```bash
uv run python -m airquality.forecasting.plot_imputation_support_analysis \
  reports/forecasting_support/YYYYMMDD_HHMMSS
```

Omit the directory to render the newest run. The report contains:

- `training_support.csv`: per-series and per-strategy support counts and ages
- `support_composition.png`: aggregate support before and after imputation
- `support_gain_by_series.png`: additional usable training days per series
- `support_age.png`: median age of filled hours by strategy
- `support_age_timeline.png`: the same age metric positioned relative to the holdout

### Train the configured forecasting models

This uses `src/airquality/train.py` and reads only from the shared config.

```bash
uv run python -m airquality.train
```

What it does:

- loads the configured dataset
- finds the longest held-out segment
- builds train/validation bundles
- trains the configured Darts models from `training.model_names`

Outputs:

- model checkpoints under `models/{ModelName}_k{size_k}.pt`
- appended training metrics in `reports/metrics/training_curves_and_times.csv` unless disabled in code/config flow

### Run the Monte Carlo imputation benchmark

This uses `src/airquality/benchmark.py`.

```bash
uv run python -m airquality.benchmark
```

What it does:

- evaluates the explicit methods from `imputation.model_names`
- validates every selected Darts artifact before parallel execution
- runs Monte Carlo imputation evaluation
- saves benchmark CSVs and rendered plots

Outputs are written to a timestamped directory like:

```text
reports/benchmark/montecarlo_YYYYMMDD_HHMMSS/
```

Typical files inside that directory:

- `results_mc.csv`: raw benchmark results
- `summary_mc.csv`: aggregated summary metrics
- `ranking_by_seed.csv`: seed-level ranking output
- `plot_images.csv`: manifest of saved plot images
- `plots/gap_*/...png`: per-series benchmark plots

### Fine-tune TSPulse

This uses `src/airquality/imputation/tspulse_finetune.py`.

```bash
uv run python -m airquality.imputation.tspulse_finetune
```

The parser intentionally re-applies config-backed defaults, so the main workflow is config-driven. The CLI currently exposes a small set of explicit flags:

```bash
uv run python -m airquality.imputation.tspulse_finetune --verbose-segment
uv run python -m airquality.imputation.tspulse_finetune --auto-lr
uv run python -m airquality.imputation.tspulse_finetune --dropout 0.1 --head-dropout 0.1
```

What it does:

- discovers configured CSV files
- loads and filters valid series
- builds a long-format train/validation dataset
- masks the held-out longest segment to avoid leakage
- fine-tunes TSPulse and writes artifacts to `tspulse.output_dir`

Notes:

- Hugging Face and `tsfm_public` imports are intentionally optional
- CPU-only workflows should still work when those dependencies are unavailable, but fine-tuning/inference features that need them will not

### Scrape station data

This uses `src/airquality/data/fetch.py`.

```bash
uv run python -m airquality.data.fetch
```

Options:

```bash
uv run python -m airquality.data.fetch --query hourly --pollutants CO NO2 PM10 O3 PM1 PM2.5 --start-date 2026-06-12 --output-dir data/raw/datos_estaciones
```

What it does:

- requests station data from the Cartagena API
- defaults to raw 5-minute data from `2024-01-01`
- fetches `CO`, `NO2`, `PM10`, and `O3` by default
- can fetch hourly SQL averages with `--query hourly`
- accepts pollutant lists with `--pollutants`; `PM2.5` and `PM2_5` are normalized to `PM25`
- saves per-station CSV files under `datos_estaciones/` by default
- accepts `--output-dir` to select another destination

Stations are still defined inside the module; use CLI args for query mode, pollutant lists, the start date, and the output directory.

## Programmatic Usage

If you want to call the workflows from Python instead of the CLI modules:

```python
from airquality.train import train_from_config
from airquality.benchmark import run_benchmark_from_config

training_artifacts = train_from_config()
benchmark_artifacts = run_benchmark_from_config()
```

Useful lower-level modules:

- `airquality.data.io.load_and_normalize_series()` for loading configured datasets
- `airquality.data.fetch.ejecutar_scraper(contaminantes=[...], fecha_inicio=..., query="hourly", output_dir=...)` for scraping from Python
- `airquality.data.segments.get_longest_segment()` for the held-out block selection
- `airquality.modeling.training.build_training_dataset_bundle()` for train/validation construction
- `airquality.imputation.run_benchmark.run_imputation_benchmark_parallel()` for direct benchmark orchestration

## Testing

Run the full suite:

```bash
uv run pytest
```

Run a single file:

```bash
uv run pytest tests/test_tspulse_helpers.py
```

Run a single test subset:

```bash
uv run pytest tests/test_tspulse_helpers.py -k split_long_train_valid
```

Useful test areas:

- `tests/test_train_cli.py`: training entrypoint behavior
- `tests/test_benchmark_cli.py`: benchmark entrypoint behavior
- `tests/test_training_pipeline_helpers.py`: training helpers
- `tests/test_run_imputation_helpers.py`: benchmark orchestration helpers
- `tests/test_config.py` and `tests/test_config_helpers.py`: config precedence and helper behavior

## Important Project Behaviors

- Tests import code from `src/` via `tests/conftest.py`; they do not require an editable install.
- Darts model configs may default to GPU or mixed precision, but runtime device helpers fall back to CPU if CUDA is unavailable.
- Benchmark loading expects trained Darts checkpoints to follow the naming convention `models/{ModelName}_k{size_k}.pt`.
- The repository contains runtime artifacts such as `models/`, `reports/`, `lightning_logs/`, and `__pycache__/`; avoid treating those as source files when navigating the project.

## Quick Start

If you already have processed data in the expected location:

1. Install dependencies with `uv sync`.
2. Review `config/pipeline.cfg` and update data paths or model settings.
3. Train models with `uv run python -m airquality.train`.
4. Run the benchmark with `uv run python -m airquality.benchmark`.
5. Inspect outputs in `models/` and `reports/benchmark/`.
