# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install optional metric extras
pip install -e ".[asr,fid,tifa,coco]"

# Run all tests
pytest tests/

# Run unit tests only (skip GPU-heavy integration tests)
pytest tests/unit/
pytest -m "not integration"

# Run a single test file
pytest tests/unit/test_benchmark_runner.py -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/eval_learn/

# Run a benchmark
eval-learn run --config examples/single_config.json
eval-learn run --config examples/matrix_config.json

# Push/pull results to HuggingFace Hub
eval-learn push <target> --run-dir <dir> --run-id <id>
eval-learn pull <target> --local-dir <dir>
```

## Architecture

This is a benchmarking framework for evaluating machine unlearning techniques applied to text-to-image diffusion models.

### Runner Modes

The framework has three runner modes, auto-detected from config structure:

- **SingleBenchmarkRunner** — 1 technique × 1 metric. Config has `technique` + `metric` keys.
- **MultiBenchmarkRunner** — 1 technique × N metrics. Config has `technique` + `metrics` (list).
- **MatrixBenchmarkRunner** — N techniques × M metrics. Config has `techniques` (list) + `metrics` (list). Unloads each technique between runs to free VRAM.

### Plugin System

Techniques, metrics, and datasets are plugins discovered via setuptools entry points (defined in `pyproject.toml`) and registered with decorators (`@register_technique`, `@register_metric`, `@register_dataset`) in `src/eval_learn/registry/local.py`. Entry point groups: `eval_learn.techniques`, `eval_learn.metrics`, `eval_learn.datasets`.

**Techniques:** `free_run`, `esd`, `mace`, `sld`, `uce`, `safree`, `concept_steerers`, `saeuron`
**Metrics:** `asr`, `fid`, `err`, `tifa`, `clip_score`, `ccrt`, `ua_ira`
**Datasets:** `i2p_csv`, `err_composite`, `tifa_json`, `coco_parquet`, `ccrt_genetic`, `ua_ira_hf`

### Validation

`src/eval_learn/runners/validation.py` enforces technique-metric compatibility rules **before** any model is loaded. Key rules:

- `ccrt` metric requires `free_run` technique only
- `asr`/`err` metrics require `nudity` concept
- Nudity-specific techniques (`sld`, `safree`, `saeuron`, `concept_steerers`) only support nudity concept
- `uce` is limited to three presets: `nudity`, `violence`, `dog`
- `ua_ira` requires CSV paths in its metric config

Full compatibility rules are documented in [docs/COMPATIBILITY_MATRIX.md](docs/COMPATIBILITY_MATRIX.md).

### Data Flow

Config (JSON/YAML) → CLI parser → Runner (validates, discovers plugins) → Instantiate technique + metric → Load dataset → Generate images → Compute metric → Save artifacts (JSON report + images) → Optional HF Hub push.

### Config Structure

Each run config specifies `output_dir`, a dataset block (`name` + `config`), and technique/metric blocks. Example:

```json
{
  "run_name": "my_run",
  "output_dir": "results/my_run",
  "dataset": {"name": "i2p_csv", "config": {}},
  "technique": {"name": "sld", "config": {"device": "cuda"}},
  "metric": {"name": "asr", "config": {"use_nudenet": false}}
}
```

See `examples/` for complete config examples for all three runner modes.

### Output

Results are saved under `results/{run_id}/report.json` and optionally `results/{run_id}/images/`. The `ArtifactWriter` in `src/eval_learn/artifacts/writer.py` manages this.

### Environment Variables

Required for HF Hub sync (via `.env`):
- `HF_TOKEN` — HuggingFace authentication
- `HF_DATASETS_REPO`, `HF_RESULTS_REPO`, `HF_IMAGES_REPO` — target repositories
