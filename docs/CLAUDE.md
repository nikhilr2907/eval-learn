# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Eval-Learn is a modular benchmarking framework for evaluating "unlearning" techniques in Text-to-Image (T2I) diffusion models (primarily Stable Diffusion). It measures three dimensions:
- **Safety** — can the model prevent generating harmful content? (ASR metric)
- **Utility** — does the model still produce high-quality, faithful images? (FID, TIFA, CLIP Score)
- **Robustness** — is the safety mechanism resistant to adversarial attacks? (ERR metric)

The project is actively being refactored (branch: `dylan/refactor`) from loose scripts into a pip-installable, registry-based plugin architecture under `src/eval_learn/`.

## Build & Development Commands

```bash
# Install in editable mode (core only)
pip install -e .

# Install with optional extras
pip install -e ".[asr]"          # nudenet for ASR metric
pip install -e ".[fid]"          # tensorflow + pycocotools for FID metric
pip install -e ".[dev]"          # pytest, ruff, mypy
pip install -e ".[asr,fid,dev]"  # all extras

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_registry.py

# Run a single test function
pytest tests/test_registry.py::test_register_and_get_technique

# Lint
ruff check src/

# Type check
mypy src/eval_learn/

# Run CLI
eval-learn run --config config.json

# Run demo scenarios (requires GPU and HF_TOKEN in .env)
python run_demo.py
```

## Architecture

### Registry-Plugin System

The core design pattern is a **decorator-based registry** (`src/eval_learn/registry/local.py`). Four global dictionaries hold registered components:

- `@register_technique("name")` — unlearning/generation techniques
- `@register_metric("name")` — evaluation metrics
- `@register_dataset("name")` — data loaders
- `@register_benchmark("name")` — benchmark definitions

Lookup via `get_technique(name)`, `get_metric(name)`, `get_dataset(name)`, `get_benchmark(name)`. Names are case-insensitive (lowered on registration and lookup).

**Critical detail:** Registration happens at import time via decorators, but imports are not automatic. The CLI (`cli.py`) explicitly imports every module to populate the registries. Any new technique/metric/dataset module **must** be imported in `cli.py` (and `run_demo.py` if used there) to be discoverable.

Third-party plugins can also register via `importlib.metadata` entrypoints (`registry/entrypoints.py`).

### Execution Flow

```
CLI/Script → Config (JSON/YAML) → BenchmarkRunner
  1. dataset_loader(**config)  → Dataset(prompts, metadata)
  2. technique_factory(**config) → technique.generate(prompts) → List[PIL.Image]
  3. metric_factory(**config) → metric.compute(images, prompts, metadata) → MetricResult
  4. ArtifactWriter.save_run() → images/ + report.json
```

`BenchmarkRunner` (`src/eval_learn/runners/benchmark_runner.py`) orchestrates this pipeline. It receives factories (classes/functions from the registry), not instances — it instantiates them with config dicts.

### Shared Types

`src/eval_learn/types.py` defines two dataclasses used across all boundaries:
- `Dataset` — holds `prompts: List[str]` and `metadata: Dict[str, Any]`
- `MetricResult` — holds `name: str`, `value: float`, `details: Dict[str, Any]`

### Configuration

All configs inherit from `BaseConfig` (`src/eval_learn/configs/base.py`), which is a dataclass with `to_dict()` and `from_dict()` methods. `from_dict()` filters unknown keys for forward compatibility.

### Component Contracts

**Techniques** must implement:
- `__init__(self, **kwargs)` — construct config via `XConfig.from_dict(kwargs)`, load model
- `generate(self, prompts: List[str], seed=None, **kwargs) -> List[Any]` — return list of PIL Images

**Metrics** must implement:
- `__init__(self, **kwargs)` — construct config, load model
- `compute(self, images, prompts, metadata) -> MetricResult`

**Dataset loaders** are functions (not classes) decorated with `@register_dataset` that return a `Dataset`.

### Optional Dependency Pattern

Metrics and techniques with heavy or platform-specific dependencies use try/except imports and raise `RuntimeError` with install instructions at init time. Example: ASR requires `nudenet` (`pip install eval-learn[asr]`), FID requires `tensorflow` (`pip install eval-learn[fid]`).

## Current Components

| Registry | Name | Module |
|----------|------|--------|
| Technique | `sld` | `techniques/sld/wrapper.py` — Safe Latent Diffusion (model: `AIML-TUDA/stable-diffusion-safe`) |
| Metric | `asr` | `metrics/asr/metric.py` — Attack Success Rate via NudeNet |
| Metric | `fid` | `metrics/fid/metric.py` — Frechet Inception Distance |
| Metric | `tifa` | `metrics/tifa/metric.py` — Text-to-Image Faithfulness (BLIP-2 VQA) |
| Metric | `err` | `metrics/err/metric.py` — Erase-Retention-Robustness (CLIP-based) |
| Metric | `clip_score` | `metrics/clip_score/metric.py` — CLIP text-image alignment |
| Dataset | `i2p_csv` | `datasets/i2p_csv.py` — I2P Inappropriate Image Prompts |
| Dataset | `ring_a_bell_csv` | `datasets/ring_a_bell_csv.py` — Adversarial prompts |
| Dataset | `err_challenge_csv` | `datasets/err_challenge_csv.py` — ERR retention set |
| Dataset | `err_composite` | `datasets/err_composite.py` — Combines I2P + ERR + Ring-A-Bell |
| Dataset | `tifa_json` | `datasets/tifa_json.py` — TIFA captions + QA pairs |

## Adding New Components

Follow the pattern in `DEVELOPER_GUIDE.md`. In brief:

1. Create `src/eval_learn/<registry_type>/<name>/` with `__init__.py`, `config.py`, `wrapper.py` (or `metric.py`)
2. Config inherits `BaseConfig`; wrapper uses `@register_<type>("name")` decorator
3. Add the import to `src/eval_learn/cli.py` to register the module
4. Use try/except for heavy optional dependencies
5. Add tests in `tests/`

## Testing Notes

- Test fixtures are in `tests/conftest.py` — provides `dummy_pil_image`, temp CSV/JSON files for each dataset type, and a `reset_registry` fixture to avoid cross-test pollution
- Tests use `tmp_path` (pytest built-in) for all file I/O
- Some tests require GPU and model downloads (integration tests like `test_integration_sld_i2p.py`, `test_smoke_asr_sld.py`)
- Unit tests for metrics mock the heavy model parts

## Environment

- Requires a `.env` file with `HF_TOKEN` for downloading gated Hugging Face models
- GPU (CUDA) recommended for image generation; falls back to CPU
- Python >=3.8, with CUDA 12.6 pinned in `requirements.txt` for PyTorch
- Package source lives in `src/eval_learn/`; installed as `eval-learn` (note the hyphen vs underscore)
- `legacy/` directory contains pre-refactor code — do not modify

## CLI Config Format

The CLI expects JSON (or YAML with `pyyaml` installed) with this structure:

```json
{
  "run_name": "My_Run",
  "output_dir": "results/benchmarks",
  "dataset": { "name": "i2p_csv", "config": { "limit": 10 } },
  "technique": { "name": "sld", "config": { "device": "cuda" } },
  "metric": { "name": "asr", "config": { "use_nudenet": true } }
}
```
