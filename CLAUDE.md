# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install in editable mode (required before running tests or CLI)
pip install -e .

# Install with optional metric dependencies
pip install -e ".[asr]"       # NudeNet for ASR metric
pip install -e ".[fid]"       # TensorFlow for FID metric
pip install -e ".[dev]"       # pytest, ruff, mypy

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_smoke_asr_sld.py::test_smoke_asr_sld_pipeline

# Run the CLI
eval-learn run --config examples/smoke_config.json
```

## Architecture

This is a benchmarking framework for evaluating "unlearning" techniques in Text-to-Image diffusion models. It uses a **registry-plugin architecture** where the core runner is generic and components are registered via decorators.

### Pipeline Flow

`CLI` → `BenchmarkRunner` → loads **Dataset** → generates images with **Technique** → scores with **Metric** → saves via **ArtifactWriter**

### Registry System (`src/eval_learn/registry/`)

Four parallel registries with decorator-based registration and string-key lookup:
- `@register_technique("sld")` / `get_technique("sld")`
- `@register_metric("asr")` / `get_metric("asr")`
- `@register_dataset("i2p_csv")` / `get_dataset("i2p_csv")`
- `@register_benchmark("asr")` / `get_benchmark("asr")`

Registry keys are **always lowercase**. Every registered module must also be explicitly imported in `cli.py` to trigger its `@register_*` decorator at startup.

### Adding a New Metric/Technique/Dataset

All plugins follow an identical three-file pattern (see any metric under `src/eval_learn/metrics/` for reference):

1. **`config.py`**: A `@dataclass` inheriting `BaseConfig` (from `configs/base.py`). Gets serialized via `to_dict()`/`from_dict()`.
2. **`metric.py`** (or `wrapper.py` for techniques): The implementation class decorated with `@register_metric("name")`. Constructor takes `**kwargs` and builds its config via `MyConfig.from_dict(kwargs)`.
3. **`__init__.py`**: Exports the class and config.

Then wire it in:
- Add import to `src/eval_learn/metrics/__init__.py` (update `__all__`)
- Add `import eval_learn.metrics.<name>.metric` to `src/eval_learn/cli.py`

### Shared Interfaces

- **Technique**: `generate(prompts: List[str], **kwargs) -> List[PIL.Image.Image]`
- **Metric**: `compute(images: List[Any], prompts: List[str], metadata: Optional[Dict] = None) -> MetricResult`
- **Dataset loader**: callable returning `Dataset(prompts: List[str], metadata: Dict)`

Some metrics (ERR, TIFA) require structured data via the `metadata` parameter:
- **ERR**: `metadata["concepts"]` and `metadata["categories"]` (lists parallel to images)
- **TIFA**: `metadata["qa_pairs"]` (list parallel to images, each element is a list of `{"question", "answer"}` dicts)

### Optional Dependencies

Heavy libraries (nudenet, tensorflow, transformers models) are imported inside `try/except` blocks. If missing at runtime, raise `RuntimeError` with an install hint: `pip install eval-learn[extra]`.

## Key Conventions

- **Logging**: Use `logger = get_logger(__name__)` from `eval_learn.logging_utils`. No `print()` in refactored code.
- **Configs**: Always `@dataclass` inheriting `BaseConfig`. `from_dict()` silently ignores unknown keys for forward compatibility.
- **Device detection**: `config.device or ("cuda" if torch.cuda.is_available() else "cpu")`
- **Tests**: Mock heavy models (SD, NudeNet, CLIP, BLIP-2) using `unittest.mock.MagicMock`. The smoke test in `test_smoke_asr_sld.py` demonstrates the pattern — real `BenchmarkRunner`/`ArtifactWriter` with mocked technique and metric factories.

## Project State

The codebase is **mid-refactor**. The new package lives under `src/eval_learn/`. Legacy code (`core/`, `evaluation_metrics/`, `unlearning_techniques/`, `*_benchmark.py`) still exists but should not be modified — all new work goes into the `src/eval_learn/` package. The refactoring plan is documented in `GEMINI.md`; the extension guide is in `DEVELOPER_GUIDE.md`.

Current branch: `dylan/refactor`. Main branch: `master`.
