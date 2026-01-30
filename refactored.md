# Refactoring Master Plan for Eval-Learn (Gemini CLI)

## 0) Role, Objective, and Operating Rules

**Role:** You are Gemini CLI acting as a Senior Python Architect + Packaging Expert.

**Primary Objective:** Refactor the existing Eval-Learn repo (currently a loose script collection) into a **modular, extensible, pip-installable library** under `src/eval_learn/` with a **registry/plugin architecture** so developers can add new unlearning techniques, datasets, and metrics without touching core runner logic.

### Non-Negotiable Operating Rules (Gemini Must Follow)
1. **Work in phases. Do not attempt the entire refactor in one go.**
2. **Do not delete legacy folders early.** Keep old code until new package path runs end-to-end.
3. After each phase, output:
   - (a) **Changed files list**
   - (b) **What changed & why**
   - (c) **Exact validation commands you ran + results**
4. **Do not introduce breaking behavior changes** unless explicitly required; preserve ASR+SLD behavior.
5. **Guard optional dependencies.** If a metric/technique needs an optional package, handle missing imports gracefully and show a helpful error message suggesting the correct `pip install eval-learn[extra]`.
6. **Keep registry keys lowercase** (e.g., `"sld"`, `"asr"`). CLI should use these keys.
7. Prefer:
   - `dataclasses` for configs
   - Python `logging` (no `print()` in production code)
   - clean type hints + docstrings
   
8. Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

---

## 1) Current State Summary (Context)
Eval-Learn benchmarks “unlearning” in text-to-image diffusion models (e.g., Stable Diffusion). It currently has:
- `core/base_benchmark.py`: benchmark loop + I/O + reporting mixed together
- `core/base_technique.py`: abstract technique interface
- `unlearning_techniques/sld_pipeline/sld_wrapper.py`: SLD technique wrapper
- `evaluation_metrics/asr_metric/asr_task.py`: ASR metric + benchmark task
- Other metric folders exist (ERR/TIFA/FID) but may be incomplete/inconsistent.

**Known issues:**
- `print()`-based logging
- raw dict configs passed around
- mixed responsibilities (dataset loading + orchestration + artifact writing)
- unclear extension points for adding techniques/metrics/datasets
- not pip-installable as a clean module

---

## 2) Target Architecture (Package-First, Publish-Later)
We will make the repo **installable locally immediately** (`pip install -e .`) and keep PyPI publishing as a later concern.

### Target Directory Structure (Recommended)
```text
eval-learn/
├── pyproject.toml
├── README.md
├── src/
│   └── eval_learn/
│       ├── __init__.py
│       ├── cli.py
│       ├── logging_utils.py
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── local.py          # local decorator registry
│       │   └── entrypoints.py    # optional: importlib.metadata entrypoints
│       ├── types.py              # shared dataclasses
│       ├── configs/
│       │   ├── __init__.py
│       │   └── base.py           # BaseConfig + (de)serialization helpers
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── base.py           # DatasetLoader protocol
│       │   └── i2p_csv.py        # I2P CSV loader used by ASR
│       ├── techniques/
│       │   ├── __init__.py
│       │   └── sld/
│       │       ├── __init__.py
│       │       ├── config.py
│       │       └── wrapper.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── asr/
│       │       ├── __init__.py
│       │       ├── config.py
│       │       └── metric.py
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── base.py           # Benchmark definition (ties dataset+metric)
│       │   └── asr_benchmark.py
│       ├── runners/
│       │   ├── __init__.py
│       │   └── benchmark_runner.py
│       └── artifacts/
│           ├── __init__.py
│           ├── writer.py          # image saving + report saving
│           └── report.py          # report dataclasses
└── tests/
    └── test_smoke_asr_sld.py
```

### Core Concepts (Minimal & Explicit)

To prevent ambiguity, implement these minimal abstractions:

**Dataset**

* `Dataset(prompts: list[str], metadata: dict[str, Any] = {})`

**Technique**

* `Technique.generate(prompts: list[str], *, seed: int | None = None, **kwargs) -> list[PIL.Image.Image]`

  * (Later can evolve to return `ImageArtifact`, but keep it simple now.)

**Metric**

* `Metric.compute(images: list[Any], prompts: list[str], metadata: dict[str, Any] | None = None) -> MetricResult`

**MetricResult**

* `MetricResult(name: str, value: float, details: dict[str, Any] = {})`

**Benchmark**

* A composition of:

  * dataset loader
  * metric(s)
  * evaluation procedure (usually just: generate → compute → report)

**Runner**

* Orchestrates: load dataset → instantiate technique + metric → run → write artifacts.

---

## 3) Plugin/Registry Strategy (Two-Step)

We will implement registry in two layers:

### Layer 1 (Required): Local Registry

* Decorators:

  * `@register_technique("sld")`
  * `@register_metric("asr")`
  * `@register_dataset("i2p_csv")`
  * `@register_benchmark("asr")`
* Accessors:

  * `get_technique(name)`, `get_metric(name)`, etc.

### Layer 2 (Optional Later): Entry Points

* Use `importlib.metadata.entry_points` to load plugins from external packages:

  * `eval_learn.techniques`
  * `eval_learn.metrics`
  * `eval_learn.datasets`
  * `eval_learn.benchmarks`

**Important:** Do NOT implement entry points until local registry works and a smoke test passes.

---

## 4) Dependencies and Optional Extras

We will use optional extras to avoid forcing heavy deps for all users.

**In pyproject.toml define extras like:**

* `asr`: NudeNet + PIL dependencies if needed
* `diffusers`: diffusers + torch + hf hub
* `dev`: ruff, mypy, pytest
* `all`: union of above

**Optional dependency behavior:**

* If `nudenet` missing and user runs ASR, raise a clear RuntimeError:

  * "ASR metric requires NudeNet. Install with: pip install eval-learn[asr]"

---

## 5) Validation and Acceptance Criteria (Per Phase)

General acceptance rules:

* Local install works: `pip install -e .`
* Imports work: `python -c "import eval_learn"`
* Smoke path works: **ASR + SLD** with a tiny prompt list generates images + report JSON
* No `print()` in refactored paths; use `logging`.

---

## 6) Execution Phases (Do in Order)

### Phase 0 — Recon (No Code Changes)

**Goal:** Understand current working flow and risks.

1. Map the current execution path for ASR+SLD.
2. Identify where I/O, configs, and logging occur.
3. List top refactor risks.

**Output:** Findings only. Do not modify code.

**Acceptance:** None (no edits).

---

### Phase 1 — Packaging Skeleton (Non-Breaking)

**Goal:** Make the repo installable locally without refactoring logic.

1. Add `pyproject.toml` (setuptools backend is fine).
2. Create `src/eval_learn/__init__.py`.
3. Add `eval_learn/cli.py` stub and console script entry `eval-learn`.
4. Add `tests/` folder with a placeholder smoke test (can be skipped until Phase 4).

**Do NOT move or delete legacy folders yet.**

**Acceptance Commands:**

* `pip install -e .`
* `python -c "import eval_learn; print(eval_learn.__version__)"` (or similar)
* `eval-learn --help`

---

### Phase 2 — Logging Standardization (Mechanical, Minimal)

**Goal:** Replace prints in the refactor path with logging.

1. Add `src/eval_learn/logging_utils.py` with `get_logger(name)` and basic config.
2. For any new refactored modules, use logging only.
3. (Optional) Leave legacy code prints for now; do not churn legacy.

**Acceptance Commands:**

* `python -c "from eval_learn.logging_utils import get_logger; get_logger('x').info('ok')"`

---

### Phase 3 — Core Types + Config Base

**Goal:** Introduce shared types + dataclass config helpers.

1. Add `src/eval_learn/types.py`:

   * `Dataset`, `MetricResult`
2. Add `src/eval_learn/configs/base.py`:

   * `BaseConfig` dataclass with `to_dict()` and `from_dict()` (best-effort)
3. Add docstrings + type hints.

**Acceptance Commands:**

* `python -c "from eval_learn.types import Dataset, MetricResult; print(Dataset(['a']).prompts)"`

---

### Phase 4 — Local Registry + Minimal Runner (Still No Heavy Refactor)

**Goal:** Establish extension points first.

1. Implement `src/eval_learn/registry/local.py`:

   * decorators + dict registries + getters
2. Implement `src/eval_learn/runners/benchmark_runner.py`:

   * loads dataset via loader
   * instantiates technique + metric
   * runs generate → compute
   * writes report via artifact writer (can be stubbed initially)

**Acceptance Commands:**

* `python -c "from eval_learn.registry.local import register_metric, get_metric"`

---

### Phase 5 — Migrate SLD (Technique) into Package

**Goal:** Get one technique working in the new system.

1. Create `src/eval_learn/techniques/sld/config.py` (dataclass config).
2. Create `src/eval_learn/techniques/sld/wrapper.py` (ported SLDWrapper) and register it:

   * `@register_technique("sld")`
3. Guard imports (`diffusers`, `torch`, `huggingface_hub`) and fail with install hints if missing.

**Acceptance Commands:**

* `python -c "from eval_learn.registry.local import get_technique; print(get_technique('sld'))"`

---

### Phase 6 — Migrate ASR Metric + Dataset Loader

**Goal:** Get one metric + dataset loader working in the new system.

1. Create `src/eval_learn/datasets/i2p_csv.py` loader (ported from ASR task).
2. Create `src/eval_learn/metrics/asr/metric.py` and register:

   * `@register_metric('asr')`
3. Add config dataclass for ASR as needed.
4. Ensure NudeNet import is optional and handled cleanly.

**Acceptance Commands:**

* `python -c "from eval_learn.registry.local import get_metric; print(get_metric('asr'))"`

---

### Phase 7 — Artifacts Writer + End-to-End Smoke Test

**Goal:** Produce images + JSON report from the new runner.

1. Implement `src/eval_learn/artifacts/writer.py`:

   * save images under `results/...`
   * save report JSON with structured config (not `str(config)`)
2. Implement a minimal benchmark definition or config-driven run.
3. Implement `tests/test_smoke_asr_sld.py`:

   * run with 2–3 prompts max
   * skip if heavy deps missing (or mock technique if needed)

**Acceptance Commands:**

* `pytest -q` (or at least the smoke test file)
* A short run that writes a report JSON

---

### Phase 8 — CLI Real Run Command

**Goal:** Run via CLI with config file.

1. Update `src/eval_learn/cli.py` to implement:

   * `eval-learn run --benchmark asr --technique sld --config path.yaml`
2. Config file supports:

   * `model_id`, `device`, `limit_prompts`, `output_dir`
   * technique config + metric config + dataset config

**Acceptance Commands:**

* `eval-learn run --help`
* `eval-learn run ...` with a tiny prompt set

---

### Phase 9 — Optional: Entry Points Plugins

**Goal:** Enable third-party plugins.
Only after Phase 7 passes:

1. Implement `src/eval_learn/registry/entrypoints.py`.
2. Load entry points and merge into registry at runtime.
3. Add built-in entry point declarations in pyproject.

**Acceptance Commands:**

* `python -c "from eval_learn.registry.entrypoints import load_entrypoints; print(load_entrypoints())"`

---

### Phase 10 — Cleanup (Only When Stable)

**Goal:** Remove legacy folders and finalize docs.

1. Remove old `core/`, `unlearning_techniques/`, `evaluation_metrics/` ONLY when:

   * new runner works end-to-end
   * smoke test passes
2. Add migration notes to README.

**Acceptance Commands:**

* `pytest -q`
* CLI run works

---

## 7) Immediate Next Instruction

**Wait for the user to specify which Phase to execute.**
When requested to execute a phase:

* Keep changes scoped to that phase only.
* Do not jump ahead.
* End with the required outputs (changed files + validation commands + results).

```
