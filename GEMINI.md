# GEMINI.md - Context for Eval-Learn

## Project Overview
**Eval-Learn** is a comprehensive benchmarking library for evaluating "unlearning" techniques in text-to-image diffusion models (e.g., Stable Diffusion). The project has been refactored from a collection of loose scripts into a modular, pip-installable Python package located in `src/eval_learn`.

The core goal is to provide a standardized way to measure how well a model "forgets" specific concepts (like nudity, artistic styles, or objects) while retaining general capability.

## Architecture
The project follows a **Registry-Plugin Architecture**:

*   **Core Logic (`src/eval_learn/`):**
    *   **Runner (`runners/benchmark_runner.py`):** Orchestrates the pipeline: `Load Dataset` -> `Init Technique` -> `Generate Images` -> `Compute Metric` -> `Save Report`.
    *   **Registry (`registry/`):** Manages plugins via decorators (`@register_technique`, `@register_metric`) and entry points.
    *   **Configuration (`configs/`):** Uses strict `dataclasses` for type-safe configuration.
    *   **Artifacts (`artifacts/`):** Handles saving images and JSON reports centrally.

*   **Plugins:**
    *   **Techniques:** Algorithms that modify the generation process (e.g., SLD).
    *   **Metrics:** Evaluators that score the output (e.g., ASR, FID).
    *   **Datasets:** Loaders for prompt collections (e.g., I2P).

## Key Components

### Techniques
*   **SLD (`sld`):** Safe Latent Diffusion. Modifies the diffusion process to suppress specific concepts.
    *   *Dependencies:* `diffusers`, `torch`, `huggingface_hub`.

### Metrics
*   **ASR (`asr`):** Attack Success Rate. Uses NudeNet to detect NSFW content.
*   **FID (`fid`):** Frechet Inception Distance. Measures image quality/diversity against a real reference set.
*   **ERR (`err`):** Erasing-Retention-Robustness. Uses CLIP to measure forgetting/retention of concepts.
*   **TIFA (`tifa`):** Text-to-Image Faithfulness. Uses BLIP-2 VQA to verify prompt adherence.

### Datasets
*   **I2P (`i2p_csv`):** Loads prompts from the Inappropriate Image Prompts (I2P) benchmark CSV.

## Installation & Setup

**Prerequisites:** Python 3.8+

1.  **Editable Install (Recommended for Dev):**
    ```bash
    pip install -e .[all]
    ```
    *   `[all]` installs dependencies for all metrics (Torch, Diffusers, TensorFlow/FID, NudeNet, etc.).
    *   Specific extras: `[asr]`, `[fid]`, `[diffusers]`, `[dev]`.

## Usage

### CLI Execution
The primary entry point is the `eval-learn` CLI.

```bash
# Run a benchmark using a config file
eval-learn run --config examples/my_config.json
```

**Example Config (`my_config.json`):**
```json
{
  "run_name": "My_Test_Run",
  "dataset": { "name": "i2p_csv", "config": { "limit": 10 } },
  "technique": { "name": "sld", "config": { "model_id": "CompVis/stable-diffusion-v1-4" } },
  "metric": { "name": "asr", "config": { "use_nudenet": false } }
}
```

### Python API
```python
from eval_learn.registry import get_technique, get_metric
from eval_learn.runners import BenchmarkRunner
# ... set up loader ...
runner = BenchmarkRunner(...)
report = runner.run()
```

## Development & Testing

*   **Tests:** Located in `tests/`.
    *   Run all tests: `pytest`
    *   Run smoke test: `pytest tests/test_smoke_asr_sld.py`
*   **Logging:** Use `eval_learn.logging_utils.get_logger(__name__)`. Do not use `print()`.
*   **Legacy Code:** The root directory contains legacy scripts (e.g., `asr_benchmark.py`, `core/`). **Do not modify these.** Focus development on `src/eval_learn`.

## Current Status (Hybrid)
The project is in a transition phase. The new architecture in `src/` is functional and tested. The legacy scripts in the root directory are deprecated and slated for removal in Phase 10 of the refactoring plan.
