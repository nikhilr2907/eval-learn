# Eval-Learn Package Integration Guide

This document explains how to build an external package (technique or metric) that integrates
seamlessly with the eval-learn benchmarking framework. Follow this exactly and the package
will be auto-discovered, configurable via JSON, and runnable through the standard runners.

---

## How the framework works

eval-learn has two layers:

1. **External packages** — standalone pip-installable packages that contain the actual
   implementation (model training, inference, attack generation, etc.). These have no
   dependency on eval-learn.

2. **Wrappers inside eval-learn** — thin adapter classes that live in
   `src/eval_learn/techniques/<name>/` or `src/eval_learn/metrics/<name>/`. These import
   from the external package and expose the standard interface the runners expect.

The runners never touch external packages directly. They call `technique.generate()` and
`metric.load_dataset()` / `metric.update()` / `metric.compute()` — only what the wrapper
exposes.

---

## Part 1: The external package

### What the external package must expose

The external package only needs to expose **one class** that the eval-learn wrapper will
import. It has no knowledge of eval-learn at all.

#### For a technique package

Expose a pipeline class that accepts config in `__init__` and implements `generate()`:

```python
# mypackage/__init__.py
from .pipeline import MyPipeline
__all__ = ["MyPipeline"]
```

```python
# mypackage/pipeline.py
from typing import List, Optional
from PIL import Image

class MyPipeline:
    def __init__(
        self,
        model_id: str,
        device: str,
        erase_concept: str,
        # ...any other params
    ):
        # Load model, apply technique, etc.
        ...

    def generate(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Return one PIL Image per prompt."""
        ...
```

The `generate()` method must return a `List[PIL.Image.Image]` with one image per prompt.
That is the only contract the wrapper depends on.

#### For a metric package (adversarial prompt generation)

The MMA-Diffusion and Ring-A-Bell packages follow this pattern — they expose a generator/
discovery class that the metric wrapper calls during `load_dataset()`.

For **MMA-Diffusion style** (GCG attack, returns a list of dicts):

```python
# mypackage/__init__.py
from .generator import MyAdversarialGenerator
__all__ = ["MyAdversarialGenerator"]
```

```python
# mypackage/generator.py
from typing import List, Dict, Any

class MyAdversarialGenerator:
    def __init__(self, clip_model_id: str, output_csv: str, **kwargs):
        ...

    def generate(
        self,
        target_prompts: List[str],
        n_steps: int = 10,
        # ...attack params
    ) -> List[Dict[str, Any]]:
        """
        Return a list of dicts. Each dict must have an 'adversarial_prompt' key.
        Any other keys are passed through as metadata.
        """
        ...
```

For **Ring-A-Bell style** (genetic algorithm, writes a CSV):

```python
# mypackage/__init__.py
from .discovery import PromptDiscovery
from .config import GAConfig
__all__ = ["PromptDiscovery", "GAConfig"]
```

```python
# mypackage/discovery.py
class PromptDiscovery:
    def __init__(
        self,
        seed_prompts_path: str,
        concept_vector_path: str,
        output_path: str,
        filter_fn,
        config=None,
    ):
        ...

    def run(self) -> None:
        """Run discovery and write results to output_path CSV (no header, one prompt per line)."""
        ...
```

### pyproject.toml

The external package needs no entry points — it is just a normal pip package. Keep
dependencies minimal: `torch`, `transformers`, `diffusers`, `Pillow` as needed.

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-technique"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "diffusers>=0.20",
    "transformers>=4.30",
    "Pillow>=9.0",
]

[tool.setuptools.packages.find]
where = ["src"]
```

---

## Part 2: The wrapper inside eval-learn

Once the external package exists, add a wrapper in eval-learn. This is always two files:
`config.py` and `wrapper.py` (for techniques) or `config.py` and `metric.py` (for metrics).

### Directory structure

```
src/eval_learn/
├── techniques/
│   └── my_technique/
│       ├── __init__.py      (empty or re-exports)
│       ├── config.py
│       └── wrapper.py
└── metrics/
    └── my_metric/
        ├── __init__.py      (empty or re-exports)
        ├── config.py
        └── metric.py
```

---

### Technique wrapper

#### config.py

Inherit from `BaseConfig`. Every field maps directly to a JSON config key. All fields
must have defaults (the runner passes config as `**kwargs`).

```python
from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class MyTechniqueConfig(BaseConfig):
    erase_concept: str = "nudity"
    device: str = "cuda"
    train_steps: int = 200
    save_path: Optional[str] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
```

#### wrapper.py

```python
from typing import List, Optional
from PIL import Image

try:
    from my_technique import MyPipeline
except ImportError:
    raise ImportError(
        "MyTechnique requires the 'my-technique' package. "
        "Install with: pip install git+https://..."
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import MyTechniqueConfig

logger = get_logger(__name__)


@register_technique("my_technique")
class MyTechniqueWrapper:
    def __init__(self, **kwargs):
        self.config = MyTechniqueConfig.from_dict(kwargs)
        logger.info(f"Initializing MyTechnique for concept '{self.config.erase_concept}'")
        self.pipeline = MyPipeline(
            erase_concept=self.config.erase_concept,
            device=self.config.device,
            train_steps=self.config.train_steps,
            save_path=self.config.save_path,
        )

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Image.Image]:
        return self.pipeline.generate(
            prompts=prompts,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=seed,
        )
```

The decorator `@register_technique("my_technique")` is the only registration needed.
The name in the decorator is the string used in the JSON config `"technique": { "name": "my_technique" }`.

---

### Metric wrapper

Metrics have a three-method interface: `load_dataset()`, `update()`, `compute()`.

- `load_dataset()` — returns a `DataLoader` where each batch is a `Dataset(prompts=[...], metadata={...})`
- `update(images, prompts, metadata)` — called once per batch after the runner generates images
- `compute()` — returns a `MetricResult(name, value, details)`

#### config.py

```python
from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class MyMetricConfig(BaseConfig):
    concept_name: str = "nudity"
    similarity_threshold: float = 0.28
    clip_model_id: str = "openai/clip-vit-large-patch14"
    device: str = "cuda"
    limit: Optional[int] = 500
```

#### metric.py

```python
from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader

from ...types import MetricResult, Dataset
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import MyMetricConfig

logger = get_logger(__name__)

try:
    from my_metric_package import MyGenerator
except ImportError:
    raise ImportError(
        "MyMetric requires the 'my-metric-package'. "
        "Install with: pip install git+https://..."
    )


@register_metric("my_metric")
class MyMetric:
    def __init__(self, **kwargs):
        self.config = MyMetricConfig.from_dict(kwargs)
        self._unsafe_count = 0
        self._total = 0

    def load_dataset(self) -> DataLoader:
        """Generate/load prompts and return them as a DataLoader."""
        self._unsafe_count = 0
        self._total = 0

        generator = MyGenerator(...)
        rows = generator.generate(...)

        def collate_fn(batch):
            return Dataset(
                prompts=[r["prompt"] for r in batch],
                metadata={"source": "my_metric", "concept": self.config.concept_name},
            )

        return DataLoader(rows, batch_size=32, shuffle=False, collate_fn=collate_fn)

    def update(
        self,
        images: List[Any],
        _prompts: List[str],
        _metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Evaluate each image and accumulate counts. Called once per batch."""
        for img in images:
            is_unsafe = self._detect(img)
            if is_unsafe:
                self._unsafe_count += 1
            self._total += 1

    def _detect(self, image) -> bool:
        # Your detection logic here (NudeNet, CLIP, classifier, etc.)
        ...

    def compute(self) -> MetricResult:
        """Return the final score. Called once after all batches."""
        if self._total == 0:
            return MetricResult(name="MyMetric", value=0.0, details={"error": "No images evaluated"})
        score = self._unsafe_count / self._total
        return MetricResult(
            name="MyMetric",
            value=score,
            details={
                "unsafe_count": self._unsafe_count,
                "total_images": self._total,
                "concept": self.config.concept_name,
            },
        )
```

The decorator `@register_metric("my_metric")` registers it. The name is the string used
in `"metric": { "name": "my_metric" }`.

---

## Part 3: Imports available in wrappers

These are the only eval-learn imports a wrapper should ever need:

```python
from ...types import MetricResult, Dataset          # return types
from ...registry import register_technique          # decorator
from ...registry import register_metric             # decorator
from ...logging_utils import get_logger             # logger
from ...configs.base import BaseConfig              # config base class
```

`Dataset` is what each DataLoader batch must yield:
```python
@dataclass
class Dataset:
    prompts: List[str]
    metadata: Dict[str, Any]  # anything you want passed through to update()
```

`MetricResult` is what `compute()` must return:
```python
@dataclass
class MetricResult:
    name: str           # display name, e.g. "MyMetric_ASR"
    value: float        # primary score, reported in the JSON output
    details: Dict[str, Any]   # any sub-scores, counts, config snapshot
```

---

## Part 4: JSON config format

Once registered, the technique/metric is usable in any run config:

```json
{
  "output_dir": "results/my_run",
  "technique": {
    "name": "my_technique",
    "config": {
      "erase_concept": "violence",
      "device": "cuda",
      "train_steps": 200
    }
  },
  "metric": {
    "name": "my_metric",
    "config": {
      "concept_name": "violence",
      "device": "cuda",
      "limit": 500
    }
  }
}
```

Config keys map 1:1 to the dataclass fields. Unknown keys are silently ignored
(`BaseConfig.from_dict` filters them out). Missing keys use the dataclass defaults.

---

## Part 5: Validation

If the technique or metric has constraints (e.g. nudity-only), add a rule to
`src/eval_learn/runners/validation.py` in `validate_technique_metric_pair()`.

The existing rules are:
- `err` metric requires `erase_concept="nudity"`
- `safree`, `sld`, `concept_steerers`, `saeuron` techniques require `erase_concept="nudity"`
- `uce` technique only supports presets `nudity`, `violence`, `dog`
- `ua_ira` metric requires `target_prompts_path` and `retain_prompts_path`

Add a new `validate_*` function following the same pattern and call it from
`validate_technique_metric_pair`.

---

## Checklist

### External package
- [ ] Single class exposed from `__init__.py`
- [ ] Technique: `generate(prompts) -> List[PIL.Image]`
- [ ] Metric generator: returns list of dicts with a prompt key, or writes a CSV
- [ ] No dependency on eval-learn
- [ ] `pyproject.toml` with correct package name and dependencies

### eval-learn wrapper
- [ ] `config.py` with a frozen `@dataclass` inheriting `BaseConfig`, all fields defaulted
- [ ] `wrapper.py` / `metric.py` with `@register_technique("name")` or `@register_metric("name")`
- [ ] Technique: `generate(prompts, seed, **kwargs) -> List[Image]`
- [ ] Metric: `load_dataset() -> DataLoader`, `update(images, prompts, metadata)`, `compute() -> MetricResult`
- [ ] `ImportError` with install instructions if external package missing
- [ ] Validation rule added if concept-restricted
