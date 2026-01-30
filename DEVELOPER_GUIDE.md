# Developer Guide: Adding New Techniques to Eval-Learn

Welcome to the **Eval-Learn Developer Guide**. This document is intended for contributors who want to extend the library by adding new **Unlearning Techniques**.

The library is designed with a plugin architecture, making it easy to add new methods without modifying the core runner logic.

---

## 1. Architecture Overview

Before adding code, understand the core flow:

1.  **Registry:** All techniques are registered via a decorator (`@register_technique`). This allows the `BenchmarkRunner` to find them by a string name (e.g., `"sld"`).
2.  **Interface:** A technique is simply a class that implements a `generate(prompts, ...)` method.
3.  **Configuration:** Techniques use strongly-typed `dataclasses` (inherited from `BaseConfig`) for configuration.

---

## 2. Directory Structure

Your new technique should live in `src/eval_learn/techniques/<your_technique_name>/`.

**Example Structure:**
```text
src/eval_learn/techniques/
└── my_new_technique/
    ├── __init__.py      # Exports
    ├── config.py        # Configuration Dataclass
    └── wrapper.py       # Main Logic (The Wrapper Class)
```

---

## 3. Step-by-Step Implementation Guide

Let's say you want to implement a technique called **" ESD "** (Erasing Stable Diffusion).

### Step 1: Create the Configuration

Create `src/eval_learn/techniques/esd/config.py`.
Inherit from `BaseConfig` to get automatic dictionary serialization.

```python
from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class ESDConfig(BaseConfig):
    """
    Configuration for Erasing Stable Diffusion (ESD).
    """
    # 1. Standard params (model, device)
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: Optional[str] = None
    
    # 2. Technique-specific hyperparameters
    train_method: str = "xattn"  # 'xattn', 'noc', 'selfattn'
    lr: float = 1e-5
    steps: int = 1000
    erase_concept: str = "nudity"
```

### Step 2: Create the Wrapper Class

Create `src/eval_learn/techniques/esd/wrapper.py`.

**Key Requirements:**
1.  **Decorate:** Use `@register_technique("esd")`.
2.  **Initialize:** Accept `**kwargs` and convert them to your Config.
3.  **Implement `generate`:** Must accept `prompts` (list of strings) and return a list of images.

```python
from typing import List, Any, Optional
import torch

# 1. Import Registry & Logger
from ...registry import register_technique
from ...logging_utils import get_logger
from .config import ESDConfig

logger = get_logger(__name__)

# 2. Register the technique with a unique string key
@register_technique("esd")
class ESDWrapper:
    def __init__(self, **kwargs):
        # 3. Load Config
        self.config = ESDConfig.from_dict(kwargs)
        
        # 4. Initialize Model (Heavy Lifting)
        logger.info(f"Initializing ESD with method: {self.config.train_method}")
        
        # ... Load your model/pipeline here ...
        # self.pipe = MyESDPipeline.from_pretrained(self.config.model_id)
        # self.pipe.to(self.config.device)
        
        pass

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:
        """
        Generate images for the given prompts.
        """
        logger.info(f"Generating {len(prompts)} images...")
        
        images = []
        # 5. Generation Loop
        for prompt in prompts:
            # Call your pipeline
            # output = self.pipe(prompt, ...).images[0]
            # images.append(output)
            pass
            
        return images
```

### Step 3: Expose the Module (Optional but Recommended)

In `src/eval_learn/techniques/__init__.py`, import your wrapper so it gets registered when someone imports `eval_learn.techniques`.

```python
# src/eval_learn/techniques/__init__.py

# ... existing imports ...
from .esd.wrapper import ESDWrapper # This triggers the @register_technique
```

---

## 4. Handling Dependencies

If your technique requires specific libraries (e.g., specific diffusers version, clip, etc.):

1.  **Do NOT** put them in `pyproject.toml`'s main dependencies unless strictly necessary for *everyone*.
2.  **Use Optional Imports:** Import them inside the file or in a `try/except` block.
3.  **Fail Gracefully:** If the user lacks the dependency, raise a clear error telling them what to install.

**Example:**
```python
try:
    import concept_erasure
except ImportError:
    raise RuntimeError(
        "ESD requires 'concept_erasure'. Install with: pip install eval-learn[esd]"
    )
```

---

## 5. Testing Your New Technique

You don't need to write a full benchmark to test your wrapper.

1.  Create a test file: `tests/test_technique_esd.py`
2.  Use the registry to load it (verifies registration works).
3.  Mock the heavy generation if possible, or run a tiny smoke test.

```python
from eval_learn.registry import get_technique

def test_esd_registration():
    # 1. Get class
    ESDClass = get_technique("esd")
    
    # 2. Instantiate with config
    esd = ESDClass(train_method="full", steps=10)
    
    # 3. Check config
    assert esd.config.train_method == "full"
    assert esd.config.steps == 10
```

---

## 6. Checklist for PRs

*   [ ] Created `config.py` with `BaseConfig`.
*   [ ] Created `wrapper.py` with `@register_technique("name")`.
*   [ ] Implemented `generate(prompts) -> List[Image]`.
*   [ ] Added graceful error handling for missing dependencies.
*   [ ] Added a basic test case.