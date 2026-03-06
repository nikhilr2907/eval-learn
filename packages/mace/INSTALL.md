# Installation Guide

## Installation Methods

### 1. Local Development (Recommended for now)

From the eval-learn repository root:

```bash
# Install the package in editable mode
pip install -e packages/mace
```

### 2. From GitHub (After uploading)

```bash
pip install git+https://github.com/your-org/mace.git
```

## Integration with Eval-Learn

After installing the standalone package, the eval-learn wrapper will automatically detect it:

```python
from eval_learn.registry import get_technique

technique_factory = get_technique("mace")
technique = technique_factory(
    model_id="CompVis/stable-diffusion-v1-4",
    erase_concept="nudity",
    train_steps=200,
)

images = technique.generate(prompts=["a photo of a person"])
```

## Saving & Loading Trained Weights

MACE trains a new UNet on each initialization. To avoid retraining:

```python
from mace import MACEPipeline

# Train and save
pipeline = MACEPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    erase_concept="nudity",
    save_path="weights/mace_nudity.pth",
)

# Later: load pre-trained weights (skips training)
pipeline = MACEPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    load_path="weights/mace_nudity.pth",
)
```

## Verifying Installation

```python
import mace
print(mace.__version__)  # Should print: 0.1.0
```

## Troubleshooting

### Import Error

If you get `ImportError: No module named 'mace'`:

1. Make sure you installed the package: `pip install -e packages/mace`
2. Check it's in your environment: `pip list | grep mace`

### CUDA Out of Memory

If you run out of GPU memory during training:

1. Reduce `train_steps`
2. Enable `use_fp16=True` (default)
