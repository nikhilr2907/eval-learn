# Installation Guide

## Installation Methods

### 1. Local Development (Recommended for now)

From the eval-learn repository root:

```bash
# Install the package in editable mode
pip install -e packages/esd
```

### 2. From GitHub (After uploading)

```bash
pip install git+https://github.com/your-org/esd.git
```

## Integration with Eval-Learn

After installing the standalone package, the eval-learn wrapper will automatically detect it:

```python
from eval_learn.registry import get_technique

technique_factory = get_technique("esd")
technique = technique_factory(
    model_id="CompVis/stable-diffusion-v1-4",
    erase_concept="nudity",
    train_method="xattn",
    train_steps=200,
)

images = technique.generate(prompts=["a photo of a person"])
```

## Saving & Loading Trained Weights

ESD trains a new UNet on each initialization. To avoid retraining:

```python
from esd import ESDPipeline

# Train and save
pipeline = ESDPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    erase_concept="nudity",
    train_method="xattn",
    save_path="weights/esd_nudity.pth",
)

# Later: load pre-trained weights (skips training)
pipeline = ESDPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    load_path="weights/esd_nudity.pth",
)
```

## Verifying Installation

```python
import esd
print(esd.__version__)  # Should print: 0.1.0
```

## Troubleshooting

### Import Error

If you get `ImportError: No module named 'esd'`:

1. Make sure you installed the package: `pip install -e packages/esd`
2. Check it's in your environment: `pip list | grep esd`

### CUDA Out of Memory

If you run out of GPU memory during training:

1. Reduce `train_steps`
2. Enable `use_fp16=True` (default)
3. Use `train_method="xattn"` (fewest parameters)
