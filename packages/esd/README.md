# Erased Stable Diffusion (ESD)

Concept erasure for Stable Diffusion via UNet fine-tuning.

## Installation

### From GitHub
```bash
pip install git+https://github.com/your-org/esd.git
```

### Local Development
```bash
git clone https://github.com/your-org/esd.git
cd esd
pip install -e .
```

## Usage

```python
from esd import ESDPipeline

# Initialize and train ESD to erase a concept
pipeline = ESDPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    erase_concept="nudity",
    train_method="xattn",
    negative_guidance=2.0,
    train_steps=200,
)

# Generate images with the concept erased
images = pipeline.generate(
    prompts=["a photo of a person"],
    seed=42,
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

## Training Methods

| Method | Layers Fine-tuned | Best For |
|--------|-------------------|----------|
| `xattn` (ESD-x) | Cross-attention K/V | Specific concepts (artists, objects) |
| `full` (ESD-u) | Entire UNet | Broad concepts (nudity, violence) |
| `selfattn` | Self-attention only | Experimental |
| `noxattn` | All except cross-attention | Experimental |

## How It Works

ESD fine-tunes the UNet so that its prediction for a target concept is pushed in the opposite direction:

```
target = pred_base - negative_guidance * (pred_concept - pred_unconditional)
```

After training, the model can no longer generate the erased concept.

## License

MIT License - See LICENSE file for details
