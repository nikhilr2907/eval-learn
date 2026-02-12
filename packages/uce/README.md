# UCE - Unified Concept Editing

Closed-form concept erasing for Stable Diffusion. Remove unwanted concepts (nudity, violence, objects) from generated images without full retraining.

## Installation

### From Local
```bash
pip install -e packages/uce
```

### From GitHub (after uploading)
```bash
pip install git+https://github.com/your-org/uce.git
```

## Quick Start

### Using Pre-trained Weights (Bundled)

The package comes with pre-trained weights for common concepts:

```python
from uce import UCEPipeline

# Use bundled weights for nudity erasure
pipeline = UCEPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    preset="nudity"  # or "violence", "dog"
)

prompts = ["a photo of a person", "a beach scene"]
images = pipeline.generate(prompts)
```

### Using Custom Weights

```python
pipeline = UCEPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    weights_path="path/to/custom_uce_weights.safetensors"
)
```

### Creating New Weights for Custom Concepts

```python
from uce import UCEWeightCreator

creator = UCEWeightCreator(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda"
)

# Create weights to erase "car" concept
creator.create_weights(
    concept="car",
    output_path="./uce_car.safetensors"
)
```

**Note**: Weight creation requires cloning the original UCE repository and takes 5-30 minutes on GPU.

## Bundled Weights

The following pre-trained weights are included:
- **nudity**: Erases NSFW/nudity content (74 MB)
- **violence**: Erases violent/graphic content (74 MB)
- **dog**: Erases dog objects (74 MB)

Total package size: ~220 MB

## How UCE Works

UCE uses a closed-form solution to modify Stable Diffusion UNet weights, making it much faster than iterative fine-tuning methods like ESD:

1. Load base Stable Diffusion model
2. Apply UCE weight modifications for target concept
3. Generate images with concept erased

**Key advantages**:
- ⚡ Fast: One-shot weight computation
- 🎯 Precise: Closed-form solution
- 💾 Efficient: Only UNet weights modified (74 MB)

## Citation

```bibtex
@article{gandikota2023unified,
  title={Unified Concept Editing in Diffusion Models},
  author={Gandikota, Rohit and Orgad, Hadas and Belinkov, Yonatan and Materzynska, Joanna and Bau, David},
  journal={arXiv preprint arXiv:2308.14761},
  year={2023}
}
```

## License

MIT License - See LICENSE file for details
