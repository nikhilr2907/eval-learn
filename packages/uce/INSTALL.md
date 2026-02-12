# UCE Installation Guide

## Quick Install

```bash
# From eval-learn repository root
pip install -e packages/uce
```

## Package Size

**Total: ~220 MB** (includes 3 pre-trained weight files)

Bundled weights:
- `uce_nudity.safetensors` - 74 MB
- `uce_violence.safetensors` - 74 MB
- `uce_dog.safetensors` - 74 MB

## Usage

### Using Bundled Weights (Instant)

```python
from uce import UCEPipeline

# Use pre-trained nudity erasure weights
pipeline = UCEPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    preset="nudity"  # or "violence", "dog"
)

images = pipeline.generate(["a photo of a person"])
```

### Creating Custom Weights

To create weights for new concepts, you need the UCE training code:

```python
from uce import UCEWeightCreator

# This will clone the UCE repo to ~/.cache/uce/ on first use
creator = UCEWeightCreator(device="cuda")

# Create weights for custom concept (takes 5-30 min on GPU)
creator.create_weights(
    concept="car",
    output_path="./uce_car.safetensors"
)

# Then use the custom weights
pipeline = UCEPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    weights_path="./uce_car.safetensors"
)
```

**Requirements for weight creation**:
- GPU (CUDA recommended)
- 5-30 minutes training time
- Will clone UCE repo (~50 MB) to `~/.cache/uce/` on first use

## Integration with Eval-Learn

After installing the UCE package, eval-learn automatically uses it:

```python
from eval_learn.registry import get_technique

technique = get_technique("uce")(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    preset="nudity"  # Uses bundled weights
)

images = technique.generate(prompts=["test prompt"])
```

## Verifying Installation

```python
import uce
print(uce.__version__)  # Should print: 0.1.0

# Check bundled weights are accessible
pipeline = uce.UCEPipeline(preset="nudity", device="cpu")
print("✓ UCE installed correctly!")
```

## Troubleshooting

### Import Error
```
ImportError: No module named 'uce'
```
**Solution**: Make sure you installed the package:
```bash
pip install -e packages/uce
pip list | grep uce
```

### Weights Not Found
```
FileNotFoundError: UCE weights not found
```
**Solution**: Verify weights are in the package:
```bash
ls packages/uce/src/uce/weights/
```

Should show:
- `uce_dog.safetensors`
- `uce_nudity.safetensors`
- `uce_violence.safetensors`

### Weight Creation Fails
```
FileNotFoundError: UCE training script not found
```
**Solution**: The UCE repository wasn't cloned. It should happen automatically, but you can manually clone:
```bash
git clone https://github.com/rohitgandikota/unified-concept-editing.git ~/.cache/uce/unified-concept-editing
```

### CUDA Out of Memory
If you run out of GPU memory during weight creation:
- Close other GPU-intensive processes
- Try with `device="cpu"` (slower but uses less memory)
- Reduce batch size in UCE training script
