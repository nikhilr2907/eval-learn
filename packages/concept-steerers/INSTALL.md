# Installation Guide

## Installation Methods

### 1. Local Development (Recommended for now)

From the eval-learn repository root:

```bash
# Install the package in editable mode
pip install -e packages/concept-steerers
```

### 2. From GitHub (After uploading)

```bash
pip install git+https://github.com/your-org/concept-steerers.git
```

### 3. From Hugging Face (After uploading)

```bash
pip install git+https://huggingface.co/your-org/concept-steerers.git
```

## SAE Checkpoint Setup

The package requires a trained Sparse Autoencoder checkpoint. You have two options:

### Option A: Copy from eval-learn (for local development)

```bash
# From eval-learn root
cp -r src/eval_learn/techniques/concept_steerers/checkpoints packages/concept-steerers/
```

### Option B: Download from Hugging Face (recommended for production)

```bash
# Download checkpoint to a local directory
mkdir -p ~/.concept_steerers/checkpoints
huggingface-cli download your-org/concept-steerers-sae \
    --local-dir ~/.concept_steerers/checkpoints/i2p_sd14_l9
```

Then specify the path when initializing:

```python
from concept_steerers import ConceptSteeringPipeline

pipeline = ConceptSteeringPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    sae_path="~/.concept_steerers/checkpoints/i2p_sd14_l9",  # Point to your checkpoint
    concept="nudity",
    multiplier=1.0
)
```

## Integration with Eval-Learn

After installing the standalone package, the eval-learn wrapper will automatically detect it:

```python
from eval_learn.registry import get_technique

# This will use the external concept-steerers package
technique_factory = get_technique("concept_steerers")
technique = technique_factory(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    sae_path="path/to/checkpoint",
    concept="nudity",
    multiplier=1.0
)

images = technique.generate(prompts=["a photo of a person"])
```

## Verifying Installation

```python
import concept_steerers
print(concept_steerers.__version__)  # Should print: 0.1.0
```

## Troubleshooting

### Import Error

If you get `ImportError: No module named 'concept_steerers'`:

1. Make sure you installed the package: `pip install -e packages/concept-steerers`
2. Check it's in your environment: `pip list | grep concept`

### SAE Checkpoint Not Found

If you get `FileNotFoundError: SAE checkpoint not found`:

1. Verify the checkpoint path exists
2. Make sure it contains `config.json` and `state_dict.pth`
3. Update `sae_path` parameter to point to the correct location

### CUDA Out of Memory

If you run out of GPU memory:

1. Use a smaller batch size
2. Switch to `device="cpu"` (slower but uses less memory)
3. Use mixed precision: `torch_dtype=torch.float16`
