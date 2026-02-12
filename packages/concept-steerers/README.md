# Concept Steerers

SAE-based concept steering for Stable Diffusion (2025 research implementation).

## Installation

### From GitHub
```bash
pip install git+https://github.com/your-org/concept-steerers.git
```

### Local Development
```bash
git clone https://github.com/your-org/concept-steerers.git
cd concept-steerers
pip install -e .
```

## Usage

```python
from concept_steerers import ConceptSteeringPipeline

# Initialize pipeline
pipeline = ConceptSteeringPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    sae_path="path/to/sae/checkpoint",
    concept="nudity",
    multiplier=1.0
)

# Generate images with concept steering
prompts = ["a photo of a person", "a landscape"]
images = pipeline.generate(prompts, num_inference_steps=50)
```

## Features

- **SAE-based steering**: Uses sparse autoencoders to identify interpretable concept directions
- **Layer 9 targeting**: Focuses on the primary bottleneck for concept steering in CLIP text encoder
- **Classifier-free guidance**: Properly handles conditional/unconditional modulation
- **Flexible multipliers**: Control steering strength with multiplier parameter

## Citation

If you use this code in your research, please cite:

```bibtex
@article{conceptsteerers2025,
  title={Concept Steering via Sparse Autoencoders},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details
