# Getting Started

## Requirements

- Python >= 3.8
- CUDA-capable GPU recommended — most techniques require one

## Installation

```bash
git clone https://github.com/your-org/eval-learn
cd eval-learn
pip install -e ".[all]"
```

Each technique is a separate package. Install only the ones you need:

```bash
pip install -e packages/mace/
pip install -e packages/esd/
pip install -e packages/uce/
pip install -e packages/advunlearn/
pip install -e packages/saeuron/
pip install -e packages/safree/
pip install -e packages/concept-steerers/
```

!!! note "Why separate packages?"
    Each technique has its own dependencies which can conflict with one another.
    Installing only what you need avoids resolution issues.

### Metric extras

Some metrics require additional dependencies not installed by default:

```bash
pip install -e ".[asr]"    # ASR — requires NudeNet
pip install -e ".[fid]"    # FID — requires torchvision
pip install -e ".[coco]"   # COCO-based metrics — requires torchvision
```

### CUDA wheels

If installing PyTorch via pip rather than conda:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu126
```

## Hugging Face authentication

Most techniques download model weights from the Hugging Face Hub.
Create a `.env` file at the project root:

```bash
HF_TOKEN=your_token_here
```

Eval-Learn loads this automatically on startup. Alternatively export it in your shell.

## Running a benchmark

Benchmarks are defined in a YAML config file and run with:

```bash
eval-learn run --config config.yaml
```

### Single metric

```yaml
technique:
  name: mace
  config:
    erase_concept: nudity
    lambda_cfr: 0.1
    save_path: checkpoints/mace_nudity.pt

metric:
  name: asr

output_dir: results/mace_nudity
```

### Multiple metrics

Replace `metric` with `metrics` as a list:

```yaml
technique:
  name: mace
  config:
    erase_concept: nudity

metrics:
  - name: asr
  - name: clip_score
  - name: fid

output_dir: results/mace_nudity
```

Results are written to `output_dir` as JSON.

## Pushing results to HF Hub

Run and push in one step:

```bash
eval-learn run --config config.yaml --hf-repo your-org/results
```

Or push an existing results directory separately:

```bash
eval-learn push --repo your-org/results --local-dir results/mace_nudity
```

## Useful commands

List all installed techniques and metrics:

```bash
eval-learn plugins
```

Show the base model each technique uses:

```bash
eval-learn models
```

## Next steps

- [Concepts](concepts.md) — what unlearning benchmarking measures
- [Techniques](techniques/index.md) — configuration reference for each technique
- [Metrics](metrics/index.md) — what each metric measures and when to use it
