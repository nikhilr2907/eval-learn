# Getting Started

## Requirements

- Python >= 3.8
- CUDA-capable GPU recommended — most techniques require one

## Installation

### 1. Install eval-learn

```bash
pip install eval-learn
```

### 2. Install technique packages

To ensure of a lightweight and clean package, the precise implementation of all unlearning techniques are in  seperate installable packages hosted on [Hugging Face](https://huggingface.co/datasets/Unlearningltd/Packages).

Install only the ones you need:

```bash
# ESD
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=esd"

# MACE
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=mace"

# UCE
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=uce"

# SAeUron
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=saeuron"

# SAFREE
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=safree"

# Concept Steerers
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=concept-steerers"

# AdvUnlearn
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=advunlearn"
```

SLD is included in `eval-learn` directly and requires no extra install as it is implemented within the [Hugging-Face] [diffusers] library, a required dependency of the package.

### 3. Metric packages.

Most metrics are fairly lightweight and their implementation does not require any standalone dependencies. However, for custom (non-nudity) unlearning, the option to create a custom set of prompts for concept unlearning testing is presented by the use of the 'asr_custom' and
'mma_diffusion' evaluation metrics, using seperate methods to generate adversarial prompts. To evaluate unlearning on these metrics, you must install them. 

```bash
# MMA-Diffusion
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=mma_diff"

# Ring-A-Bell
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=RING_A_BELL"
```

### 4. Metric extras

Some metrics require additional dependencies:

```bash
pip install "eval-learn[asr]"    # ASR — requires NudeNet
pip install "eval-learn[fid]"    # FID — requires torchvision
pip install "eval-learn[coco]"   # COCO-based metrics — requires torchvision
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

Benchmarks are defined in a config file and run with:

```bash
eval-learn run --config config.yaml   # or config.json
```

Both YAML and JSON are supported and equivalent.

### Single metric

=== "YAML"

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

=== "JSON"

    ```json
    {
      "technique": {
        "name": "mace",
        "config": {
          "erase_concept": "nudity",
          "lambda_cfr": 0.1,
          "save_path": "checkpoints/mace_nudity.pt"
        }
      },
      "metric": {
        "name": "asr"
      },
      "output_dir": "results/mace_nudity"
    }
    ```

### Multiple metrics

Replace `metric` with `metrics` as a list:

=== "YAML"

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

=== "JSON"

    ```json
    {
      "technique": {
        "name": "mace",
        "config": {
          "erase_concept": "nudity"
        }
      },
      "metrics": [
        { "name": "asr" },
        { "name": "clip_score" },
        { "name": "fid" }
      ],
      "output_dir": "results/mace_nudity"
    }
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
