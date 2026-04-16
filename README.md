# eval-learn

A benchmarking framework for evaluating concept-unlearning techniques in text-to-image diffusion models.

Unlearning techniques modify or constrain Stable Diffusion to suppress specific concepts — nudity, violence, artistic styles, named individuals. eval-learn provides a common interface to run, compare, and evaluate these techniques under consistent conditions.

---

## Techniques

| Technique | Key |
|-----------|-----|
| Erased Stable Diffusion | `esd` |
| Mass Concept Erasure | `mace` |
| Unified Concept Editing | `uce` |
| Selective Synaptic Dampening | `ssd` |
| Concept Ablation | `ca` |
| CoGFD | `cogfd` |
| TraSCE | `trasce` |
| SAFREE | `safree` |
| Safe Latent Diffusion | `sld` |
| AdvUnlearn | `advunlearn` |
| Concept Steerers | `concept_steerers` |
| SAeUron | `saeuron` |
| Free Run (custom model) | `free_run` |

## Metrics

| Metric | Key | What it measures |
|--------|-----|-----------------|
| ASR — I2P | `asr_i2p` | Attack success rate on I2P prompts |
| ASR — P4D | `asr_p4d` | Attack success rate via P4D adversarial prompts |
| ASR — MMA Diffusion | `asr_mma_diffusion` | Attack success rate via MMA-Diffusion GCG attack |
| ASR — Ring-A-Bell | `asr_ring_a_bell` | Attack success rate via genetic adversarial prompt discovery |
| Erasure Retention Rate | `err` | Concept erasure vs. unrelated concept retention |
| FID | `fid` | Image quality vs. COCO reference |
| CLIP Score | `clip_score` | Prompt-image alignment |
| UA-IRA | `ua_ira` | Unsafe concept alignment vs. retain concept alignment |
| TIFA | `tifa` | Text-image faithfulness via VQA |

---

## Installation

### 1. Install eval-learn

```bash
pip install eval-learn
```

### 2. Install technique packages

Technique implementations are hosted on [Hugging Face](https://huggingface.co/datasets/Unlearningltd/Packages). Install only what you need:

```bash
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=esd"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=mace"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=uce"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=ssd"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=ca"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=cogfd"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=trasce"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=saeuron"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=safree"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=concept-steerers"
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=advunlearn"
```

SLD is built into eval-learn via the `diffusers` library and requires no extra install.

### 3. Install metric packages

```bash
# P4D adversarial attack
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=p4d"

# MMA-Diffusion adversarial attack
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=mma_diff"

# Ring-A-Bell adversarial prompt discovery
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=RING_A_BELL"

# Q16 classifier (used by P4D and Ring-A-Bell for non-nudity concepts)
pip install "git+https://huggingface.co/datasets/Unlearningltd/Packages#subdirectory=Q16"

# NudeNet (nudity ASR)
pip install "eval-learn[asr]"

# FID / COCO metrics
pip install "eval-learn[fid,coco]"
```

### 4. Hugging Face authentication

Create a `.env` file in the directory you run `eval-learn run` from:

```
HF_TOKEN=your_token_here
```

---

## Quick start

Benchmarks are defined in a JSON or YAML config file:

```json
{
  "output_dir": "results/esd_nudity",
  "technique": {
    "name": "esd",
    "config": { "erase_concept": "nudity", "train_method": "noxattn", "device": "cuda" }
  },
  "metrics": [
    { "name": "asr_i2p",    "config": { "concept_name": "nudity", "device": "cuda" } },
    { "name": "fid",        "config": { "device": "cuda" } },
    { "name": "clip_score", "config": { "device": "cuda" } }
  ]
}
```

Run it:

```bash
eval-learn run --config config.json
```

Results are written to `output_dir` as JSON.

### Useful commands

```bash
eval-learn plugins   # list installed techniques and metrics
eval-learn models    # show the base model each technique targets
```

---

## Examples

The [`examples/`](examples/) directory contains ready-to-run configs for all techniques across nudity and violence concepts:

```
examples/
  nudity/     one config per technique (esd.json, mace.json, ...)
  violence/   same, for violence concept
  data/       seed prompts and concept vectors used by the configs
```

Run all nudity benchmarks in sequence:

```bash
python nudity_unlearning_demo.py
```

Run all violence benchmarks:

```bash
python nudity_unlearning_demo_violence.py
```

---

## Documentation

Full configuration reference, technique guides, metric descriptions, and experiment recipes:

**https://eval-learn.readthedocs.io**

Key pages:

- [Getting started](docs/docs/getting-started.md)
- [Technique-metric compatibility](docs/docs/running-experiments/compatibility.md)
- [Caching adversarial prompts and technique weights](docs/docs/running-experiments/caching-adversarial-prompts.md)

---

## License

MIT
