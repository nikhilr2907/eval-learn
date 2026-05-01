# CoGFD — Concept-combination Graph-based Feature Decoupling

## Overview

CoGFD erases *concept combinations* rather than individual atomic concepts. The key
insight is that a harmful theme such as "nudity" is not a single token but a
combination of component concepts (person + unclothed). Erasing the atom "person"
collaterally damages general generation; erasing the *combination* preserves each
component while making the model unable to produce them together.

**Training modifies only cross-attention K/Q/V projections (attn2 layers).** All other
UNet parameters are frozen, keeping fine-tuning surgical.

### Loss

```
L = λ_erase · L_erase + λ_preserve · L_preserve + λ_decouple · L_decouple
```
| Term | What it does |
|------|-------------|
| **L_erase** | Pulls every combination-prompt response toward the frozen null output, erasing the harmful combination |
| **L_preserve** | Keeps each individual/component concept unchanged vs the frozen original model |
| **L_decouple** | Minimises cosine similarity between combination directions and individual-concept directions in noise-prediction space — ensures decoupling rather than mere suppression |

`λ_preserve` defaults to `2.0` (higher than `λ_erase = 1.0`) to prevent the
preservation term from being overwhelmed. This ratio is intentional.

### Concept logic graph

In the original paper, Stage 1 uses an LLM to enumerate every visual phrasing of the
harmful theme (e.g. "naked woman", "person without clothes", ...), making erasure
robust to prompt rephrasing. This implementation uses hardcoded defaults for `"nudity"`
and `"violence"`. For any other concept, supply `combination_prompts` manually or
replace the defaults with LLM-generated expansions.

**Base model:** `CompVis/stable-diffusion-v1-4`  
**Paper:** Nie et al. "Erasing Concept Combination from Text-to-Image Diffusion Model" (ICLR 2025)
---
## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Any I2P concept | NudeNet for nudity; Q16 for all others |
| ASR P4D | Any | |
| ASR MMA-Diffusion | Any | Requires explicit target prompts for non-nudity |
| ASR Ring-a-Bell | Any | |
| ERR | nudity only | Requires `erase_concept="nudity"` |
| FID | Any | |
| CLIP Score | Any | |
| UA-IRA | Any | Requires custom prompt CSVs |
| TIFA | Any | |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `erase_concept` | `str` | `"nudity"` | High-level concept to erase. Used to look up built-in defaults if `combination_prompts` is empty. |
| `combination_prompts` | `list[str]` | `[]` | Prompts expressing the harmful concept combination (the concept logic graph). Falls back to built-in defaults for `"nudity"` and `"violence"`. Required for all other concepts. |
| `preserve_concepts` | `list[str]` | `[]` | Individual component concepts to keep intact (e.g. `["a person", "a woman"]`). Falls back to built-in defaults for `"nudity"` and `"violence"`. Leave empty to skip preservation. |
| `lambda_erase` | `float` | `1.0` | Weight for the combination erasure loss. Must be >= 0. |
| `lambda_preserve` | `float` | `2.0` | Weight for the individual preservation loss. Must be >= 0. Set higher than `lambda_erase` to prevent collateral concept loss. |
| `lambda_decouple` | `float` | `0.5` | Weight for the feature decoupling loss. Must be >= 0. |
| `train_steps` | `int` | `150` | Training iterations. Must be > 0. |
| `learning_rate` | `float` | `1e-5` | Optimiser learning rate. Must be > 0. |
| `load_path` | `str \| None` | `None` | Path to a directory saved by a previous CoGFD run (via `save_path`). Must contain a `unet/` subdirectory in HuggingFace `save_pretrained` format. If set, training is skipped entirely. |
| `save_path` | `str \| None` | `None` | Directory to save the modified UNet using HuggingFace `save_pretrained` after training. Produces a `unet/` subdirectory — not a single `.pt` file. Only used when training runs (i.e. `load_path` is not set). Skipped if `None`. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation. |
| `guidance_scale` | `float` | `7.5` | CFG scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision on CUDA. |
| `device` | `str \| None` | `None` | Device for training and inference. Auto-detects CUDA/MPS if `None`. |

### Built-in concept defaults

For `erase_concept="nudity"` and `erase_concept="violence"`, `combination_prompts` and
`preserve_concepts` are populated automatically if left empty. For all other values of
`erase_concept`, both lists must be supplied explicitly.

| `erase_concept` | Built-in `combination_prompts` | Built-in `preserve_concepts` |
|-----------------|-------------------------------|------------------------------|
| `"nudity"` | 5 prompts (nude/naked variants) | `["a person", "a woman", "a man", "a human body"]` |
| `"violence"` | 4 prompts (violent scene variants) | `["a person", "a scene", "an action scene"]` |
| anything else | *none — must be supplied* | *none — must be supplied* |

---

## Warnings

!!! warning "Checkpoint format differs from ESD and MACE"
    CoGFD uses HuggingFace `save_pretrained` format rather than a bare `.pt` state dict.
    `save_path` produces a directory containing a `unet/` subdirectory (with `config.json`
    and weight files). Pass that same directory as `load_path` on subsequent runs — do not
    point `load_path` at the `unet/` subdirectory itself.

!!! warning "Custom concepts require explicit combination_prompts"
    If `erase_concept` is not `"nudity"` or `"violence"`, `combination_prompts` must be
    provided. Without it the only combination prompt is the raw `erase_concept` string,
    which provides poor coverage and weakens erasure robustness.

!!! warning "lambda_preserve should stay above lambda_erase"
    The preservation loss counteracts collateral erasure of component concepts. If
    `lambda_preserve` is set too low relative to `lambda_erase`, individual concepts
    (e.g. "a person") will degrade alongside the combination. The defaults
    (`lambda_preserve=2.0`, `lambda_erase=1.0`) reflect the paper's tuning.

!!! warning "train_steps=150 is a moderate default"
    Unlike AdvUnlearn where the default is intentionally minimal, 150 steps is a
    reasonable starting point for CoGFD given its lighter per-step budget (3 UNet
    passes total regardless of prompt count). Published results use 100–500 steps.
    Increase for thorough erasure.

---

## Examples

### Nudity (built-in defaults)

```json
{
  "output_dir": "results/cogfd_nudity",
  "technique": {
    "name": "cogfd",
    "config": {
      "erase_concept": "nudity",
      "train_steps": 150,
      "save_path": "checkpoints/cogfd_nudity",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "asr_i2p",
    "config": {
      "device": "cuda",
      "limit": 500
    }
  }
}
```

### Violence (built-in defaults)

```json
{
  "output_dir": "results/cogfd_violence",
  "technique": {
    "name": "cogfd",
    "config": {
      "erase_concept": "violence",
      "train_steps": 200,
      "device": "cuda"
    }
  },
  "metric": {
    "name": "ua_ira",
    "config": {
      "target_prompts_path": "data/violence_target_prompts.csv",
      "retain_prompts_path": "data/violence_retain_prompts.csv",
      "target_concept": "violence",
      "retain_concept": "action scene",
      "device": "cuda"
    }
  }
}
```

### Custom concept

```json
{
  "output_dir": "results/cogfd_vangogh",
  "technique": {
    "name": "cogfd",
    "config": {
      "erase_concept": "Van Gogh",
      "combination_prompts": [
        "a painting in the style of Van Gogh",
        "a Van Gogh style landscape",
        "swirling brushstrokes in the style of Van Gogh",
        "a starry night Van Gogh painting"
      ],
      "preserve_concepts": [
        "a landscape painting",
        "an oil painting",
        "a painting with brushstrokes"
      ],
      "train_steps": 200,
      "device": "cuda"
    }
  },
  "metric": {
    "name": "ua_ira",
    "config": {
      "target_prompts_path": "data/vangogh_target_prompts.csv",
      "retain_prompts_path": "data/vangogh_retain_prompts.csv",
      "target_concept": "Van Gogh painting",
      "retain_concept": "landscape painting",
      "device": "cuda"
    }
  }
}
```

### Multi-metric run

```json
{
  "output_dir": "results/cogfd_nudity_multi",
  "technique": {
    "name": "cogfd",
    "config": {
      "erase_concept": "nudity",
      "lambda_erase": 1.0,
      "lambda_preserve": 2.0,
      "lambda_decouple": 0.5,
      "train_steps": 150,
      "save_path": "checkpoints/cogfd_nudity",
      "device": "cuda"
    }
  },
  "metrics": [
    { "name": "asr_i2p", "config": { "device": "cuda", "limit": 500 } },
    { "name": "fid", "config": { "device": "cuda", "limit": 1000 } },
    { "name": "clip_score", "config": { "device": "cuda", "limit": 300 } },
    {
      "name": "ua_ira",
      "config": {
        "target_prompts_path": "data/nudity_target_prompts.csv",
        "retain_prompts_path": "data/nudity_retain_prompts.csv",
        "target_concept": "nudity",
        "retain_concept": "person",
        "device": "cuda"
      }
    },
    { "name": "tifa", "config": { "device": "cuda", "limit": 200 } }
  ]
}
```


---

!!! tip "Reusing trained weights across runs"
    Set `save_path` on the first run to persist the trained weights, then use `load_path`
    on all subsequent runs to skip retraining. This is especially useful when benchmarking
    multiple metrics against the same trained model. See
    [Caching adversarial prompts and technique weights](../running-experiments/caching-adversarial-prompts.md)
    for the full workflow.
