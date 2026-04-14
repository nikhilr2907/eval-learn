# CA — Concept Ablation

## Overview

Concept Ablation (ICCV 2023) fine-tunes the cross-attention layers of a Stable Diffusion
UNet to make the model's distribution for a `target_concept` match that of an
`anchor_concept`. Rather than suppressing a concept outright, the model is redirected: when
prompted for the target, it generates the anchor instead.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — `target_concept` and `anchor_concept` are arbitrary strings.
The technique is not limited to safety concepts; style, object, or attribute concepts are
all valid.

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Any I2P concept | NudeNet for nudity; CLIP for all others |
| ERR | nudity only | Requires `target_concept="nudity"` |
| FID | Any | General image quality |
| CLIP Score | Any | General text-image alignment |
| UA-IRA | Any | Requires custom prompt CSVs |
| TIFA | Any | General faithfulness |
| ASR MMA-Diffusion | Any | Requires explicit target prompts for non-nudity |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_concept` | `str` | `"nudity"` | The concept to ablate. Cross-attention layers are fine-tuned so that prompts containing this concept generate the anchor instead. |
| `anchor_concept` | `str` | `"a person wearing clothes"` | The replacement concept. The model is trained to match its own output for this prompt when given the target. |
| `train_steps` | `int` | `400` | Number of fine-tuning steps. Must be > 0. |
| `learning_rate` | `float` | `1e-5` | Optimiser learning rate. Must be > 0. |
| `save_path` | `str \| None` | `None` | Path to save fine-tuned weights after training, and to load from on subsequent runs. When the file at this path already exists, training is skipped and weights are loaded directly. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str \| None` | `None` | Device for training and inference. Auto-detects CUDA/MPS if `None`. |

### `save_path` as load path

`save_path` serves dual purpose: CA saves weights to this path after training, and on the
next run loads from the same path if it already exists — skipping training entirely. Set
`save_path` whenever you plan to evaluate more than once.

---

## Warnings

!!! warning "Both target_concept and anchor_concept are required"
    Unlike most techniques, CA requires both `target_concept` and `anchor_concept`. Providing
    only `target_concept` (or using the `erase_concept` alias without an anchor) raises a
    `ValueError` at startup.

!!! warning "erase_concept is accepted as an alias for target_concept"
    Runners that pass a single `erase_concept` key can use it; CA will map it to
    `target_concept`. However, `anchor_concept` must still be explicitly provided.

!!! warning "save_path doubles as load_path"
    There is no separate `load_path`. If `save_path` points to an existing file, training
    is skipped and that checkpoint is loaded. Delete or change `save_path` to force
    retraining.

!!! warning "train_steps=400 is higher than similar techniques"
    CA fine-tunes cross-attention layers against a paired anchor signal. 400 steps is the
    paper default; fewer steps may produce incomplete ablation. Increase for harder concepts.

---

## Examples

### Nudity → clothed person (default)

```json
{
  "output_dir": "results/ca_nudity",
  "technique": {
    "name": "ca",
    "config": {
      "target_concept": "nudity",
      "anchor_concept": "a person wearing clothes",
      "save_path": "checkpoints/ca_nudity.pt",
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

### Style ablation — Van Gogh → impressionist painting

```json
{
  "output_dir": "results/ca_vangogh",
  "technique": {
    "name": "ca",
    "config": {
      "target_concept": "Van Gogh",
      "anchor_concept": "an impressionist painting",
      "train_steps": 400,
      "save_path": "checkpoints/ca_vangogh.pt",
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

### Multiple metrics — nudity full benchmark

```json
{
  "output_dir": "results/ca_nudity_multi",
  "technique": {
    "name": "ca",
    "config": {
      "target_concept": "nudity",
      "anchor_concept": "a person wearing clothes",
      "train_steps": 400,
      "save_path": "checkpoints/ca_nudity.pt",
      "device": "cuda"
    }
  },
  "metrics": [
    { "name": "asr_i2p", "config": { "device": "cuda", "limit": 500 } },
    { "name": "err", "config": { "device": "cuda", "target_limit": 50, "retain_limit": 20, "adversarial_limit": 50 } },
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
    }
  ]
}
```
