# RECE — Reliable and Efficient Concept Erasure

## Overview

RECE (Reliable and Efficient Concept Erasure, ECCV 2024) erases concepts from a Stable
Diffusion UNet by iterating a closed-form adversarial embedding update with a UCE-style
weight edit. Each epoch computes the worst-case adversarial text embedding for the current
model state, then solves a closed-form weight update to erase that embedding — making the
erasure robust to adversarial prompt rephrasing without requiring gradient-based training.

The algorithm:

1. **Adversarial embedding** — given the current model, compute a text embedding that
   maximally recovers the erased concept (closed-form solve, no gradient descent).
2. **UCE edit** — apply a closed-form weight update to the cross-attention K/V projections
   that maps the adversarial embedding away from the concept, while preserving retain
   concept embeddings.
3. **Iterate** — repeat for `epochs` rounds (default 3). Each round tightens the erasure
   against the updated adversarial embedding.

Three regularization strategies are available for the adversarial embedding solve,
controlled by `emb_computing`.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — supply `erase_concept` for inline weight creation, or
`load_path` to load pre-built weights.

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Any I2P concept | NudeNet for nudity; CLIP for all others |
| ERR | nudity only | Requires `erase_concept="nudity"` |
| FID | Any | General image quality |
| CLIP Score | Any | General text-image alignment |
| UA_IRA | Any | Requires custom prompt CSVs |
| TIFA | Any | General faithfulness |
| ASR Custom | Any | Concept-agnostic via CLIP |
| MMA-Diffusion | Any | Requires explicit target prompts for non-nudity |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_path` | `str \| None` | `None` | Path to a pre-built `.safetensors` weight file. Skips weight creation entirely. Mutually exclusive with `erase_concept`. |
| `erase_concept` | `str \| None` | `None` | Concept to erase. Triggers inline weight creation; requires `save_path`. Mutually exclusive with `load_path`. |
| `concept_type` | `str` | `"object"` | Type of concept for weight creation. One of `"object"`, `"style"`, `"attribute"`. |
| `emb_computing` | `str` | `"close_regzero"` | Regularization strategy for the adversarial embedding solve. One of `"close_regzero"` (no regularization), `"close_standardreg"` (standard regularization with old target), `"close_surrogatereg"` (surrogate regularization without old target). |
| `save_path` | `str \| None` | `None` | Path to save weights created via `erase_concept`. Required when `erase_concept` is set. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run generation in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

### `emb_computing` strategies

| Value | Regularization | When to use |
|-------|---------------|-------------|
| `"close_regzero"` | None — adversarial embedding solved without regularization. | Default; good general-purpose erasure. |
| `"close_standardreg"` | Standard — regularizes toward the original target concept embedding. | When over-erasure of adjacent concepts is observed. |
| `"close_surrogatereg"` | Surrogate — regularizes without an old target concept reference. | When no meaningful old-target anchor exists or single-concept erasure is required. |

---

## Warnings

!!! warning "At least one weight source is required"
    RECE will raise a `ValueError` at startup if neither `load_path` nor `erase_concept`
    is specified.

!!! warning "load_path and erase_concept are mutually exclusive"
    Both provide the erased-concept weights — providing both raises a `ValueError`. Use
    one or the other.

!!! warning "save_path required for inline creation"
    When using `erase_concept`, `save_path` must be specified. Weight creation runs up to
    `epochs` rounds of closed-form adversarial embedding solves and can take 5–30 minutes.
    Persist the result via `save_path` and reload with `load_path` on subsequent runs.

!!! warning "ERR requires nudity concept"
    ERR is nudity-specific. Pair it with `erase_concept="nudity"` or a nudity-specific
    `load_path`. Use FID, CLIP Score, or UA_IRA for other concepts.

---

## Examples

### Inline creation — ASR (nudity)

```json
{
  "output_dir": "results/rece_asr",
  "technique": {
    "name": "rece",
    "config": {
      "erase_concept": "nudity",
      "concept_type": "object",
      "save_path": "weights/rece_nudity.safetensors",
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

### Load pre-built weights

```json
{
  "output_dir": "results/rece_asr_fast",
  "technique": {
    "name": "rece",
    "config": {
      "load_path": "weights/rece_nudity.safetensors",
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

### Surrogate regularization — custom concept

```json
{
  "output_dir": "results/rece_violence",
  "technique": {
    "name": "rece",
    "config": {
      "erase_concept": "violence",
      "concept_type": "object",
      "emb_computing": "close_surrogatereg",
      "save_path": "weights/rece_violence.safetensors",
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

### Multiple metrics — nudity full benchmark

```json
{
  "output_dir": "results/rece_nudity_multi",
  "technique": {
    "name": "rece",
    "config": {
      "erase_concept": "nudity",
      "concept_type": "object",
      "save_path": "weights/rece_nudity.safetensors",
      "device": "cuda",
      "num_inference_steps": 50,
      "guidance_scale": 7.5
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
    Set `save_path` on the first run to persist the weights, then use `load_path` on all
    subsequent runs to skip the adversarial training loop entirely. See
    [Caching adversarial prompts and technique weights](../running-experiments/caching-adversarial-prompts.md)
    for the full workflow.
