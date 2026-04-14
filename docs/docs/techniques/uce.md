# UCE — Unified Concept Editing

## Overview

UCE (Unified Concept Editing) edits the cross-attention layers of a Stable Diffusion UNet
using pre-computed concept vectors. Rather than fine-tuning from scratch, UCE loads a set
of pre-trained editing weights and applies them to steer generation away from the target
concept.

**Base model:** `CompVis/stable-diffusion-v1-4`

UCE supports three initialisation paths:

1. **Preset** — load bundled weights for one of three pre-defined concepts (`nudity`, `violence`, `dog`). No extra files needed.
2. **Load path** — supply your own pre-built `.safetensors` file via `load_path`. Skips weight creation entirely.
3. **Inline creation** — provide `erase_concept` (and `save_path`) to run UCEWeightCreator and build weights on the fly. Takes 5–30 minutes on GPU; result is persisted to `save_path` for reuse.

`preset` and `load_path` are mutually exclusive weight sources. `erase_concept` is independent and may be combined with either to carry concept metadata (e.g. for metric routing). When provided alone, `erase_concept` triggers inline weight creation.

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | `nudity` or `violence` preset | NudeNet for nudity; CLIP for violence |
| ERR | `nudity` preset only | Nudity-specific datasets |
| FID | Any preset | General image quality |
| CLIP Score | Any preset | General text-image alignment |
| UA_IRA | Any preset | Requires custom prompt CSVs |
| TIFA | Any preset | General faithfulness |
| ASR Custom | Any preset | Concept-agnostic via CLIP |
| MMA-Diffusion | Any preset | Requires explicit target prompts for non-nudity presets |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `preset` | `str \| None` | `None` | Bundled concept preset. One of `"nudity"`, `"violence"`, `"dog"`. Mutually exclusive with `load_path` / `erase_concept`. |
| `load_path` | `str \| None` | `None` | Path to a pre-built `.safetensors` weight file. Skips weight creation. |
| `erase_concept` | `str \| None` | `None` | Concept to erase inline via UCEWeightCreator. Requires `save_path`. Takes 5–30 minutes on GPU. |
| `concept_type` | `str` | `"object"` | Type of concept for inline creation. One of `"object"`, `"style"`, `"attribute"`. |
| `save_path` | `str \| None` | `None` | Path to save weights created via `erase_concept`. Required when `erase_concept` is set. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

---

## Warnings

!!! warning "At least one of preset, load_path, or erase_concept is required"
    UCE will raise a `ValueError` at startup if none of the three are specified.

!!! warning "preset and load_path are mutually exclusive"
    Both resolve pre-built weights — providing both raises a `ValueError`. Use one or the other.

!!! warning "Only three bundled presets"
    Passing any value other than `"nudity"`, `"violence"`, or `"dog"` to `preset` raises a
    `ValueError`. For arbitrary concept erasure use `load_path` (pre-built weights) or
    `erase_concept` (inline creation).

!!! warning "save_path required for inline creation"
    When using `erase_concept`, `save_path` must be specified. Weight creation takes 5–30
    minutes and the result is persisted so it can be loaded directly on subsequent runs
    via `load_path`.

!!! warning "ERR requires nudity preset"
    ERR is nudity-specific. Using it with `preset="violence"` or `preset="dog"` will raise
    a `ValidationError`. Use ASR I2P, UA_IRA, FID, or CLIP Score for non-nudity presets.

!!! warning "ASR I2P concept must match preset"
    ASR I2P filters I2P prompts by concept. The `concept` in the metric config must match
    the preset used — e.g. `preset="violence"` should pair with `concept="violence"`.
    The `dog` preset has no matching I2P category; use UA_IRA or CLIP Score instead.

---

## Examples

### Preset — ASR (nudity)

```json
{
  "output_dir": "results/uce_asr",
  "technique": {
    "name": "uce",
    "config": {
      "preset": "nudity",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "asr",
    "config": {
      "device": "cuda",
      "limit": 500
    }
  }
}
```

### Preset — FID (violence)

```json
{
  "output_dir": "results/uce_violence_fid",
  "technique": {
    "name": "uce",
    "config": {
      "preset": "violence",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "fid",
    "config": {
      "device": "cuda",
      "limit": 1000
    }
  }
}
```

### Load path — custom pre-built weights

```json
{
  "output_dir": "results/uce_custom_load",
  "technique": {
    "name": "uce",
    "config": {
      "load_path": "weights/uce_car.safetensors",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "clip_score",
    "config": { "device": "cuda", "limit": 300 }
  }
}
```

### Inline creation — custom concept

```json
{
  "output_dir": "results/uce_custom_create",
  "technique": {
    "name": "uce",
    "config": {
      "erase_concept": "car",
      "concept_type": "object",
      "save_path": "weights/uce_car.safetensors",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "clip_score",
    "config": { "device": "cuda", "limit": 300 }
  }
}
```

### Multiple metrics — nudity full benchmark

```json
{
  "output_dir": "results/uce_nudity_multi",
  "technique": {
    "name": "uce",
    "config": {
      "preset": "nudity",
      "device": "cuda",
      "num_inference_steps": 50,
      "guidance_scale": 7.5
    }
  },
  "metrics": [
    { "name": "asr", "config": { "device": "cuda", "limit": 500 } },
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
    },
    { "name": "tifa", "config": { "device": "cuda", "limit": 200 } }
  ]
}
```
