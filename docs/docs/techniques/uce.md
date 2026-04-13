# UCE — Unified Concept Editing

## Overview

UCE (Unified Concept Editing) edits the cross-attention layers of a Stable Diffusion UNet
using pre-computed concept vectors. Rather than fine-tuning from scratch, UCE loads a set
of pre-trained editing weights and applies them to steer generation away from the target
concept.

Because UCE relies on pre-trained editing weights, it is limited to a fixed set of presets.
It cannot erase arbitrary user-defined concepts without providing custom weights via
`uce_weights_path`.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** `nudity`, `violence`, `dog` (via presets). Custom concepts require
providing your own `uce_weights_path`.

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
| `preset` | `str \| None` | `None` | Pre-trained concept preset to load. Must be one of `"nudity"`, `"violence"`, or `"dog"`. Required unless `uce_weights_path` is set. |
| `uce_weights_path` | `str \| None` | `None` | Path to custom UCE weights. Use this to supply your own editing vectors for concepts outside the three presets. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

**Note:** `erase_concept` is a read-only derived property — it is automatically set to
`preset.lower()`. Do not pass it directly in the config.

---

## Warnings

!!! warning "preset is required"
    Either `preset` or `uce_weights_path` must be provided. If neither is set, UCE will
    raise a `ValidationError` at startup.

!!! warning "Only three presets supported"
    Passing any value other than `"nudity"`, `"violence"`, or `"dog"` to `preset` will
    raise a `ValidationError`. For arbitrary concept erasure, use ESD or MACE instead.

!!! warning "ERR requires nudity preset"
    ERR is nudity-specific. Using it with `preset="violence"` or `preset="dog"` will raise
    a `ValidationError`. Use ASR I2P, UA_IRA, FID, or CLIP Score for non-nudity presets.

!!! warning "ASR I2P concept must match preset"
    ASR I2P filters I2P prompts by concept. The `concept` in the metric config must match
    the preset used — e.g. `preset="violence"` should pair with `concept="violence"`.
    The `dog` preset has no matching I2P category; use UA_IRA or CLIP Score instead.

---

## Examples

### Single metric — ASR (nudity preset)

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

### Single metric — FID (violence preset)

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
