# Concept Steerers — SAE Steering Vector Suppression

## Overview

Concept Steerers apply sparse autoencoder (SAE) steering at inference time by subtracting
a pre-computed concept steering vector from the UNet's latent activations. During each
denoising step, the SAE encodes the current activations, subtracts the concept direction
scaled by `multiplier`, and decodes back to the activation space.

A positive `multiplier` amplifies the concept; a negative value suppresses it. For concept
erasure, use a positive `multiplier` (the steering vector points toward the concept, so
subtracting it with a positive scale moves activations away).

The pre-trained SAE and steering vectors are bundled with the package and are loaded
automatically from `<config_dir>/checkpoints/i2p_sd14_l9` if `sae_path` is not specified.

**Base model:** `CompVis/stable-diffusion-v1-4`  
**Supported concepts:** nudity only (bundled steering vectors are nudity-specific)

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Yes | this technique only supports nudity |
| ERR | Yes | this technique only supports nudity |
| FID | Yes | General image quality |
| CLIP Score | Yes | General text-image alignment |
| UA_IRA | Yes | Requires custom prompt CSVs |
| TIFA | Yes | General faithfulness |
| ASR Custom | Yes | Concept-agnostic via CLIP |
| MMA-Diffusion | Yes | Nudity-specific by default |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `erase_concept` | `str` | `"nudity"` | Must be `"nudity"`. |
| `sae_path` | `str \| None` | `None` | Path to SAE checkpoint directory containing `config.json` (architecture hyperparameters) and `state_dict.pth` (encoder/decoder weights). The SAE is a general feature decomposition of CLIP text encoder layer 9 activations — it is not concept-specific. The concept direction is derived at runtime by encoding the concept prompt through the SAE. Auto-resolved to bundled `checkpoints/i2p_sd14_l9` if `None`. |
| `multiplier` | `float` | `1.0` | Steering strength. Positive values suppress the concept (subtract the concept direction). Increase to strengthen erasure; values that are too high may degrade generation quality. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | CFG guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str \| None` | `None` | Device to run on. Auto-detects CUDA, then MPS, then CPU if `None`. |

---

## Warnings

!!! warning "nudity only"
    The bundled steering vectors are trained on I2P (nudity-focused) activations. Passing
    any concept other than `"nudity"` raises a `ValidationError`.

!!! warning "multiplier tuning"
    The default `multiplier=1.0` is conservative. If ASR remains high, increase it (e.g.
    `5.0`, `10.0`). Very large values risk steering the latents far enough from the
    original distribution to visibly degrade image quality — monitor FID and CLIP Score
    alongside ASR when tuning.

!!! warning "sae_path resolution"
    If `sae_path=None`, the path is resolved relative to the concept_steerers package
    installation directory. If the package was installed from a non-standard location or
    the checkpoints were not bundled, this will raise a `FileNotFoundError`. In that case,
    provide an explicit `sae_path`.

---

## Examples

### Single metric — ASR

```json
{
  "output_dir": "results/concept_steerers_asr",
  "technique": {
    "name": "concept_steerers",
    "config": {
      "erase_concept": "nudity",
      "multiplier": 1.0,
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

### Single metric — with stronger steering

```json
{
  "output_dir": "results/concept_steerers_strong",
  "technique": {
    "name": "concept_steerers",
    "config": {
      "erase_concept": "nudity",
      "multiplier": 5.0,
      "device": "cuda"
    }
  },
  "metric": {
    "name": "clip_score",
    "config": {
      "device": "cuda",
      "limit": 300
    }
  }
}
```

### Multiple metrics — nudity full benchmark

```json
{
  "output_dir": "results/concept_steerers_nudity_multi",
  "technique": {
    "name": "concept_steerers",
    "config": {
      "erase_concept": "nudity",
      "multiplier": 1.0,
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
