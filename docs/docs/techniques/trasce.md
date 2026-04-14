# TraSCE — Trajectory Steering for Concept Erasure

## Overview

TraSCE (Trajectory Steering for Concept Erasure, 2024) is an inference-time technique
that steers the denoising trajectory away from a target concept at each step. It does not
modify model weights — erasure is applied live during generation.

At each denoising step, TraSCE runs three UNet forward passes:

| Pass | Conditioning | Role |
|------|-------------|------|
| Unconditional | `""` | CFG baseline |
| Conditional | user prompt | CFG signal |
| Negative | `erase_concept` | Concept direction to steer away from |

The steering gradient is:

```
loss = -guidance_loss_scale * exp(-||noise_pred_text - noise_pred_neg|| / sigma)
latents -= discriminator_guidance_scale * ∂loss/∂latents
```

This pushes the latent trajectory away from the concept direction at every step. CFG then
applies normally using the unconditional and conditional passes.

Because `noise_pred_neg` is computed from the erase concept embedding, the technique
generalises to any concept string — no pre-computed statistics or checkpoints are needed.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — `erase_concept` is used directly as the negative conditioning text.

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
| `erase_concept` | `str` | `"nudity"` | Concept to erase. Used as the negative conditioning text at every denoising step. |
| `guidance_loss_scale` | `float` | `15.0` | Controls how strongly the latents are steered away from the concept direction. Higher = stronger erasure. Must not be 0. |
| `discriminator_guidance_scale` | `float` | `5.0` | Scales the gradient update applied to latents at each step. Higher = larger per-step correction. |
| `sigma` | `float` | `1.0` | Normalisation term in the exponential loss. Controls the sensitivity of the loss to the distance between concept and prompt noise predictions. Must be > 0. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation. |
| `use_fp16` | `bool` | `True` | Run in half precision on CUDA. |
| `device` | `str` | `"cuda"` | Device to run on. |

### Parameter interactions

`guidance_loss_scale` and `discriminator_guidance_scale` jointly control erasure strength:

- `guidance_loss_scale` scales the loss magnitude — higher values make the exponential term
  more sensitive to similarity between concept and prompt directions.
- `discriminator_guidance_scale` scales how much each gradient step moves the latent — acts
  as a step size for the steering update.

Increasing either increases erasure but may degrade prompt fidelity. Check CLIP Score and
FID alongside ASR when tuning.

---

## Warnings

!!! warning "No save/load — inference-time only"
    TraSCE does not modify weights. There is nothing to save or load. Every generation
    run applies steering live. Startup cost is model loading only.

!!! warning "guidance_loss_scale must be > 0"
    Setting `guidance_loss_scale` to `0` disables steering entirely; a negative value
    inverts the steering direction. Both are rejected by the config.

!!! warning "Three UNet forward passes per step"
    TraSCE runs three UNet passes per denoising step (unconditional, conditional, negative)
    versus two for standard CFG. Expect roughly 1.5× the generation time of an unmodified
    pipeline at the same `num_inference_steps`.

---

## Examples

### Single metric — ASR (nudity)

```json
{
  "output_dir": "results/trasce_asr",
  "technique": {
    "name": "trasce",
    "config": {
      "erase_concept": "nudity",
      "guidance_loss_scale": 15.0,
      "discriminator_guidance_scale": 5.0,
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

### Single metric — custom concept

```json
{
  "output_dir": "results/trasce_vangogh",
  "technique": {
    "name": "trasce",
    "config": {
      "erase_concept": "Van Gogh",
      "guidance_loss_scale": 15.0,
      "discriminator_guidance_scale": 5.0,
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
  "output_dir": "results/trasce_nudity_multi",
  "technique": {
    "name": "trasce",
    "config": {
      "erase_concept": "nudity",
      "guidance_loss_scale": 15.0,
      "discriminator_guidance_scale": 5.0,
      "sigma": 1.0,
      "device": "cuda"
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
