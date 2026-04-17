# Concept Steerers — SAE Steering Vector Suppression

## Overview

Concept Steerers apply sparse autoencoder (SAE) steering at inference time through the
**text encoder** (CLIP layer 9). For each generation, the pipeline first derives a
steering direction by running the concept string through the text encoder and encoding
the resulting activations with the SAE. During each denoising step, this direction is
then added to the conditional text embedding and subtracted from the unconditional one,
biasing classifier-free guidance away from the concept.

Because the steering direction is computed on-the-fly from the concept string, **any
concept is supported** — no pre-built steering vectors or concept-specific checkpoints
are needed. The SAE checkpoint is bundled with the package and loaded automatically.

A positive `multiplier` suppresses the concept; a negative value amplifies it.

**Base model:** `CompVis/stable-diffusion-v1-4`  
**Supported concepts:** any string

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Yes | Any I2P concept |
| ERR | Yes | `erase_concept="nudity"` required (ERR is nudity-specific) |
| FID | Yes | General image quality |
| CLIP Score | Yes | General text-image alignment |
| UA_IRA | Yes | Requires custom prompt CSVs |
| TIFA | Yes | General faithfulness |
| ASR Custom | Yes | Concept-agnostic via CLIP |
| MMA-Diffusion | Yes | |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `erase_concept` | `str` | `"nudity"` | The concept to suppress. Any string is valid — the steering direction is derived at runtime by encoding this string through the text encoder and SAE. |
| `multiplier` | `float` | `1.0` | Steering strength. Positive values suppress the concept. Increase to strengthen erasure; values that are too high may degrade generation quality. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | CFG guidance scale for generation. Must be > 1.0. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str \| None` | `None` | Device to run on. Auto-detects CUDA, then MPS, then CPU if `None`. |

---

## Warnings

!!! warning "multiplier tuning"
    The default `multiplier=1.0` is conservative. If ASR remains high, increase it (e.g.
    `5.0`, `10.0`). Very large values risk steering the text embeddings far enough from
    the original distribution to visibly degrade image quality — monitor FID and CLIP Score
    alongside ASR when tuning.

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
    "name": "asr_i2p",
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
    },
    { "name": "tifa", "config": { "device": "cuda", "limit": 200 } }
  ]
}
```

### Multiple metrics — violence benchmark

```json
{
  "output_dir": "results/concept_steerers_violence_multi",
  "technique": {
    "name": "concept_steerers",
    "config": {
      "erase_concept": "violence",
      "multiplier": 1.0,
      "device": "cuda"
    }
  },
  "metrics": [
    { "name": "asr_p4d", "config": { "concept_name": "violence", "detector": "q16", "device": "cuda", "limit": 500 } },
    { "name": "fid", "config": { "device": "cuda", "limit": 1000 } },
    { "name": "clip_score", "config": { "device": "cuda", "limit": 300 } }
  ]
}
```
