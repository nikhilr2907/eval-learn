# SAeUron — Sparse Autoencoder Unlearning

## Overview

SAeUron uses a pre-trained Sparse Autoencoder (SAE) to intercept and ablate concept-specific
features in the UNet's intermediate activations at inference time. A forward hook is
registered on a specific UNet layer. At each denoising step, the hook:

1. Projects the conditional activations into the SAE's sparse latent space
2. Suppresses the feature indices associated with the target concept
3. Reconstructs the modified activations with a residual correction to preserve unrelated structure

The hook operates only on the conditional half of the CFG batch, leaving the unconditional
half untouched.

Because ablation is applied via a hook at inference time, SAeUron does not retrain the model.
The SAE checkpoint, target layer, and all internal parameters are bundled and resolved
automatically — only the concept and suppression strength are user-facing.

**Base model:** `CompVis/stable-diffusion-v1-4`  
**Supported concepts:** any string — see below

---

## Concept support

The SAE itself is concept-agnostic: it learns a sparse decomposition of UNet activations
regardless of content. Which features correspond to a concept is determined by an activation
cache — a record of which SAE features fire strongly across images of that concept.

**Bundled concepts** (`nudity`): feature indices are loaded instantly from the bundled
activation cache.

**Any other concept**: feature indices are computed on-the-fly during initialisation. The
pipeline generates 20 images with the concept as the prompt, hooks the SAE layer at
denoising step 10 to collect sparse activations, and selects the top features by mean
activation magnitude. This takes a few minutes and progress is printed to stdout. You
will be informed at config creation time if on-the-fly computation will be needed.

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
| `erase_concept` | `str` | `"nudity"` | Concept to suppress. Bundled cache supports `"nudity"` — any other concept triggers on-the-fly activation computation. |
| `multiplier` | `float` | `-20.0` | Feature scaling factor applied to the target SAE latents. Negative values suppress the concept; positive values amplify it. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | CFG scale. Must be > 1.0 — SAeUron requires CFG to be active. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str \| None` | `None` | Device to run on. Auto-detects CUDA, then MPS, then CPU if `None`. |

---

## Warnings

!!! warning "guidance_scale must be > 1.0"
    SAeUron splits the inference batch into unconditional and conditional halves along
    the CFG axis. If `guidance_scale <= 1.0`, CFG is inactive and the batch has only one
    chunk — the hook will fail to split it. The config enforces this at construction time.

!!! warning "on-the-fly computation for custom concepts"
    For any concept outside the bundled cache, 20 images are generated at initialisation
    to compute activation statistics. This is GPU-intensive and takes a few minutes.
    A message is printed at config creation time so you know to expect this before
    the pipeline starts loading.

---

## Examples

### Single metric — ASR

```json
{
  "output_dir": "results/saeuron_asr",
  "technique": {
    "name": "saeuron",
    "config": {
      "erase_concept": "nudity",
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
  "output_dir": "results/saeuron_nudity_multi",
  "technique": {
    "name": "saeuron",
    "config": {
      "erase_concept": "nudity",
      "multiplier": -20.0,
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

### Custom concept — violence (on-the-fly activation computation)

```json
{
  "output_dir": "results/saeuron_violence_multi",
  "technique": {
    "name": "saeuron",
    "config": {
      "erase_concept": "violence",
      "multiplier": -20.0,
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
