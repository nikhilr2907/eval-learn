# SAeUron — Sparse Autoencoder Unlearning

## Overview

SAeUron uses a pre-trained Sparse Autoencoder (SAE) to intercept and ablate concept-specific
features in the UNet's intermediate activations at inference time. A forward hook is
registered on the target UNet module. At each denoising step, the hook:

1. Projects the activation into the SAE's sparse latent space
2. Identifies and suppresses features associated with the target concept (determined
   by pre-cached activation statistics and a percentile threshold)
3. Reconstructs the modified activations with a residual correction to preserve
   unrelated structure

The technique splits batches along the classifier-free guidance axis (unconditioned and
conditioned chunks) and applies ablation only to the conditioned chunk, preserving
unconditional generation quality.

Because ablation is applied via a hook at inference time, SAeUron does not retrain the
model. However, it requires pre-computed SAE weights and concept activation statistics,
which are loaded from bundled checkpoints.

**Base model:** `CompVis/stable-diffusion-v1-4` (the SAE is trained on SD 1.4 activations)  
**Supported concepts:** nudity only (cached activations only available for nudity)

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR | Yes | nudity only |
| ERR | Yes | nudity only |
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
| `sae_path` | `str \| None` | `None` | Path to SAE checkpoint directory containing `cfg.json` and `sae.safetensors`. Auto-resolved to bundled checkpoints if `None`. |
| `acts_path` | `str \| None` | `None` | Path to cached concept activation file (`.pkl`). Auto-resolved to bundled `cls_latents_dict_mini.pkl` if `None`. |
| `position` | `str` | `"unet.up_blocks.1.attentions.1"` | Dot-path to the UNet submodule where the hook is registered. |
| `multiplier` | `float` | `-20.0` | Feature scaling factor. Negative values ablate (suppress) features; positive values amplify. |
| `percentile` | `float` | `99.99` | Percentile threshold for feature selection. Only features with activation above this percentile are ablated. |
| `target_latents` | `list[int]` | `[]` | Explicit list of SAE latent indices to ablate. If empty, latents are selected automatically from `acts_path` using `percentile`. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | CFG guidance scale. Must be > 1.0. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

---

## Warnings

!!! warning "nudity only"
    The bundled activation cache (`cls_latents_dict_mini.pkl`) contains statistics for
    nudity only. Passing any other concept raises a `ValidationError`.

!!! warning "guidance_scale must be > 1.0"
    SAeUron splits the inference batch into unconditioned and conditioned halves along
    the CFG axis. If `guidance_scale <= 1.0`, CFG is inactive and the batch has only one
    chunk — the hook will fail to locate the conditioned half. Always use
    `guidance_scale > 1.0` (default 7.5 is safe).

!!! warning "target_latents and acts_path"
    If `target_latents` is empty (default), latents are auto-computed from `acts_path`
    using `percentile`. If `acts_path` is also `None`, the bundled path is used. If you
    provide `target_latents` explicitly, `acts_path` is ignored. Providing an empty
    list with an invalid or missing `acts_path` will raise a runtime error.

!!! warning "position module path"
    The default `position` corresponds to `unet.up_blocks.1.attentions.1` in SD 1.4.
    Changing this to an invalid module path will cause an `AttributeError` when the
    hook is registered. Only modify this if you know the exact submodule structure of
    your target model.

---

## Examples

### Single metric — ASR (default bundled checkpoints)

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
    "name": "asr",
    "config": {
      "device": "cuda",
      "limit": 500
    }
  }
}
```

### Single metric — with custom checkpoint paths

```json
{
  "output_dir": "results/saeuron_custom",
  "technique": {
    "name": "saeuron",
    "config": {
      "erase_concept": "nudity",
      "sae_path": "checkpoints/saeuron/",
      "acts_path": "checkpoints/saeuron/cls_latents_dict.pkl",
      "multiplier": -20.0,
      "percentile": 99.99,
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
  "output_dir": "results/saeuron_nudity_multi",
  "technique": {
    "name": "saeuron",
    "config": {
      "erase_concept": "nudity",
      "multiplier": -20.0,
      "percentile": 99.99,
      "device": "cuda",
      "num_inference_steps": 50,
      "guidance_scale": 7.5
    }
  },
  "metrics": [
    { "name": "asr", "config": { "device": "cuda", "limit": 500 } },
    { "name": "err", "config": { "device": "cuda", "target_limit": 100, "retain_limit": 100, "adversarial_limit": 100 } },
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
