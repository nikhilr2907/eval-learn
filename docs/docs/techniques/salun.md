# SalUn — Saliency Unlearning

## Overview

SalUn (ICLR 2024) erases a concept by fine-tuning only the UNet weights most responsible
for generating that concept, identified via gradient-magnitude saliency.

The algorithm runs in two phases:

1. **Saliency mask (Phase 1):** Forward and backward passes are run on the forget concept images.
   Gradient magnitudes over all UNet parameters are accumulated, then thresholded to keep the
   top-`threshold` fraction — these are the weights most responsible for the concept.
2. **Masked fine-tune (Phase 2):** The UNet is fine-tuned with mask-gated gradient updates:
   - *Forget loss* — pulls `UNet(erase_concept)` toward `UNet(anchor_concept).detach()`, so the
     model redirects the forget concept toward a safe substitute.
   - *Retain loss* — standard diffusion MSE on the anchor concept, preserving general quality.
   Only the masked weights receive gradient updates; all other parameters are frozen.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — requires paired image datasets for the forget and retain concepts.

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
| MMA-Diffusion | Any | Requires explicit target prompts for non-nudity |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `erase_concept` | `str` | `"nudity"` | The concept to forget. Used as the forget text embedding in Phase 2. |
| `anchor_concept` | `str` | `"a person fully clothed"` | Safe substitute concept. The forget loss pulls UNet outputs toward this anchor's predictions. |
| `forget_data_path` | `str` | — | Path to a directory of images representing the forget concept. Used in Phase 1 (saliency) and Phase 2 (forget loss). Required. |
| `retain_data_path` | `str` | — | Path to a directory of diverse benign images. Used in Phase 2 retain loss. Required. |
| `threshold` | `float` | `0.5` | Top fraction of UNet weights to include in the saliency mask. `0.5` keeps the top 50% by gradient magnitude. Lower values are more selective. |
| `alpha` | `float` | `0.5` | Weight of the retain loss relative to the forget loss: `loss = forget_loss + alpha * retain_loss`. |
| `lr` | `float` | `1e-5` | Learning rate for the Phase 2 fine-tune. |
| `epochs` | `int` | `5` | Number of fine-tuning epochs in Phase 2. |
| `batch_size` | `int` | `4` | Batch size for both Phase 1 and Phase 2 data loaders. |
| `c_guidance` | `float` | `7.5` | CFG scale used during Phase 1 saliency gradient computation. |
| `train_method` | `str` | `"full"` | `"full"` updates all masked UNet params; `"xattn"` restricts updates to cross-attention layers only. |
| `save_path` | `str \| None` | `None` | Directory to save the trained UNet checkpoint and saliency mask. Strongly recommended for repeated evaluation. |
| `load_path` | `str \| None` | `None` | Path to a pre-trained UNet checkpoint or `.safetensors` file, skipping Phase 1 and Phase 2 entirely. |
| `image_size` | `int` | `512` | Image resize target for both forget and retain data loaders. |
| `use_fp16` | `bool` | `False` | Run in half precision. Disabled by default because fp16 training can be numerically unstable. |
| `device` | `str` | `"cuda"` | Device to run on. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |

---

## Examples

### Single metric — ASR (nudity)

```json
{
  "output_dir": "results/salun_asr",
  "technique": {
    "name": "salun",
    "config": {
      "erase_concept": "nudity",
      "anchor_concept": "a person fully clothed",
      "forget_data_path": "data/nudity_images",
      "retain_data_path": "data/retain_images",
      "threshold": 0.5,
      "alpha": 0.5,
      "epochs": 5,
      "save_path": "checkpoints/salun_nudity",
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
  "output_dir": "results/salun_nudity_multi",
  "technique": {
    "name": "salun",
    "config": {
      "erase_concept": "nudity",
      "anchor_concept": "a person fully clothed",
      "forget_data_path": "data/nudity_images",
      "retain_data_path": "data/retain_images",
      "threshold": 0.5,
      "alpha": 0.5,
      "epochs": 5,
      "save_path": "checkpoints/salun_nudity",
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

!!! warning "Training runs at construction time"
    Both phases execute when the technique is instantiated. Training can take tens of minutes
    depending on dataset size and epoch count. Always set `save_path` so the trained UNet
    and saliency mask are persisted, then use `load_path` on subsequent runs to skip training.

!!! warning "Image datasets required"
    Unlike prompt-only techniques (SSD, ESD), SalUn requires actual image directories for
    `forget_data_path` and `retain_data_path`. These must be populated before running.

!!! tip "Reusing trained weights across runs"
    Set `save_path` on the first run to persist the trained weights, then use `load_path`
    on all subsequent runs to skip retraining. See
    [Caching adversarial prompts and technique weights](../running-experiments/caching-adversarial-prompts.md)
    for the full workflow.
