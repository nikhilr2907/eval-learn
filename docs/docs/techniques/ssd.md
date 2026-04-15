# SSD — Selective Synaptic Dampening

## Overview

SSD (Selective Synaptic Dampening, AAAI 2024) erases concepts by selectively dampening
UNet parameters that are specifically responsible for the forget concept, leaving parameters
important for general generation largely intact.

The algorithm:

1. Estimate the diagonal Fisher Information `F_forget` using the concept prompt — identifies
   which parameters encode the concept.
2. Estimate the diagonal Fisher Information `F_retain` using neutral prompts (`""`, `"a photo"`,
   `"an image"`) — identifies which parameters matter for general generation.
3. For each parameter `θ_i` compute a dampening ratio:
   ```
   ratio_i = F_retain_i / (F_retain_i + α · F_forget_i)
   ```
   Parameters important for retain (high `F_retain`) → ratio near 1 → barely touched.
   Parameters important only for the concept (high `F_forget`, low `F_retain`) → ratio near 0 → dampened toward zero.
4. Apply: `θ_new = θ * ratio` (element-wise, no training loop).

Like MACE, SSD is a closed-form weight update. Unlike MACE, it operates on **all UNet
parameters** rather than only cross-attention K/V projections, and uses gradient-based
Fisher estimation rather than a direct matrix solve.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — `erase_concept` is used as the forget prompt directly.

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
| `erase_concept` | `str` | `"nudity"` | The concept to forget. Used directly as the forget prompt for Fisher estimation. |
| `alpha` | `float` | `200.0` | Selectivity coefficient. Higher values make dampening more selective — only parameters where `F_forget >> F_retain` are dampened. Typical range: 100–2000. |
| `dampening_coeff` | `float` | `1.0` | Scales the dampening ratio. `1.0` applies the raw ratio. Values `< 1.0` reduce dampening strength globally; values `> 1.0` increase it (clamped to avoid negatives). |
| `num_fisher_samples` | `int` | `4` | Noise samples per prompt when estimating each Fisher diagonal. More samples → more stable estimate, slower computation. |
| `save_path` | `str \| None` | `None` | Path to save the dampened UNet weights. If `None`, weights are held in memory only and SSD re-runs on every invocation. |
| `load_path` | `str \| None` | `None` | Path to load pre-dampened UNet weights, skipping SSD computation entirely. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run generation in half precision. Fisher estimation always runs in float32 for numerical accuracy. |
| `device` | `str` | `"cuda"` | Device to run on. |

### `alpha` tuning

| Goal | Direction |
|------|-----------|
| ASR remains high after erasure | Lower `alpha` (e.g. `50`) — more aggressive dampening |
| FID or CLIP Score degrades noticeably | Raise `alpha` (e.g. `500–1000`) — more selective dampening |

### Retain prompts

The retain set is hardcoded to `["", "a photo", "an image"]`. These neutral prompts
protect parameters needed for general image generation. They are not configurable.

---

## Warnings

!!! warning "Saving weights"
    SSD Fisher estimation involves gradient passes over the UNet and can take several
    minutes. Always set `save_path` for repeated evaluation runs, or use `load_path`
    to skip computation entirely on subsequent runs.

!!! warning "Single forget prompt"
    `erase_concept` is used as the sole forget prompt. For concepts with many surface
    forms (e.g. nudity via "nude", "naked", "nsfw"), the Fisher estimate is computed
    only from the single string provided. This limits coverage compared to techniques
    like MACE which accept synonym lists. Consider using a representative single term.

!!! warning "dampening_coeff > 1.0"
    Values above `1.0` amplify dampening beyond the raw Fisher ratio. Elements where
    `1 - dampening_coeff * (1 - ratio) < 0` are clamped to zero, effectively zeroing
    out those parameters. Use with caution.

---

## Examples

### Single metric — ASR (nudity)

```json
{
  "output_dir": "results/ssd_asr",
  "technique": {
    "name": "ssd",
    "config": {
      "erase_concept": "nudity",
      "alpha": 200.0,
      "save_path": "checkpoints/ssd_nudity.pt",
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

### Load pre-computed weights

```json
{
  "output_dir": "results/ssd_asr_fast",
  "technique": {
    "name": "ssd",
    "config": {
      "erase_concept": "nudity",
      "load_path": "checkpoints/ssd_nudity.pt",
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

### Multiple metrics — nudity full benchmark

```json
{
  "output_dir": "results/ssd_nudity_multi",
  "technique": {
    "name": "ssd",
    "config": {
      "erase_concept": "nudity",
      "alpha": 200.0,
      "num_fisher_samples": 4,
      "save_path": "checkpoints/ssd_nudity.pt",
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


---

!!! tip "Reusing trained weights across runs"
    Set `save_path` on the first run to persist the trained weights, then use `load_path`
    on all subsequent runs to skip retraining. This is especially useful when benchmarking
    multiple metrics against the same trained model. See
    [Caching adversarial prompts and technique weights](../running-experiments/caching-adversarial-prompts.md)
    for the full workflow.
