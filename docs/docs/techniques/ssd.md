# SSD — Selective Synaptic Dampening

## Overview

SSD (Selective Synaptic Dampening, AAAI 2024) erases concepts by selectively dampening
UNet parameters that are specifically responsible for the forget concept, leaving parameters
important for general generation largely intact.

The algorithm:

1. Estimate the diagonal Fisher Information `F_forget` using the concept prompt — identifies
   which parameters encode the concept.
2. Estimate the diagonal Fisher Information `F_retain` using neutral retain prompts — identifies
   which parameters matter for general generation.
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

**Supported concepts:** Any — `erase_concept` is used as the fallback forget prompt if `forget_prompts` is not set.

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
| `erase_concept` | `str` | `"nudity"` | The concept to forget. Used as the fallback forget prompt if `forget_prompts` is not set. |
| `forget_prompts` | `list[str] \| None` | `None` | Varied phrasings of the concept to erase, used to estimate `F_forget`. Defaults to `[erase_concept]` with a warning if unset. Recommended: 5–10 prompts (synonyms, descriptions, contextual phrasings). |
| `retain_prompts` | `list[str] \| None` | `None` | Diverse benign prompts used to estimate `F_retain`. Defaults to `["", "a photo", "an image"]` with a warning if unset. Recommended: 10–20 prompts across objects, scenes, people, and animals. |
| `alpha` | `float` | `1.0` | Selectivity coefficient. Unlike classification networks, diffusion UNet features are highly entangled — most parameters contribute to every concept. High alpha drives nearly all params toward zero, causing model collapse. Recommended range: 1–20. |
| `dampening_coeff` | `float` | `1.0` | Scales the dampening ratio. `1.0` applies the raw ratio. Values `< 1.0` reduce dampening strength globally; values `> 1.0` increase it (clamped to avoid negatives). |
| `num_fisher_samples` | `int` | `50` | Noise samples per prompt when estimating each Fisher diagonal. More samples → more stable estimate, slower computation. |
| `save_path` | `str \| None` | `None` | Path to save the dampened UNet weights. If `None`, weights are held in memory only and SSD re-runs on every invocation. |
| `load_path` | `str \| None` | `None` | Path to load pre-dampened UNet weights, skipping SSD computation entirely. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run generation in half precision. Fisher estimation always runs in float32 for numerical accuracy. |
| `device` | `str` | `"cuda"` | Device to run on. |

### `alpha` tuning

`alpha` controls how aggressively `F_forget` overrides `F_retain` in the dampening ratio. For shared params where `F_forget ≈ F_retain`, the ratio simplifies to `1 / (1 + alpha)` — so `alpha=100` drives shared params to ~1% of their original value, causing model collapse.

| Goal | Direction |
|------|-----------|
| Images are coloured static / model collapse | Lower `alpha` (start at `1`) |
| General image quality degrades (FID/TIFA) | Lower `alpha` |
| Concept still appears after erasure (high ASR) | Raise `alpha` gradually (e.g. `5`, `10`, `20`) |

### Prompt set sizing

Fisher estimation quality depends on prompt diversity, not just sample count (`num_fisher_samples`).

- **`forget_prompts`** — 5–10 varied phrasings ensure `F_forget` covers the full activation footprint of the concept, not just one text embedding.
- **`retain_prompts`** — 10–20 diverse benign prompts ensure `F_retain` is non-trivially large for general-purpose parameters, preventing them from being over-dampened.

---

## Warnings

!!! warning "Saving weights"
    SSD Fisher estimation involves gradient passes over the UNet and can take several
    minutes. Always set `save_path` for repeated evaluation runs, or use `load_path`
    to skip computation entirely on subsequent runs.

!!! warning "Model collapse at high alpha"
    Diffusion UNet features are highly entangled — almost every parameter contributes
    to every concept. High `alpha` values (e.g. 100+) that work for classification
    networks cause near-total weight collapse in UNets, producing coloured static output.
    Start at `alpha=1` and increase gradually.

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
      "forget_prompts": ["nudity", "naked person", "nude figure", "explicit nudity"],
      "retain_prompts": ["a dog", "a car", "a mountain landscape", "a portrait of a person",
                         "a bowl of fruit", "a city street", "a cat", "a sunset", "a forest path"],
      "alpha": 1.0,
      "num_fisher_samples": 50,
      "save_path": "checkpoints/ssd_nudity.pt",
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
  "output_dir": "results/ssd_nudity_multi",
  "technique": {
    "name": "ssd",
    "config": {
      "erase_concept": "nudity",
      "forget_prompts": ["nudity", "naked person", "nude figure", "explicit nudity"],
      "retain_prompts": ["a dog", "a car", "a mountain landscape", "a portrait of a person",
                         "a bowl of fruit", "a city street", "a cat", "a sunset", "a forest path"],
      "alpha": 1.0,
      "num_fisher_samples": 50,
      "save_path": "checkpoints/ssd_nudity.pt",
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

!!! tip "Reusing trained weights across runs"
    Set `save_path` on the first run to persist the trained weights, then use `load_path`
    on all subsequent runs to skip retraining. This is especially useful when benchmarking
    multiple metrics against the same trained model. See
    [Caching adversarial prompts and technique weights](../running-experiments/caching-adversarial-prompts.md)
    for the full workflow.
