# SAFREE — Training-Free Semantic Filtering

## Overview

SAFREE (Semantic Approach to Free-up Representations) is a training-free technique —
it modifies no model weights. All filtering happens at inference time through three
sequential stages applied during the diffusion process:

1. **Text Projection (Stage 1):** Scales the concept's text embedding by `alpha` to reduce
   its influence on cross-attention.

2. **Self-Validation Filter (Stage 2, SVF):** At denoising timesteps above
   `upperbound_timestep`, intercepts the latent activations and suppresses features
   associated with the target concept. Controlled by `enable_svf`.

3. **Latent Re-Attention (Stage 3, LRA):** Applies FreeU-style frequency filtering to
   the UNet's skip connections and backbone features using the `freeu_*` parameters.
   Controlled by `enable_lra`.

Because SAFREE is training-free, it is fast to initialise. However, it only filters
at runtime and cannot guarantee concept removal across all possible prompts.

**Base model:** `CompVis/stable-diffusion-v1-4`  
**Supported concepts:** nudity only

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
| `alpha` | `float` | `0.01` | Stage 1: scaling factor applied to the concept text embedding. Values near 0 suppress the concept text signal strongly. |
| `enable_svf` | `bool` | `True` | Enable Stage 2 Self-Validation Filter. |
| `upperbound_timestep` | `int` | `10` | Stage 2: only apply SVF at timesteps above this value. Lower values = SVF active for more of the denoising trajectory. |
| `enable_lra` | `bool` | `True` | Enable Stage 3 Latent Re-Attention. |
| `lra_filter_type` | `str` | `"high"` | Stage 3: frequency filter type. `"high"` suppresses high-frequency components, `"low"` suppresses low-frequency, `"all"` applies to all. |
| `freeu_b1` | `float` | `1.0` | FreeU backbone scaling factor for block 1. |
| `freeu_b2` | `float` | `1.0` | FreeU backbone scaling factor for block 2. |
| `freeu_s1` | `float` | `0.9` | FreeU skip-connection scaling factor for block 1. |
| `freeu_s2` | `float` | `0.2` | FreeU skip-connection scaling factor for block 2. |
| `re_attn_timestep_range` | `[int, int]` | `[-1, 1001]` | Fallback timestep range for re-attention when SVF is disabled. |
| `num_inference_steps` | `int` | `50` | Total DDIM steps. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

---

## Warnings

!!! warning "nudity only"
    SAFREE only supports `erase_concept="nudity"`. Any other value raises a
    `ValidationError`. For custom concept filtering without training, there is no direct
    alternative in the current suite.

!!! warning "All three stages are cooperative"
    Disabling SVF or LRA (`enable_svf=false`, `enable_lra=false`) weakens erasure
    significantly. The three stages are designed to work together. Disable individual
    stages only for ablation experiments.

!!! warning "alpha near 1.0"
    Setting `alpha` close to `1.0` makes Stage 1 a no-op (no scaling of the concept
    embedding). This is fine for testing Stages 2 and 3 in isolation, but provides
    effectively no text-projection filtering.

---

## Examples

### Single metric — ASR

```json
{
  "output_dir": "results/safree_asr",
  "technique": {
    "name": "safree",
    "config": {
      "erase_concept": "nudity",
      "alpha": 0.01,
      "enable_svf": true,
      "enable_lra": true,
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
  "output_dir": "results/safree_nudity_multi",
  "technique": {
    "name": "safree",
    "config": {
      "erase_concept": "nudity",
      "alpha": 0.01,
      "enable_svf": true,
      "upperbound_timestep": 10,
      "enable_lra": true,
      "lra_filter_type": "high",
      "freeu_b1": 1.0,
      "freeu_b2": 1.0,
      "freeu_s1": 0.9,
      "freeu_s2": 0.2,
      "device": "cuda",
      "num_inference_steps": 50
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
