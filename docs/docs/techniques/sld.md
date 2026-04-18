# SLD — Safe Latent Diffusion

## Overview

SLD (Safe Latent Diffusion) is an inference-time technique — it does not modify model
weights at all. Instead, it injects a safety guidance signal into the diffusion process
at each denoising step, steering the latent trajectory away from unsafe content.

The guidance operates by computing the score difference between the original conditioning
and a safety-negative conditioning, then applying it as an additive correction to the
predicted noise. A warmup period (`sld_warmup_steps`) allows early denoising steps (which
determine overall composition) to proceed unmodified, with safety correction kicking in
only after the specified step.

Because SLD is inference-time only, it does not require checkpoints or fine-tuning and is
the fastest technique in the suite. The trade-off is that it cannot fully suppress concepts
from the model — it only redirects generation at runtime.

SLD ships with five named presets that cover a range of safety strengths. These can be
used as-is or overridden by setting individual parameters directly.

**Base model:** `AIML-TUDA/stable-diffusion-safe`  
**Supported concepts:** nudity, violence, hate, disturbing

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Yes | nudity, violence, hate, disturbing |
| ERR | Yes | nudity only (NudeNet-based) |
| FID | Yes | General image quality |
| CLIP Score | Yes | General text-image alignment |
| UA_IRA | Yes | Requires custom prompt CSVs |
| TIFA | Yes | General faithfulness |
| ASR Custom | Yes | Concept-agnostic via CLIP |
| MMA-Diffusion | Yes | Nudity-specific by default |

SLD suppresses nudity, violence, hate, and disturbing content simultaneously. The `erase_concept` field indicates the primary category being benchmarked.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `erase_concept` | `str` | `"nudity"` | Concept to benchmark. One of `"nudity"`, `"violence"`, `"hate"`, `"disturbing"`. |
| `preset` | `str \| None` | `None` | Named preset. If set, overrides all SLD parameter fields below. One of `"none"`, `"weak"`, `"medium"`, `"strong"`, `"max"`. |
| `sld_guidance_scale` | `float` | `5000` | Safety guidance strength. Higher = stronger steering. Overridden by preset. |
| `sld_warmup_steps` | `int` | `0` | Denoising steps to skip before applying safety guidance. Overridden by preset. |
| `sld_threshold` | `float` | `1.0` | Activation threshold for safety guidance. Overridden by preset. |
| `sld_momentum_scale` | `float` | `0.5` | Momentum applied to guidance signal. Overridden by preset. |
| `sld_mom_beta` | `float` | `0.7` | Momentum decay factor. Overridden by preset. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

### Preset parameter values

| Preset | `sld_guidance_scale` | `sld_warmup_steps` | `sld_threshold` | `sld_momentum_scale` | `sld_mom_beta` |
|--------|---------------------|-------------------|----------------|---------------------|----------------|
| `none` | 0 | 0 | 0.0 | 0.0 | 0.0 |
| `weak` | 200 | 15 | 0.0 | 0.0 | 0.0 |
| `medium` | 1000 | 10 | 0.01 | 0.3 | 0.4 |
| `strong` | 2000 | 7 | 0.025 | 0.5 | 0.7 |
| `max` | 5000 | 0 | 1.0 | 0.5 | 0.7 |

---

## Warnings

!!! warning "Supported concepts"
    SLD accepts `erase_concept` of `"nudity"`, `"violence"`, `"hate"`, or `"disturbing"`.
    Any other value raises a `ValidationError`. Note that SLD suppresses all four categories
    simultaneously regardless of which is specified — `erase_concept` indicates which category
    is being benchmarked, not which content is filtered.

!!! warning "Preset fills unspecified parameters only"
    If `preset` is set, it provides default values for any SLD parameter fields you have
    not explicitly specified. Explicitly set fields always take priority over the preset.
    To use a preset with no overrides, omit all individual SLD parameter fields.

!!! warning "Different base model"
    SLD uses `AIML-TUDA/stable-diffusion-safe` rather than `CompVis/stable-diffusion-v1-4`.
    This is a modified SD checkpoint with SafetyNet baked in. Do not substitute a different
    model ID — SLD's guidance hooks are designed for this specific variant.

---

## Examples

### Single metric — ASR with max preset

```json
{
  "output_dir": "results/sld_asr",
  "technique": {
    "name": "sld",
    "config": {
      "erase_concept": "nudity",
      "preset": "max",
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

### Single metric — with manual parameters

```json
{
  "output_dir": "results/sld_manual",
  "technique": {
    "name": "sld",
    "config": {
      "erase_concept": "nudity",
      "preset": null,
      "sld_guidance_scale": 1500,
      "sld_warmup_steps": 8,
      "sld_threshold": 0.015,
      "sld_momentum_scale": 0.4,
      "sld_mom_beta": 0.6,
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
  "output_dir": "results/sld_nudity_multi",
  "technique": {
    "name": "sld",
    "config": {
      "erase_concept": "nudity",
      "preset": "max",
      "device": "cuda"
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
