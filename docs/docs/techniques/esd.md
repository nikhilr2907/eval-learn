# ESD — Erased Stable Diffusion

## Overview

ESD (Erased Stable Diffusion) fine-tunes selected layers of a Stable Diffusion UNet using
a negative guidance objective. During training, the model is pushed to predict noise
conditioned on the target concept as if it had received strong negative guidance — effectively
teaching it to anti-generate the concept.

ESD was introduced to handle two distinct erasure regimes, controlled by `train_method`:

- **ESD-x** (`xattn`) — fine-tunes only the cross-attention layers. Best for erasing specific
  objects, artistic styles, or named entities where the concept is primarily encoded in
  text-to-image cross-attention.
- **ESD-u** (`noxattn`) — fine-tunes all layers except cross-attention. Better for broad
  semantic concepts like nudity where the concept manifests in deeper UNet activations.

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — ESD makes no assumptions about the concept. Any string passed
as `erase_concept` is used as the conditioning text during training.

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
| `erase_concept` | `str` | `"nudity"` | The concept to erase. Used as the conditioning text during fine-tuning. |
| `erase_from` | `str \| None` | `None` | The broader concept to erase from. If `None`, defaults to `erase_concept`. Set this to erase a sub-concept (e.g. erase `"Van Gogh"` from `"art"`). |
| `train_method` | `str` | `"xattn"` | Which UNet layers to fine-tune. See options below. |
| `negative_guidance` | `float` | `2.0` | Scale of the negative guidance during training. Higher values = stronger push away from the concept. |
| `train_steps` | `int` | `200` | Number of fine-tuning steps. |
| `learning_rate` | `float` | `5e-5` | Optimiser learning rate. |
| `use_fp16` | `bool` | `True` | Run in half precision. Disable if you encounter NaN losses. |
| `load_path` | `str \| None` | `None` | Path to a `.pt` file containing a pre-trained UNet state dict (saved by a previous ESD run). If set, training is skipped entirely and these weights are loaded directly. |
| `save_path` | `str \| None` | `None` | Path to save the fine-tuned UNet state dict as a `.pt` file after training. Only used when training runs (i.e. `load_path` is not set). |
| `num_inference_steps` | `int` | `50` | DDIM steps used during image generation for evaluation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `device` | `str` | `"cuda"` | Device to run on. |

### `train_method` options

| Value | Layers trained | Use when |
|-------|---------------|----------|
| `"xattn"` | Cross-attention only | Erasing styles, artists, specific objects |
| `"noxattn"` | All layers except cross-attention | Erasing broad semantic concepts (nudity, violence) |
| `"selfattn"` | Self-attention only | Rarely needed; experimental |
| `"full"` | All UNet layers | Maximum erasure, highest risk of over-erasure |

---

## Warnings

!!! warning "load_path and save_path are distinct"
    `load_path` and `save_path` serve different purposes and should not be set to the same
    file. Set `save_path` on a training run to persist the weights. On all subsequent runs,
    set `load_path` to those same weights to skip retraining. If neither is set, ESD retrains
    from scratch on every run.

!!! warning "Checkpoint format"
    Both `load_path` and `save_path` refer to a single `.pt` file containing the full UNet
    state dict (`torch.save(unet.state_dict(), path)`). Do not point `load_path` at a
    HuggingFace model directory or a partial state dict — it will fail silently or raise a
    key mismatch error.

!!! warning "train_method and over-erasure"
    Using `"full"` trains all layers and carries a higher risk of degrading general image quality.
    Prefer `"noxattn"` for nudity and `"xattn"` for style/object erasure. Check FID and CLIP Score
    alongside erasure metrics to detect over-erasure.

!!! warning "NaN losses with fp16"
    On some hardware, `use_fp16=True` can produce NaN losses during training, causing the
    technique to silently produce a degraded model. If CLIP Score collapses to near zero,
    retry with `use_fp16=false`.

---

## Examples

### Single metric — ASR (nudity)

```json
{
  "output_dir": "results/esd_asr",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "nudity",
      "train_method": "noxattn",
      "negative_guidance": 2.0,
      "train_steps": 200,
      "learning_rate": 5e-5,
      "save_path": "checkpoints/esd_nudity.pt",
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

### Single metric — UA_IRA (custom concept)

```json
{
  "output_dir": "results/esd_vangogh_ua_ira",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "Van Gogh",
      "train_method": "xattn",
      "train_steps": 200,
      "save_path": "checkpoints/esd_vangogh.pt",
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
  "output_dir": "results/esd_nudity_multi",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "nudity",
      "train_method": "noxattn",
      "negative_guidance": 2.0,
      "train_steps": 200,
      "learning_rate": 5e-5,
      "save_path": "checkpoints/esd_nudity.pt",
      "device": "cuda",
      "num_inference_steps": 50,
      "guidance_scale": 7.5
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
