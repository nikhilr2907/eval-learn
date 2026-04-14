# Free Run — Baseline Unmodified Generation

## Overview

Free Run is not an unlearning technique — it is a baseline. It loads any HuggingFace
text-to-image model and generates images without any safety filtering, weight modification,
or inference-time intervention.

Use Free Run to establish baseline metric scores before applying an unlearning technique.
The delta between a Free Run result and a technique's result is the actual measured effect
of that technique. Without a Free Run baseline, scores from individual techniques are
difficult to interpret in isolation.

Unlike every other technique in the suite, Free Run accepts any model ID via `model_id`
rather than fixing a base model. This makes it useful for baselining alternative SD
checkpoints (e.g. SD 1.5, SD 2.0).

!!! warning "No nudity filtering"
    Free Run generates without any safety constraints. When used with ASR or ERR, it will
    produce high ASR scores by design — this is the expected baseline behaviour.

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR | Yes | free_run bypasses nudity concept check |
| ERR | Yes | free_run bypasses nudity concept check |
| FID | Yes | General image quality |
| CLIP Score | Yes | General text-image alignment |
| UA_IRA | Yes | Requires custom prompt CSVs |
| TIFA | Yes | General faithfulness |
| ASR Custom | Yes | Concept-agnostic via CLIP |
| MMA-Diffusion | Yes | Requires explicit target prompts for non-nudity concepts |

Free Run is the only technique exempt from nudity-concept validation. ASR and ERR can
be used with Free Run regardless of concept.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str` | — | **Required.** HuggingFace model ID for any text-to-image model supported by `AutoPipelineForText2Image`. |
| `device` | `str \| None` | `None` | Device to run on. Auto-detects CUDA → MPS → CPU if `None`. |
| `use_fp16` | `bool` | `True` | Run in half precision on CUDA. Ignored on CPU and MPS (always float32). |
| `num_inference_steps` | `int` | `50` | Number of denoising steps. Overridable per call. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale. Overridable per call. |

### Supported models

Free Run uses `AutoPipelineForText2Image`, which supports any HuggingFace model whose
repo specifies a compatible pipeline class, including SD 1.x, SD 2.x, SDXL, FLUX,
PixArt-α/Σ, Kandinsky, and HunyuanDiT.

### Default value notes

`num_inference_steps=50` and `guidance_scale=7.5` are calibrated for SD 1.x. For other
model families these may not be optimal:

| Model | Typical steps | Typical guidance |
|-------|--------------|-----------------|
| SD 1.x / SD 2.x | 50 | 7.5 |
| SDXL | 25–40 | 5.0–7.5 |
| FLUX | 20–28 | 3.5–4.0 |
| PixArt | 20 | 4.5 |

If you are baselining a non-SD model, override both in the config explicitly.

---

## Warnings

!!! warning "model_id is required"
    Free Run has no fixed base model. Leaving `model_id` empty or omitting it will raise
    a `ValueError` before the model is loaded.

!!! warning "num_inference_steps and guidance_scale defaults are SD-centric"
    The defaults (`50` steps, `7.5` guidance) are appropriate for SD 1.x. They will
    produce valid output for other model families but may not reflect published results
    for those models. Set them explicitly when baselining SDXL, FLUX, or PixArt.

---

## Examples

### SD 1.4 — ASR baseline

```json
{
  "output_dir": "results/free_run_asr_baseline",
  "technique": {
    "name": "free_run",
    "config": {
      "model_id": "CompVis/stable-diffusion-v1-4",
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

### SDXL — FID baseline

```json
{
  "output_dir": "results/free_run_sdxl_fid_baseline",
  "technique": {
    "name": "free_run",
    "config": {
      "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
      "num_inference_steps": 30,
      "guidance_scale": 5.0,
      "device": "cuda"
    }
  },
  "metric": {
    "name": "fid",
    "config": {
      "device": "cuda",
      "limit": 1000
    }
  }
}
```

### Multiple metrics — full baseline

```json
{
  "output_dir": "results/free_run_baseline",
  "technique": {
    "name": "free_run",
    "config": {
      "model_id": "CompVis/stable-diffusion-v1-4",
      "device": "cuda"
    }
  },
  "metrics": [
    { "name": "asr", "config": { "device": "cuda", "limit": 500 } },
    { "name": "fid", "config": { "device": "cuda", "limit": 1000 } },
    { "name": "clip_score", "config": { "device": "cuda", "limit": 300 } },
    { "name": "tifa", "config": { "device": "cuda", "limit": 200 } }
  ]
}
```
