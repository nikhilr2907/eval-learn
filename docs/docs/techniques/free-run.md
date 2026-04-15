# Free Run — Custom Model Evaluation

## Overview

Free Run loads any HuggingFace text-to-image model and generates images without any weight
modification or inference-time intervention. Use it to evaluate a model you already have —
an external checkpoint, a fine-tuned variant, or any T2I model not covered by the built-in
technique wrappers.

Unlike every other technique, Free Run does not fix a base model. You supply the `model_id`
and Free Run loads it via `AutoPipelineForText2Image`, which supports SD 1.x, SD 2.x, SDXL,
FLUX, PixArt-α/Σ, Kandinsky, HunyuanDiT, and any other model whose HuggingFace repo
specifies a compatible pipeline class.

No erasure, filtering, or safety mechanism is applied — what the model generates is what
gets evaluated.

---

## Compatible metrics

All metrics are compatible. Free Run is exempt from nudity-concept validation, so ASR and
ERR can be used regardless of concept.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str` | — | **Required.** HuggingFace model ID for any T2I model supported by `AutoPipelineForText2Image`. |
| `device` | `str \| None` | `None` | Device to run on. Auto-detects CUDA → MPS → CPU if `None`. |
| `use_fp16` | `bool` | `True` | Run in half precision on CUDA. Ignored on CPU and MPS (always float32). |
| `num_inference_steps` | `int` | `50` | Number of denoising steps. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale. |

### Default value notes

`num_inference_steps=50` and `guidance_scale=7.5` are calibrated for SD 1.x. Set them
explicitly when evaluating other model families:

| Model | Typical steps | Typical guidance |
|-------|--------------|-----------------|
| SD 1.x / SD 2.x | 50 | 7.5 |
| SDXL | 25–40 | 5.0–7.5 |
| FLUX | 20–28 | 3.5–4.0 |
| PixArt | 20 | 4.5 |

---

## Warnings

!!! warning "model_id is required"
    Free Run has no fixed base model. Omitting `model_id` raises a `ValueError` at
    initialisation.

!!! warning "No safety filtering"
    Free Run disables the safety checker on any model that has one. Images are generated
    without constraint — high ASR scores are expected when evaluating concepts the model
    has not been trained to suppress.

!!! warning "num_inference_steps and guidance_scale defaults are SD-centric"
    The defaults produce valid output for other model families but may not reflect
    published results for those models. Set them explicitly when evaluating SDXL, FLUX,
    or PixArt.

---

## Examples

### Evaluating a custom model checkpoint

```json
{
  "output_dir": "results/my_model_nudity",
  "technique": {
    "name": "free_run",
    "config": {
      "model_id": "my-org/my-finetuned-sd",
      "device": "cuda"
    }
  },
  "metrics": [
    {
      "name": "asr_i2p",
      "config": { "concept_name": "nudity", "device": "cuda", "limit": 500 }
    },
    {
      "name": "fid",
      "config": { "device": "cuda", "limit": 1000 }
    }
  ]
}
```

### SDXL

```json
{
  "output_dir": "results/sdxl_eval",
  "technique": {
    "name": "free_run",
    "config": {
      "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
      "num_inference_steps": 30,
      "guidance_scale": 5.0,
      "device": "cuda"
    }
  },
  "metrics": [
    { "name": "fid", "config": { "device": "cuda", "limit": 1000 } },
    { "name": "clip_score", "config": { "device": "cuda", "limit": 500 } }
  ]
}
```
