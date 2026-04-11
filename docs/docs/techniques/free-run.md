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
| `model_id` | `str` | `""` | **Required.** HuggingFace model ID for any text-to-image diffusion model. |
| `device` | `str \| None` | `None` | Device to run on. Auto-detects CUDA → MPS → CPU if `None`. |
| `use_fp16` | `bool` | `True` | Run in half precision. |

---

## Warnings

!!! warning "model_id is required"
    Free Run has no fixed base model. Leaving `model_id` as an empty string or omitting it
    will raise an error when the pipeline is loaded. Always specify a valid HuggingFace
    model ID.

!!! warning "No num_inference_steps or guidance_scale"
    Unlike other techniques, Free Run does not expose `num_inference_steps` or
    `guidance_scale` in its config. These are controlled by each metric's own generation
    parameters or the runner defaults.

---

## Examples

### Single metric — ASR baseline

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

### Single metric — FID baseline with SD 1.5

```json
{
  "output_dir": "results/free_run_fid_baseline",
  "technique": {
    "name": "free_run",
    "config": {
      "model_id": "runwayml/stable-diffusion-v1-5",
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
