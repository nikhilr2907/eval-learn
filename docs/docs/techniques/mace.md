# MACE — Mass Concept Erasure

## Overview

MACE (Mass Concept Erasure, CVPR 2024) erases concepts through a closed-form weight update
rather than gradient-based fine-tuning. It analytically computes a modification to the
key and value projection matrices in the UNet's cross-attention layers, remapping the
concept's token representations to a neutral (empty) representation.

Because it is closed-form, MACE is deterministic and significantly faster than fine-tuning
approaches like ESD or AdvUnlearn — a single erasure typically completes in seconds rather
than minutes. The `lambda_cfr` parameter controls the conservatism of the update: higher
values preserve the original weights more aggressively, at the cost of weaker erasure.

MACE also supports erasing multiple synonyms of a concept simultaneously by passing a list
to `erase_concept`, which is useful when a concept has many surface forms (e.g.
`["nude", "naked", "nudity", "nsfw"]`).

**Base model:** `CompVis/stable-diffusion-v1-4`

**Supported concepts:** Any — both single strings and lists of synonyms are accepted.

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Any I2P concept | NudeNet for nudity; CLIP for all others |
| ERR | nudity only | Requires `erase_concept` to be or contain `"nudity"` |
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
| `erase_concept` | `str \| list[str]` | `"nudity"` | Concept(s) to erase. A single string or a list of synonyms. All listed terms are mapped to the neutral representation. |
| `erase_from` | `str \| list[str] \| None` | `None` | Scope restriction. If set, only erases the concept when it appears in the context of this broader concept. Defaults to `None` (fully erase to neutral with no scope). |
| `lambda_cfr` | `float` | `0.1` | CFR regularisation strength. Higher = more conservative update, weaker erasure. Lower = more aggressive erasure, higher risk of side effects on related concepts. |
| `save_path` | `str \| None` | `None` | Path to save modified UNet weights. If `None`, weights are held in memory only. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | Classifier-free guidance scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

---

## Warnings

!!! warning "Saving weights"
    If `save_path` is `None`, the CFR computation runs on every invocation. While MACE is
    fast, set `save_path` for any repeated evaluation runs.

!!! warning "lambda_cfr tuning"
    The default `lambda_cfr=0.1` is a reasonable starting point. If ASR remains high after
    erasure, lower it (e.g. `0.01`). If FID or CLIP Score degrades noticeably, raise it
    (e.g. `0.5`). The right value depends on the concept.

!!! warning "Synonym lists and ERR compatibility"
    When `erase_concept` is a list, the validation layer extracts the first element to
    determine concept compatibility with ERR (nudity-specific). Ensure the first element
    is `"nudity"` if using ERR with a synonym list. ASR I2P has no such restriction.

---

## Examples

### Single metric — ASR (nudity)

```json
{
  "output_dir": "results/mace_asr",
  "technique": {
    "name": "mace",
    "config": {
      "erase_concept": "nudity",
      "lambda_cfr": 0.1,
      "save_path": "checkpoints/mace_nudity.pt",
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

### Single metric — with synonym list

```json
{
  "output_dir": "results/mace_nudity_synonyms",
  "technique": {
    "name": "mace",
    "config": {
      "erase_concept": ["nudity", "nude", "naked", "nsfw"],
      "lambda_cfr": 0.1,
      "save_path": "checkpoints/mace_nudity_synonyms.pt",
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
  "output_dir": "results/mace_nudity_multi",
  "technique": {
    "name": "mace",
    "config": {
      "erase_concept": "nudity",
      "lambda_cfr": 0.1,
      "save_path": "checkpoints/mace_nudity.pt",
      "device": "cuda",
      "num_inference_steps": 50,
      "guidance_scale": 7.5
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
