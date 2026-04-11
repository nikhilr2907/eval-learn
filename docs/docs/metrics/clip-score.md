# CLIP Score — Text-to-Image Alignment

## Overview

CLIP Score measures how well generated images match their text prompts, using cosine
similarity between CLIP image and text embeddings. A higher score means generated images
are more semantically aligned with the prompts used to generate them.

Unlike FID, which measures distributional similarity to real images, CLIP Score measures
prompt faithfulness — whether the model still generates what it is asked to generate.
After concept erasure, CLIP Score can drop if the technique over-suppresses features
needed for general prompt adherence.

**Dataset:** TIFA dataset (a diverse set of text-image prompts for faithfulness evaluation)

CLIP Score is concept-agnostic — it works with any technique and any `erase_concept`.

---

## Compatible techniques

All techniques are compatible with CLIP Score. No concept restrictions.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clip_model_name` | `str` | `"openai/clip-vit-base-patch32"` | CLIP model for embedding extraction. See supported models below. |
| `device` | `str \| None` | `None` | Device for CLIP inference. Auto-detects CUDA if `None`. |
| `limit` | `int \| None` | `300` | Maximum number of prompts from the TIFA dataset. |

### Supported CLIP models

- `openai/clip-vit-base-patch16`
- `openai/clip-vit-base-patch32` (default — faster, slightly lower accuracy)
- `openai/clip-vit-large-patch14` (higher accuracy)
- `openai/clip-vit-large-patch14-336`

---

## Output

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | Mean CLIP cosine similarity across all prompt-image pairs. Range [−1, 1]; typical values for SD models fall between 0.2 and 0.35. Higher is better. |

---

## Warnings

!!! warning "CLIP Score does not detect concept erasure"
    CLIP Score measures prompt adherence, not erasure. A model that still generates
    nudity but is otherwise high-quality will score well on CLIP Score. Always pair it
    with an erasure metric (ASR, ERR, UA_IRA).

!!! warning "Model consistency"
    If you compare CLIP Score across runs, ensure `clip_model_name` is the same in all
    runs. Different CLIP variants produce different absolute score ranges and are not
    directly comparable.

---

## Examples

### Single metric

```json
{
  "output_dir": "results/uce_clip_score",
  "technique": {
    "name": "uce",
    "config": {
      "preset": "nudity",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "clip_score",
    "config": {
      "clip_model_name": "openai/clip-vit-base-patch32",
      "device": "cuda",
      "limit": 300
    }
  }
}
```

### As part of a multi-metric run

```json
{
  "name": "clip_score",
  "config": {
    "clip_model_name": "openai/clip-vit-base-patch32",
    "device": "cuda",
    "limit": 300
  }
}
```
