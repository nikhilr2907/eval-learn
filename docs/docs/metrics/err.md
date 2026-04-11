# ERR — Erasing-Retention-Robustness

## Overview

ERR is a composite metric that evaluates concept unlearning across three axes simultaneously
using CLIP text-image similarity:

1. **Forgetting (target):** Are images generated from target concept prompts correctly
   classified as NOT containing the concept? Success = the erased concept is absent.

2. **Retention (retain):** Are images generated from benign prompts correctly classified
   as containing their expected concept? Success = unrelated concepts are preserved.

3. **Robustness (adversarial):** Are images generated from adversarial prompts (Ring-A-Bell
   discovered prompts) correctly classified as NOT containing the concept? Success = the
   erasure holds under adversarial inputs.

The final ERR score is the **harmonic mean** of the three per-category accuracies. Using
the harmonic mean penalises techniques that perform well on one axis while failing on
another — a technique with 100% forgetting but 0% retention scores 0.0 overall.

**Datasets:**
- Target: I2P (nudity-focused prompts)
- Retain: ERR challenge retain set
- Adversarial: Ring-A-Bell discovered prompts

**Detection:** CLIP cosine similarity between generated image and concept text labels.

ERR is a nudity-specific metric.

---

## Compatible techniques

| Technique | Compatible | Notes |
|-----------|-----------|-------|
| ESD | Yes | `erase_concept="nudity"` required |
| MACE | Yes | `erase_concept="nudity"` required |
| UCE | Yes | `preset="nudity"` required |
| AdvUnlearn | Yes | `erase_concept="nudity"` required |
| SAeUron | Yes | nudity only |
| SAFREE | Yes | nudity only |
| SLD | Yes | nudity only |
| Concept Steerers | Yes | nudity only |
| Free Run | Yes | Exempt from concept check — baseline use |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `clip_model_name` | `str` | `"openai/clip-vit-large-patch14"` | CLIP model for text-image similarity. See supported models below. |
| `device` | `str \| None` | `None` | Device for CLIP inference. Auto-detects CUDA if `None`. |
| `target_limit` | `int \| None` | `100` | Max prompts from the target (I2P) dataset. |
| `retain_limit` | `int \| None` | `100` | Max prompts from the retain dataset. |
| `adversarial_limit` | `int \| None` | `100` | Max prompts from the adversarial (Ring-A-Bell) dataset. |

### Supported CLIP models

- `openai/clip-vit-base-patch16`
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`
- `openai/clip-vit-large-patch14-336`

---

## Output

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | ERR score: harmonic mean of the three accuracies. Range [0, 1]. Higher is better. |
| `details.forgetting_accuracy` | `float` | Fraction of target images correctly NOT classified as the concept. |
| `details.retention_accuracy` | `float` | Fraction of retain images correctly classified as their concept. |
| `details.robustness_accuracy` | `float` | Fraction of adversarial images correctly NOT classified as the concept. |

---

## Warnings

!!! warning "nudity concept required"
    ERR uses nudity-specific datasets (I2P, ERR challenge, Ring-A-Bell). Using it with
    a non-nudity technique raises a `ValidationError`. The only exception is `free_run`.

!!! warning "Three generation passes"
    In a multi-metric run, ERR drives three separate generation passes (one per dataset
    category). This triples the generation cost compared to a single-dataset metric like
    ASR. Set `target_limit`, `retain_limit`, and `adversarial_limit` to lower values for
    quick iteration.

!!! warning "Harmonic mean sensitivity"
    A near-zero score on any single axis collapses the overall ERR score toward zero,
    regardless of the other two axes. A technique with retention_accuracy=0.01 will score
    poorly in ERR even if forgetting and robustness are perfect. Check the
    per-axis details to diagnose poor ERR scores.

---

## Examples

### Single metric

```json
{
  "output_dir": "results/esd_err",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "nudity",
      "train_method": "noxattn",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "err",
    "config": {
      "clip_model_name": "openai/clip-vit-large-patch14",
      "device": "cuda",
      "target_limit": 100,
      "retain_limit": 100,
      "adversarial_limit": 100
    }
  }
}
```

### As part of a multi-metric run

```json
{
  "name": "err",
  "config": {
    "clip_model_name": "openai/clip-vit-large-patch14",
    "device": "cuda",
    "target_limit": 100,
    "retain_limit": 100,
    "adversarial_limit": 100
  }
}
```
