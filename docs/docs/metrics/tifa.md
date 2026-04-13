# TIFA — Text-to-Image Faithfulness Assessment

## Overview

TIFA evaluates prompt faithfulness using Visual Question Answering (VQA) rather than
embedding similarity. For each generated image, BLIP-2 is asked a set of questions
derived from the generation prompt. The score is the fraction of questions answered
correctly across all images.

For example, for the prompt "a red bicycle in a park", TIFA might ask:
- "What colour is the bicycle?" → expected: "red"
- "Where is the bicycle?" → expected: "park"

This QA-based approach captures fine-grained semantic faithfulness that embedding-based
metrics like CLIP Score can miss — particularly for attribute binding (colour, count,
spatial relationships).

**Dataset:** TIFA dataset (with pre-annotated QA pairs per prompt)  
**VQA model:** BLIP-2 (Salesforce/blip2-flan-t5-xl by default)

TIFA is concept-agnostic and compatible with all techniques.

---

## Compatible techniques

All techniques are compatible with TIFA. No concept restrictions.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vqa_model_name` | `str` | `"Salesforce/blip2-flan-t5-xl"` | HuggingFace model ID for the BLIP-2 VQA model. |
| `device` | `str \| None` | `None` | Device for BLIP-2 inference. Auto-detects CUDA if `None`. |
| `limit` | `int \| None` | `200` | Maximum number of prompts from the TIFA dataset. |

---

## Output

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | TIFA score: fraction of QA pairs answered correctly. Range [0, 1]. Higher is better. Typical SD baselines score 0.7–0.85. |

---

## Warnings

!!! warning "BLIP-2 GPU memory"
    `Salesforce/blip2-flan-t5-xl` is a large model (~15B parameters). It requires
    substantial GPU memory (16GB+ recommended). In a multi-metric run, TIFA should
    be listed last to avoid GPU memory conflicts with other metrics that are still loaded.

!!! warning "qa_pairs metadata requirement"
    TIFA requires each batch to carry `qa_pairs` metadata — lists of
    `{"question": str, "answer": str}` dicts parallel to the images. The TIFA dataset
    provides these automatically. If you supply a custom dataset without `qa_pairs`,
    TIFA will fail with a `KeyError` on the metadata dict.

!!! warning "Slow evaluation"
    BLIP-2 inference is slow compared to CLIP-based metrics. With `limit=200`, expect
    TIFA to take several minutes on a GPU. Reduce `limit` for faster iteration.

---

## Examples

### Single metric

```json
{
  "output_dir": "results/esd_tifa",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "nudity",
      "train_method": "noxattn",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "tifa",
    "config": {
      "vqa_model_name": "Salesforce/blip2-flan-t5-xl",
      "device": "cuda",
      "limit": 200
    }
  }
}
```

### As part of a multi-metric run

```json
{
  "name": "tifa",
  "config": {
    "vqa_model_name": "Salesforce/blip2-flan-t5-xl",
    "device": "cuda",
    "limit": 200
  }
}
```
