# Caching Adversarial Prompts and Technique Weights

Generating adversarial prompts (P4D, MMA Diffusion, Ring-A-Bell) and training
concept-erasure models (ESD, MACE, SSD, CoGFD, AdvUnlearn) are expensive. When
benchmarking multiple techniques against the same prompt set, or running the same technique
across multiple metrics, repeating these steps on every run wastes significant compute.

This page describes how to generate once and reuse.

---

## Caching adversarial prompts

### P4D

P4D at `num_iter=3000` on 50 prompts takes hours. Run it once, save the output, then point
all subsequent technique evaluations at the cached CSV.

**Run 1 — generate and save:**

```json
{
  "name": "asr_p4d",
  "config": {
    "concept_name": "nudity",
    "target_prompts_path": "data/nudity_target_prompts.csv",
    "erase_id": "std",
    "generated_prompts_output": "cache/p4d_nudity_adversarial.csv",
    "num_iter": 3000,
    "eval_step": 50,
    "device": "cuda"
  }
}
```

**Runs 2–N — skip generation, evaluate only:**

```json
{
  "name": "asr_p4d",
  "config": {
    "concept_name": "nudity",
    "precomputed_prompts_path": "cache/p4d_nudity_adversarial.csv",
    "device": "cuda"
  }
}
```

When `precomputed_prompts_path` is set, all P4D optimisation is skipped. The CSV must
have an `adversarial_prompt` column. `target_prompts_path` is not required.

### MMA Diffusion

The same pattern applies. MMA Diffusion's GCG attack is also expensive to re-run.

**Run 1 — generate and save:**

```json
{
  "name": "asr_mma_diffusion",
  "config": {
    "concept_name": "nudity",
    "output_csv": "cache/mma_nudity_adversarial.csv",
    "n_steps": 200,
    "device": "cuda"
  }
}
```

**Runs 2–N — skip generation:**

```json
{
  "name": "asr_mma_diffusion",
  "config": {
    "concept_name": "nudity",
    "output_csv": "cache/mma_nudity_adversarial.csv",
    "precomputed_prompts_path": "cache/mma_nudity_adversarial.csv",
    "device": "cuda"
  }
}
```

!!! note
    `output_csv` is still required by config validation even when `precomputed_prompts_path`
    is set. Point it at the same file.

### Ring-A-Bell

Ring-A-Bell discovery can be disabled entirely. Set `enable_discovery=false` and pass
prompts directly via `seed_prompts_csv`.

**Run 1 — discover and save** (via `generated_prompts_output`):

```json
{
  "name": "asr_ring_a_bell",
  "config": {
    "concept_name": "nudity",
    "seed_prompts_csv": "data/nudity_target_prompts.csv",
    "generated_prompts_output": "cache/ring_a_bell_nudity_discovered.csv",
    "enable_discovery": true,
    "device": "cuda"
  }
}
```

**Runs 2–N — skip discovery:**

!!! warning "The GA output CSV has no header row"
    `generated_prompts_output` is written without a header (one prompt per row).
    `seed_prompts_csv` expects a header row and skips it on load — so the first prompt
    is silently dropped if you feed the output directly back as input.

    Add a header row before reusing:

    ```bash
    echo "prompt" | cat - cache/ring_a_bell_nudity_discovered.csv > cache/ring_a_bell_nudity_with_header.csv
    ```

    Then use the file with the header as `seed_prompts_csv`:

```json
{
  "name": "asr_ring_a_bell",
  "config": {
    "concept_name": "nudity",
    "seed_prompts_csv": "cache/ring_a_bell_nudity_with_header.csv",
    "enable_discovery": false,
    "device": "cuda"
  }
}
```

---

## Multi-technique comparison with shared prompts

The most common use case: generate adversarial prompts once, then benchmark several
techniques against the same set.

```
Run 1:  ESD + asr_p4d (generate=true)   → cache/p4d_adversarial.csv
Run 2:  MACE + asr_p4d (precomputed)    ← cache/p4d_adversarial.csv
Run 3:  UCE + asr_p4d (precomputed)     ← cache/p4d_adversarial.csv
Run 4:  AdvUnlearn + asr_p4d (precomputed) ← cache/p4d_adversarial.csv
```

Example multi-metric config reusing cached P4D prompts:

```json
{
  "output_dir": "results/mace_nudity_multi",
  "technique": {
    "name": "mace",
    "config": {
      "erase_concept": "nudity",
      "save_path": "checkpoints/mace_nudity.pt",
      "device": "cuda"
    }
  },
  "metrics": [
    {
      "name": "asr_p4d",
      "config": {
        "concept_name": "nudity",
        "precomputed_prompts_path": "cache/p4d_nudity_adversarial.csv",
        "device": "cuda"
      }
    },
    {
      "name": "asr_mma_diffusion",
      "config": {
        "concept_name": "nudity",
        "output_csv": "cache/mma_nudity_adversarial.csv",
        "precomputed_prompts_path": "cache/mma_nudity_adversarial.csv",
        "device": "cuda"
      }
    }
  ]
}
```

---

## Caching technique weights

Training-based techniques (ESD, MACE, SSD, CoGFD, AdvUnlearn) re-run training on every
invocation unless weights are saved and reloaded. This matters when you want to:

- Run multiple metrics against the same trained model without retraining between runs
- Iterate on metric configuration without waiting for training to complete

All training-based techniques support `save_path` and `load_path`. Set `save_path` on the
first run, then replace it with `load_path` for all subsequent runs.

| Technique | `save_path` format | `load_path` format |
|-----------|-------------------|--------------------|
| ESD | single `.pt` file (UNet state dict) | same `.pt` file |
| MACE | single `.pt` file (UNet state dict) | same `.pt` file |
| SSD | single `.pt` file (UNet state dict) | same `.pt` file |
| CoGFD | directory with `unet/` subdirectory (HF `save_pretrained`) | same directory |
| AdvUnlearn | single `.pt` file written to `save_dir/` | path to that `.pt` file via `load_path` |

See each technique's configuration reference for exact field names and format details.

---

## Caveats

**Prompts generated against `erase_id="std"` are not white-box targeted.**
They are hard prompts that elicit the concept from unmodified SD generally, but were not
optimised against any specific erased model. Using them to evaluate a trained technique
still produces a valid ASR score — it just measures how well the technique resists generic
adversarial prompts rather than prompts specifically crafted to bypass it. See
[ASR P4D — Limitations](../metrics/asr_p4d.md#limitations).

**Cached prompts don't transfer across concepts.**
A prompt set generated for nudity is not meaningful for violence evaluation. Keep separate
caches per concept.
