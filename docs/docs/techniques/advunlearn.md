# AdvUnlearn — Adversarially Robust Concept Unlearning

## Overview

AdvUnlearn fine-tunes Stable Diffusion with a dual objective: erase the target concept
while simultaneously defending against adversarial prompt attacks. At each training step
it runs an inner attack loop (PGD or fast_at) to find adversarial embeddings that could
recover the concept, then updates the model to resist those embeddings.

The result is a model that is more robust than standard fine-tuning approaches — naïve
adversarial prompts are less likely to bypass the erasure. The trade-off is substantially
higher training time compared to ESD or MACE, due to the inner attack loop.

AdvUnlearn also supports retention: a subset of the training process uses a retention
dataset (COCO objects or ImageNet) to maintain general generation quality.

**Base model:** `CompVis/stable-diffusion-v1-4`  
**Supported concepts:** nudity (default); other concepts can be specified via `erase_concept`
but the pre-built retention datasets are curated for nudity erasure.

---

## Compatible metrics

| Metric | Compatible | Notes |
|--------|-----------|-------|
| ASR I2P | Any I2P concept | NudeNet for nudity; Q16 for all others |
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
| `erase_concept` | `str` | `"nudity"` | Concept to erase. |
| `train_method` | `str` | `"text_encoder_full"` | Layers to train. See options below. |
| `dataset_retain` | `str` | `"coco_object"` | Retention dataset. One of `"coco_object"`, `"imagenet243"`, `"coco_object_no_filter"`, `"imagenet243_no_filter"`. |
| `retain_train` | `str` | `"iter"` | Retention training schedule. `"iter"` = interleaved, `"reg"` = regularisation. |
| `retain_batch` | `int` | `5` | Batch size for retention samples. Must be > 0. |
| `retain_step` | `int` | `1` | How often (in steps) to apply retention. Must be > 0. |
| `retain_loss_w` | `float` | `1.0` | Weight of the retention loss term. |
| `start_guidance` | `float` | `3.0` | Initial classifier-free guidance scale during training. |
| `negative_guidance` | `float` | `1.0` | Guidance scale for the erasing objective. |
| `train_steps` | `int` | `5` | Total training iterations. Must be > 0. |
| `learning_rate` | `float` | `1e-5` | Learning rate. Must be > 0. |
| `attack_method` | `str` | `"pgd"` | Inner-loop attack algorithm. `"pgd"` (Projected Gradient Descent) or `"fast_at"`. |
| `attack_step` | `int` | `30` | Number of attack steps per iteration. Must be > 0. |
| `attack_lr` | `float` | `1e-3` | Step size for the attack optimiser. Must be > 0. |
| `attack_type` | `str` | `"prefix_k"` | How the attack modifies the prompt embedding. See options below. |
| `attack_init` | `str` | `"latest"` | Attack initialisation strategy. |
| `attack_embd_type` | `str` | `"word_embd"` | Embedding space for the attack. Only `"word_embd"` is supported. |
| `adv_prompt_num` | `int` | `1` | Number of adversarial prompts generated per step. Must be > 0. |
| `adv_prompt_update_step` | `int` | `1` | How often to refresh adversarial prompts. |
| `warmup_iter` | `int` | `1` | Warmup iterations before adversarial training begins. Must be < `train_steps`. |
| `component` | `str` | `"all"` | UNet component to modify. `"all"`, `"ffn"`, or `"attn"`. |
| `norm_layer` | `bool` | `False` | Include layer normalisation in trainable parameters. |
| `ddim_steps` | `int` | `50` | DDIM steps during training rollouts. |
| `save_interval` | `int` | `1` | Checkpoint save frequency (in iterations). |
| `save_dir` | `str \| None` | `None` | Directory to save checkpoints. After training, a `.pt` file is written here named `{concept}_{text_encoder\|unet}.pt` (e.g. `nudity_text_encoder.pt`). The suffix depends on `train_method`: `text_encoder` for any `text_encoder_*` method, `unet` for all others. |
| `load_path` | `str \| None` | `None` | Path to a `.pt` file saved by a previous AdvUnlearn run (from `save_dir`). If the file exists, training is skipped and these weights are loaded. Must match the current `train_method` — a text-encoder checkpoint cannot be loaded with a UNet `train_method` and vice versa. |
| `num_inference_steps` | `int` | `50` | DDIM steps for image generation during evaluation. |
| `guidance_scale` | `float` | `7.5` | CFG scale for generation. |
| `use_fp16` | `bool` | `True` | Run in half precision. |
| `device` | `str` | `"cuda"` | Device to run on. |

### `train_method` options

| Value | Description |
|-------|-------------|
| `"text_encoder_full"` | Fine-tune full text encoder (default, recommended) |
| `"noxattn"` | All UNet layers except cross-attention |
| `"selfattn"` | Self-attention only |
| `"xattn"` | Cross-attention only |
| `"full"` | All UNet layers |
| `"notime"` | All layers except time embeddings |
| `"xlayer"` | Cross-attention at specific layers |
| `"selflayer"` | Self-attention at specific layers |
| `"text_encoder_layer<N>"` | Specific text encoder layer (replace `<N>` with layer index) |

### `attack_type` options

| Value | Description |
|-------|-------------|
| `"prefix_k"` | Prepend adversarial tokens to the prompt |
| `"suffix_k"` | Append adversarial tokens |
| `"replace_k"` | Replace tokens at position k |
| `"add"` | Additive perturbation to embeddings |
| `"mid_k"` | Insert at midpoint |
| `"insert_k"` | Insert at position k |
| `"per_k_words"` | Perturb every k-th word |

---

## Warnings

!!! warning "Checkpoint format and train_method compatibility"
    `load_path` must point to a `.pt` file written by `save_dir` from a previous run.
    The file contains a state dict keyed with `text_encoder.*` prefixes (for `text_encoder_*`
    methods) or bare UNet parameter names (for UNet methods). Loading a text-encoder
    checkpoint with a UNet `train_method` (or vice versa) will raise a key mismatch error.
    Always use the same `train_method` when resuming from a checkpoint.

!!! warning "warmup_iter must be less than train_steps"
    If `warmup_iter >= train_steps`, AdvUnlearn raises a `ValidationError` on startup.
    With `train_steps=5`, `warmup_iter` must be at most `4`.

!!! warning "Training time"
    AdvUnlearn is significantly slower than MACE or UCE. Each outer iteration runs
    `attack_step` inner PGD steps. With defaults (`train_steps=5`, `attack_step=30`),
    expect substantially longer wall-clock time than other techniques. Reduce `attack_step`
    for faster (but less robust) training.

!!! warning "Low train_steps default"
    The default `train_steps=5` is minimal. Published results typically use 100–1000
    steps. The default is set low to keep demo runs feasible — increase it for
    production benchmarks.

!!! warning "attack_embd_type"
    Only `"word_embd"` is supported for `attack_embd_type`. Any other value will cause
    a runtime error inside the attack loop.

---

## Examples

### Single metric — ASR (nudity)

```json
{
  "output_dir": "results/advunlearn_asr",
  "technique": {
    "name": "advunlearn",
    "config": {
      "erase_concept": "nudity",
      "train_method": "text_encoder_full",
      "train_steps": 100,
      "attack_step": 30,
      "warmup_iter": 5,
      "save_dir": "checkpoints/advunlearn_nudity",
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
  "output_dir": "results/advunlearn_nudity_multi",
  "technique": {
    "name": "advunlearn",
    "config": {
      "erase_concept": "nudity",
      "train_method": "text_encoder_full",
      "train_steps": 100,
      "learning_rate": 1e-5,
      "attack_method": "pgd",
      "attack_step": 30,
      "attack_lr": 1e-3,
      "warmup_iter": 5,
      "retain_loss_w": 1.0,
      "save_dir": "checkpoints/advunlearn_nudity",
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


---

!!! tip "Reusing trained weights across runs"
    Set `save_path` on the first run to persist the trained weights, then use `load_path`
    on all subsequent runs to skip retraining. This is especially useful when benchmarking
    multiple metrics against the same trained model. See
    [Caching adversarial prompts and technique weights](../running-experiments/caching-adversarial-prompts.md)
    for the full workflow.
