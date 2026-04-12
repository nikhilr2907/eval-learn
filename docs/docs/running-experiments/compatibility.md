# Compatibility

Not all technique–metric combinations are valid. This page documents the constraints.

---

## Technique constraints

### Nudity-only techniques

The following techniques only support `erase_concept="nudity"` and cannot be used with other concepts:

| Technique | Reason |
|-----------|--------|
| SAFREE | Training-free self-guidance is tuned for nudity filtering |
| SLD | Safe latent diffusion suppression targets sexual content |
| Concept Steerers | Steering vectors shipped with the package target nudity |
| SAeUron | SAE features identified for nudity concept only |

### UCE presets

UCE is limited to three fixed pretrained presets: `nudity`, `violence`, `dog`. Custom concepts are not supported — use ESD or MACE instead.

---

## Metric constraints

### ERR

ERR is nudity-specific. It uses NudeNet detection and the I2P dataset's `sexual` category. It cannot be used with other concepts.

### ASR I2P

ASR I2P supports all I2P concept categories: `nudity`, `harassment`, `hate`, `illegal activity`, `self-harm`, `shocking`, `violence`. The `concept` in the metric config must match the technique's `erase_concept`.

- `nudity` → NudeNet detector
- All other concepts → CLIP similarity

### ASR MMA Diffusion

ASR MMA Diffusion can be run against any concept. You can supply your own `target_prompts` as seed inputs for GCG optimisation.

!!! warning "Custom seed prompts reduce attack effectiveness"
    MMA Diffusion's GCG algorithm optimises adversarial suffixes starting from your seed prompts.
    Using generic or non-adversarial seed prompts will produce weaker attacks and may
    underestimate the true adversarial ASR. For best results, use prompts that already
    elicit the concept from an unmodified base model.

For nudity, built-in seed prompts from the MMA-Diffusion paper are used by default. For
other concepts, `target_prompts` must be provided.

### ASR Ring A Bell

ASR Ring A Bell can be run against any concept with a matching CLIP concept vector. Custom prompts can be used directly via `enable_discovery=false`.

!!! warning "Custom prompts bypass the discovery phase"
    Ring A Bell's genetic algorithm is designed to discover adversarial prompts by evolving
    against the concept's CLIP vector. Supplying your own prompts via `seed_prompts_csv`
    with `enable_discovery=false` skips this entirely. Manually written prompts are typically
    weaker adversarial signals and will likely produce lower ASR scores than properly
    discovered prompts — this is not evidence of better erasure.

---

## Compatibility matrix

| Technique | ASR I2P (nudity) | ASR I2P (other) | ASR MMA Diffusion | ASR Ring A Bell | ERR | UA_IRA | FID | CLIP Score | TIFA |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ESD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MACE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| UCE (nudity/violence) | ✓ | ✓* | ✓ | ✓ | ✓* | ✓ | ✓ | ✓ | ✓ |
| AdvUnlearn | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SAFREE | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| SLD | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Concept Steerers | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| SAeUron | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Free Run | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

\* UCE only supports its fixed presets (`nudity`, `violence`, `dog`) — the metric concept must match the preset used.
