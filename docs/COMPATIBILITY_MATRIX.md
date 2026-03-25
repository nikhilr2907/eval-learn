# Eval-Learn Compatibility Matrix

## Technique Capabilities

### Concept Flexibility Summary

| Technique | Config Name | Hardcoded? | Supported Concepts | Notes |
|-----------|-------------|-----------|-------------------|-------|
| **ESD** | `erase_concept` | ❌ No | Any string | Flexible; trained on nudity but can target other concepts |
| **MACE** | `erase_concept` | ❌ No | Any string (or list) | Flexible; supports single concept or list of synonyms |
| **UCE** | `preset` | ✅ Yes (3) | nudity, violence, dog | Limited to 3 pre-trained presets |
| **SAeUron** | `erase_concept` | ✅ Yes (1) | nudity only | Pickle contains only nudity latents in style_latents_dict |
| **SLD** | `erase_concept` | ✅ Yes (1) | nudity only | Safe Latent Diffusion - nudity safety classifier |
| **SAFREE** | `erase_concept` | ✅ Yes (1) | nudity only | Stages tuned for nudity erasure |
| **ConceptSteerers** | `erase_concept` | ✅ Yes (1) | nudity only | SAE checkpoint trained on nudity only |
| **Free_run** | N/A | N/A | N/A | Baseline generation (no erasure) |

### Detailed Technique Specifications

#### ESD (Erased Stable Diffusion)
- **Config Parameters:**
  - `erase_concept`: str (any concept name)
  - `train_method`: "xattn" | "noxattn" | "selfattn" | "full"
  - `negative_guidance`: float (default 2.0)
  - `train_steps`, `learning_rate`, `use_fp16`: training hyperparameters
- **Flexibility:** ✅ Fully flexible - concept agnostic
- **Training Methods:**
  - `xattn`: Fine-tunes cross-attention (best for specific objects, artists)
  - `noxattn`: Skips cross-attention (good for general concepts)
  - `selfattn`: Only self-attention layers
  - `full`: All layers (most aggressive)

#### MACE (Mass Concept Erasure)
- **Config Parameters:**
  - `erase_concept`: str | List[str] (single or multiple concepts)
  - `lambda_cfr`: float (regularization strength, default 0.1)
- **Flexibility:** ✅ Fully flexible - accepts any concept(s)
- **Key Feature:** Closed-Form Refinement (CFR) - no training loop, purely analytical

#### UCE (Unlearning with Concept Erasure)
- **Config Parameters:**
  - `preset`: "nudity" | "violence" | "dog" (REQUIRED)
  - `erase_concept`: property (derived from preset)
- **Flexibility:** 🔒 Limited to 3 pre-trained presets
- **Pre-trained Weights:** Only for nudity, violence, and dog

#### SAeUron (Sparse Autoencoder Unlearning)
- **Config Parameters:**
  - `erase_concept`: "nudity" (FIXED)
  - `position`: str (UNet hook location, default "unet.up_blocks.1.attentions.1")
  - `multiplier`: float (default -20.0 for ablation)
  - `percentile`: float (latent selection threshold, default 99.99)
  - `target_latents`: Optional[List[int]] (explicit latents to target)
  - `acts_path`: Path to cached activations pickle
- **Flexibility:** 🔒 Nudity only - cls_latents_dict_mini.pkl contains only nudity
- **Activation Lookup:** Dynamically reads cached importance scores from pickle

#### SLD (Safe Latent Diffusion)
- **Config Parameters:**
  - `erase_concept`: "nudity" (FIXED)
  - `preset`: "none" | "weak" | "medium" | "strong" | "max"
  - `sld_guidance_scale`, `sld_warmup_steps`, etc.
- **Flexibility:** 🔒 Nudity only - inherent to the safety classifier
- **Model:** Uses AIML-TUDA/stable-diffusion-safe

#### SAFREE (Selective and Attribute Free)
- **Config Parameters:**
  - `erase_concept`: "nudity" (FIXED)
  - `alpha`, `enable_svf`, `enable_lra`: Stage-specific parameters
  - `lra_filter_type`, `freeu_*`: Latent re-attention hyperparameters
- **Flexibility:** 🔒 Nudity only - all stages tuned for nudity
- **Architecture:** 3-stage pipeline (Text Projection, SVF, LRA)

#### ConceptSteerers
- **Config Parameters:**
  - `erase_concept`: "nudity" (FIXED)
  - `multiplier`: float (strength of steering)
  - `sae_path`: Path to SAE checkpoint (default: i2p_sd14_l9)
- **Flexibility:** 🔒 Nudity only - SAE checkpoint trained on nudity
- **Based On:** SAE-based steering approach

---

## Metric Compatibility

### Metric Requirements & Restrictions

| Metric | Requires Technique | Requires Concept | Dataset Req | Notes |
|--------|-------------------|-----------------|-------------|-------|
| **CCRT** | free_run only | N/A | HuggingFace I2P | Genetic search; requires original + erased comparison |
| **ASR** | Any | nudity | I2P (hardcoded) | Nudity-specific detector (NudeNet) |
| **ERR** | Any | nudity | I2P (hardcoded) | Nudity-specific detector (NudeNet) |
| **UA_IRA** | Any | Any | User CSV | CLIP-based; flexible to any concept via CSV paths |
| **FID** | Any | Any | Any | Model-agnostic; compares diversity |
| **Others** | Any | Any | Flexible | Framework-specific metrics |

### Metric-Technique Compatibility Rules

**CCRT (Concept Correction through Regeneration):**
- ✅ **Required:** `technique_name == "free_run"`
- ❌ **Blocked:** All other techniques
- **Reason:** Requires original unmodified model for comparison

**ASR/ERR (Attack/Erase Robustness):**
- ✅ **Allowed:** Any technique with `erase_concept == "nudity"`
- ❌ **Blocked:** Techniques erasing non-nudity concepts
- **Reason:** Uses hardcoded I2P nudity dataset + NudeNet detector

**UA_IRA (Unlearning/Retention Accuracy):**
- ✅ **Allowed:** Any technique, any `erase_concept`
- **Requirement:** User must provide CSV paths in metric config:
  - `target_prompts_path`: CSV with prompts for target concept
  - `retain_prompts_path`: CSV with prompts for retain concept

---

## Runner Validation Logic

### SingleBenchmarkRunner

```python
def _validate(self):
    # 1. Get technique and metric factories
    technique_factory = get_technique(technique_name)
    metric_factory = get_metric(metric_name)

    # 2. CCRT requires free_run
    if metric_name == "ccrt" and technique_name != "free_run":
        raise ValueError("CCRT requires 'free_run' technique")

    # 3. ASR/ERR require nudity
    if metric_name in ["asr", "err"]:
        erase_concept = technique_config.get("erase_concept", "").lower()
        if erase_concept and erase_concept != "nudity":
            raise ValueError(
                f"Metric '{metric_name}' requires nudity concept. "
                f"Got erase_concept='{erase_concept}'"
            )
```

### MultiBenchmarkRunner

```python
def _validate(self):
    # 1. Check all metric factories
    for name in metric_names:
        get_metric(name)

    # 2. ASR/ERR require nudity (checks technique config)
    nudity_metrics = {"asr", "err"}
    used_nudity_metrics = set(metric_names) & nudity_metrics
    if used_nudity_metrics:
        erase_concept = technique_config.get("erase_concept", "").lower()
        if erase_concept and erase_concept != "nudity":
            raise ValueError(
                f"Metrics {used_nudity_metrics} require nudity. "
                f"Got erase_concept='{erase_concept}'"
            )
```

### MatrixBenchmarkRunner

```python
def _validate(self):
    # 1. CCRT requires all techniques to be free_run
    if "ccrt" in metric_names:
        non_free = [t for t in technique_names if t != "free_run"]
        if non_free:
            raise ValueError(
                "CCRT requires all techniques to be 'free_run'. "
                f"Got: {non_free}"
            )

    # 2. ASR/ERR require nudity for each technique
    nudity_metrics = {"asr", "err"}
    used_nudity_metrics = set(metric_names) & nudity_metrics
    if used_nudity_metrics:
        for technique_name in technique_names:
            tech_config = technique_configs.get(technique_name, {})
            erase_concept = tech_config.get("erase_concept", "").lower()
            if erase_concept and erase_concept != "nudity":
                raise ValueError(
                    f"Metrics {used_nudity_metrics} require nudity. "
                    f"Got technique '{technique_name}' with erase_concept='{erase_concept}'"
                )
```

---

## Configuration Examples

### Example 1: Flexible Concept (ESD with Custom Concept)

```python
single_runner = SingleBenchmarkRunner(
    technique_name="esd",
    metric_name="ua_ira",
    technique_config={
        "erase_concept": "artistic_style",
        "train_method": "xattn",
        "train_steps": 200,
    },
    metric_config={
        "target_prompts_path": "path/to/artistic_targets.csv",
        "retain_prompts_path": "path/to/artistic_retain.csv",
        "target_concept_name": "Picasso style",
        "retain_concept_name": "general art",
    },
)
```

### Example 2: Nudity-Specific (SAeUron + ASR)

```python
single_runner = SingleBenchmarkRunner(
    technique_name="saeuron",
    metric_name="asr",
    technique_config={
        # erase_concept is fixed to "nudity" - validation ensures this
        "erase_concept": "nudity",
        "multiplier": -20.0,
        "percentile": 99.99,
    },
    metric_config={
        # ASR uses hardcoded I2P dataset + NudeNet
    },
)
```

### Example 3: UCE with Preset

```python
single_runner = SingleBenchmarkRunner(
    technique_name="uce",
    metric_name="fid",
    technique_config={
        "preset": "violence",  # Pre-trained violence weights
        # erase_concept property will be "violence"
    },
    metric_config={
        # FID is concept-agnostic
    },
)
```

### Example 4: Matrix Benchmark (Mixed Concepts - Invalid)

```python
# This will FAIL validation
matrix_runner = MatrixBenchmarkRunner(
    technique_names=["esd", "mace"],
    metric_names=["asr"],  # Requires nudity
    technique_configs={
        "esd": {"erase_concept": "violence"},
        "mace": {"erase_concept": "violence"},
    },
)
# Error: ASR metric requires nudity concept, got violence
```

### Example 5: Matrix Benchmark (CCRT - Invalid)

```python
# This will FAIL validation
matrix_runner = MatrixBenchmarkRunner(
    technique_names=["esd", "free_run"],
    metric_names=["ccrt"],  # Requires free_run only
)
# Error: CCRT metric only works with free_run technique
```

---

## Summary Table: What You Can Mix

| Technique | With CCRT? | With ASR/ERR? | With UA_IRA? | With FID? |
|-----------|-----------|--------------|------------|----------|
| **free_run** | ✅ Only this | ❌ (not nudity) | ✅ | ✅ |
| **ESD** (any concept) | ❌ | ❌ (unless nudity) | ✅ | ✅ |
| **ESD** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **MACE** (any concept) | ❌ | ❌ (unless nudity) | ✅ | ✅ |
| **MACE** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **SAeUron** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **SLD** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **SAFREE** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **ConceptSteerers** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **UCE** (nudity) | ❌ | ✅ | ✅ | ✅ |
| **UCE** (violence) | ❌ | ❌ | ✅ | ✅ |
| **UCE** (dog) | ❌ | ❌ | ✅ | ✅ |
