# Metrics

Metrics evaluate how well a technique has erased a concept and how much it has degraded general image quality.

## Erasure metrics

| Metric | Description |
|---|---|
| [ASR I2P](asr_i2p.md) | Attack success rate on the I2P benchmark using a nudity/violence classifier |
| [ASR P4D](asr_p4d.md) | Attack success rate using P4D adversarial prompts |
| [ASR Ring-a-Bell](asr-ring-a-bell.md) | Attack success rate using Ring-a-Bell concept vectors |
| [ASR MMA-Diffusion](asr_mma-diffusion.md) | Attack success rate using MMA-Diffusion adversarial prompts |
| [ERR](err.md) | Erasure recall rate — fraction of concept prompts successfully suppressed |

## Quality metrics

| Metric | Description |
|---|---|
| [FID](fid.md) | Fréchet Inception Distance — distributional similarity to a reference set |
| [CLIP Score](clip-score.md) | Prompt–image alignment measured via CLIP embeddings |
| [TIFA](tifa.md) | Text-image faithfulness via VQA |
| [UA-IRA](ua-ira.md) | Unlearning accuracy vs. image retention accuracy trade-off |
