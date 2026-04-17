# Techniques

Each technique is a concept unlearning method that modifies a Stable Diffusion model to suppress generation of a target concept. They are distributed as separate installable packages.

## Training-based

| Technique | Paper | Description |
|---|---|---|
| [ESD](esd.md) | [ICCV 2023](https://arxiv.org/abs/2303.07345) | Fine-tunes UNet layers to erase a concept via guided score distillation |
| [MACE](mace.md) | [CVPR 2024](https://arxiv.org/abs/2403.06135) | Closed-form rank-one edits to cross-attention weights |
| [CA](ca.md) | [ICCV 2023](https://arxiv.org/abs/2303.13516) | Ablates concept by fine-tuning cross-attention to map it to an anchor |
| [CoGFD](cogfd.md) | [ICLR 2025](https://openreview.net/forum?id=OBjF5I4PWg) | Erases concept *combinations* while preserving individual components |
| [AdvUnlearn](advunlearn.md) | [NeurIPS 2024](https://arxiv.org/abs/2405.15234) | Adversarially robust unlearning via text-encoder fine-tuning |

## Inference-time

| Technique | Paper | Description |
|---|---|---|
| [SSD](ssd.md) | [AAAI 2024](https://arxiv.org/abs/2308.07707) | Selectively dampens UNet parameters using diagonal Fisher information |
| [UCE](uce.md) | [WACV 2024](https://arxiv.org/abs/2308.14761) | Closed-form weight update using concept projection |
| [SAFREE](safree.md) | [ICLR 2025](https://arxiv.org/abs/2410.12761) | Self-supervised token filtering at inference time |
| [SLD](sld.md) | [CVPR 2023](https://arxiv.org/abs/2211.05105) | Suppresses concepts via classifier-free guidance manipulation |
| [SAeUron](saeuron.md) | [ICML 2025](https://arxiv.org/abs/2501.18052) | Sparse autoencoder feature suppression |
| [Concept Steerers](concept-steerers.md) | [arXiv 2025](https://arxiv.org/abs/2501.19066) | Steers activations away from concept directions at inference |
| [TraSCE](trasce.md) | [arXiv 2024](https://arxiv.org/abs/2412.07658) | Training-free concept erasure via trajectory steering |
| [Free Run](free-run.md) | — |Custom Model Evaluation|
