import os
import logging
from typing import List, Optional, Union

import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

logger = logging.getLogger(__name__)


class MACEPipeline:
    """
    MACE: Mass Concept Erasure in Diffusion Models (CVPR 2024).

    Implements Stage 1 of MACE: Closed-Form Refinement (CFR) of the
    cross-attention K/V projection matrices. This is the core MACE
    contribution — no training loop is required.

    For each cross-attention K/V weight matrix W, CFR solves:

        W_new = (λW + W @ D @ C^T) @ inv(λI + C @ C^T)

    where:
        C = concatenated token embeddings for ALL concepts  [d_text, 77*N]
        D = concatenated token embeddings for ALL targets   [d_text, 77*N]
        λ = regularization (lambda_cfr), controls erasure strength vs preservation

    Multiple concepts are handled in a single matrix solve — no extra cost
    per additional concept.

    Reference: Lu et al., "MACE: Mass Concept Erasure in Diffusion Models",
    CVPR 2024. https://arxiv.org/abs/2403.06135

    Args:
        model_id: HuggingFace model ID for Stable Diffusion.
        device: Device to run on ('cuda', 'cpu', 'mps', or None for auto).
        erase_concept: Concept(s) to erase. A single string or a list of
                       strings (e.g. ['nudity', 'naked', 'bare skin']).
        erase_from: Target(s) to map each concept to. A single string
                    (applied to all concepts), a matching list, or None
                    (defaults to '' — fully erase).
        lambda_cfr: Regularization strength for CFR. Higher = more conservative
                    (preserves more, erases less). Default 0.1.
        save_path: Optional path to save the modified UNet weights.
        load_path: Optional path to load a pre-modified UNet (skips CFR).
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: Optional[str] = None,
        erase_concept: Union[str, List[str]] = "nudity",
        erase_from: Optional[Union[str, List[str]]] = None,
        lambda_cfr: float = 0.1,
        save_path: Optional[str] = None,
        load_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.lambda_cfr = lambda_cfr
        self.save_path = save_path

        # Normalise erase_concept → always a list internally
        self.erase_concepts: List[str] = (
            [erase_concept] if isinstance(erase_concept, str) else list(erase_concept)
        )

        # Normalise erase_from → list matching erase_concepts length
        if erase_from is None:
            self.erase_targets: List[str] = [""] * len(self.erase_concepts)
        elif isinstance(erase_from, str):
            self.erase_targets = [erase_from] * len(self.erase_concepts)
        else:
            if len(erase_from) != len(self.erase_concepts):
                raise ValueError(
                    f"erase_from has {len(erase_from)} entries but "
                    f"erase_concept has {len(self.erase_concepts)}. Lengths must match."
                )
            self.erase_targets = list(erase_from)

        # Keep a single string for logging / generate() message
        self.erase_concept = ", ".join(f"'{c}'" for c in self.erase_concepts)

        if device:
            self.device = device
        else:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        logger.info(
            f"MACEPipeline: model={model_id}, concepts={self.erase_concepts}, "
            f"targets={self.erase_targets}, lambda_cfr={lambda_cfr}, device={self.device}"
        )

        if load_path:
            self._load_weights(load_path)
        else:
            self._apply_cfr()

        # Build generation pipeline with the modified UNet
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=self.unet,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

    def _load_weights(self, load_path: str):
        """Load a pre-modified UNet, skipping CFR."""
        logger.info(f"Loading pre-modified MACE UNet from {load_path}")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)
        self.unet.load_state_dict(torch.load(load_path, map_location=self.device))
        self.unet.eval()

    def _get_token_embeddings(self, text: str, tokenizer, text_encoder) -> torch.Tensor:
        """
        Return CLIP hidden states for all 77 tokens as a column matrix.

        Returns:
            Tensor of shape [d_text, 77], in float32 for numerical stability.
        """
        enc = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(self.device)
        with torch.no_grad():
            hidden = text_encoder(input_ids).last_hidden_state[0]  # [77, d_text]
        return hidden.T.float()  # [d_text, 77]

    def _apply_cfr(self):
        """
        Stage 1: Closed-Form Refinement of cross-attention K/V matrices.

        For each attn2 (cross-attention) layer's to_k and to_v:

            W_new = (λW + W @ D @ C^T) @ inv(λI + C @ C^T)

        For multiple concepts, C and D are the column-wise concatenation of
        all concept / target embeddings:

            C = [C_1 | C_2 | ... | C_N]   [d_text, 77*N]
            D = [D_1 | D_2 | ... | D_N]   [d_text, 77*N]

        The shared factor inv(λI + C @ C^T) is [d_text, d_text] regardless
        of N, so erasing 10 concepts costs the same as erasing 1.
        """
        logger.info(
            f"Loading model {self.model_id} for CFR "
            f"({len(self.erase_concepts)} concept(s))..."
        )

        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        ).to(self.device)
        text_encoder.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)

        # Encode every concept and its matching target → each [d_text, 77]
        # Concatenate along the token axis → C, D: [d_text, 77*N]
        C = torch.cat(
            [self._get_token_embeddings(c, tokenizer, text_encoder) for c in self.erase_concepts],
            dim=1,
        )
        D = torch.cat(
            [self._get_token_embeddings(t, tokenizer, text_encoder) for t in self.erase_targets],
            dim=1,
        )

        d_text = C.shape[0]  # CLIP hidden dim, e.g. 768 for SD 1.x
        lam = self.lambda_cfr

        # Pre-compute the shared right-hand factor: inv(λI + C @ C^T)
        # Shape: [d_text, d_text] — computed ONCE, reused for every K/V layer
        mat2 = lam * torch.eye(d_text, device=self.device) + C @ C.T  # [d_text, d_text]
        inv_mat2 = torch.inverse(mat2)  # [d_text, d_text]

        # Pre-compute DC_T = D @ C^T — also [d_text, d_text], reused for every layer
        DC_T = D @ C.T  # [d_text, d_text]

        updated = 0
        for name, module in self.unet.named_modules():
            # Only modify cross-attention (attn2) K/V layers, not self-attention
            if "attn2" not in name:
                continue

            for proj_name in ("to_k", "to_v"):
                proj = getattr(module, proj_name, None)
                if proj is None or not isinstance(proj, torch.nn.Linear):
                    continue

                # W: [d_out, d_text] — cast to float32 for numerical stability
                W = proj.weight.data.float()

                # W_new = (λW + W @ D @ C^T) @ inv(λI + C @ C^T)
                #       = (λW + W @ DC_T) @ inv_mat2
                mat1 = lam * W + W @ DC_T  # [d_out, d_text]
                proj.weight.data = (mat1 @ inv_mat2).to(proj.weight.dtype)
                updated += 1

        logger.info(f"CFR complete: updated {updated} cross-attention K/V matrices.")

        # Save modified weights if requested
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else ".", exist_ok=True)
            torch.save(self.unet.state_dict(), self.save_path)
            logger.info(f"Saved modified UNet weights to {self.save_path}")

        # Free components no longer needed
        del tokenizer, text_encoder

        self.unet.eval()

    def generate(
        self,
        prompts: List[str],
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images with the concept-erased model.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            **kwargs: Additional arguments forwarded to the diffusers pipeline.

        Returns:
            List of PIL Images.
        """
        logger.info(f"Generating {len(prompts)} images ('{self.erase_concept}' erased via CFR)")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        images = []
        for prompt in prompts:
            result = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            ).images[0]
            images.append(result)

        return images
