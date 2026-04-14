import os
import torch
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image
from tqdm import tqdm

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    raise ImportError("CAWrapper requires the 'diffusers' package.")

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import CAConfig

logger = get_logger(__name__)


@register_technique("ca")
class CAWrapper:
    """
    Wrapper for Concept Ablation (CA) technique.
    
    Implements dynamic fine-tuning of cross-attention layers upon initialization
    if trained weights are not already present at the specified save_path.
    """

    def __init__(self, **kwargs):
        """Initialize the pipeline, apply ablation, and handle weight caching."""
        self.config = CAConfig.from_dict(kwargs)

        logger.info(f"Initializing CA: {self.config.model_id}")

        # 1. Load Base Pipeline
        dtype = torch.float16 if self.config.use_fp16 else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id, torch_dtype=dtype
        ).to(self.config.device)
        self.pipeline.set_progress_bar_config(disable=True)

        # 2. Check for cached weights to save time
        if self.config.save_path and os.path.exists(self.config.save_path):
            logger.info(f"Loading cached ablated weights from {self.config.save_path}")
            self.pipeline.unet.load_state_dict(torch.load(self.config.save_path))
        else:
            # 3. Perform fine-tuning if no cached weights are found
            logger.info(
                f"Starting Concept Ablation: forcing target '{self.config.target_concept}' "
                f"to align with anchor '{self.config.anchor_concept}'"
            )
            self._train_ablation()

            # 4. Cache weights for future runs
            if self.config.save_path:
                os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)
                torch.save(self.pipeline.unet.state_dict(), self.config.save_path)
                logger.info(f"Saved newly ablated weights to {self.config.save_path}")

    def _train_ablation(self):
        """Core internal method to fine-tune the UNet cross-attention layers."""
        unet = self.pipeline.unet
        text_encoder = self.pipeline.text_encoder
        tokenizer = self.pipeline.tokenizer
        scheduler = self.pipeline.scheduler

        # Step A: Freeze all UNet parameters initially
        unet.requires_grad_(False)

        # Step B: Unfreeze ONLY Cross-Attention (attn2) 'to_k' and 'to_v' layers
        unfrozen_params_count = 0
        for name, param in unet.named_parameters():
            if "attn2" in name and ("to_k" in name or "to_v" in name):
                param.requires_grad = True
                unfrozen_params_count += 1
                
        logger.info(f"Unfrozen {unfrozen_params_count} cross-attention projection layers.")

        # Step C: Setup Optimizer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=self.config.learning_rate,
        )

        # Helper to get text embeddings
        def get_embeds(prompt_text):
            text_inputs = tokenizer(
                prompt_text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            return text_encoder(text_inputs.input_ids)[0]

        # Step D: Pre-compute target and anchor embeddings
        with torch.no_grad():
            target_embeds = get_embeds(self.config.target_concept)
            anchor_embeds = get_embeds(self.config.anchor_concept)

        unet.train()
        logger.info(f"Training for {self.config.train_steps} steps...")

        # Step E: Training Loop
        for step in tqdm(range(self.config.train_steps), desc="Ablating Concept"):
            optimizer.zero_grad()

            # Sample random noise and timesteps
            latents = torch.randn(
                (1, unet.config.in_channels, 64, 64), 
                device=self.config.device, 
                dtype=unet.dtype
            )
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (1,), 
                device=self.config.device
            ).long()

            # Forward pass 1: Anchor concept (Ground Truth, no gradient computation needed)
            with torch.no_grad():
                noise_pred_anchor = unet(
                    latents, timesteps, encoder_hidden_states=anchor_embeds
                ).sample

            # Forward pass 2: Target concept (The one we are modifying)
            noise_pred_target = unet(
                latents, timesteps, encoder_hidden_states=target_embeds
            ).sample

            # Compute MSE Loss to align target generation trajectory to anchor trajectory
            loss = F.mse_loss(noise_pred_target, noise_pred_anchor)
            loss.backward()
            optimizer.step()

        # Switch back to eval mode to prevent accidental modifications during generation
        unet.eval()
        logger.info("Concept Ablation fine-tuning completed.")

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using the ablated model.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            List of PIL Images.
        """
        num_inference_steps = kwargs.pop(
            "num_inference_steps", self.config.num_inference_steps
        )
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)

        logger.info(
            f"Generating {len(prompts)} images ('{self.config.target_concept}' ablated)"
        )

        # Standard diffusers generation call
        return self.pipeline(
            prompt=prompts,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images