import os
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

logger = logging.getLogger(__name__)

# Available training methods for ESD
TRAIN_METHODS = ["xattn", "noxattn", "selfattn", "full"]


class ESDPipeline:
    """
    Erased Stable Diffusion (ESD) pipeline.

    Trains the UNet to erase a specified concept, then generates images
    with the concept-erased model.

    Training methods:
    - xattn (ESD-x): Fine-tunes cross-attention K/V layers
    - noxattn: Fine-tunes all layers except cross-attention
    - selfattn: Fine-tunes only self-attention layers
    - full (ESD-u): Fine-tunes entire UNet

    Args:
        model_id: HuggingFace model ID for Stable Diffusion.
        device: Device to run on ('cuda', 'cpu', 'mps', or None for auto).
        erase_concept: The concept to erase (e.g. 'nudity').
        erase_from: Target concept to erase from (defaults to erase_concept).
        train_method: Which UNet layers to fine-tune.
        negative_guidance: Strength of the erasure signal.
        train_steps: Number of training steps.
        learning_rate: Learning rate for training.
        use_fp16: Whether to use fp16 for faster training on CUDA.
        save_path: Optional path to save trained UNet weights.
        load_path: Optional path to load pre-trained UNet weights (skips training).
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: Optional[str] = None,
        erase_concept: str = "nudity",
        erase_from: Optional[str] = None,
        train_method: str = "xattn",
        negative_guidance: float = 2.0,
        train_steps: int = 200,
        learning_rate: float = 5e-5,
        use_fp16: bool = True,
        save_path: Optional[str] = None,
        load_path: Optional[str] = None,
    ):
        if train_method not in TRAIN_METHODS:
            raise ValueError(
                f"Unknown train_method '{train_method}'. "
                f"Available: {TRAIN_METHODS}"
            )

        self.model_id = model_id
        self.erase_concept = erase_concept
        self.erase_from = erase_from or erase_concept
        self.train_method = train_method
        self.negative_guidance = negative_guidance
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.use_fp16 = use_fp16
        self.save_path = save_path

        # Resolve device
        if device:
            self.device = device
        else:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        # Train or load UNet
        if load_path:
            self._load_weights(load_path)
        else:
            self._train()

        # Build generation pipeline with the modified UNet
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=self.unet,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

    def _load_weights(self, load_path: str):
        """Load pre-trained UNet weights, skipping training."""
        logger.info(f"Loading pre-trained ESD weights from {load_path}")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)
        self.unet.load_state_dict(torch.load(load_path, map_location=self.device))
        self.unet.eval()

    def _train(self):
        """Train UNet to erase the specified concept."""
        logger.info(f"Training ESD to erase: '{self.erase_concept}'")
        logger.info(
            f"Method: {self.train_method}, Steps: {self.train_steps}, "
            f"LR: {self.learning_rate}"
        )

        # Load model components
        logger.info(f"Loading model from {self.model_id}...")
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        ).to(self.device)
        text_encoder.requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)
        scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        # fp16 for faster training on CUDA
        use_fp16 = self.use_fp16 and self.device == "cuda"
        if use_fp16:
            text_encoder.half()
            logger.info("Using fp16 for faster training")

        # Pre-compute text embeddings
        def get_emb(prompt):
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            with torch.no_grad():
                return text_encoder(tokens)[0]

        emb_concept = get_emb(self.erase_concept)
        emb_empty = get_emb("")
        emb_erase_from = get_emb(self.erase_from)

        # Select which parameters to train based on method
        params = []
        for name, p in self.unet.named_parameters():
            should_train = (
                self.train_method == "full"
                or (self.train_method == "xattn" and "attn2" in name
                    and ("to_k" in name or "to_v" in name))
                or (self.train_method == "selfattn" and "attn1" in name)
                or (self.train_method == "noxattn"
                    and ("attn2" not in name
                         or ("to_k" not in name and "to_v" not in name)))
            )
            p.requires_grad = should_train
            if should_train:
                params.append(p)

        logger.info(f"Training {sum(p.numel() for p in params):,} parameters")
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

        # Training loop: push UNet prediction away from the concept direction
        self.unet.train()
        for step in range(self.train_steps):
            optimizer.zero_grad()

            t = torch.randint(
                0, scheduler.config.num_train_timesteps, (1,), device=self.device
            )
            latent = torch.randn(1, 4, 64, 64, device=self.device)
            noisy_latent = scheduler.add_noise(latent, torch.randn_like(latent), t)

            # Compute erased target direction (no grad)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
                pred_concept = self.unet(
                    noisy_latent, t, encoder_hidden_states=emb_concept
                ).sample
                pred_empty = self.unet(
                    noisy_latent, t, encoder_hidden_states=emb_empty
                ).sample
                pred_erase_from = self.unet(
                    noisy_latent, t, encoder_hidden_states=emb_erase_from
                ).sample

                target = pred_erase_from - self.negative_guidance * (
                    pred_concept - pred_empty
                )

            # Train UNet to match erased target
            with torch.amp.autocast("cuda", enabled=use_fp16):
                pred_esd = self.unet(
                    noisy_latent, t, encoder_hidden_states=emb_erase_from
                ).sample
                loss = F.mse_loss(pred_esd, target)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if (step + 1) % 50 == 0 or step == 0:
                logger.info(
                    f"Step {step + 1}/{self.train_steps}, Loss: {loss.item():.6f}"
                )

        self.unet.eval()
        logger.info("ESD training complete.")

        # Save weights if requested
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            torch.save(self.unet.state_dict(), self.save_path)
            logger.info(f"Saved trained weights to {self.save_path}")

        # Free training-only resources
        del tokenizer, text_encoder, scheduler

    def generate(
        self,
        prompts: List[str],
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images using the concept-erased model.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            **kwargs: Additional arguments forwarded to the diffusers pipeline.

        Returns:
            List of PIL Images.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        images = []
        for prompt in prompts:
            output = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            ).images[0]
            images.append(output)

        return images
