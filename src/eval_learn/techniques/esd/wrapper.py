import os
from typing import List, Any, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import ESDConfig


logger = get_logger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from huggingface_hub import login
except ImportError as e:
    logger.error("Optional dependencies for ESD missing.")
    raise RuntimeError(
        "ESD technique requires 'diffusers', 'torch', 'transformers', and 'huggingface_hub'. "
        "Install with: pip install eval-learn[diffusers]"
    ) from e


@register_technique("esd")
class ESDWrapper:
    """
    Erased Stable Diffusion (ESD) - trains to erase custom concepts.

    When initialized, ESD will:
    1. Train the UNet to erase the specified concept
    2. Build a pipeline for generation

    Training methods:
    - xattn (ESD-x): Fine-tunes cross-attention layers
    - noxattn: Fine-tunes all except cross-attention
    - selfattn: Fine-tunes only self-attention
    - full (ESD-u): Fine-tunes entire UNet
    """

    def __init__(self, **kwargs):
        self.config = ESDConfig.from_dict(kwargs)

        # Setup Device
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

        logger.info(f"Initializing ESD on {self.device}")
        logger.info(f"Concept to erase: '{self.config.erase_concept}'")

        # Auth (Optional)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        # Train ESD to erase concept
        self._train_esd()
        
        # Build pipeline
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(    
                self.config.model_id,
                unet=self.unet,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load ESD model:{e}")

    def _train_esd(self):
        """Train UNet to erase the specified concept."""
        logger.info(f"Training ESD to erase: '{self.config.erase_concept}'")
        logger.info(f"Method: {self.config.train_method}, Steps: {self.config.train_steps}, LR: {self.config.learning_rate}")

        # Load model components
        logger.info(f"Loading model from {self.config.model_id}...")
        tokenizer = CLIPTokenizer.from_pretrained(self.config.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.config.model_id, subfolder="text_encoder").to(self.device)
        text_encoder.requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(self.config.model_id, subfolder="unet").to(self.device)
        scheduler = DDPMScheduler.from_pretrained(self.config.model_id, subfolder="scheduler")

        # half bits to decrease training time (may drop precision but faster on cuda)
        use_fp16 = self.config.use_fp16 and self.device == "cuda"
        if use_fp16:
            text_encoder.half()
            logger.info("Using fp16 for faster training")

        # Get text embeddings
        def get_emb(prompt):
            tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                return text_encoder(tokens)[0]

        emb_concept = get_emb(self.config.erase_concept)
        emb_empty = get_emb("")
        emb_erase_from = get_emb(self.config.erase_from or self.config.erase_concept)

        # Select which parameters to train based on method
        params = []
        for name, p in self.unet.named_parameters():
            method = self.config.train_method
            should_train = (
                method == "full" or
                (method == "xattn" and "attn2" in name and ("to_k" in name or "to_v" in name)) or
                (method == "selfattn" and "attn1" in name) or
                (method == "noxattn" and ("attn2" not in name or ("to_k" not in name and "to_v" not in name)))
            )
            p.requires_grad = should_train
            if should_train:
                params.append(p)

        logger.info(f"Training {sum(p.numel() for p in params):,} parameters")
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        # Use GradScaler to fix precision problem from fp16
        scaler = torch.amp.GradScaler('cuda') if use_fp16 else None

        # Train UNet to output opposite direction when target concept is prompted
        self.unet.train()
        for step in range(self.config.train_steps):
            optimizer.zero_grad()

            t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=self.device)
            latent = torch.randn(1, 4, 64, 64, device=self.device)
            noisy_latent = scheduler.add_noise(latent, torch.randn_like(latent), t)

            # Get current model directions for each prompt
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_fp16):
                pred_concept = self.unet(noisy_latent, t, encoder_hidden_states=emb_concept).sample
                pred_empty = self.unet(noisy_latent, t, encoder_hidden_states=emb_empty).sample
                pred_erase_from = self.unet(noisy_latent, t, encoder_hidden_states=emb_erase_from).sample

                # target = base - negative guidance *(concept - empty) 
                target = pred_erase_from - self.config.negative_guidance * (pred_concept - pred_empty)

            # Learn to match erased target
            with torch.amp.autocast('cuda', enabled=use_fp16):
                pred_esd = self.unet(noisy_latent, t, encoder_hidden_states=emb_erase_from).sample
                loss = F.mse_loss(pred_esd, target)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if (step + 1) % 50 == 0 or step == 0:
                logger.info(f"Step {step + 1}/{self.config.train_steps}, Loss: {loss.item():.6f}")

        self.unet.eval()
        logger.info("ESD training complete.")

        # Save weights if requested
        if self.config.save_path:
            os.makedirs(os.path.dirname(self.config.save_path) or ".", exist_ok=True)
            torch.save(self.unet.state_dict(), self.config.save_path)
            logger.info(f"Saved trained weights to {self.config.save_path}")

        del tokenizer, text_encoder, scheduler

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:
        """Generate images using the concept-erased model."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(f"Generating {len(prompts)} images ('{self.config.erase_concept}' erased)")

        images = []
        for prompt in prompts:
            try:
                output = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs
                ).images[0]
                images.append(output)
            except Exception as e:
                logger.error(f"Error generating image for prompt '{prompt}': {e}")
                raise

        return images
