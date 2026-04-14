import torch
from typing import List, Optional, Any
from PIL import Image

# Import standard registry and logging from your eval_learn framework
from ...registry import register_technique
from ...logging_utils import get_logger

# Import the SAeUron-specific components from the installed package
from saeuron.core.model import SparseAutoencoder
from saeuron.core.utils import get_target_latents
from diffusers import StableDiffusionPipeline

# Import local integration config
from .config import SAeUronConfig

logger = get_logger(__name__)


@register_technique("saeuron")
class SAeUronTechnique:
    """
    Wrapper for the SAeUron (Sparse Autoencoder Unlearning) technique.

    This class intercepts the forward pass of a specified diffusion model layer,
    projects the activations into a sparse latent space, modifies specific features
    representing the target concept, and reconstructs the activations.
    """

    def __init__(self, **kwargs):
        """
        Initializes the technique, loads the models, and identifies target latents.
        """
        # 1. Parse configuration
        self.config = SAeUronConfig.from_dict(kwargs)
        self.hook_handles = []

        # 2. Load the base Diffusion Pipeline
        # (Assuming you have a standard pipeline loading utility in eval_learn,
        # otherwise we initialize it here).
        self.pipe = self._load_pipeline()

        # 3. Load the Sparse Autoencoder from the core/ directory logic
        logger.info(f"Loading SAE from {self.config.sae_path}")
        torch_dtype = torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
        self.sae = SparseAutoencoder.from_pretrained(
            self.config.sae_path, device=self.config.device, dtype=torch_dtype
        )

        # 4. Determine which latents to steer/ablate
        if self.config.target_latents:
            self.target_latents = self.config.target_latents
            logger.info(
                f"Using {len(self.target_latents)} explicitly provided target latents."
            )
        else:
            logger.info(
                f"Dynamically computing target latents for concept: '{self.config.erase_concept}'"
            )
            self.target_latents = get_target_latents(
                acts_path=self.config.acts_path,
                target_concept=self.config.erase_concept,
                percentile=self.config.percentile,
            )
            logger.info(f"Identified {len(self.target_latents)} latents to steer.")

        if not self.target_latents:
            raise ValueError(
                f"No target latents identified for concept '{self.config.erase_concept}'. "
                f"Check the acts_path and percentile ({self.config.percentile}) configuration."
            )

    def _sae_intervention_hook(
        self, module: torch.nn.Module, input: Any, output: Any
    ) -> Any:
        """
        The core hook applied to the target layer during inference.
        It isolates the conditional activations, ablates target concepts, and
        preserves reconstruction quality using residual compensation.
        """
        # Handle Diffusers output formats (often tuples)
        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output

        # Split into Unconditional and Conditional chunks for Classifier-Free Guidance
        # Standard SD input format is usually [Uncond_Batch, Cond_Batch]
        uncond, cond = act.chunk(2)

        # --- SAeUron Mathematical Intervention ---

        # Reshape spatial activations for SAE: (b, c, h, w) -> (b, h*w, c)
        b, c, h, w = cond.shape
        cond_flat = cond.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # 1. Project conditional activations to the sparse latent space
        orig_latents = self.sae.encode(cond_flat)

        # 2. Calculate the original SAE reconstruction to capture the residual.
        # This residual contains high-frequency details the SAE cannot perfectly reconstruct.
        # Adding it back prevents image quality degradation.
        orig_reconstruction = self.sae.decode(orig_latents)
        residual = cond_flat - orig_reconstruction

        # 3. Apply the intervention (Ablation / Steering)
        orig_latents[:, :, self.target_latents] *= self.config.multiplier

        # 4. Reconstruct the modified activations and add the residual back
        modified_cond_flat = self.sae.decode(orig_latents) + residual

        # Reshape back to spatial: (b, h*w, c) -> (b, c, h, w)
        modified_cond = modified_cond_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # ----------------------------------------

        # Re-concatenate the unmodified unconditional chunk and the modified conditional chunk
        modified_act = torch.cat([uncond, modified_cond], dim=0)

        return (modified_act,) if is_tuple else modified_act

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Runs the generation process with the SAeUron intervention applied.
        """
        logger.info(
            f"Applying SAeUron at {self.config.position} (Multiplier: {self.config.multiplier})"
        )

        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        target_layer = self._get_module_by_path(self.pipe, self.config.position)

        images = []
        try:
            handle = target_layer.register_forward_hook(self._sae_intervention_hook)
            self.hook_handles = [handle]

            for i, prompt in enumerate(prompts):
                generator = (
                    torch.Generator(device=self.config.device).manual_seed(seed + i)
                    if seed is not None
                    else None
                )
                output = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs,
                )
                images.append(output.images[0])

        finally:
            # CRITICAL: Always remove hooks to prevent interference with other benchmarks
            for h in self.hook_handles:
                h.remove()
            self.hook_handles = []

        return images

    def _get_module_by_path(self, model: Any, path: str) -> torch.nn.Module:
        """Helper to navigate the model tree using a dot-separated string."""
        for part in path.split("."):
            model = getattr(model, part)
        return model

    def _load_pipeline(self) -> Any:
        """
        Loads the base diffusion pipeline.
        Note: You can replace this with your framework's global model loader if eval_learn uses one.
        """

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=(
                torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
            ),
        )
        pipe = pipe.to(self.config.device)
        return pipe
