import torch
from typing import List, Optional, Any
from PIL import Image

# Import standard registry and logging from your eval_learn framework
from ...registry import register_technique
from ...logging_utils import get_logger

# Import the SAeUron-specific components from the installed package
from saeuron.core.model import SparseAutoencoder
from saeuron.core.utils import get_target_latents

# Import local integration config
from .config import SAeUronConfig

logger = get_logger(__name__)

@register_technique("saeuron")
class SAeUronWrapper:
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
        self.sae = SparseAutoencoder.from_pretrained(
            self.config.sae_path, device=self.config.device
        )
        
        # 4. Determine which latents to steer/ablate
        if self.config.target_latents:
            self.target_latents = self.config.target_latents
            logger.info(f"Using {len(self.target_latents)} explicitly provided target latents.")
        else:
            logger.info(f"Dynamically computing target latents for concept: '{self.config.erase_concept}'")
            self.target_latents = get_target_latents(
                acts_path=self.config.acts_path,
                target_concept=self.config.erase_concept,
                percentile=self.config.percentile
            )
            logger.info(f"Identified {len(self.target_latents)} latents to steer.")

    def _sae_intervention_hook(self, module: torch.nn.Module, input: Any, output: Any) -> Any:
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
        
        # 1. Project conditional activations to the sparse latent space
        orig_latents = self.sae.encode(cond)
        
        # 2. Calculate the original SAE reconstruction to capture the residual.
        # This residual contains high-frequency details the SAE cannot perfectly reconstruct.
        # Adding it back prevents image quality degradation.
        orig_reconstruction = self.sae.decode(orig_latents)
        residual = cond - orig_reconstruction 
        
        # 3. Apply the intervention (Ablation / Steering)
        modified_latents = orig_latents.clone()
        modified_latents[:, :, self.target_latents] *= self.config.multiplier
        
        # 4. Reconstruct the modified activations and add the residual back
        modified_cond = self.sae.decode(modified_latents) + residual
        
        # ----------------------------------------

        # Re-concatenate the unmodified unconditional chunk and the modified conditional chunk
        modified_act = torch.cat([uncond, modified_cond], dim=0)
        
        return (modified_act,) if is_tuple else modified_act

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Image.Image]:
        """
        Runs the generation process with the SAeUron intervention applied.
        """
        logger.info(f"Applying SAeUron at {self.config.position} (Multiplier: {self.config.multiplier})")
        
        # Dynamically locate the PyTorch module defined in the config
        target_layer = self._get_module_by_path(self.pipe, self.config.position)
        
        try:
            # Register the forward hook
            handle = target_layer.register_forward_hook(self._sae_intervention_hook)
            self.hook_handles.append(handle)
            
            # Execute standard generation through the pipeline
            generator = torch.Generator(device=self.config.device).manual_seed(seed) if seed else None
            images = self.pipe(prompts, generator=generator, **kwargs).images
            return images
            
        finally:
            # CRITICAL: Always remove hooks to prevent interference with other benchmarks
            for h in self.hook_handles:
                h.remove()
            self.hook_handles = []

    def _get_module_by_path(self, model: Any, path: str) -> torch.nn.Module:
        """Helper to navigate the model tree using a dot-separated string."""
        for part in path.split('.'):
            model = getattr(model, part)
        return model

    def _load_pipeline(self) -> Any:
        """
        Loads the base diffusion pipeline. 
        Note: You can replace this with your framework's global model loader if eval_learn uses one.
        """
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id, 
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
        )
        pipe = pipe.to(self.config.device)
        return pipe