import torch
import os
from typing import List, Optional
from PIL import Image

from .sdlens.hooked_sd_pipeline import HookedStableDiffusionPipeline
from .training.k_sparse_autoencoder import SparseAutoencoder
from .hooks import add_feature_on_text_prompt, minus_feature_on_text_prompt


class ConceptSteeringPipeline:
    """
    SAE-based Concept Steering Pipeline for Stable Diffusion.

    Integrates Sparse Autoencoder (SAE) feature modulation into the
    text encoder of Stable Diffusion to steer generated images away
    from or towards specific concepts.
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cuda",
        sae_path: Optional[str] = None,
        concept: str = "nudity",
        multiplier: float = 1.0,
    ):
        """
        Initialize the Concept Steering Pipeline.

        Args:
            model_id: HuggingFace model identifier for Stable Diffusion.
            device: Device to run inference on ('cuda' or 'cpu').
            sae_path: Path to SAE checkpoint directory.
            concept: Concept to steer (used as prompt for feature extraction).
            multiplier: Strength of steering (positive = add, negative = subtract).
        """
        self.model_id = model_id
        self.device = device
        self.concept = concept
        self.multiplier = multiplier

        # Initialize Hooked SD Pipeline
        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32
        ).to(device)

        # Layer 9 is the primary bottleneck for concept steering
        self.target_block = 'text_encoder.text_model.encoder.layers.9'

        # Load SAE
        if sae_path and os.path.exists(sae_path):
            self.sae = SparseAutoencoder.load_from_disk(sae_path).to(device)
        else:
            raise FileNotFoundError(
                f"SAE checkpoint not found at: {sae_path}\n"
                "Please provide a valid sae_path to a trained SAE checkpoint."
            )

    def _get_steering_feature(self, concept_prompt: str, seed: int) -> torch.Tensor:
        """
        Extract semantic direction vector from SAE for a given concept.

        Args:
            concept_prompt: Text prompt representing the concept to steer.
            seed: Random seed for reproducibility.

        Returns:
            Steering feature tensor to be added/subtracted.
        """
        # Cache activation for the concept prompt
        _, cache = self.pipe.run_with_cache(
            concept_prompt,
            positions_to_cache=[self.target_block],
            save_output=True,
            num_inference_steps=1,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        )

        # Extract activations and encode via SAE
        activations = cache['output'][self.target_block][:, 0, :].squeeze(0)
        with torch.no_grad():
            # Get k-sparse latent representation
            activated = self.sae.encode_without_topk(activations)

        # Scale by multiplier and project back using SAE decoder
        steering_feature = (activated * self.multiplier) @ self.sae.decoder.weight.T
        return steering_feature

    def _create_modulate_hook(self, steering_feature: torch.Tensor):
        """
        Create toggle hook for classifier-free guidance modulation.

        CFG calls text encoder twice per denoising step:
        - First call (conditional): Add steering feature
        - Second call (unconditional): Subtract steering feature

        Args:
            steering_feature: The steering vector to apply.

        Returns:
            Hook function that modulates based on call count.
        """
        call_counter = {"count": 0}

        def hook_function(*args, **kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                # Conditional pass: Add feature
                return add_feature_on_text_prompt(self.sae, steering_feature, *args, **kwargs)
            else:
                # Unconditional pass: Subtract feature
                return minus_feature_on_text_prompt(self.sae, steering_feature, *args, **kwargs)

        return hook_function

    def generate(
        self,
        prompts: List[str],
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate images with concept steering applied.

        Args:
            prompts: List of text prompts to generate images from.
            seed: Base random seed (default: 42).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            List of PIL Images.
        """
        images = []
        base_seed = seed if seed is not None else 42

        # Pre-calculate steering feature for the target concept
        steering_feature = self._get_steering_feature(self.concept, base_seed)

        for i, prompt in enumerate(prompts):
            # Run inference with hook injection
            output = self.pipe.run_with_hooks(
                prompt,
                position_hook_dict={
                    self.target_block: self._create_modulate_hook(steering_feature)
                },
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(base_seed + i)
            )
            images.append(output.images[0])

        return images
