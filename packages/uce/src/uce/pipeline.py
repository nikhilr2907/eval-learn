import os
from typing import List, Optional
from pathlib import Path
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file as load_safetensors


# Bundled pre-trained weights
_BUNDLED_WEIGHTS = {
    "nudity": "uce_nudity.safetensors",
    "violence": "uce_violence.safetensors",
    "dog": "uce_dog.safetensors",
}


class UCEPipeline:
    """
    Unified Concept Editing Pipeline for Stable Diffusion.

    Applies closed-form weight modifications to erase specific concepts
    from generated images.
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cuda",
        preset: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        """
        Initialize UCE Pipeline.

        Args:
            model_id: HuggingFace model identifier for Stable Diffusion.
            device: Device to run inference on ('cuda' or 'cpu').
            preset: Name of bundled preset ("nudity", "violence", "dog").
            weights_path: Path to custom UCE weights file (.safetensors).

        Either `preset` or `weights_path` must be provided.
        """
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve weights path
        if weights_path:
            self.weights_path = weights_path
        elif preset:
            if preset.lower() not in _BUNDLED_WEIGHTS:
                raise ValueError(
                    f"Unknown preset '{preset}'. "
                    f"Available: {list(_BUNDLED_WEIGHTS.keys())}"
                )
            # Get bundled weight path
            package_dir = Path(__file__).parent
            self.weights_path = package_dir / "weights" / _BUNDLED_WEIGHTS[preset.lower()]
        else:
            raise ValueError("Either 'preset' or 'weights_path' must be provided")

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"UCE weights not found at: {self.weights_path}")

        # Load base pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float32,
        ).to(self.device)

        # Apply UCE weights
        self._load_uce_weights(self.weights_path)

    def _load_uce_weights(self, weights_path: str):
        """Load and apply UCE weights to the UNet."""
        # Load UCE weights from safetensors
        uce_state_dict = load_safetensors(str(weights_path))

        # Get current UNet state dict
        unet_state_dict = self.pipe.unet.state_dict()

        # Update UNet with UCE-modified weights
        for key in uce_state_dict:
            if key in unet_state_dict:
                unet_state_dict[key] = uce_state_dict[key]

        # Load modified weights into UNet
        self.pipe.unet.load_state_dict(unet_state_dict)

    def generate(
        self,
        prompts: List[str],
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate images with concept erased.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            **kwargs: Additional arguments passed to pipeline.

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
                **kwargs
            )
            images.append(output.images[0])

        return images
