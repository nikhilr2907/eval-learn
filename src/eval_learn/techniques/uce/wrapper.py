import os
from typing import List, Any, Optional
from .config import UCEConfig
from pathlib import Path
from ...registry import register_technique
from ...logging_utils import get_logger
logger = get_logger(__name__)

# weights created by UCE are loaded using safetensors
try:
    import torch
    from diffusers import DiffusionPipeline
    from safetensors.torch import load_file as load_safetensors
    from huggingface_hub import login
except ImportError as e:
    logger.error("Optional dependencies for UCE missing.")
    raise RuntimeError(
        "UCE technique requires 'diffusers', 'torch', 'safetensors', and 'huggingface_hub'. "
        "Install with: pip install eval-learn[diffusers]"
    ) from e

@register_technique("uce")
class UCEWrapper:
    """
    Wrapper for Unified Concept Editting (UCE) pipeline.
    UCE is a closed-form and training free unlearning technique by modifying the weights of the UNet
    Adapted from SLD wrapper but made changes for UCE
    """
    def __init__(self, **kwargs):
        # 1. Parse Config
        # We accept kwargs to be flexible with the registry instantiation
        # calls UCEConfig from config.py
        self.config = UCEConfig.from_dict(kwargs)
        
        # 2. Setup Device
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

        logger.info(f"Initialising UCE on {self.device} with model {self.config.model_id}")

        # 3. Auth (Optional)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")    
        
        # 4. Load Pipeline
        # Note: We disable the standard safety_checker because UCE enforces safety by using modified weights
        # a new model is loaded first, afterwhich UCE weights are applied
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.config.model_id,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        except Exception as e:
             raise RuntimeError(f"Failed to load UCE model, Error: {e}")
        
        logger.info("UCE Model successfully loaded")
        # weights have to be created beforehand
        # these editted weights are altered to unlearn specific content
        # afterwhich create weights path object and verifying its existence
        weights_path = Path(self.config.uce_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f'{weights_path} does not exist')
        # try to load UCE weights
        try:
            self._load_uce_weights(weights_path)
            logger.info(f'UCE Weights applied from {weights_path}')
        except Exception as e:
            raise RuntimeError(f" Failed to load UCE Weight, Error: {e}")
    
    # Load UCE weights to modify the weights in the UNet, to alter image generation
    def _load_uce_weights(self, weights_path):
        # load weights from the safetensors file
        uce_state_dict = load_safetensors(str(weights_path))
        unet_state_dict = self.pipe.unet.state_dict()
        # update the unet state dictionary weights with that of the uce editted weights
        for key in uce_state_dict:
            if key in unet_state_dict:
                unet_state_dict[key] = uce_state_dict[key]
            else:
                logger.warning(f"Key {key} from UCE Weights not found in UNet, going to the next key...")

        # update the new unet weights to the pipeline
        self.pipe.unet.load_state_dict(unet_state_dict)
        logger.debug("UCE weights loaded into UNet.")

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:
        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)
        generator = None
        if seed is not None:
            generator = torch.Generator(device = self.device).manual_seed(seed)
        
        # Generate new images after applying the UCE weights to the model
        logger.info('Generating new images after applying UCE weights...')
        images = []
        for i, prompt in enumerate(prompts):
            try:
                # generate image for each prompt
                output = self.pipe(
                    prompt = prompt, num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    generator=generator, **kwargs,
                ).images[0]
                images.append(output)
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images