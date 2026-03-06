import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) model for feature extraction and concept unlearning.
    
    This architecture maps dense, uninterpretable activations from a specific layer 
    of the diffusion model into a high-dimensional, sparse latent space. In this space, 
    individual active features often correspond to interpretable concepts (e.g., objects or styles).
    """
    
    def __init__(self, d_model: int, d_sae: int):
        """
        Initializes the SAE architecture.
        
        Args:
            d_model (int): The hidden dimension of the original diffusion model layer 
                           (e.g., 320, 640, or 1280 depending on the UNet block).
            d_sae (int): The expanded, sparse dimension of the autoencoder 
                         (typically d_model multiplied by an expansion factor like 16 or 32).
        """
        super().__init__()
        
        # Encoder: Projects the original dense activations to the high-dimensional sparse space
        self.encoder = nn.Linear(d_model, d_sae)
        
        # Activation: Enforces sparsity (ensuring only a few features are active at once)
        self.relu = nn.ReLU()
        
        # Decoder: Projects the (potentially manipulated) sparse latents back to the original dimension
        self.decoder = nn.Linear(d_sae, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps original model activations to sparse feature latents.
        
        Args:
            x (torch.Tensor): Original activations from the hooked diffusion model layer.
            
        Returns:
            torch.Tensor: Sparse feature representations.
        """
        return self.relu(self.encoder(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the model activations from the sparse feature latents.
        
        Args:
            latents (torch.Tensor): The sparse features (which may have been ablated or steered).
            
        Returns:
            torch.Tensor: Reconstructed dense activations to be passed back to the diffusion model.
        """
        return self.decoder(latents)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: str = "cuda") -> "SparseAutoencoder":
        """
        Loads the SAE architecture and weights from a specified directory.
        Adapted to read 'cfg.json' and Hugging Face's 'sae.safetensors' format.
        
        Args:
            checkpoint_dir (str): Path to the directory containing the SAE files.
            device (str): Device to load the model onto ("cpu" or "cuda").
            
        Returns:
            SparseAutoencoder: The loaded and initialized SAE model set to evaluation mode.
        """
        config_path = os.path.join(checkpoint_dir, "cfg.json")
        weights_path = os.path.join(checkpoint_dir, "sae.safetensors")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing SAE config file at: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing SAE safetensors weights file at: {weights_path}")

        # 1. Load configuration to determine architecture dimensions
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        # Parse dimensions. SAE config formats vary, so we check multiple common keys.
        d_model = cfg.get("d_model", cfg.get("d_in"))
        d_sae = cfg.get("d_sae", cfg.get("dict_size", cfg.get("d_hidden")))
        
        if d_model is None or d_sae is None:
            raise ValueError(
                f"Could not parse model dimensions from cfg.json. "
                f"Available keys are: {list(cfg.keys())}"
            )

        # 2. Initialize the PyTorch module with the parsed dimensions
        model = cls(d_model=d_model, d_sae=d_sae)
        
        # 3. Load the pre-trained weights using the safetensors library
        state_dict = load_file(weights_path, device=device)
        model.load_state_dict(state_dict)
        
        # 4. Move to the target device and set to evaluation mode (crucial for inference)
        model.to(device)
        model.eval()
        
        return model