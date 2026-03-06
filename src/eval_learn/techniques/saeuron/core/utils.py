import torch
import pickle
import os
from typing import Dict, List, Union

def compute_feature_importance(
    style_latents_dict: Dict[str, torch.Tensor], 
    target_style: str, 
    timestep: int, 
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Computes the importance score for each SAE feature.
    Supports both multi-concept (difference-based) and single-concept (proportion-based) modes.
    """
    if target_style not in style_latents_dict:
        raise ValueError(f"target_style '{target_style}' not found. Available keys: {list(style_latents_dict.keys())}")

    # 1. compute the mean activation for the target style (shape: [num_features])
    latents_x = style_latents_dict[target_style][:, timestep, :].float()
    mean_x = latents_x.mean(dim=0)

    other_styles = [s for s in style_latents_dict if s != target_style]
    
    # ==========================================
    # Single Concept Mode
    # ==========================================
    if not other_styles:
        total_x = mean_x.sum() + epsilon
        p_x = mean_x / total_x
        return p_x  

    # ==========================================
    #Multi-Concept Mode
    # ==========================================
    latents_others = torch.cat(
        [style_latents_dict[s][:, timestep, :].float() for s in other_styles], dim=0
    )
    mean_others = latents_others.mean(dim=0)

    total_x = mean_x.sum() + epsilon
    total_others = mean_others.sum() + epsilon

    p_x = mean_x / total_x
    p_others = mean_others / total_others

    scores = p_x - p_others

    return scores

def get_percentile_threshold(scores: torch.Tensor, percentile: float = 95.0) -> float:
    """Returns the threshold value for a given percentile."""
    fraction = percentile / 100.0
    threshold = torch.quantile(scores, fraction)
    return threshold.item()

def get_target_latents(
    acts_path: str, 
    target_concept: str, 
    timestep: int = 10, 
    percentile: float = 99.99
) -> List[int]:
    """Helper function to extract target latent indices from a cached .pkl file."""
    if not os.path.exists(acts_path):
        raise FileNotFoundError(f"Activations file not found at {acts_path}")
        
    with open(acts_path, "rb") as f:
        style_latents_dict = pickle.load(f)
        
    scores = compute_feature_importance(style_latents_dict, target_concept, timestep)
    threshold = get_percentile_threshold(scores, percentile)
    
    target_indices = torch.where(scores >= threshold)[0].tolist()
    
    return target_indices