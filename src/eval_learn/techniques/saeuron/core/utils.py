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
    Computes the difference-based importance score for each SAE feature.
    It compares how often a feature activates for the target concept vs. all other concepts.
    
    Args:
        style_latents_dict (Dict[str, torch.Tensor]): Dictionary mapping concept names 
            to their cached SAE activations (shape: [batch, timesteps, num_features]).
        target_style (str): The specific concept/style to unlearn.
        timestep (int): The specific diffusion timestep to analyze.
        epsilon (float): Small value to prevent division by zero.
        
    Returns:
        torch.Tensor: A 1D tensor of importance scores for each feature.
    """
    if target_style not in style_latents_dict:
        raise ValueError(f"target_style '{target_style}' not found in the provided dictionary.")

    # Mean activation for the target style (shape: [num_features])
    latents_x = style_latents_dict[target_style][:, timestep, :].float()
    mean_x = latents_x.mean(dim=0)

    # All other styles
    other_styles = [s for s in style_latents_dict if s != target_style]
    if not other_styles:
        # If there's only one style, we can't compare. Return raw means.
        return mean_x  

    # Mean activation for the combined "others"
    latents_others = torch.cat(
        [style_latents_dict[s][:, timestep, :].float() for s in other_styles], dim=0
    )
    mean_others = latents_others.mean(dim=0)

    # Denominators: total activation across all features
    total_x = mean_x.sum() + epsilon
    total_others = mean_others.sum() + epsilon

    # Proportions of activation
    p_x = mean_x / total_x
    p_others = mean_others / total_others

    # Difference-based score (high score = highly specific to the target concept)
    scores = p_x - p_others

    return scores

def get_percentile_threshold(scores: torch.Tensor, percentile: float = 95.0) -> float:
    """
    Returns the threshold value for a given percentile.

    Args:
        scores (torch.Tensor): 1D tensor of unnormalized scores, shape [num_features].
        percentile (float): Percentile in [0, 100].

    Returns:
        float: The score value at the given percentile.
    """
    fraction = percentile / 100.0
    # Use PyTorch's built-in quantile function
    threshold = torch.quantile(scores, fraction)
    return threshold.item()

def get_target_latents(
    acts_path: str, 
    target_concept: str, 
    timestep: int = 10, 
    percentile: float = 99.99
) -> List[int]:
    """
    Helper function to load cached activations from disk and return the indices 
    of the features that should be ablated/steered.
    
    Args:
        acts_path (str): Path to the .pkl file containing cached activations.
        target_concept (str): The concept name (must match a key in the .pkl file).
        timestep (int): Which diffusion timestep to use for scoring.
        percentile (float): Only features scoring above this percentile are selected.
        
    Returns:
        List[int]: A list of feature indices to target for unlearning.
    """
    if not os.path.exists(acts_path):
        raise FileNotFoundError(f"Activations file not found at {acts_path}")
        
    with open(acts_path, "rb") as f:
        style_latents_dict = pickle.load(f)
        
    # 1. Compute importance scores
    scores = compute_feature_importance(style_latents_dict, target_concept, timestep)
    
    # 2. Find the threshold based on the requested percentile
    threshold = get_percentile_threshold(scores, percentile)
    
    # 3. Extract the indices of features that meet or exceed the threshold
    # torch.where returns a tuple, we take the first element [0] and convert to list
    target_indices = torch.where(scores >= threshold)[0].tolist()
    
    return target_indices