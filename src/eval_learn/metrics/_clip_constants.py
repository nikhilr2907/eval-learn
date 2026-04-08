"""Shared CLIP model constants for metric configs."""

# Validated OpenAI CLIP model IDs supported across all metrics.
# These are the only architectures confirmed to work with CLIPModel/CLIPProcessor
# from transformers in the context of concept detection and scoring.
SUPPORTED_CLIP_MODELS = frozenset({
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
})

# Fixed model for SD 1.x text encoder alignment (used by mma_diffusion).
# All Stable Diffusion 1.x models use this exact CLIP text encoder.
SD1X_CLIP_MODEL = "openai/clip-vit-large-patch14"


def validate_clip_model(model_id: str, field_name: str = "clip_model_name") -> None:
    """Raise ValueError if model_id is not in the supported CLIP allowlist."""
    if model_id not in SUPPORTED_CLIP_MODELS:
        raise ValueError(
            f"Unsupported CLIP model '{model_id}' for '{field_name}'. "
            f"Supported models: {sorted(SUPPORTED_CLIP_MODELS)}"
        )