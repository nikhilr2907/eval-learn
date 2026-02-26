from typing import List, Any, Dict, Optional
from ...types import Dataset, MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import CLIPScoreConfig

logger = get_logger(__name__)

# Optional imports
try:
    import torch
except ImportError:
    torch = None

try:
    from torchmetrics.multimodal.clip_score import CLIPScore
except ImportError:
    CLIPScore = None

try:
    from torchvision import transforms
except ImportError:
    transforms = None

try:
    from PIL import Image
except ImportError:
    Image = None

# TODO: Add batch processing for efficiency
@register_metric("clip_score")
class CLIPScoreMetric:
    """
    CLIP Score Metric.

    Measures text-to-image alignment by computing the cosine similarity
    between CLIP embeddings of each generated image and its prompt.
    Higher scores indicate better prompt-image faithfulness.

    Uses ``torchmetrics.multimodal.CLIPScore`` under the hood
    (ported from the legacy ``ClipScore`` / ``CLIP_score_calculation`` code).
    """

    def __init__(self, **kwargs):
        self.config = CLIPScoreConfig.from_dict(kwargs)

        # Validate required dependencies
        for name, mod in [("torch", torch), ("torchmetrics", CLIPScore),
                          ("torchvision", transforms), ("Pillow", Image)]:
            if mod is None:
                raise RuntimeError(
                    f"CLIP Score metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device_str

        logger.info("Loading CLIPScore model '%s' on %s...",
                     self.config.clip_model_name, self.device)
        self._clip_score_fn = CLIPScore(
            model_name_or_path=self.config.clip_model_name
        ).to(self.device)
        logger.info("CLIPScoreMetric ready.")

    def load_dataset(self) -> Dataset:
        """Stream TIFA prompts from HuggingFace and collect into a Dataset."""
        from ...datasets.tifa_json import load_tifa_json

        loader = load_tifa_json(limit=self.config.limit)

        all_prompts = []
        for dataset_batch in loader:
            all_prompts.extend(dataset_batch.prompts)

        logger.info("Loaded %d prompts from TIFA.", len(all_prompts))
        return Dataset(
            prompts=all_prompts,
            metadata={"source": "tifa_hf", "total_loaded": len(all_prompts)},
        )

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(img) -> Optional["Image.Image"]: # type: ignore
        """Return a PIL Image from a path string or PIL Image."""
        if Image and isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            try:
                return Image.open(img).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not load image %s: %s", img, e)
                return None
        return None

    def _pil_to_tensor(self, pil_img: "Image.Image") -> "torch.Tensor": # type: ignore
        """Convert a PIL image to a uint8 tensor on the configured device.

        This matches the legacy code:
            image_tensor = transforms.ToTensor()(image)
            image_tensor = (image_tensor * 255).to(torch.uint8).to(device)
        """
        tensor = transforms.ToTensor()(pil_img)
        tensor = (tensor * 255).to(torch.uint8).to(self.device)
        return tensor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        images: List[Any],
        prompts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute average CLIP Score across image-prompt pairs.

        Args:
            images: List of generated images (file paths or PIL Images).
            prompts: List of text prompts, parallel to *images*.
            metadata: Optional (unused by this metric).

        Returns:
            MetricResult with the average CLIP score (higher is better).
        """
        if not images:
            return MetricResult(name="CLIPScore", value=0.0,
                                details={"error": "No images provided"})

        if len(images) != len(prompts):
            return MetricResult(
                name="CLIPScore", value=0.0,
                details={"error": f"images length ({len(images)}) must match prompts length ({len(prompts)})"},
            )

        logger.info("Computing CLIP Score for %d image-prompt pairs...", len(images))

        scores: List[Optional[float]] = []
        total_score = 0.0
        evaluated = 0

        for idx, (img, prompt) in enumerate(zip(images, prompts)):
            pil_img = self._to_pil(img)
            if pil_img is None:
                logger.warning("Skipping image %d: could not load.", idx)
                scores.append(None)
                continue

            try:
                image_tensor = self._pil_to_tensor(pil_img)
                score = self._clip_score_fn(image_tensor, prompt)
                score_val = score.item()
                scores.append(score_val)
                total_score += score_val
                evaluated += 1
            except Exception as e:
                logger.error("Error scoring image %d: %s", idx, e)
                scores.append(None)

        avg_score = total_score / evaluated if evaluated > 0 else 0.0
        logger.info("CLIP Score: %.4f (evaluated %d/%d)", avg_score, evaluated, len(images))

        return MetricResult(
            name="CLIPScore",
            value=avg_score,
            details={
                "per_image_scores": scores,
                "evaluated": evaluated,
                "total_images": len(images),
                "config": self.config.to_dict(),
            },
        )
