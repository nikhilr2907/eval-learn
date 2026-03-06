from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import UAIRAConfig

logger = get_logger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPModel = None
    CLIPProcessor = None

try:
    from PIL import Image
except ImportError:
    Image = None


@register_metric("ua_ira")
class UAIRAMetric:
    """
    Unlearning Accuracy (UA) and In-domain Retain Accuracy (IRA) Metric.

    UA: ratio of images NOT classified as target concept / total target images
    IRA: ratio of images correctly classified as retain concept / total retain images

    The runner loads the dataset via load_dataset(), then calls update() once per batch,
    and finally compute() to get the result.
    """

    def __init__(self, **kwargs):
        self.config = UAIRAConfig.from_dict(kwargs)

        # Ensure required dependencies are loaded
        for name, mod in [("torch", torch), ("transformers", CLIPModel),
                          ("transformers", CLIPProcessor), ("Pillow", Image)]:
            if mod is None:
                raise RuntimeError(
                    f"UA_IRA metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        logger.info("Initializing CLIP model '%s' on %s...", self.config.clip_model, self.device)
        self.model = CLIPModel.from_pretrained(self.config.clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model)
        self.model.eval()

        # State for accumulation across batches
        self._target_prompt_end_index = 0
        self._target_correct_count = 0
        self._target_total_count = 0
        self._retain_correct_count = 0
        self._retain_total_count = 0

        logger.info("UA_IRA Metric ready.")

    def load_dataset(self) -> DataLoader:
        """
        Load target and retain prompts from HuggingFace.

        Returns a DataLoader that yields Dataset batches with prompts and metadata.
        """
        from ...datasets.ua_ira_hf import load_ua_ira_hf

        # Reset counters
        self._target_correct_count = 0
        self._target_total_count = 0
        self._retain_correct_count = 0
        self._retain_total_count = 0

        return load_ua_ira_hf(
            target_concept=self.config.target_concept,
            retain_concept=self.config.retain_concept,
            target_limit=self.config.target_prompt_count,
            retain_limit=self.config.retain_prompt_count,
        )

    def update(self, images: List[Any], _prompts: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Run CLIP classification on each image and accumulate target/retain correct counts.

        Args:
            images:    Generated PIL Images or file paths.
            _prompts:  Unused.
            metadata:  Must contain ``target_prompt_end_index`` to split target vs retain images.
        """
        metadata = metadata or {}

        # Get the split index on first call
        if self._target_prompt_end_index == 0:
            self._target_prompt_end_index = metadata.get("target_prompt_end_index", 0)

        if not images:
            return

        # Caption prompts for CLIP classification
        caption_prompts = [
            f"Image of {self.config.target_concept}",
            f"Image of {self.config.retain_concept}",
        ]

        # Split images by target/retain based on metadata
        target_prompt_end_index = metadata.get("target_prompt_end_index", self._target_prompt_end_index)
        target_images = images[:target_prompt_end_index]
        retain_images = images[target_prompt_end_index:]

        # Evaluate target images
        for img in target_images:
            pil_img = self._to_pil(img)
            if pil_img is None:
                continue
            self._target_total_count += 1
            is_correct = self._classify_image(pil_img, caption_prompts)
            if is_correct:
                self._target_correct_count += 1

        # Evaluate retain images
        for img in retain_images:
            pil_img = self._to_pil(img)
            if pil_img is None:
                continue
            self._retain_total_count += 1
            is_correct = self._classify_image_retain(pil_img, caption_prompts)
            if is_correct:
                self._retain_correct_count += 1

    def _classify_image(self, pil_img: "Image.Image", caption_prompts: List[str]) -> bool:
        """
        Classify image via CLIP. Returns True if NOT classified as target (correct unlearning).
        """
        try:
            inputs = self.processor(
                text=caption_prompts,
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs)
                predicted_index = output.logits_per_image.softmax(dim=1).argmax(dim=1).item()
            # Correct if predicted as retain (index 1), not target (index 0)
            return predicted_index == 1
        except Exception as e:
            logger.warning("Error classifying image: %s", e)
            return False

    def _classify_image_retain(self, pil_img: "Image.Image", caption_prompts: List[str]) -> bool:
        """
        Classify image via CLIP. Returns True if classified as retain (correct retention).
        """
        try:
            inputs = self.processor(
                text=caption_prompts,
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs)
                predicted_index = output.logits_per_image.softmax(dim=1).argmax(dim=1).item()
            # Correct if predicted as retain (index 1)
            return predicted_index == 1
        except Exception as e:
            logger.warning("Error classifying image: %s", e)
            return False

    def compute(self) -> MetricResult:
        """
        Compute UA and IRA scores from accumulated counts.

        Returns:
            MetricResult with average score and individual UA/IRA scores in details.
        """
        # Calculate individual scores
        ua_score = (
            self._target_correct_count / self._target_total_count
            if self._target_total_count > 0
            else 0.0
        )
        ira_score = (
            self._retain_correct_count / self._retain_total_count
            if self._retain_total_count > 0
            else 0.0
        )

        avg_score = (ua_score + ira_score) / 2

        logger.info(
            "UA_IRA Score: %.4f  (UA: %.4f, IRA: %.4f)",
            avg_score, ua_score, ira_score,
        )

        return MetricResult(
            name="UA_IRA",
            value=avg_score,
            details={
                "ua_score": ua_score,
                "ira_score": ira_score,
                "target_correct": self._target_correct_count,
                "target_total": self._target_total_count,
                "retain_correct": self._retain_correct_count,
                "retain_total": self._retain_total_count,
                "config": self.config.to_dict(),
            },
        )

    @staticmethod
    def _to_pil(img) -> Optional["Image.Image"]:
        """Convert image to PIL. Returns None if conversion fails."""
        if Image and isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            try:
                return Image.open(img).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not load image %s: %s", img, e)
                return None
        return None