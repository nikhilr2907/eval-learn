from typing import List, Any, Dict, Optional
from ...types import Dataset, MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import CCRTConfig

logger = get_logger(__name__)

try:
    from ccrt.genetic.search import run_genetic_search
    from ccrt.genetic.llm_prompts import generate_prompts
except ImportError as e:
    raise RuntimeError(
        "CCRT metric requires the 'ccrt' package. "
        "Install it before running this metric."
    ) from e

# Optional imports
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None

try:
    from PIL import Image
except ImportError:
    Image = None


@register_metric("ccrt")
class CCRTMetric:
    """
    Cross-Category Retention Test (CCRT) Metric.

    TODO: add description of what CCRT measures.

    The ``compute()`` method expects ``metadata`` to contain:
    - ``concepts``: list of concept strings, parallel to ``images``.
    - ``categories``: list of category strings, parallel to ``images``.
    """

    def __init__(self, **kwargs):
        self.config = CCRTConfig.from_dict(kwargs)

        for name, mod in [("torch", torch), ("transformers", CLIPModel), ("Pillow", Image)]:
            if mod is None:
                raise RuntimeError(
                    f"CCRT metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        logger.info("Initializing CLIP model '%s' on %s...", self.config.clip_model_name, self.device)
        self.model = CLIPModel.from_pretrained(self.config.clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.model.eval()
        logger.info("CCRTMetric ready.")

    def load_dataset(self) -> Dataset:
        """Run the genetic search and LLM prompt generation to build the eval dataset.

        Steps:
          1. Load reference images from config.reference_imgs.
          2. Run the genetic concept search against both models — survivors are
             concepts where the erased model's noise predictions diverge most
             from the original model (high MSE = concept still leaks).
          3. Generate prompts from survivors via GPT-3.5.
          4. Generate baseline images from the original (unmodified) model and
             store them as self._baseline_images for use in compute().
          5. Unload the baseline pipeline and clear VRAM before returning.

        Returns:
            Dataset with LLM-generated prompts and parallel seeds in metadata.
        """
        import gc
        import os
        from PIL import Image as PILImage

        # ccrt package — adjust import path once the package is available

        os.makedirs(self.config.output_dir, exist_ok=True)

        # --- 1. Reference images ---
        if not os.path.isdir(self.config.reference_imgs):
            raise FileNotFoundError(
                f"reference_imgs directory not found: {self.config.reference_imgs}"
            )
        ref_paths = sorted([
            os.path.join(self.config.reference_imgs, f)
            for f in os.listdir(self.config.reference_imgs)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if len(ref_paths) < 3:
            raise ValueError(
                f"At least 3 reference images required in {self.config.reference_imgs}, "
                f"found {len(ref_paths)}."
            )
        self._reference_images = [PILImage.open(p).convert("RGB") for p in ref_paths]
        logger.info("Loaded %d reference images from %s.", len(self._reference_images), self.config.reference_imgs)

        # --- 2. Genetic search ---
        logger.info(
            "Running genetic search: original=%s, erased=%s, concept=%s",
            self.config.original_model_id, self.config.erased_model_id, self.config.concept_name,
        )
        entities = run_genetic_search(
            original_model_id=self.config.original_model_id,
            erased_model_id=self.config.erased_model_id,
            concept_name=self.config.concept_name,
            vocab_dir=self.config.vocab_dir,
            output_dir=self.config.output_dir,
            max_iterations=self.config.genetic_iterations,
            top_k=self.config.genetic_top_k,
        )
        logger.info("Genetic search complete: %d survivors.", len(entities))

        # Free both diffusion models loaded by the genetic search before continuing
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 3. LLM prompt generation ---
        logger.info("Generating prompts via LLM...")
        prompts, seeds = generate_prompts(
            entities=entities,
            api_key=self.config.llm_api_key,
            output_dir=self.config.output_dir,
            limit=self.config.limit,
        )
        logger.info("Generated %d prompts.", len(prompts))

        # --- 4. Baseline image generation ---
        logger.info("Generating baseline images from original model %s...", self.config.original_model_id)
        from ...techniques.free_run.wrapper import FreeRunTechnique

        baseline_technique = FreeRunTechnique(model_id=self.config.original_model_id)
        self._baseline_images = baseline_technique.generate(prompts=prompts)
        logger.info("Generated %d baseline images.", len(self._baseline_images))

        # --- 5. Unload baseline pipeline ---
        del baseline_technique
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Baseline pipeline unloaded.")

        return Dataset(
            prompts=prompts,
            metadata={
                "source": "ccrt_genetic",
                "concept_name": self.config.concept_name,
                "concept_desc": self.config.concept_desc,
                "seeds": seeds,
                "total_loaded": len(prompts),
            },
        )

    # ------------------------------------------------------------------
    # CCRT calculation
    # ------------------------------------------------------------------

    def _calculate_ccrt(self, images: List[Any], concepts: List[str], categories: List[str]) -> Dict[str, Any]:
        """
        TODO: implement the core CCRT scoring logic here.

        Args:
            images: List of generated images (PIL Images or file paths).
            concepts: Concept strings parallel to images.
            categories: Category strings parallel to images.

        Returns:
            Dict with at minimum a ``"CCRT_Score"`` key.
        """
        raise NotImplementedError("_calculate_ccrt not yet implemented.")

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
        Compute CCRT score.

        Args:
            images: List of generated images (file paths or PIL Images).
            prompts: List of prompts (kept for interface compatibility).
            metadata: Must contain:
                - ``concepts``: list[str] parallel to *images*.
                - ``categories``: list[str] parallel to *images*.

        Returns:
            MetricResult with the CCRT score.
        """
        if not images:
            return MetricResult(name="CCRT", value=0.0, details={"error": "No images provided"})

        metadata = metadata or {}
        concepts = metadata.get("concepts")
        categories = metadata.get("categories")

        if not concepts or not categories:
            return MetricResult(
                name="CCRT",
                value=0.0,
                details={"error": "metadata must contain 'concepts' and 'categories' lists"},
            )

        if len(concepts) != len(images) or len(categories) != len(images):
            return MetricResult(
                name="CCRT",
                value=0.0,
                details={"error": "concepts/categories length must match images length"},
            )

        logger.info("Computing CCRT for %d images...", len(images))

        try:
            result = self._calculate_ccrt(images, concepts, categories)
            score = result["CCRT_Score"]

            logger.info("CCRT Score: %.4f", score)

            return MetricResult(
                name="CCRT",
                value=score,
                details={
                    **{k: v for k, v in result.items() if k != "CCRT_Score"},
                    "config": self.config.to_dict(),
                },
            )
        except Exception as e:
            logger.exception("CCRT computation failed.")
            return MetricResult(
                name="CCRT",
                value=0.0,
                details={"error": str(e)},
            )