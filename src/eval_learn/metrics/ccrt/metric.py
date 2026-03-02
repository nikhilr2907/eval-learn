import gc
import os
from typing import List, Any, Dict, Optional

import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader

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

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


@register_metric("ccrt")
class CCRTMetric:
    """
    Cross-Category Retention Test (CCRT) Metric.

    load_dataset() runs the genetic concept search and LLM prompt generation,
    generates baseline images from the original model, then returns a DataLoader
    of the resulting prompt batches for generation by the erased technique.

    update() accumulates generated images alongside their prompts and seeds.
    compute() calls _calculate_ccrt() — currently a stub pending full implementation.
    """

    def __init__(self, **kwargs):
        self.config = CCRTConfig.from_dict(kwargs)

        for name, mod in [("transformers", CLIPModel), ("Pillow", PILImage)]:
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

        self._baseline_images: List[Any] = []
        self._reference_images: List[Any] = []
        self._pending_images: List[Any] = []
        self._pending_prompts: List[str] = []
        self._pending_seeds: List[Any] = []
        logger.info("CCRTMetric ready.")

    def load_dataset(self) -> DataLoader:
        """
        Run genetic search, generate LLM prompts, generate baseline images,
        then return a DataLoader of prompt batches for the erased technique.

        Steps:
          1. Load reference images from config.reference_imgs.
          2. Run genetic concept search — finds concepts where the erased model's
             noise predictions diverge most from the original.
          3. Generate prompts from survivors via GPT-3.5.
          4. Generate baseline images from the original (unmodified) model.
          5. Unload baseline pipeline and clear VRAM.
          6. Return a DataLoader of (prompt, seed) batches.
        """
        self._pending_images = []
        self._pending_prompts = []
        self._pending_seeds = []

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
        logger.info("Loaded %d reference images.", len(self._reference_images))

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

        gc.collect()
        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Baseline pipeline unloaded.")

        # --- 6. Wrap prompts + seeds in a DataLoader ---
        class _PromptSeedDataset(torch.utils.data.Dataset):
            def __init__(self, prompts, seeds):
                self._prompts = prompts
                self._seeds = seeds

            def __len__(self):
                return len(self._prompts)

            def __getitem__(self, idx):
                return self._prompts[idx], self._seeds[idx]

        concept_name = self.config.concept_name
        concept_desc = self.config.concept_desc
        batch_size = getattr(self.config, "batch_size", len(prompts))

        def collate_fn(batch):
            batch_prompts, batch_seeds = zip(*batch)
            return Dataset(
                prompts=list(batch_prompts),
                metadata={
                    "source": "ccrt_genetic",
                    "concept_name": concept_name,
                    "concept_desc": concept_desc,
                    "seeds": list(batch_seeds),
                },
            )

        return DataLoader(
            _PromptSeedDataset(prompts, seeds),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, images: List[Any], prompts: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Accumulate a batch of generated images and their metadata.

        Args:
            images:   Generated PIL Images from the erased technique.
            prompts:  Prompts used to generate the images.
            metadata: Must contain ``seeds`` parallel to images.
        """
        metadata = metadata or {}
        seeds = metadata.get("seeds", [None] * len(images))
        self._pending_images.extend(images)
        self._pending_prompts.extend(prompts)
        self._pending_seeds.extend(seeds)

    def _calculate_ccrt(self, images: List[Any], prompts: List[str], seeds: List[Any]) -> Dict[str, Any]:
        """
        TODO: implement the core CCRT scoring logic here.

        Args:
            images:  Generated images from the erased technique.
            prompts: Prompts used to generate the images.
            seeds:   Seed values parallel to images.

        Returns:
            Dict with at minimum a ``"CCRT_Score"`` key.
        """
        raise NotImplementedError("_calculate_ccrt not yet implemented.")

    def compute(self) -> MetricResult:
        """
        Compute CCRT score from all accumulated update() calls.
        """
        if not self._pending_images:
            return MetricResult(name="CCRT", value=0.0, details={"error": "No images accumulated"})

        logger.info("Computing CCRT for %d images...", len(self._pending_images))

        try:
            result = self._calculate_ccrt(
                self._pending_images, self._pending_prompts, self._pending_seeds
            )
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
