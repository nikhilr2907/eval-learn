import os
import json
from typing import List, Any, Dict, Optional
from ..logging_utils import get_logger

logger = get_logger(__name__)


class ArtifactWriter:
    """
    Handles saving of benchmark artifacts (images and reports).

    Folder layout::

        <base_dir>/
          <technique>_<metric>_<run_id>/
            images/
              <category>/        ← only if metadata has categories
                0.png, 1.png
              0.png, 1.png       ← flat if no categories
            <run_id>_report.json
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir

    def save_run(
        self,
        run_id: str,
        technique_name: str,
        metric_name: str,
        images: List[Any],
        report: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save images and report for a benchmark run.

        Args:
            run_id: Short hash identifying this run.
            technique_name: Name of the technique (e.g. ``"sld"``).
            metric_name: Name of the metric (e.g. ``"asr"``).
            images: List of generated PIL images.
            report: Result dictionary to persist as JSON.
            metadata: Dataset metadata. If it contains a ``categories``
                key (list parallel to *images*), images are saved into
                per-category subdirectories.

        Returns:
            Path to the saved report JSON.
        """
        metadata = metadata or {}
        categories = metadata.get("categories")

        # Build folder: <base_dir>/<technique>_<metric>_<run_id>/
        folder_name = f"{technique_name}_{metric_name}_{run_id}"
        run_dir = os.path.join(self.base_dir, folder_name)
        images_dir = os.path.join(run_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Save images — category-aware if metadata provides categories
        logger.info(f"Saving {len(images)} images to {images_dir}...")
        image_paths = []

        if categories and len(categories) == len(images):
            # Category-aware: save into subdirectories
            category_counters: Dict[str, int] = {}
            for img, cat in zip(images, categories):
                cat_dir = os.path.join(images_dir, cat.lower())
                os.makedirs(cat_dir, exist_ok=True)
                idx = category_counters.get(cat.lower(), 0)
                category_counters[cat.lower()] = idx + 1
                path = os.path.join(cat_dir, f"{idx}.png")
                image_paths.append(self._save_image(img, path, idx))
        else:
            # Flat: save numbered images directly
            for i, img in enumerate(images):
                path = os.path.join(images_dir, f"{i}.png")
                image_paths.append(self._save_image(img, path, i))

        # Filter out None (failed saves)
        image_paths = [p for p in image_paths if p is not None]

        # Update report and save
        report["image_paths"] = image_paths
        report["run_id"] = run_id

        report_path = os.path.join(run_dir, f"{run_id}_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        return report_path

    @staticmethod
    def _save_image(img: Any, path: str, index: int) -> Optional[str]:
        """Save a single image, returning the path or None on failure."""
        try:
            if hasattr(img, "save"):
                img.save(path)
                return path
            else:
                logger.warning(f"Image {index} does not support save(), skipping.")
                return None
        except Exception as e:
            logger.error(f"Failed to save image {index}: {e}")
            return None
