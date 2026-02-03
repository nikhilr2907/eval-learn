import os
import json
import time
from typing import List, Any, Dict, Optional
from ..logging_utils import get_logger

logger = get_logger(__name__)

class ArtifactWriter:
    """
    Handles saving of benchmark artifacts (images and reports).
    """
    def __init__(self, base_dir: str = "results/benchmarks"):
        self.base_dir = base_dir
        
    def save_run(self, 
                 run_name: str, 
                 images: List[Any], 
                 report: Dict[str, Any],
                 timestamp: Optional[float] = None):
        """
        Saves images and report for a specific run.
        
        Args:
            run_name: Name of the benchmark run (e.g., "ASR_Benchmark").
            images: List of generated images.
            report: The result dictionary to save as JSON.
            timestamp: Optional timestamp for unique naming (default: current time).
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Create run-specific directory
        run_dir = os.path.join(self.base_dir, run_name)
        images_dir = os.path.join(run_dir, "images", f"run_{int(timestamp)}")
        os.makedirs(images_dir, exist_ok=True)
        
        # 1. Save Images
        logger.info(f"Saving {len(images)} images to {images_dir}...")
        image_paths = []
        for i, img in enumerate(images):
            try:
                # Assuming PIL Image or similar
                path = os.path.join(images_dir, f"{i}.png")
                if hasattr(img, 'save'):
                    img.save(path)
                    image_paths.append(path)
                else:
                    logger.warning(f"Image {i} does not support save(), skipping.")
            except Exception as e:
                logger.error(f"Failed to save image {i}: {e}")

        # 2. Update Report with Image Paths
        report['image_paths'] = image_paths
        report['timestamp'] = timestamp
        
        # 3. Save Report JSON
        report_path = os.path.join(run_dir, f"report_{int(timestamp)}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            
        return report_path
