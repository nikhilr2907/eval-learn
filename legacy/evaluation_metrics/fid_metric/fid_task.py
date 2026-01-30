from typing import List, Any
import os
import tempfile
import tensorflow as tf
import numpy as np
import torch
from scipy import linalg
from PIL import Image
import requests
from io import BytesIO
from pycocotools.coco import COCO
import zipfile
from core.base_benchmark import BenchmarkTask


class FIDMetric:
    """
    FID Metric Calculator.
    Measures the quality and diversity of generated images compared to real images.
    Lower FID scores indicate better image quality.
    """
    def __init__(self, batch_size: int = 32, device: str = None):
        self.batch_size = batch_size
        self.inception_model = None
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_inception(self):
        """Load InceptionV3 model for feature extraction."""
        if self.inception_model is None:
            if not tf.config.list_physical_devices('GPU'):
                print("Warning: No GPU found for TensorFlow. FID calculation will be slow.")
            self.inception_model = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet", pooling="avg", input_shape=(299, 299, 3),
            )
        return self.inception_model
    
    def _load_process_image(self, path):
        """Load and preprocess image for InceptionV3."""
        if isinstance(path, str):
            image = Image.open(path).convert("RGB").resize((299, 299), Image.Resampling.BICUBIC)
        elif isinstance(path, Image.Image):
            image = path.convert("RGB").resize((299, 299), Image.Resampling.BICUBIC)
        else:
            raise ValueError(f"Invalid input type: {type(path)}")
        
        image = np.array(image)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image
    
    def _get_activations(self, image_paths, model):
        """Extract InceptionV3 features from images."""
        n = len(image_paths)
        if n == 0: return np.array([])
        
        activation_lists = []
        for i in range(0, n, self.batch_size):
            batch_path = image_paths[i : i + self.batch_size]
            batch_images = [self._load_process_image(p) for p in batch_path]
            batch_images = np.stack(batch_images, axis=0)
            activations = model(batch_images, training=False).numpy()
            activation_lists.append(activations)
        
        activations = np.concatenate(activation_lists, axis=0)
        return activations
    
    def _calculate_fid_score(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Frechet Inception Distance between two distributions."""
        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
             if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
                return float("inf")
             covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return float(fid)
    
    def _prepare_image_paths(self, images, temp_dir, prefix):
        """Convert PIL Images to file paths, saving to temp directory if needed."""
        paths = []
        for i, img in enumerate(images):
            if isinstance(img, str):
                paths.append(img)
            elif isinstance(img, Image.Image):
                if temp_dir is None: temp_dir = tempfile.mkdtemp()
                os.makedirs(temp_dir, exist_ok=True)
                save_path = os.path.join(temp_dir, f"{prefix}_{i}.png")
                img.save(save_path)
                paths.append(save_path)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        return paths
    
    def calculate(self, generated_images, real_images):
        """Calculate FID score between generated images and real images."""
        if not generated_images: return float('inf')
        if real_images is None or len(real_images) == 0:
            print("No real images provided for FID calculation")
            return float('inf')
            
        with tempfile.TemporaryDirectory() as temp_dir:
            gen_paths = self._prepare_image_paths(generated_images, temp_dir, prefix="gen")
            real_paths = self._prepare_image_paths(real_images, temp_dir, prefix="real")
            
            print(f"Computing FID with {len(real_paths)} real vs {len(gen_paths)} generated images...")
            model = self._load_inception()
            real_activations = self._get_activations(real_paths, model)
            gen_activations = self._get_activations(gen_paths, model)

            mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
            mu_gen, sigma_gen = np.mean(gen_activations, axis=0), np.cov(gen_activations, rowvar=False)
            
            fid_score = self._calculate_fid_score(mu_real, sigma_real, mu_gen, sigma_gen)
            print(f"  FID Result: {fid_score:.4f}")
            return fid_score

class FIDBenchmarkTask(BenchmarkTask):
    """
    Benchmark Task for FID evaluation using COCO dataset API.
    Only downloads specific images needed for the test.
    """
    def __init__(self, num_samples=50, dataset_split="validation", batch_size=32, seed=42, exclude_categories=None):
        super().__init__(name="FID_Benchmark")
        self.num_samples = num_samples
        self.dataset_split = dataset_split
        self.seed = seed
        self.exclude_categories = exclude_categories or []
        
        self.metric_calculator = FIDMetric(batch_size=batch_size)
        self.real_images: List[Image.Image] = []
        self.real_images_dir = os.path.join("results", "benchmarks", "FID_Benchmark", "real_images")
    
    def _load_data(self) -> List[str]:
        """
        Load prompts and real images from COCO dataset using COCO API.
        Downloads annotations once, then fetches images on demand.
        """
        prompts = []
        self.real_images = []

        coco_dir = os.path.join("data", "COCO")
        os.makedirs(coco_dir, exist_ok=True)

        annotations_zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        annotations_json_path = os.path.join(coco_dir, "annotations", "captions_val2017.json")

        if not os.path.exists(annotations_json_path):
            try:
                zip_path = os.path.join(coco_dir, "annotations.zip")
                response = requests.get(annotations_zip_url, stream=True)
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(coco_dir)
                os.remove(zip_path)
            except Exception as e:
                print(f"Failed to download annotations: {e}")
                return self._fallback_prompts()

        # Initialize COCO api
        coco = COCO(annotations_json_path)
        imgIds = coco.getImgIds()
        
        count = 0
        for img_id in imgIds:
            if count >= self.num_samples:
                break

            # Get captions for this image
            annIds = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(annIds)
            if not anns:
                continue

            caption = anns[0]['caption']

            # Skip excluded categories (e.g., "dog" if testing unlearning)
            if self.exclude_categories:
                skip = False
                for category in self.exclude_categories:
                    if category.lower() in caption.lower():
                        skip = True
                        break
                if skip:
                    continue
            
            # Download the specific image
            try:
                img_data = coco.loadImgs(img_id)[0]
                img_url = img_data['coco_url']

                response = requests.get(img_url, timeout=10)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                self.real_images.append(img)
                prompts.append(caption)
                count += 1
                
                if count % 10 == 0 or count == self.num_samples:
                    print(f"  Downloaded {count}/{self.num_samples} images")
                    
            except Exception as e:
                continue
        return prompts

    def _calculate_metric(self, generated_images: List[Any], prompts: List[str]) -> float:
        """Calculate FID between generated and real images."""
        if not self.real_images:
            print("No real images available for FID calculation")
            return float('inf')
        return self.metric_calculator.calculate(generated_images, self.real_images)

    def _save_images(self, images: List[Any], run_name: str):
        """Save generated images and real reference images."""
        super()._save_images(images, run_name)
        
        # Save real images once for reference
        if not os.path.exists(self.real_images_dir) and self.real_images:
            os.makedirs(self.real_images_dir, exist_ok=True)
            for i, img in enumerate(self.real_images):
                try:
                    img.save(os.path.join(self.real_images_dir, f"real_{i:03d}.png"))
                except Exception as e:
                    print(f"Failed to save real image {i}: {e}")