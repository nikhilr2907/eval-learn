import tensorflow as tf
import numpy as np
from scipy import linalg
from PIL import Image

def inception_extractor():
    """
    Create InceptionV3 model for feature extraction.
    
    Returns:
        TensorFlow model that outputs 2048-dimensional feature vectors
    """
    model = tf.keras.applications.InceptionV3(
        include_top = False,
        weights = "imagenet",
        pooling = "avg",
        input_shape = (299, 299, 3),
    )

    return model

def load_process_image(path):
    """Load and preprocess a single image for InceptionV3.
    
    Args:
        path: Path to image file or PIL Image object
        
    Returns:
        Preprocessed image array ready for InceptionV3
    """
    if isinstance(path, str):
        image = Image.open(path).convert("RGB").resize((299, 299), Image.BILINEAR)
    elif isinstance(path, Image.Image):
        image = path.convert("RGB").resize((299, 299), Image.BILINEAR)
    else:
        raise ValueError(f"Invalid input type: {type(path)}")
    
    image = np.array(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    
    return image

def get_activations(image_paths, model, batch_size=32):
    """
    Extract InceptionV3 features from images.
    
    Args:
        image_paths: List of image paths or PIL Image objects
        model: InceptionV3 model from inception_extractor()
        batch_size: Number of images to process at once
        
    Returns:
        numpy array of shape (N, 2048) containing features
    """
    n = len(image_paths)
    activation_lists = []
    
    for i in range(0, n, batch_size):
        batch_path = image_paths[i : i + batch_size]
        batch_images = [load_process_image(p) for p in batch_path]
        batch_images = np.stack(batch_images, axis=0)
        
        activations = model(batch_images, training=False).numpy()
        activation_lists.append(activations)
    
    activations = np.concatenate(activation_lists, axis=0)
    return activations

def compute_mean_sigma(x):
    """Calculate mean and covariance matrix of features.
    
    Args:
        x: Feature array
        
    Returns:
        Tuple of (mean, covariance)
    """
    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    return mean, cov

def calculate_FID(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Inception Distance between two distributions.
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance matrix of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance matrix of second distribution
        eps: Small epsilon for numerical stability
        
    Returns:
        FID score
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    # Calculate squared difference of means
    diff = mu1 - mu2
    
    # Compute square root of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical instability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def calculate_fid_from_paths(real_image_paths, 
                             generated_image_paths,
                             batch_size: int = 32) -> float:
    """
    Calculate FID score between real and generated images
    
    Args:
        real_image_paths: List of paths to real images or PIL Image objects
        generated_image_paths: List of paths to generated images or PIL Image objects
        batch_size: Batch size for processing
        
    Returns:
        FID score
    """
    model = inception_extractor()
    real_activations = get_activations(real_image_paths, model, batch_size)
    gen_activations = get_activations(generated_image_paths, model, batch_size)

    mu_real, sigma_real = compute_mean_sigma(real_activations)
    mu_gen, sigma_gen = compute_mean_sigma(gen_activations)

    fid_score = calculate_FID(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_score

