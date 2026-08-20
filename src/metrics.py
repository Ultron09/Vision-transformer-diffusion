"""
Evaluation Metrics for Generative Diffusion Models (Pure NumPy)

Implements foundational image synthesis quality metrics:
1. Fréchet Inception Distance (FID) - Heusel et al., 2017
2. Inception Score (IS) - Salimans et al., 2016
3. Peak Signal-to-Noise Ratio (PSNR)
4. Structural Similarity Index Measure (SSIM)
"""

import numpy as np


def matrix_sqrt(matrix, eps=1e-6):
    """
    Compute matrix square root for symmetric / positive semi-definite matrices.
    Uses eigenvalue decomposition with numerical clipping.
    
    Args:
        matrix: 2D square matrix (d, d)
        eps: Small constant for numerical stability
        
    Returns:
        Square root of matrix (d, d)
    """
    # Symmetrize to eliminate asymmetric numerical noise
    sym_matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)
    
    # Clip negative eigenvalues due to numerical imprecision
    eigenvalues = np.maximum(eigenvalues, eps)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two Gaussian distributions:
    
        d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁ Σ₂)^(1/2))
        
    Args:
        mu1: Mean vector of generated feature distribution (d,)
        sigma1: Covariance matrix of generated features (d, d)
        mu2: Mean vector of real feature distribution (d,)
        sigma2: Covariance matrix of real features (d, d)
        eps: Small stability constant
        
    Returns:
        Scalar Fréchet Inception Distance (FID) score
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    mean_diff_sq = np.dot(diff, diff)
    
    # Product of covariances
    covmean = matrix_sqrt(sigma1 @ sigma2, eps=eps)
    
    # Check for complex numbers from numerical artifacts
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    fid = mean_diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(np.maximum(fid, 0.0))


def calculate_inception_score(class_probabilities, num_splits=10, eps=1e-16):
    """
    Calculate Inception Score (IS) from classification probabilities.
    
        IS = exp( E_{x} [ KL( p(y|x) || p(y) ) ] )
        
    Args:
        class_probabilities: Array of softmax probabilities (N, num_classes)
        num_splits: Number of partitions to compute mean and standard error
        eps: Epsilon to prevent log(0)
        
    Returns:
        Tuple of (mean_score, std_score)
    """
    num_samples = class_probabilities.shape[0]
    split_size = num_samples // num_splits
    scores = []
    
    for i in range(num_splits):
        part = class_probabilities[i * split_size:(i + 1) * split_size]
        if len(part) == 0:
            continue
            
        # Marginal class distribution: p(y) = 1/N * sum_x p(y|x)
        p_y = np.mean(part, axis=0, keepdims=True)
        
        # KL Divergence: sum_y p(y|x) * (log p(y|x) - log p(y))
        kl_div = part * (np.log(np.maximum(part, eps)) - np.log(np.maximum(p_y, eps)))
        kl_div = np.sum(kl_div, axis=1)
        
        scores.append(np.exp(np.mean(kl_div)))
        
    return float(np.mean(scores)), float(np.std(scores))


def compute_psnr(img1, img2, data_range=2.0):
    """
    Compute Peak Signal-to-Noise Ratio between two image batches.
    
    Args:
        img1: Array of shape (B, H, W, C)
        img2: Array of shape (B, H, W, C)
        data_range: Dynamic range of input images (e.g. 2.0 for [-1, 1])
        
    Returns:
        Mean PSNR in decibels (dB)
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2, axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-10)
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(np.mean(psnr))


def compute_ssim(img1, img2, data_range=2.0, k1=0.01, k2=0.03):
    """
    Compute mean Structural Similarity Index (SSIM) across batches.
    
    Args:
        img1: First image batch (B, H, W, C)
        img2: Second image batch (B, H, W, C)
        data_range: Range of image pixel values
        k1: SSIM stability parameter
        k2: SSIM stability parameter
        
    Returns:
        Mean SSIM score in range [-1, 1]
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    mu1 = np.mean(img1, axis=(1, 2), keepdims=True)
    mu2 = np.mean(img2, axis=(1, 2), keepdims=True)
    
    sigma1_sq = np.var(img1, axis=(1, 2), keepdims=True)
    sigma2_sq = np.var(img2, axis=(1, 2), keepdims=True)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(1, 2), keepdims=True)
    
    numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / (denominator + 1e-8)
    return float(np.mean(ssim_map))
