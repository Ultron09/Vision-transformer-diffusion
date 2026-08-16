"""
2D Rotary Position Embedding (2D RoPE) for Vision Transformers

Extends 1D RoPE (Su et al., 2021) to 2D spatial image grids (Heo et al., 2024).
Decomposes token embeddings into height (y) and width (x) coordinates to preserve
relative spatial distances and translation invariance across variable patch resolutions.

Mathematical Formulation:
    For a patch at grid coordinate (h, w) and token feature vector x:
        x_rotated = x * cos(Θ_{h,w}) + rotate_half(x) * sin(Θ_{h,w})
        
    Where Θ_{h,w} is formed by concatenating frequencies for h and w:
        Θ_{h,w} = [θ_{y, 0}, ..., θ_{y, d/4-1}, θ_{x, 0}, ..., θ_{x, d/4-1}]
"""

import numpy as np


def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input tensor.
    [-x2, x1] formulation for complex multiplication.
    
    Args:
        x: Array of shape (..., d)
        
    Returns:
        Rotated array of shape (..., d)
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return np.concatenate((-x2, x1), axis=-1)


class RotaryEmbedding2D:
    """
    2D Rotary Positional Embedding (RoPE-2D)
    
    Computes spatial frequency matrices and applies rotation to Q and K tensors
    in Vision Transformer attention layers.
    """
    
    def __init__(self, dim, base=10000.0):
        """
        Initialize 2D RoPE
        
        Args:
            dim: Dimension of attention head (must be divisible by 4 for 2D decomposition)
            base: Base frequency for geometric progression
        """
        assert dim % 4 == 0, f"Head dimension ({dim}) must be divisible by 4 for 2D RoPE."
        self.dim = dim
        self.base = base
        
        # Quarter dimension for x, quarter dimension for y (each paired with sin/cos)
        self.dim_spatial = dim // 2
        self.inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim_spatial, 2, dtype=np.float32) / self.dim_spatial))
        self.cache = {}
        
    def _compute_freqs_grid(self, height_patches, width_patches):
        """
        Compute 2D frequency grid (cos, sin) for spatial coordinates.
        
        Args:
            height_patches: Number of vertical patches
            width_patches: Number of horizontal patches
            
        Returns:
            Tuple of (cos_grid, sin_grid) of shape (1, 1, num_patches, dim)
        """
        cache_key = (height_patches, width_patches)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Grid coordinates
        y_pos = np.arange(height_patches, dtype=np.float32)
        x_pos = np.arange(width_patches, dtype=np.float32)
        
        # Meshgrid: shape (H, W)
        grid_y, grid_x = np.meshgrid(y_pos, x_pos, indexing='ij')
        grid_y = grid_y.flatten()  # (N,)
        grid_x = grid_x.flatten()  # (N,)
        
        # Outer product with frequencies: (N, dim_spatial // 2)
        freqs_y = np.outer(grid_y, self.inv_freq)
        freqs_x = np.outer(grid_x, self.inv_freq)
        
        # Duplicate for sin/cos pairs: (N, dim_spatial)
        freqs_y = np.repeat(freqs_y, 2, axis=-1)
        freqs_x = np.repeat(freqs_x, 2, axis=-1)
        
        # Combine y and x frequencies into full head dim: (N, dim)
        freqs = np.concatenate([freqs_y, freqs_x], axis=-1)
        
        cos = np.cos(freqs)[np.newaxis, np.newaxis, :, :]  # (1, 1, N, dim)
        sin = np.sin(freqs)[np.newaxis, np.newaxis, :, :]  # (1, 1, N, dim)
        
        self.cache[cache_key] = (cos, sin)
        return cos, sin
        
    def apply_rotary_emb(self, q, k, height_patches, width_patches):
        """
        Apply 2D RoPE rotation to query and key tensors.
        
        Args:
            q: Query tensor of shape (B, num_heads, num_patches, head_dim)
            k: Key tensor of shape (B, num_heads, num_patches, head_dim)
            height_patches: Grid height in patches
            width_patches: Grid width in patches
            
        Returns:
            Tuple of rotated (q_rot, k_rot)
        """
        cos, sin = self._compute_freqs_grid(height_patches, width_patches)
        
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        
        return q_rot, k_rot

    def backward(self, grad_q_rot, grad_k_rot, height_patches, width_patches):
        """
        Backward pass for 2D RoPE.
        Since RoPE is an orthogonal rotation matrix R, the adjoint / inverse is R^T = R(-Θ).
        
        Args:
            grad_q_rot: Gradient w.r.t rotated Q
            grad_k_rot: Gradient w.r.t rotated K
            height_patches: Grid height
            width_patches: Grid width
            
        Returns:
            Tuple of (grad_q, grad_k)
        """
        cos, sin = self._compute_freqs_grid(height_patches, width_patches)
        
        # Inverse rotation uses -sin
        grad_q = (grad_q_rot * cos) + (rotate_half(grad_q_rot) * (-sin))
        grad_k = (grad_k_rot * cos) + (rotate_half(grad_k_rot) * (-sin))
        
        return grad_q, grad_k
