"""
Neural Network Layers for Vision Transformer Diffusion Model

This module implements fundamental building blocks used in transformers and neural networks:
- Linear (fully connected) layers
- Layer Normalization
- Dropout
- Patch Embedding (for Vision Transformers)
- Positional Embeddings

Each layer includes forward and backward pass for gradient computation.
"""

import numpy as np
from . import initializers


class Linear:
    """
    Fully Connected (Dense) Layer
    
    Performs linear transformation: y = xW + b
    
    Where:
        - x: input with shape (batch_size, in_features)
        - W: weight matrix with shape (in_features, out_features)
        - b: bias vector with shape (out_features,)
        - y: output with shape (batch_size, out_features)
    
    Backpropagation:
        - ∂L/∂x = ∂L/∂y · W^T
        - ∂L/∂W = x^T · ∂L/∂y
        - ∂L/∂b = sum(∂L/∂y, axis=0)
    """
    
    def __init__(self, in_features, out_features, bias=True, init_method='xavier_normal'):
        """
        Initialize Linear layer
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
            init_method: Weight initialization method
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights
        if init_method == 'xavier_normal':
            self.weight = initializers.xavier_normal((in_features, out_features))
        elif init_method == 'xavier_uniform':
            self.weight = initializers.xavier_uniform((in_features, out_features))
        elif init_method == 'he_normal':
            self.weight = initializers.he_normal((in_features, out_features))
        elif init_method == 'he_uniform':
            self.weight = initializers.he_uniform((in_features, out_features))
        else:
            self.weight = initializers.xavier_normal((in_features, out_features))
        
        # Initialize bias
        if self.use_bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None
        
        # Cache for backward pass
        self.cache = None
        
        # Gradients
        self.grad_weight = None
        self.grad_bias = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, in_features) or (..., in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features) or (..., out_features)
        """
        # Save input for backward pass
        self.cache = x
        
        # Compute output: y = xW + b
        output = np.dot(x, self.weight)
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer, shape (..., out_features)
            
        Returns:
            Gradient with respect to input, shape (..., in_features)
        """
        x = self.cache
        
        # Compute gradients
        # ∂L/∂W = x^T · ∂L/∂y
        self.grad_weight = np.dot(x.reshape(-1, x.shape[-1]).T, 
                                   grad_output.reshape(-1, grad_output.shape[-1]))
        
        # ∂L/∂b = sum(∂L/∂y) over batch dimension
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
        
        # ∂L/∂x = ∂L/∂y · W^T
        grad_input = np.dot(grad_output, self.weight.T)
        
        return grad_input
    
    def parameters(self):
        """Return dictionary of parameters"""
        params = {'weight': self.weight}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def gradients(self):
        """Return dictionary of gradients"""
        grads = {'weight': self.grad_weight}
        if self.use_bias:
            grads['bias'] = self.grad_bias
        return grads
    
    def __call__(self, x):
        return self.forward(x)


class LayerNorm:
    """
    Layer Normalization
    
    Normalizes inputs across the feature dimension:
    
    y = γ * (x - μ) / √(σ² + ε) + β
    
    Where:
        - μ: mean across features
        - σ²: variance across features
        - γ: learnable scale parameter
        - β: learnable shift parameter
        - ε: small constant for numerical stability
    
    Used extensively in transformers as it normalizes each sample independently
    (unlike batch norm which normalizes across the batch).
    
    Reference:
        Ba et al., 2016: "Layer Normalization"
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize LayerNorm
        
        Args:
            normalized_shape: Size of features to normalize (last dimension)
            eps: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters (initialized to 1 and 0)
        self.gamma = np.ones(normalized_shape)  # Scale
        self.beta = np.zeros(normalized_shape)   # Shift
        
        # Cache for backward pass
        self.cache = None
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized output with same shape as input
        """
        # Compute mean and variance across last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        # Cache for backward pass
        self.cache = (x, x_normalized, mean, variance)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        x, x_normalized, mean, variance = self.cache
        N = self.normalized_shape
        
        # Gradient w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * x_normalized, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
        
        # Gradient w.r.t. normalized input
        grad_x_normalized = grad_output * self.gamma
        
        # Gradient w.r.t. variance
        grad_variance = np.sum(grad_x_normalized * (x - mean) * -0.5 * 
                               np.power(variance + self.eps, -1.5), axis=-1, keepdims=True)
        
        # Gradient w.r.t. mean
        grad_mean = np.sum(grad_x_normalized * -1.0 / np.sqrt(variance + self.eps), 
                           axis=-1, keepdims=True)
        grad_mean += grad_variance * np.sum(-2.0 * (x - mean), axis=-1, keepdims=True) / N
        
        # Gradient w.r.t. input
        grad_input = grad_x_normalized / np.sqrt(variance + self.eps)
        grad_input += grad_variance * 2.0 * (x - mean) / N
        grad_input += grad_mean / N
        
        return grad_input
    
    def parameters(self):
        """Return dictionary of parameters"""
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def gradients(self):
        """Return dictionary of gradients"""
        return {'gamma': self.grad_gamma, 'beta': self.grad_beta}
    
    def __call__(self, x):
        return self.forward(x)


class Dropout:
    """
    Dropout Regularization
    
    During training:
        - Randomly sets elements to zero with probability p
        - Scales remaining elements by 1/(1-p) to maintain expected value
    
    During inference:
        - Acts as identity function (no dropout)
    
    Prevents overfitting and improves generalization.
    
    Reference:
        Srivastava et al., 2014: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    """
    
    def __init__(self, p=0.1):
        """
        Initialize Dropout
        
        Args:
            p: Probability of dropping an element (0 to 1)
        """
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Output with same shape as input
        """
        if not self.training or self.p == 0:
            return x
        
        # Generate random mask
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        
        # Apply mask and scale
        output = x * self.mask / (1 - self.p)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        if not self.training or self.p == 0:
            return grad_output
        
        # Apply same mask to gradient
        grad_input = grad_output * self.mask / (1 - self.p)
        
        return grad_input
    
    def train(self):
        """Set to training mode"""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode"""
        self.training = False
    
    def __call__(self, x):
        return self.forward(x)


class PatchEmbedding:
    """
    Patch Embedding for Vision Transformers
    
    Converts an image into a sequence of patch embeddings:
    1. Divide image into non-overlapping patches
    2. Flatten each patch
    3. Project to embedding dimension using linear layer
    
    For image of size (H, W, C) with patch size P:
        - Number of patches: N = (H/P) * (W/P)
        - Patch dimension: P * P * C
        - Output: (N, embed_dim)
    
    Example:
        224x224x3 image with 16x16 patches:
        - 196 patches of size 16x16x3 = 768 dims
        - Project to embed_dim (e.g., 512)
    """
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        """
        Initialize PatchEmbedding
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        
        # Linear projection layer
        self.projection = Linear(self.patch_dim, embed_dim)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images of shape (batch_size, height, width, channels)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        batch_size, H, W, C = x.shape
        P = self.patch_size
        
        assert H == W == self.img_size, f"Input size {H}x{W} doesn't match expected {self.img_size}"
        assert C == self.in_channels, f"Input channels {C} doesn't match expected {self.in_channels}"
        
        # Reshape to patches
        # (B, H, W, C) -> (B, H/P, P, W/P, P, C) -> (B, H/P, W/P, P, P, C)
        x = x.reshape(batch_size, H // P, P, W // P, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H/P, W/P, P, P, C)
        
        # Flatten patches: (B, H/P, W/P, P*P*C)
        x = x.reshape(batch_size, (H // P) * (W // P), P * P * C)
        
        # Save for backward
        self.cache = (batch_size, H, W, C)
        
        # Project to embedding dimension
        embeddings = self.projection(x)  # (B, N, embed_dim)
        
        return embeddings
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient of shape (batch_size, num_patches, embed_dim)
            
        Returns:
            Gradient w.r.t. input image of shape (batch_size, H, W, C)
        """
        batch_size, H, W, C = self.cache
        P = self.patch_size
        
        # Backprop through projection
        grad_patches = self.projection.backward(grad_output)  # (B, N, P*P*C)
        
        # Reshape back to image
        grad_patches = grad_patches.reshape(batch_size, H // P, W // P, P, P, C)
        grad_patches = grad_patches.transpose(0, 1, 3, 2, 4, 5)  # (B, H/P, P, W/P, P, C)
        grad_input = grad_patches.reshape(batch_size, H, W, C)
        
        return grad_input
    
    def parameters(self):
        """Return dictionary of parameters"""
        return self.projection.parameters()
    
    def gradients(self):
        """Return dictionary of gradients"""
        return self.projection.gradients()
    
    def __call__(self, x):
        return self.forward(x)


class PositionalEmbedding:
    """
    Positional Embeddings for Transformers
    
    Adds positional information to patch embeddings since self-attention is permutation-invariant.
    
    Two variants:
    1. Learnable: Trainable position embeddings
    2. Sinusoidal: Fixed sine/cosine embeddings (used in original Transformer)
    
    For ViT, learnable embeddings are more common.
    """
    
    def __init__(self, num_positions, embed_dim, learnable=True):
        """
        Initialize PositionalEmbedding
        
        Args:
            num_positions: Maximum number of positions (num_patches + 1 for class token)
            embed_dim: Embedding dimension
            learnable: Whether embeddings are learnable or fixed
        """
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        self.learnable = learnable
        
        if learnable:
            # Learnable position embeddings
            self.pos_embed = initializers.truncated_normal((num_positions, embed_dim), std=0.02)
        else:
            # Sinusoidal position embeddings
            self.pos_embed = self._create_sinusoidal_embeddings(num_positions, embed_dim)
        
        # Gradients (only for learnable)
        self.grad_pos_embed = None
    
    def _create_sinusoidal_embeddings(self, num_positions, embed_dim):
        """
        Create sinusoidal position embeddings
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        """
        position = np.arange(num_positions)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pos_embed = np.zeros((num_positions, embed_dim))
        pos_embed[:, 0::2] = np.sin(position * div_term)
        pos_embed[:, 1::2] = np.cos(position * div_term)
        
        return pos_embed
    
    def forward(self, x):
        """
        Forward pass - adds position embeddings to input
        
        Args:
            x: Input of shape (batch_size, num_positions, embed_dim)
            
        Returns:
            Output with position embeddings added
        """
        batch_size, seq_len, embed_dim = x.shape
        assert seq_len <= self.num_positions, f"Sequence length {seq_len} exceeds max {self.num_positions}"
        
        # Add position embeddings (broadcast across batch)
        output = x + self.pos_embed[:seq_len, :]
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input (same as grad_output for addition)
        """
        if self.learnable:
            # Accumulate gradient for position embeddings
            batch_size, seq_len, embed_dim = grad_output.shape
            self.grad_pos_embed = np.sum(grad_output, axis=0)  # Sum over batch
        
        # Gradient w.r.t. input is just pass-through (addition)
        return grad_output
    
    def parameters(self):
        """Return dictionary of parameters"""
        if self.learnable:
            return {'pos_embed': self.pos_embed}
        return {}
    
    def gradients(self):
        """Return dictionary of gradients"""
        if self.learnable:
            return {'pos_embed': self.grad_pos_embed}
        return {}
    
    def __call__(self, x):
        return self.forward(x)
