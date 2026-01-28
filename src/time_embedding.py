"""
Time Embeddings for Diffusion Models

In diffusion models, the denoising network needs to know what timestep it's operating on.
Time embeddings encode the timestep as a vector that can be injected into the network.

Two common approaches:
1. Sinusoidal embeddings (used in original DDPM, similar to positional embeddings in transformers)
2. Learned embeddings

We use sinusoidal embeddings as they work well without requiring training.

Mathematical Background:
    - Encode timestep t as multiple sine/cosine waves of different frequencies
    - Provides smooth, continuous representation across timesteps
    - Similar to positional encodings in "Attention Is All You Need"
"""

import numpy as np
from .layers import Linear
from .activations import SiLU


class SinusoidalTimeEmbedding:
    """
    Sinusoidal Time Embedding
    
    Encodes timestep t as a vector using sine and cosine functions:
    
    embedding[2i] = sin(t / 10000^(2i/d))
    embedding[2i+1] = cos(t / 10000^(2i/d))
    
    Where d is the embedding dimension.
    
    Properties:
    - Deterministic (no learnable parameters)
    - Smooth across timesteps
    - Captures multiple frequencies
    """
    
    def __init__(self, embed_dim):
        """
        Initialize Sinusoidal Time Embedding
        
        Args:
            embed_dim: Dimension of time embeddings
        """
        self.embed_dim = embed_dim
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        frequencies = np.exp(-np.log(10000.0) * np.arange(half_dim) / half_dim)
        self.frequencies = frequencies
    
    def forward(self, timesteps):
        """
        Compute time embeddings for given timesteps
        
        Args:
            timesteps: Array of timesteps, shape (batch_size,) or scalar
            
        Returns:
            Time embeddings of shape (batch_size, embed_dim)
        """
        # Ensure timesteps is array
        if np.isscalar(timesteps):
            timesteps = np.array([timesteps])
        else:
            timesteps = np.array(timesteps)
        
        # Reshape for broadcasting: (batch_size, 1)
        timesteps = timesteps.reshape(-1, 1)
        
        # Compute arguments: (batch_size, half_dim)
        args = timesteps * self.frequencies
        
        # Compute sin and cos
        sin_embeddings = np.sin(args)
        cos_embeddings = np.cos(args)
        
        # Concatenate: (batch_size, embed_dim)
        embeddings = np.concatenate([sin_embeddings, cos_embeddings], axis=-1)
        
        return embeddings
    
    def __call__(self, timesteps):
        return self.forward(timesteps)


class TimeEmbeddingMLP:
    """
    Time Embedding with MLP Projection
    
    Takes sinusoidal time embeddings and projects them through an MLP:
    
    t_embed = SiLU(Linear(sinusoidal(t)))
    t_embed = Linear(t_embed)
    
    This adds learnable parameters and non-linearity to time conditioning.
    """
    
    def __init__(self, time_embed_dim, hidden_dim=None):
        """
        Initialize Time Embedding MLP
        
        Args:
            time_embed_dim: Dimension of time embeddings
            hidden_dim: Hidden dimension (default: 4 * time_embed_dim)
        """
        self.time_embed_dim = time_embed_dim
        hidden_dim = hidden_dim or 4 * time_embed_dim
        
        # Sinusoidal embedding
        self.sinusoidal_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # MLP for projection
        self.mlp = [
            Linear(time_embed_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim)
        ]
        
        self.cache = None
    
    def forward(self, timesteps):
        """
        Forward pass
        
        Args:
            timesteps: Timesteps array (batch_size,)
            
        Returns:
            Time embeddings (batch_size, hidden_dim)
        """
        # Get sinusoidal embeddings
        x = self.sinusoidal_embed(timesteps)
        
        # Pass through MLP
        activations = [x]
        for layer in self.mlp:
            x = layer(x) if hasattr(layer, 'forward') else layer.forward(x)
            activations.append(x)
        
        self.cache = activations
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. timesteps (not used, since timesteps are discrete)
        """
        grad = grad_output
        
        # Backprop through MLP (in reverse)
        for layer in reversed(self.mlp):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        
        # No gradient w.r.t. timesteps (discrete input)
        return None
    
    def parameters(self):
        """Return learnable parameters"""
        params = {}
        layer_idx = 0
        for layer in self.mlp:
            if hasattr(layer, 'parameters'):
                layer_params = layer.parameters()
                for key, val in layer_params.items():
                    params[f'layer{layer_idx}_{key}'] = val
                layer_idx += 1
        return params
    
    def gradients(self):
        """Return gradients"""
        grads = {}
        layer_idx = 0
        for layer in self.mlp:
            if hasattr(layer, 'gradients'):
                layer_grads = layer.gradients()
                for key, val in layer_grads.items():
                    grads[f'layer{layer_idx}_{key}'] = val
                layer_idx += 1
        return grads
    
    def __call__(self, timesteps):
        return self.forward(timesteps)


class AdaptiveLayerNorm:
    """
    Adaptive Layer Normalization (AdaLN)
    
    Modulates layer normalization with time embeddings:
    
    AdaLN(x, t) = scale(t) * LayerNorm(x) + shift(t)
    
    Where scale(t) and shift(t) are learned functions of time embedding.
    
    This allows the network to adapt its normalization based on the timestep,
    which is crucial for diffusion models operating across different noise levels.
    """
    
    def __init__(self, normalized_shape, time_embed_dim):
        """
        Initialize Adaptive Layer Normalization
        
        Args:
            normalized_shape: Size of features to normalize
            time_embed_dim: Dimension of time embeddings
        """
        from .layers import LayerNorm
        
        self.norm = LayerNorm(normalized_shape)
        
        # Linear layer to predict scale and shift from time embedding
        # Outputs 2 * normalized_shape (for scale and shift)
        self.time_mlp = Linear(time_embed_dim, 2 * normalized_shape)
        
        self.cache = None
    
    def forward(self, x, time_embed):
        """
        Forward pass
        
        Args:
            x: Input tensor (..., normalized_shape)
            time_embed: Time embeddings (batch_size, time_embed_dim)
            
        Returns:
            Modulated output
        """
        # Normalize input
        x_norm = self.norm(x)
        
        # Get scale and shift from time embedding
        time_out = self.time_mlp(time_embed)  # (batch_size, 2 * normalized_shape)
        
        # Split into scale and shift
        scale, shift = np.split(time_out, 2, axis=-1)  # Each: (batch_size, normalized_shape)
        
        # Reshape for broadcasting if needed
        if x_norm.ndim > 2:
            # Add dimensions for broadcasting: (batch_size, 1, ..., 1, normalized_shape)
            for _ in range(x_norm.ndim - 2):
                scale = scale[:, np.newaxis, :]
                shift = shift[:, np.newaxis, :]
        
        # Apply modulation
        output = scale * x_norm + shift
        
        self.cache = (x, x_norm, scale, shift, time_embed)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            (grad_x, grad_time_embed) tuple
        """
        x, x_norm, scale, shift, time_embed = self.cache
        
        # Gradient w.r.t. scale
        grad_scale = grad_output * x_norm
        
        # Gradient w.r.t. shift
        grad_shift = grad_output
        
        # Sum over spatial dimensions if present
        if grad_output.ndim > 2:
            axes_to_sum = tuple(range(1, grad_output.ndim - 1))
            grad_scale = np.sum(grad_scale, axis=axes_to_sum)
            grad_shift = np.sum(grad_shift, axis=axes_to_sum)
        
        # Concatenate scale and shift gradients
        grad_time_out = np.concatenate([grad_scale, grad_shift], axis=-1)
        
        # Backprop through time MLP
        grad_time_embed = self.time_mlp.backward(grad_time_out)
        
        # Gradient w.r.t. normalized input
        grad_x_norm = grad_output * scale
        
        # Backprop through layer norm
        grad_x = self.norm.backward(grad_x_norm)
        
        return grad_x, grad_time_embed
    
    def parameters(self):
        """Return parameters"""
        params = {}
        
        norm_params = self.norm.parameters()
        for key, val in norm_params.items():
            params[f'norm_{key}'] = val
        
        mlp_params = self.time_mlp.parameters()
        for key, val in mlp_params.items():
            params[f'time_mlp_{key}'] = val
        
        return params
    
    def gradients(self):
        """Return gradients"""
        grads = {}
        
        norm_grads = self.norm.gradients()
        for key, val in norm_grads.items():
            grads[f'norm_{key}'] = val
        
        mlp_grads = self.time_mlp.gradients()
        for key, val in mlp_grads.items():
            grads[f'time_mlp_{key}'] = val
        
        return grads
    
    def __call__(self, x, time_embed):
        return self.forward(x, time_embed)
