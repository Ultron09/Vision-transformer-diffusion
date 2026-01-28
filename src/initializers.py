"""
Weight Initialization Strategies

This module implements various weight initialization methods used in deep learning.
Proper initialization is crucial for:
- Preventing vanishing/exploding gradients
- Faster convergence during training
- Better final performance

Mathematical Background:
- Xavier/Glorot: Variance = 2 / (fan_in + fan_out)
- He: Variance = 2 / fan_in (for ReLU-like activations)
- Truncated Normal: Clips values beyond 2 standard deviations
"""

import numpy as np


def xavier_uniform(shape, gain=1.0):
    """
    Xavier/Glorot Uniform Initialization
    
    Samples from uniform distribution U(-a, a) where:
    a = gain * sqrt(6 / (fan_in + fan_out))
    
    Best for:
    - Tanh activations
    - Sigmoid activations
    - Linear layers with symmetric activations
    
    Args:
        shape: Shape of weight tensor (e.g., (in_features, out_features))
        gain: Scaling factor (default: 1.0)
        
    Returns:
        Initialized weight array
        
    Reference:
        Glorot & Bengio, 2010: "Understanding the difficulty of training deep feedforward neural networks"
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(shape)
    std = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-std, std, size=shape)


def xavier_normal(shape, gain=1.0):
    """
    Xavier/Glorot Normal Initialization
    
    Samples from normal distribution N(0, σ²) where:
    σ = gain * sqrt(2 / (fan_in + fan_out))
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor (default: 1.0)
        
    Returns:
        Initialized weight array
    """
    fan_in, fan_out = _calculate_fan_in_fan_out(shape)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std


def he_uniform(shape, gain=1.0):
    """
    He Uniform Initialization (Kaiming Uniform)
    
    Samples from uniform distribution U(-a, a) where:
    a = gain * sqrt(6 / fan_in)
    
    Best for:
    - ReLU activations
    - LeakyReLU activations
    - Any ReLU-variant
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor (default: 1.0, use sqrt(2) for ReLU)
        
    Returns:
        Initialized weight array
        
    Reference:
        He et al., 2015: "Delving Deep into Rectifiers"
    """
    fan_in, _ = _calculate_fan_in_fan_out(shape)
    std = gain * np.sqrt(6.0 / fan_in)
    return np.random.uniform(-std, std, size=shape)


def he_normal(shape, gain=1.0):
    """
    He Normal Initialization (Kaiming Normal)
    
    Samples from normal distribution N(0, σ²) where:
    σ = gain * sqrt(2 / fan_in)
    
    Args:
        shape: Shape of weight tensor
        gain: Scaling factor (default: 1.0, use sqrt(2) for ReLU)
        
    Returns:
        Initialized weight array
    """
    fan_in, _ = _calculate_fan_in_fan_out(shape)
    std = gain * np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def truncated_normal(shape, mean=0.0, std=0.02, clip=2.0):
    """
    Truncated Normal Initialization
    
    Samples from normal distribution but clips values beyond 'clip' standard deviations.
    This prevents extreme outliers that can destabilize early training.
    
    Commonly used in:
    - ViT (Vision Transformers)
    - BERT and transformer models
    - Position embeddings
    
    Args:
        shape: Shape of weight tensor
        mean: Mean of distribution (default: 0.0)
        std: Standard deviation (default: 0.02)
        clip: Number of std devs to clip at (default: 2.0)
        
    Returns:
        Initialized weight array
    """
    # Generate samples
    samples = np.random.randn(*shape) * std + mean
    
    # Clip to [-clip*std, clip*std] range
    lower_bound = mean - clip * std
    upper_bound = mean + clip * std
    samples = np.clip(samples, lower_bound, upper_bound)
    
    return samples


def constant(shape, value=0.0):
    """
    Constant Initialization
    
    Initialize all weights to a constant value.
    
    Common uses:
    - Bias initialization (usually zeros)
    - LayerNorm gamma (usually ones)
    - LayerNorm beta (usually zeros)
    
    Args:
        shape: Shape of weight tensor
        value: Constant value (default: 0.0)
        
    Returns:
        Initialized weight array
    """
    return np.full(shape, value, dtype=np.float32)


def zeros(shape):
    """Initialize all weights to zero"""
    return np.zeros(shape, dtype=np.float32)


def ones(shape):
    """Initialize all weights to one"""
    return np.ones(shape, dtype=np.float32)


def orthogonal(shape, gain=1.0):
    """
    Orthogonal Initialization
    
    Initialize weights as orthogonal matrices. For non-square matrices,
    uses QR decomposition of random matrix.
    
    Properties:
    - Preserves norm of input vector
    - Helps with gradient flow
    - Good for recurrent networks
    
    Args:
        shape: Shape of weight tensor (must be 2D)
        gain: Scaling factor
        
    Returns:
        Initialized weight array
    """
    if len(shape) != 2:
        raise ValueError("Orthogonal initialization requires 2D shape")
    
    # Generate random matrix
    a = np.random.randn(*shape)
    
    # QR decomposition
    q, r = np.linalg.qr(a)
    
    # Make sure Q has the right shape and sign
    d = np.diag(r)
    q *= np.sign(d)
    
    return gain * q


def _calculate_fan_in_fan_out(shape):
    """
    Calculate fan_in and fan_out for a weight tensor
    
    For 2D tensors (Linear layers): (fan_in, fan_out)
    For 4D tensors (Conv layers): fan = kernel_size * kernel_size * channels
    
    Args:
        shape: Shape tuple of weight tensor
        
    Returns:
        (fan_in, fan_out) tuple
    """
    if len(shape) == 2:
        # Linear layer: (in_features, out_features)
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        # Conv layer: (out_channels, in_channels, kernel_h, kernel_w)
        receptive_field_size = shape[2] * shape[3]
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        # General case: treat first dim as fan_in, last as fan_out
        fan_in = shape[0]
        fan_out = shape[-1]
    
    return fan_in, fan_out


# Gain values for different activations (for He/Xavier init)
ACTIVATION_GAINS = {
    'linear': 1.0,
    'tanh': 5.0 / 3.0,
    'sigmoid': 1.0,
    'relu': np.sqrt(2.0),
    'leaky_relu': np.sqrt(2.0 / (1 + 0.01**2)),  # assuming negative_slope=0.01
    'selu': 3.0 / 4.0,
    'gelu': 1.0,
    'silu': 1.0,
}


def get_activation_gain(activation):
    """
    Get the recommended gain value for an activation function
    
    Args:
        activation: Name of activation function
        
    Returns:
        Gain value for initialization
    """
    return ACTIVATION_GAINS.get(activation.lower(), 1.0)
