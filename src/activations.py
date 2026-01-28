"""
Activation Functions for Vision Transformer Diffusion Model

This module implements various activation functions used in modern neural networks,
particularly in Vision Transformers and Diffusion Models, along with their derivatives
for backpropagation.

Mathematical Background:
- GELU: Gaussian Error Linear Unit, approximates x * Φ(x) where Φ is CDF of N(0,1)
- SiLU/Swish: Smooth activation function x * σ(x) where σ is sigmoid
- Softmax: Converts logits to probability distribution
"""

import numpy as np


class GELU:
    """
    Gaussian Error Linear Unit (GELU) Activation
    
    GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
    
    Approximation used (faster):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Properties:
    - Smooth, non-monotonic activation
    - Used extensively in transformers (BERT, GPT, ViT)
    - Derivative is continuous everywhere
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass of GELU activation
        
        Args:
            x: Input array of any shape
            
        Returns:
            Activated output with same shape as input
        """
        # Using tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        tanh_input = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
        tanh_output = np.tanh(tanh_input)
        output = 0.5 * x * (1.0 + tanh_output)
        
        # Cache for backward pass
        self.cache = (x, tanh_input, tanh_output)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass of GELU activation
        
        Derivative: d(GELU)/dx = 0.5 * (1 + tanh(...)) + x * derivative_of_tanh_part
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        x, tanh_input, tanh_output = self.cache
        
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        
        # Derivative of tanh(u) = 1 - tanh^2(u)
        sech_squared = 1.0 - np.power(tanh_output, 2)
        
        # Derivative of inner function: sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        inner_derivative = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
        
        # Full derivative
        gelu_grad = 0.5 * (1.0 + tanh_output) + 0.5 * x * sech_squared * inner_derivative
        
        return grad_output * gelu_grad
    
    def __call__(self, x):
        return self.forward(x)


class SiLU:
    """
    Sigmoid Linear Unit (SiLU) / Swish Activation
    
    SiLU(x) = x * σ(x) where σ(x) = 1 / (1 + exp(-x))
    
    Also known as Swish activation, commonly used in:
    - Diffusion models (U-Net architectures)
    - EfficientNet and modern CNNs
    - Some transformer variants
    
    Properties:
    - Smooth and non-monotonic
    - Self-gated (uses input to gate itself)
    - Unbounded above, bounded below
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass of SiLU activation
        
        Args:
            x: Input array of any shape
            
        Returns:
            Activated output with same shape as input
        """
        # Numerically stable sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        output = x * sigmoid
        
        # Cache for backward pass
        self.cache = (x, sigmoid)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass of SiLU activation
        
        Derivative: d(SiLU)/dx = σ(x) + x * σ(x) * (1 - σ(x))
                                = σ(x) * (1 + x * (1 - σ(x)))
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        x, sigmoid = self.cache
        
        # Derivative of SiLU
        silu_grad = sigmoid * (1.0 + x * (1.0 - sigmoid))
        
        return grad_output * silu_grad
    
    def __call__(self, x):
        return self.forward(x)


class Softmax:
    """
    Softmax Activation Function
    
    Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Converts logits to probability distribution:
    - Output values are in (0, 1)
    - Output values sum to 1
    
    Used in:
    - Attention mechanisms (for attention weights)
    - Classification heads
    
    Implementation uses numerical stability trick: subtract max before exp
    """
    
    def __init__(self, axis=-1):
        """
        Args:
            axis: Axis along which to apply softmax (default: -1, last axis)
        """
        self.axis = axis
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass of Softmax
        
        Args:
            x: Input logits, shape (..., num_classes)
            
        Returns:
            Probabilities with same shape as input
        """
        # Numerical stability: subtract max
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        
        # Compute exp and normalize
        exp_x = np.exp(x_shifted)
        output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        
        # Cache for backward pass
        self.cache = output
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass of Softmax
        
        For Jacobian of softmax:
        ∂y_i/∂x_j = y_i * (δ_ij - y_j)
        
        Where δ_ij is Kronecker delta (1 if i=j, 0 otherwise)
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input logits
        """
        output = self.cache
        
        # grad_input = output * (grad_output - (output * grad_output).sum())
        sum_term = np.sum(output * grad_output, axis=self.axis, keepdims=True)
        grad_input = output * (grad_output - sum_term)
        
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)


class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation
    
    ReLU(x) = max(0, x)
    
    Simple but effective activation function.
    Included for completeness, though GELU and SiLU are preferred in modern architectures.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass: max(0, x)"""
        output = np.maximum(0, x)
        self.cache = x
        return output
    
    def backward(self, grad_output):
        """Backward pass: gradient is 1 where x > 0, else 0"""
        x = self.cache
        grad_input = grad_output * (x > 0)
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)


# Standalone functions for convenience
def gelu(x):
    """Standalone GELU function (forward only)"""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    tanh_input = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(tanh_input))


def silu(x):
    """Standalone SiLU function (forward only)"""
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return x * sigmoid


def softmax(x, axis=-1):
    """Standalone Softmax function (forward only)"""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x):
    """Standalone ReLU function (forward only)"""
    return np.maximum(0, x)
