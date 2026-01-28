"""
Optimization Algorithms for Training Neural Networks

This module implements optimization algorithms used for training deep neural networks,
specifically Adam optimizer which is the standard choice for transformers and diffusion models.

Mathematical Background:
- Adam: Adaptive Moment Estimation, combines momentum and RMSprop
- Uses first moment (momentum) and second moment (variance) estimates
- Includes bias correction for early iterations
"""

import numpy as np


class Adam:
    """
    Adam Optimizer (Adaptive Moment Estimation)
    
    Algorithm:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        # First moment (momentum)
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       # Second moment (variance)
        m̂_t = m_t / (1 - β₁^t)                     # Bias correction
        v̂_t = v_t / (1 - β₂^t)                     # Bias correction
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)     # Parameter update
    
    Where:
        - g_t: Gradient at time t
        - α: Learning rate
        - β₁: Decay rate for first moment (default: 0.9)
        - β₂: Decay rate for second moment (default: 0.999)
        - ε: Small constant for numerical stability (default: 1e-8)
    
    Reference:
        Kingma & Ba, 2014: "Adam: A Method for Stochastic Optimization"
    """
    
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0, clip_norm=None):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: Step size (α)
            beta1: Decay rate for first moment estimate
            beta2: Decay rate for second moment estimate
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient (default: 0.0)
            clip_norm: Maximum gradient norm for clipping (default: None)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        
        # State variables (will be initialized on first update)
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
    
    def update(self, params, grads):
        """
        Update parameters using Adam algorithm
        
        Args:
            params: Dictionary of parameters {name: array}
            grads: Dictionary of gradients {name: array}
            
        Returns:
            Updated parameters dictionary
        """
        self.t += 1
        
        # Apply gradient clipping if specified
        if self.clip_norm is not None:
            grads = self._clip_gradients(grads, self.clip_norm)
        
        updated_params = {}
        
        for name, param in params.items():
            # Skip if no gradient
            if name not in grads or grads[name] is None:
                updated_params[name] = param
                continue
            
            grad = grads[name]
            
            # Apply weight decay (L2 regularization) if specified
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize moment estimates on first update
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def _clip_gradients(self, grads, max_norm):
        """
        Clip gradients by global norm
        
        Scales down gradients if their global norm exceeds max_norm:
        
        global_norm = sqrt(Σ ||g_i||²)
        if global_norm > max_norm:
            g_i = g_i * (max_norm / global_norm)
        
        Args:
            grads: Dictionary of gradients
            max_norm: Maximum allowed norm
            
        Returns:
            Clipped gradients dictionary
        """
        # Compute global norm
        total_norm = 0.0
        for grad in grads.values():
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            clipped_grads = {
                name: grad * clip_coef if grad is not None else None
                for name, grad in grads.items()
            }
            return clipped_grads
        
        return grads
    
    def get_lr(self):
        """Get current learning rate"""
        return self.learning_rate
    
    def set_lr(self, learning_rate):
        """Set learning rate"""
        self.learning_rate = learning_rate
    
    def zero_grad(self):
        """Reset optimizer state (useful for fresh start)"""
        self.m = {}
        self.v = {}
        self.t = 0


class SGD:
    """
    Stochastic Gradient Descent with Momentum
    
    Algorithm:
        v_t = μ * v_{t-1} + g_t              # Momentum update
        θ_t = θ_{t-1} - α * v_t              # Parameter update
    
    Simpler than Adam but can work well with proper learning rate scheduling.
    """
    
    def __init__(self, learning_rate=1e-2, momentum=0.9, weight_decay=0.0, clip_norm=None):
        """
        Initialize SGD optimizer
        
        Args:
            learning_rate: Step size
            momentum: Momentum coefficient (0 for vanilla SGD)
            weight_decay: L2 regularization coefficient
            clip_norm: Maximum gradient norm for clipping
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        
        self.velocity = {}  # Momentum buffer
    
    def update(self, params, grads):
        """Update parameters using SGD with momentum"""
        # Apply gradient clipping if specified
        if self.clip_norm is not None:
            grads = self._clip_gradients(grads, self.clip_norm)
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in grads or grads[name] is None:
                updated_params[name] = param
                continue
            
            grad = grads[name]
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize velocity on first update
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(param)
            
            # Update velocity with momentum
            self.velocity[name] = self.momentum * self.velocity[name] + grad
            
            # Update parameters
            updated_params[name] = param - self.learning_rate * self.velocity[name]
        
        return updated_params
    
    def _clip_gradients(self, grads, max_norm):
        """Clip gradients by global norm"""
        total_norm = 0.0
        for grad in grads.values():
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            return {name: grad * clip_coef if grad is not None else None 
                    for name, grad in grads.items()}
        return grads
    
    def get_lr(self):
        return self.learning_rate
    
    def set_lr(self, learning_rate):
        self.learning_rate = learning_rate


class LRScheduler:
    """
    Learning Rate Schedulers
    
    Provides various learning rate scheduling strategies:
    - Cosine Annealing: Smooth cosine decay
    - Linear Warmup: Gradual increase from 0 to base_lr
    - Step Decay: Multiplicative decay at specified steps
    """
    
    @staticmethod
    def cosine_annealing(optimizer, current_step, total_steps, min_lr=0.0):
        """
        Cosine Annealing Learning Rate Schedule
        
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t / T))
        
        Where t is current step and T is total steps.
        
        Args:
            optimizer: Optimizer instance
            current_step: Current training step
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
        """
        base_lr = optimizer.get_lr()
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * current_step / total_steps))
        optimizer.set_lr(lr)
        return lr
    
    @staticmethod
    def linear_warmup(optimizer, current_step, warmup_steps, base_lr):
        """
        Linear Warmup Learning Rate Schedule
        
        Linearly increases learning rate from 0 to base_lr over warmup_steps.
        
        Args:
            optimizer: Optimizer instance
            current_step: Current training step
            warmup_steps: Number of warmup steps
            base_lr: Target learning rate after warmup
        """
        if current_step < warmup_steps:
            lr = base_lr * (current_step / warmup_steps)
            optimizer.set_lr(lr)
            return lr
        return base_lr
    
    @staticmethod
    def warmup_cosine(optimizer, current_step, warmup_steps, total_steps, base_lr, min_lr=0.0):
        """
        Warmup + Cosine Annealing Schedule
        
        Combines linear warmup with cosine annealing decay.
        Common in transformer training.
        
        Args:
            optimizer: Optimizer instance
            current_step: Current training step
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            base_lr: Maximum learning rate (after warmup)
            min_lr: Minimum learning rate (at end)
        """
        if current_step < warmup_steps:
            # Linear warmup
            lr = base_lr * (current_step / warmup_steps)
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        
        optimizer.set_lr(lr)
        return lr
    
    @staticmethod
    def step_decay(optimizer, current_step, step_size, gamma=0.1):
        """
        Step Decay Learning Rate Schedule
        
        Multiplies learning rate by gamma every step_size steps.
        
        Args:
            optimizer: Optimizer instance
            current_step: Current training step
            step_size: Steps between each decay
            gamma: Multiplicative factor
        """
        base_lr = optimizer.get_lr()
        num_decays = current_step // step_size
        lr = base_lr * (gamma ** num_decays)
        optimizer.set_lr(lr)
        return lr
