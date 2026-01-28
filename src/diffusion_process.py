"""
Gaussian Diffusion Process

Implements the forward and reverse diffusion processes used in DDPM
(Denoising Diffusion Probabilistic Models).

Forward Process (Adding Noise):
    q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t)I)
    
    Where:
    - x_0: Original clean image
    - x_t: Noisy image at timestep t
    - ᾱ_t: Cumulative product of (1 - β_t)
    - β_t: Noise schedule

Reverse Process (Denoising):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    
    The model predicts noise ε_θ(x_t, t) to compute mean μ_θ

Training Objective:
    L = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]
    
    Predict the noise that was added to get x_t from x_0.

Reference:
    Ho et al., 2020: "Denoising Diffusion Probabilistic Models"
    Song et al., 2020: "Denoising Diffusion Implicit Models"
"""

import numpy as np


class GaussianDiffusion:
    """
    Gaussian Diffusion Process for DDPM
    
    Implements:
    - Forward diffusion (noise addition with schedule)
    - Reverse diffusion (denoising steps)
    - Various noise schedules (linear, cosine)
    - Sampling algorithms (DDPM, DDIM)
    """
    
    def __init__(self, num_timesteps=1000, beta_schedule='linear',
                 beta_start=0.0001, beta_end=0.02, s=0.008):
        """
        Initialize Gaussian Diffusion
        
        Args:
            num_timesteps: Number of diffusion timesteps (T)
            beta_schedule: Type of noise schedule ('linear' or 'cosine')
            beta_start: Starting value of beta (for linear schedule)
            beta_end: Ending value of beta (for linear schedule)
            s: Small offset for cosine schedule
        """
        self.num_timesteps = num_timesteps
        
        # Compute beta schedule
        if beta_schedule == 'linear':
            self.betas = self._linear_beta_schedule(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, s)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute useful quantities for diffusion
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for reverse diffusion (posterior mean)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Clip for numerical stability
        self.posterior_variance = np.clip(self.posterior_variance, 1e-20, None)
        
        self.posterior_log_variance_clipped = np.log(self.posterior_variance)
        
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) /
            (1.0 - self.alphas_cumprod)
        )
    
    def _linear_beta_schedule(self, beta_start, beta_end, num_timesteps):
        """
        Linear schedule from beta_start to beta_end
        
        Simple linear interpolation between start and end values.
        Used in original DDPM paper.
        """
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    
    def _cosine_beta_schedule(self, num_timesteps, s=0.008):
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
        
        Better than linear schedule as it adds less noise at beginning and end.
        
        f(t) = cos((t/T + s) / (1 + s) * π/2)²
        α_t = f(t) / f(0)
        β_t = 1 - α_t / α_{t-1}
        """
        steps = num_timesteps + 1
        t = np.linspace(0, num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos((t / num_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0)
        
        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        
        Where ε ~ N(0, I)
        
        Args:
            x_start: Clean images (batch_size, ...)
            t: Timesteps (batch_size,)
            noise: Optional pre-generated noise (same shape as x_start)
            
        Returns:
            Noisy images x_t
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        
        # Extract coefficients for batch of timesteps
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Apply forward diffusion
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)
        
        This is the "true" posterior when x_0 is known (used for computing loss).
        
        Args:
            x_start: Clean images
            x_t: Noisy images at timestep t
            t: Timesteps
            
        Returns:
            (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise
        
        x_0 = (x_t - √(1 - ᾱ_t) * ε) / √(ᾱ_t)
        
        Args:
            x_t: Noisy images at timestep t
            t: Timesteps
            noise: Predicted noise from model
            
        Returns:
            Predicted x_0
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        """
        Compute predicted mean and variance for p(x_{t-1} | x_t)
        
        Uses model to predict noise, then computes x_0 and posterior mean/variance.
        
        Args:
            model: Denoising model (takes x_t and t, returns predicted noise)
            x_t: Noisy images at timestep t
            t: Timesteps
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            
        Returns:
            (model_mean, posterior_variance, posterior_log_variance, pred_x_start)
        """
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Predict x_0
        pred_x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        # Clip if requested
        if clip_denoised:
            pred_x_start = np.clip(pred_x_start, -1, 1)
        
        # Compute posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(pred_x_start, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance, pred_x_start
    
    def p_sample(self, model, x_t, t, clip_denoised=True):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) using the model
        
        This is one denoising step.
        
        Args:
            model: Denoising model
            x_t: Noisy images at timestep t
            t: Timesteps
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            Denoised images x_{t-1}
        """
        # Get mean and variance
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised
        )
        
        # Sample noise
        noise = np.random.randn(*x_t.shape)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).astype(np.float32)
        # Reshape for broadcasting
        for _ in range(len(x_t.shape) - 1):
            nonzero_mask = nonzero_mask[:, np.newaxis]
        
        # Sample x_{t-1}
        pred_img = model_mean + nonzero_mask * np.exp(0.5 * model_log_variance) * noise
        
        return pred_img
    
    def p_sample_loop(self, model, shape, clip_denoised=True, progress=False):
        """
        Generate samples by running full reverse diffusion process
        
        Start from random noise and iteratively denoise for T steps.
        
        Args:
            model: Denoising model
            shape: Shape of images to generate (batch_size, H, W, C)
            clip_denoised: Whether to clip predicted x_0
            progress: Whether to show progress (currently unused in NumPy version)
            
        Returns:
            Generated images
        """
        batch_size = shape[0]
        
        # Start from random noise
        img = np.random.randn(*shape)
        
        # Reverse diffusion
        for i in reversed(range(self.num_timesteps)):
            t = np.full((batch_size,), i, dtype=np.int32)
            img = self.p_sample(model, img, t, clip_denoised=clip_denoised)
        
        return img
    
    def training_losses(self, model, x_start, t, noise=None):
        """
        Compute training loss for diffusion model
        
        Loss = ||ε - ε_θ(x_t, t)||²
        
        Where:
        - ε: True noise added to x_0
        - ε_θ: Noise predicted by model
        
        Args:
            model: Denoising model
            x_start: Clean images
            t: Timesteps
            noise: Optional pre-generated noise
            
        Returns:
            MSE loss between true and predicted noise
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        
        # Forward diffusion: add noise to get x_t
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Compute MSE loss
        loss = np.mean((noise - predicted_noise) ** 2)
        
        return loss
    
    def _extract(self, arr, timesteps, broadcast_shape):
        """
        Extract values from array at indices specified by timesteps,
        and reshape for broadcasting.
        
        Args:
            arr: Array to extract from (length T)
            timesteps: Indices to extract (batch_size,)
            broadcast_shape: Shape to broadcast to (batch_size, ...)
            
        Returns:
            Extracted and reshaped values
        """
        # Extract values
        res = arr[timesteps]
        
        # Reshape for broadcasting: (batch_size, 1, 1, ...)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., np.newaxis]
        
        return res
