"""
Sampling Algorithms for Diffusion Models

Implements various sampling methods to generate images from trained diffusion models:
1. DDPM Sampling: Full reverse diffusion process
2. DDIM Sampling: Deterministic, faster sampling (optional for education)

These samplers take a trained denoising model and iteratively denoise
random noise into coherent images.
"""

import numpy as np
from .diffusion_process import GaussianDiffusion


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Model) Sampler
    
    Implements the reverse diffusion process as described in Ho et al., 2020.
    
    Algorithm:
        1. Start with x_T ~ N(0, I) (pure noise)
        2. For t = T, T-1, ..., 1:
            - Predict noise ε_θ(x_t, t)
            - Compute mean μ_θ(x_t, t)
            - Sample x_{t-1} ~ N(μ_θ, σ_t²I)
        3. Return x_0
    
    This is a stochastic sampler (includes noise at each step).
    """
    
    def __init__(self, diffusion):
        """
        Initialize DDPM Sampler
        
        Args:
            diffusion: GaussianDiffusion instance
        """
        self.diffusion = diffusion
    
    def sample(self, model, shape, clip_denoised=True, progress_callback=None):
        """
        Generate samples using DDPM sampling
        
        Args:
            model: Trained denoising model
            shape: Shape of images to generate (batch_size, H, W, C)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            progress_callback: Optional callback function(step, x_t) for visualization
            
        Returns:
            Generated images of shape (batch_size, H, W, C)
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = np.random.randn(*shape)
        
        # Reverse diffusion
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = np.full((batch_size,), i, dtype=np.int32)
            
            # Denoise one step
            x = self.diffusion.p_sample(model, x, t, clip_denoised=clip_denoised)
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(i, x)
        
        return x
    
    def sample_progressive(self, model, shape, clip_denoised=True, save_interval=50):
        """
        Generate samples and save intermediate steps
        
        Args:
            model: Trained denoising model
            shape: Shape of images to generate
            clip_denoised: Whether to clip predicted x_0
            save_interval: Save image every N steps
            
        Returns:
            List of (timestep, image) tuples showing progressive generation
        """
        batch_size = shape[0]
        x = np.random.randn(*shape)
        
        progressive_images = [(self.diffusion.num_timesteps, x.copy())]
        
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = np.full((batch_size,), i, dtype=np.int32)
            x = self.diffusion.p_sample(model, x, t, clip_denoised=clip_denoised)
            
            if i % save_interval == 0 or i == 0:
                progressive_images.append((i, x.copy()))
        
        return progressive_images


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Model) Sampler
    
    Implements deterministic sampling as described in Song et al., 2020.
    
    Key advantages:
    - Deterministic (same noise -> same image)
    - Faster sampling (can skip timesteps)
    - Better for image editing tasks
    
    Algorithm uses a non-Markovian forward process that allows
    skipping timesteps while maintaining sample quality.
    """
    
    def __init__(self, diffusion, eta=0.0):
        """
        Initialize DDIM Sampler
        
        Args:
            diffusion: GaussianDiffusion instance
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
        """
        self.diffusion = diffusion
        self.eta = eta
    
    def sample(self, model, shape, num_steps=None, clip_denoised=True, progress_callback=None):
        """
        Generate samples using DDIM sampling
        
        Args:
            model: Trained denoising model
            shape: Shape of images to generate
            num_steps: Number of sampling steps (None = use all timesteps)
            clip_denoised: Whether to clip predicted x_0
            progress_callback: Optional callback for progress
            
        Returns:
            Generated images
        """
        batch_size = shape[0]
        
        # Determine which timesteps to use
        if num_steps is None:
            num_steps = self.diffusion.num_timesteps
            timesteps = np.arange(self.diffusion.num_timesteps)
        else:
            # Uniformly subsample timesteps
            skip = self.diffusion.num_timesteps // num_steps
            timesteps = np.arange(0, self.diffusion.num_timesteps, skip)
        
        # Start from pure noise
        x = np.random.randn(*shape)
        
        # Reverse diffusion with DDIM
        for i in reversed(range(len(timesteps))):
            t = np.full((batch_size,), timesteps[i], dtype=np.int32)
            
            # Get previous timestep
            if i > 0:
                t_prev = np.full((batch_size,), timesteps[i - 1], dtype=np.int32)
            else:
                t_prev = np.full((batch_size,), -1, dtype=np.int32)
            
            # DDIM denoising step
            x = self._ddim_step(model, x, t, t_prev, clip_denoised)
            
            if progress_callback is not None:
                progress_callback(timesteps[i], x)
        
        return x
    
    def _ddim_step(self, model, x_t, t, t_prev, clip_denoised=True):
        """
        Single DDIM denoising step
        
        Args:
            model: Denoising model
            x_t: Noisy image at timestep t
            t: Current timestep
            t_prev: Previous timestep (t-1 in subsampled sequence)
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            x_{t_prev}: Denoised image at previous timestep
        """
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Predict x_0
        pred_x_start = self.diffusion.predict_start_from_noise(x_t, t, predicted_noise)
        
        # Clip if requested
        if clip_denoised:
            pred_x_start = np.clip(pred_x_start, -1, 1)
        
        # Get alpha values
        alpha_t = self.diffusion.alphas_cumprod[t]
        
        if t_prev[0] >= 0:
            alpha_t_prev = self.diffusion.alphas_cumprod[t_prev]
        else:
            alpha_t_prev = np.ones_like(alpha_t)
        
        # Reshape for broadcasting
        alpha_t = alpha_t.reshape(-1, 1, 1, 1)
        alpha_t_prev = alpha_t_prev.reshape(-1, 1, 1, 1)
        
        # Compute variance
        sigma_t = self.eta * np.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        
        # Compute direction pointing to x_t
        dir_xt = np.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise
        
        # Compute x_{t-1}
        noise = np.random.randn(*x_t.shape) if self.eta > 0 else 0
        x_prev = np.sqrt(alpha_t_prev) * pred_x_start + dir_xt + sigma_t * noise
        
        return x_prev


def generate_samples(model, diffusion, num_samples, img_size, channels=3,
                     sampler='ddpm', **sampler_kwargs):
    """
    Convenience function to generate samples
    
    Args:
        model: Trained denoising model
        diffusion: GaussianDiffusion instance
        num_samples: Number of images to generate
        img_size: Image size (height and width)
        channels: Number of channels
        sampler: Sampler type ('ddpm' or 'ddim')
        **sampler_kwargs: Additional arguments for sampler
        
    Returns:
        Generated images (num_samples, img_size, img_size, channels)
    """
    shape = (num_samples, img_size, img_size, channels)
    
    if sampler == 'ddpm':
        sampler_obj = DDPMSampler(diffusion)
        return sampler_obj.sample(model, shape, **sampler_kwargs)
    elif sampler == 'ddim':
        sampler_obj = DDIMSampler(diffusion, eta=sampler_kwargs.pop('eta', 0.0))
        return sampler_obj.sample(model, shape, **sampler_kwargs)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


def visualize_generation_process(model, diffusion, img_size, channels=3,
                                   save_interval=50):
    """
    Generate a single image and return intermediate steps for visualization
    
    Args:
        model: Trained denoising model
        diffusion: GaussianDiffusion instance
        img_size: Image size
        channels: Number of channels
        save_interval: Interval to save intermediate images
        
    Returns:
        List of (timestep, image) tuples
    """
    shape = (1, img_size, img_size, channels)
    sampler = DDPMSampler(diffusion)
    return sampler.sample_progressive(model, shape, save_interval=save_interval)
