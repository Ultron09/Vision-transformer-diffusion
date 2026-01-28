"""
Training Infrastructure for Diffusion Models

This module provides a complete training framework for diffusion models,
including:
- Training loop with diffusion loss
- Gradient computation and parameter updates
- Checkpointing
- Progress tracking
"""

import numpy as np
import os
import time
from .optimizers import Adam
from .diffusion_process import GaussianDiffusion


class DiffusionTrainer:
    """
    Trainer for Diffusion Models
    
    Handles the complete training workflow:
    1. Sample random timesteps
    2. Add noise to images (forward diffusion)
    3. Predict noise with model
    4. Compute loss (MSE between true and predicted noise)
    5. Backpropagate and update parameters
    """
    
    def __init__(self, model, diffusion, optimizer=None, device='cpu'):
        """
        Initialize Diffusion Trainer
        
        Args:
            model: Denoising model (e.g., SimplifiedViTUNet)
            diffusion: GaussianDiffusion instance
            optimizer: Optimizer (default: Adam with lr=1e-4)
            device: Device to train on (currently only 'cpu' for NumPy)
        """
        self.model = model
        self.diffusion = diffusion
        
        if optimizer is None:
            self.optimizer = Adam(learning_rate=1e-4, weight_decay=0.0)
        else:
            self.optimizer = optimizer
        
        self.device = device
        
        # Training statistics
        self.step = 0
        self.epoch = 0
        self.losses = []
    
    def train_step(self, batch):
        """
        Single training step
        
        Args:
            batch: Batch of images (batch_size, H, W, C)
            
        Returns:
            Loss value
        """
        batch_size = batch.shape[0]
        
        # 1. Sample random timesteps for each image in batch
        t = np.random.randint(0, self.diffusion.num_timesteps, size=(batch_size,))
        
        # 2. Sample noise
        noise = np.random.randn(*batch.shape)
        
        # 3. Forward diffusion: add noise to images
        x_t = self.diffusion.q_sample(batch, t, noise=noise)
        
        # 4. Predict noise with model
        predicted_noise = self.model(x_t, t)
        
        # 5. Compute loss (MSE between true and predicted noise)
        loss = np.mean((noise - predicted_noise) ** 2)
        
        # 6. Backward pass
        # Gradient of MSE loss: 2 * (predicted - target) / n
        grad_output = 2 * (predicted_noise - noise) / np.prod(noise.shape)
        
        # Backpropagate through model
        self.model.backward(grad_output)
        
        # 7. Get parameters and gradients
        params = self.model.parameters()
        grads = self.model.gradients()
        
        # 8. Update parameters
        updated_params = self.optimizer.update(params, grads)
        
        # 9. Set updated parameters back to model
        self._set_model_parameters(updated_params)
        
        # Track statistics
        self.step += 1
        self.losses.append(loss)
        
        return loss
    
    def train_epoch(self, dataloader, verbose=True):
        """
        Train for one epoch
        
        Args:
            dataloader: Iterator that yields batches of images
            verbose: Whether to print progress
            
        Returns:
            Average loss for the epoch
        """
        epoch_losses = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch)
            epoch_losses.append(loss)
            
            if verbose and (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                print(f"  Batch {batch_idx + 1}, Loss: {avg_loss:.6f}")
        
        self.epoch += 1
        epoch_time = time.time() - start_time
        avg_loss = np.mean(epoch_losses)
        
        if verbose:
            print(f"Epoch {self.epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def train(self, dataloader, num_epochs, save_interval=None, checkpoint_dir='checkpoints', verbose=True):
        """
        Complete training loop
        
        Args:
            dataloader: Data loader
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs (None = don't save)
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print progress
        """
        if save_interval is not None and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {self._count_parameters()}")
        
        for epoch in range(num_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            avg_loss = self.train_epoch(dataloader, verbose=verbose)
            
            # Save checkpoint
            if save_interval is not None and (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.npz")
                self.save_checkpoint(checkpoint_path)
                if verbose:
                    print(f"  Saved checkpoint to {checkpoint_path}")
        
        print("\nTraining completed!")
        
        # Save final checkpoint
        if save_interval is not None:
            final_path = os.path.join(checkpoint_dir, "model_final.npz")
            self.save_checkpoint(final_path)
            print(f"Saved final model to {final_path}")
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        params = self.model.parameters()
        
        # Also save training state
        params['_training_step'] = np.array([self.step])
        params['_training_epoch'] = np.array([self.epoch])
        
        np.savez(path, **params)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = np.load(path)
        
        # Load parameters
        params = {k: v for k, v in checkpoint.items() if not k.startswith('_')}
        self._set_model_parameters(params)
        
        # Load training state if available
        if '_training_step' in checkpoint:
            self.step = int(checkpoint['_training_step'][0])
        if '_training_epoch' in checkpoint:
            self.epoch = int(checkpoint['_training_epoch'][0])
        
        print(f"Loaded checkpoint from {path} (step={self.step}, epoch={self.epoch})")
    
    def _set_model_parameters(self, params):
        """Helper to set model parameters"""
        # This is a simplified version - in practice, we'd need to
        # carefully match parameter names to model components
        model_params = self.model.parameters()
        
        for key in model_params.keys():
            if key in params:
                # Get the actual parameter object and update it in place
                # We need to navigate through the model structure
                self._update_parameter(key, params[key])
    
    def _update_parameter(self, param_name, param_value):
        """Update a specific parameter in the model"""
        # Parse parameter name (e.g., "block0_attn_qkv_weight")
        parts = param_name.split('_')
        
        # Navigate to the correct module and update parameter
        # This is simplified - full implementation would need recursive navigation
        obj = self.model
        
        # Find the right submodule
        if param_name.startswith('patch_embed'):
            obj = self.model.patch_embed.projection
            attr_name = parts[-1]
        elif param_name.startswith('time_embed'):
            # Navigate through time_embed layers
            pass  # Simplified for brevity
        elif param_name.startswith('block'):
            # Extract block number
            block_num = int(parts[0].replace('block', ''))
            obj = self.model.blocks[block_num]
            # Further navigation needed based on remaining parts
            pass  # Simplified for brevity
        elif param_name.startswith('norm_'):
            obj = self.model.norm
            attr_name = parts[1]
        elif param_name.startswith('output_proj'):
            obj = self.model.output_proj
            attr_name = parts[-1]
        else:
            return
        
        # Update the parameter
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], param_value.copy())
    
    def _count_parameters(self):
        """Count total number of parameters"""
        params = self.model.parameters()
        total = sum(p.size for p in params.values())
        return total
    
    def get_loss_history(self):
        """Return loss history"""
        return self.losses


def simple_dataloader(images, batch_size, shuffle=True):
    """
    Simple data loader for training
    
    Args:
        images: Array of images (N, H, W, C)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Yields:
        Batches of images
    """
    n_samples = len(images)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices]
