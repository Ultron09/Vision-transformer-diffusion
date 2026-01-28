"""
Training Example for Vision Transformer Diffusion Model

This script demonstrates how to train a ViT-based diffusion model from scratch
using pure NumPy.

Educational Note:
- Training will be slow compared to GPU-accelerated frameworks
- This is designed for understanding, not production use
- Start with small images (32x32) and few epochs for testing
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.unet_vit import SimplifiedViTUNet
from src.diffusion_process import GaussianDiffusion
from src.trainer import DiffusionTrainer, simple_dataloader
from src.optimizers import Adam
from src.data_utils import (
    create_synthetic_dataset,
    load_mnist_style_data,
    save_images_grid,
    SimpleDataset
)


def main():
    """Main training function"""
    
    # ========== Configuration ==========
    print("=" * 60)
    print("Vision Transformer Diffusion Model - Training")
    print("=" * 60)
    
    # Data configuration
    img_size = 32  # Image size (start small for NumPy!)
    channels = 3   # RGB images
    num_samples = 200  # Number of training samples
    batch_size = 4    # Small batch size for NumPy
    
    # Model configuration
    patch_size = 4
    embed_dim = 128  # Small for educational purposes
    depth = 4        # Number of transformer blocks
    num_heads = 4
    
    # Diffusion configuration
    num_timesteps = 100  # Fewer timesteps for faster training
    beta_schedule = 'linear'
    
    # Training configuration
    num_epochs = 5  # Start with few epochs
    learning_rate = 1e-3
    save_interval = 2  # Save every 2 epochs
    checkpoint_dir = 'checkpoints'
    
    print(f"\nConfiguration:")
    print(f"  Image size: {img_size}x{img_size}x{channels}")
    print(f"  Training samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: ViT (embed_dim={embed_dim}, depth={depth}, heads={num_heads})")
    print(f"  Diffusion timesteps: {num_timesteps}")
    print(f"  Training epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # ========== Create/Load Dataset ==========
    print(f"\n{'-' * 60}")
    print("Loading dataset...")
    
    # Option 1: Synthetic dataset
    dataset_type = 'synthetic'  # Change to 'mnist' for digit-like images
    
    if dataset_type == 'synthetic':
        images = create_synthetic_dataset(
            num_samples=num_samples,
            img_size=img_size,
            pattern='circles',  # 'random', 'circles', 'squares', 'gradients'
            channels=channels
        )
        print(f"  Created synthetic dataset with {num_samples} {img_size}x{img_size} images")
    
    elif dataset_type == 'mnist':
        images = load_mnist_style_data(
            num_samples=num_samples,
            img_size=img_size
        )
        channels = 1  # MNIST is grayscale
        print(f"  Created MNIST-style dataset with {num_samples} images")
    
    print(f"  Dataset shape: {images.shape}")
    print(f"  Value range: [{images.min():.2f}, {images.max():.2f}]")
    
    # Save sample images
    os.makedirs('samples', exist_ok=True)
    save_images_grid(images[:min(64, len(images))], 'samples/training_data.png', nrow=8)
    print(f"  Saved training data samples to samples/training_data.png")
    
    # ========== Initialize Model ==========
    print(f"\n{'-' * 60}")
    print("Initializing model...")
    
    model = SimplifiedViTUNet(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=channels,
        out_channels=channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    num_params = sum(p.size for p in model.parameters().values())
    print(f"  Model created with {num_params:,} parameters")
    
    # ========== Initialize Diffusion Process ==========
    print(f"\n{'-' * 60}")
    print("Initializing diffusion process...")
    
    diffusion = GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    print(f"  Diffusion process with {num_timesteps} timesteps")
    print(f"  Beta schedule: {beta_schedule}")
    
    # ========== Initialize Optimizer ==========
    optimizer = Adam(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        clip_norm=1.0
    )
    
    # ========== Initialize Trainer ==========
    print(f"\n{'-' * 60}")
    print("Initializing trainer...")
    
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer
    )
    
    # ========== Create DataLoader ==========
    def dataloader_generator():
        """Generator for training batches"""
        return simple_dataloader(images, batch_size=batch_size, shuffle=True)
    
    # ========== Training Loop ==========
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}\n")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        dataloader = dataloader_generator()
        
        for batch_idx, batch in enumerate(dataloader):
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                print(f"  Batch {batch_idx + 1}, Loss: {avg_loss:.6f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.6f}\n")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.npz')
            trainer.save_checkpoint(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}\n")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'model_final.npz')
    trainer.save_checkpoint(final_path)
    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Final model saved to {final_path}")
    print(f"{'=' * 60}")
    
    # ========== Quick Generation Test ==========
    print(f"\nGenerating test samples...")
    from src.sampler import DDPMSampler
    
    sampler = DDPMSampler(diffusion)
    test_samples = sampler.sample(model, (4, img_size, img_size, channels), clip_denoised=True)
    
    save_images_grid(test_samples, 'samples/generated_after_training.png', nrow=2)
    print(f"Saved generated samples to samples/generated_after_training.png")
    
    print(f"\nDone! Check the 'samples' and 'checkpoints' directories.")


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    main()
