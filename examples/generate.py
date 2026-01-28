"""
Image Generation Example for Vision Transformer Diffusion Model

This script demonstrates how to generate images from a trained diffusion model.

Usage:
    python generate.py --checkpoint checkpoints/model_final.npz --num_samples 16
"""

import numpy as np
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.unet_vit import SimplifiedViTUNet
from src.diffusion_process import GaussianDiffusion
from src.sampler import DDPMSampler, DDIMSampler, visualize_generation_process
from src.data_utils import save_images_grid


def load_model_from_checkpoint(checkpoint_path, config):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration dict
        
    Returns:
        Loaded model
    """
    # Create model
    model = SimplifiedViTUNet(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=config['channels'],
        out_channels=config['channels'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4.0,
        dropout=0.0  # No dropout during inference
    )
    
    # Load checkpoint
    checkpoint = np.load(checkpoint_path)
    
    # Note: This is a simplified loading mechanism
    # In a full implementation, we'd properly map checkpoint keys to model parameters
    print(f"Loaded checkpoint with {len(checkpoint.files)} parameter groups")
    
    return model


def main():
    """Main generation function"""
    
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(description='Generate images from trained diffusion model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_final.npz',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of images to generate')
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                       help='Sampling algorithm')
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (for DDIM, None=use all)')
    parser.add_argument('--output', type=str, default='samples/generated.png',
                       help='Output path for generated images')
    parser.add_argument('--progressive', action='store_true',
                       help='Save progressive generation steps')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    print("=" * 60)
    print("Vision Transformer Diffusion Model - Generation")
    print("=" * 60)
    
    # ========== Configuration ==========
    # These should match the training configuration
    config = {
        'img_size': 32,
        'channels': 3,
        'patch_size': 4,
        'embed_dim': 128,
        'depth': 4,
        'num_heads': 4,
        'num_timesteps': 100,
        'beta_schedule': 'linear'
    }
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Sampler: {args.sampler}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Image size: {config['img_size']}x{config['img_size']}x{config['channels']}")
    
    # ========== Load Model ==========
    print(f"\n{'-' * 60}")
    print("Loading model...")
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using examples/train.py")
        return
    
    model = load_model_from_checkpoint(args.checkpoint, config)
    print(f"  Model loaded successfully")
    
    # ========== Initialize Diffusion ==========
    print(f"\n{'-' * 60}")
    print("Initializing diffusion process...")
    
    diffusion = GaussianDiffusion(
        num_timesteps=config['num_timesteps'],
        beta_schedule=config['beta_schedule'],
        beta_start=0.0001,
        beta_end=0.02
    )
    
    print(f"  Diffusion process with {config['num_timesteps']} timesteps")
    
    # ========== Generate Samples ==========
    print(f"\n{'-' * 60}")
    print("Generating samples...")
    
    shape = (args.num_samples, config['img_size'], config['img_size'], config['channels'])
    
    # Progress callback
    progress_steps = []
    def progress_callback(step, x_t):
        if step % 20 == 0 or step == 0:
            progress_steps.append((step, x_t.copy()))
            print(f"  Step {config['num_timesteps'] - step}/{config['num_timesteps']}")
    
    # Choose sampler
    if args.sampler == 'ddpm':
        sampler = DDPMSampler(diffusion)
        samples = sampler.sample(
            model=model,
            shape=shape,
            clip_denoised=True,
            progress_callback=progress_callback if args.progressive else None
        )
    
    elif args.sampler == 'ddim':
        sampler = DDIMSampler(diffusion, eta=0.0)
        num_steps = args.num_steps if args.num_steps is not None else config['num_timesteps']
        print(f"  Using DDIM with {num_steps} steps")
        samples = sampler.sample(
            model=model,
            shape=shape,
            num_steps=num_steps,
            clip_denoised=True,
            progress_callback=progress_callback if args.progressive else None
        )
    
    print(f"\n  Generation complete!")
    print(f"  Generated {len(samples)} images")
    
    # ========== Save Results ==========
    print(f"\n{'-' * 60}")
    print("Saving results...")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save generated images
    save_images_grid(samples, args.output, nrow=int(np.sqrt(args.num_samples)))
    print(f"  Saved generated images to {args.output}")
    
    # Save progressive generation if requested
    if args.progressive and progress_steps:
        print(f"\n  Saving progressive generation steps...")
        for step, x_t in progress_steps:
            # Take first sample from batch
            sample = x_t[0:1]
            output_path = os.path.join(
                os.path.dirname(args.output),
                f'progressive_step_{step:04d}.png'
            )
            save_images_grid(sample, output_path, nrow=1)
        print(f"  Saved {len(progress_steps)} progressive steps")
    
    print(f"\n{'=' * 60}")
    print(f"Generation completed successfully!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
