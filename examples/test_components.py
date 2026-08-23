"""
Quick test script to verify all components work correctly

This script tests individual modules without running full training.
Useful for debugging and understanding how components interact.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_activations():
    """Test activation functions"""
    print("Testing activations...")
    from src.activations import GELU, SiLU, Softmax
    
    x = np.random.randn(2, 4)
    
    # Test GELU
    gelu = GELU()
    y = gelu.forward(x)
    grad = gelu.backward(np.ones_like(y))
    print(f"  ✓ GELU: input shape {x.shape} -> output shape {y.shape}")
    
    # Test SiLU
    silu = SiLU()
    y = silu.forward(x)
    grad = silu.backward(np.ones_like(y))
    print(f"  ✓ SiLU: input shape {x.shape} -> output shape {y.shape}")
    
    # Test Softmax
    softmax = Softmax()
    y = softmax.forward(x)
    assert np.allclose(y.sum(axis=-1), 1.0), "Softmax should sum to 1"
    print(f"  ✓ Softmax: output sums to 1.0")


def test_layers():
    """Test neural network layers"""
    print("\nTesting layers...")
    from src.layers import Linear, LayerNorm, Dropout
    
    batch_size, seq_len, dim = 2, 4, 8
    x = np.random.randn(batch_size, seq_len, dim)
    
    # Test Linear
    linear = Linear(dim, dim * 2)
    y = linear.forward(x)
    grad = linear.backward(np.ones_like(y))
    print(f"  ✓ Linear: {x.shape} -> {y.shape}")
    
    # Test LayerNorm
    ln = LayerNorm(dim)
    y = ln.forward(x)
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-5), "LayerNorm mean should be ~0"
    assert np.allclose(y.std(axis=-1), 1, atol=1e-2), "LayerNorm std should be ~1"
    print(f"  ✓ LayerNorm: mean≈0, std≈1")
    
    # Test Dropout
    dropout = Dropout(p=0.5)
    dropout.training = True
    y = dropout.forward(x)
    print(f"  ✓ Dropout: {x.shape} -> {y.shape}")


def test_attention():
    """Test multi-head self-attention"""
    print("\nTesting attention...")
    from src.attention import MultiHeadSelfAttention
    
    batch_size, seq_len, embed_dim = 2, 8, 64
    num_heads = 4
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    attn = MultiHeadSelfAttention(embed_dim, num_heads)
    y = attn.forward(x)
    grad = attn.backward(np.ones_like(y))
    
    assert y.shape == x.shape, f"Attention output shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ MultiHeadSelfAttention: {x.shape} -> {y.shape}")
    print(f"    Number of heads: {num_heads}, head_dim: {embed_dim // num_heads}")


def test_transformer():
    """Test transformer block"""
    print("\nTesting transformer...")
    from src.transformer import TransformerBlock, VisionTransformer
    
    batch_size, seq_len, embed_dim = 2, 16, 64
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Test TransformerBlock
    block = TransformerBlock(embed_dim, num_heads=4)
    y = block.forward(x)
    grad = block.backward(np.ones_like(y))
    print(f"  ✓ TransformerBlock: {x.shape} -> {y.shape}")
    
    # Test VisionTransformer
    img_size, patch_size, channels = 32, 4, 3
    images = np.random.randn(batch_size, img_size, img_size, channels)
    
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=channels,
        embed_dim=64,
        depth=2,
        num_heads=4
    )
    features = vit.forward(images)
    num_patches = (img_size // patch_size) ** 2
    expected_shape = (batch_size, num_patches, 64)
    assert features.shape == expected_shape, f"ViT output shape: {features.shape} vs {expected_shape}"
    print(f"  ✓ VisionTransformer: {images.shape} -> {features.shape}")


def test_diffusion():
    """Test diffusion process"""
    print("\nTesting diffusion process...")
    from src.diffusion_process import GaussianDiffusion
    
    batch_size, img_size, channels = 2, 32, 3
    images = np.random.randn(batch_size, img_size, img_size, channels)
    
    diffusion = GaussianDiffusion(num_timesteps=100, beta_schedule='linear')
    
    # Test forward diffusion
    t = np.array([10, 50])
    noisy_images = diffusion.q_sample(images, t)
    assert noisy_images.shape == images.shape
    print(f"  ✓ Forward diffusion: {images.shape} -> {noisy_images.shape}")
    
    # Test noise prediction
    noise = np.random.randn(*images.shape)
    pred_x0 = diffusion.predict_start_from_noise(noisy_images, t, noise)
    print(f"  ✓ Noise prediction: recovered x0 shape {pred_x0.shape}")
    
    print(f"    Beta schedule: {diffusion.betas[:5]} ... {diffusion.betas[-5:]}")
    print(f"    Alpha_cumprod range: [{diffusion.alphas_cumprod.min():.4f}, {diffusion.alphas_cumprod.max():.4f}]")


def test_time_embedding():
    """Test time embeddings"""
    print("\nTesting time embeddings...")
    from src.time_embedding import SinusoidalTimeEmbedding, TimeEmbeddingMLP
    
    batch_size = 4
    timesteps = np.array([0, 25, 50, 99])
    
    # Test sinusoidal embedding
    sin_embed = SinusoidalTimeEmbedding(embed_dim=128)
    embeddings = sin_embed.forward(timesteps)
    print(f"  ✓ SinusoidalTimeEmbedding: timesteps {timesteps.shape} -> {embeddings.shape}")
    
    # Test time MLP
    time_mlp = TimeEmbeddingMLP(time_embed_dim=128)
    processed = time_mlp.forward(timesteps)
    print(f"  ✓ TimeEmbeddingMLP: {timesteps.shape} -> {processed.shape}")


def test_unet_vit():
    """Test ViT-UNet denoising network"""
    print("\nTesting ViT-UNet...")
    from src.unet_vit import SimplifiedViTUNet
    
    batch_size = 2
    img_size, patch_size, channels = 32, 4, 3
    images = np.random.randn(batch_size, img_size, img_size, channels)
    timesteps = np.array([10, 50])
    
    model = SimplifiedViTUNet(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=channels,
        out_channels=channels,
        embed_dim=64,
        depth=2,
        num_heads=4
    )
    
    # Forward pass
    predicted_noise = model.forward(images, timesteps)
    assert predicted_noise.shape == images.shape, f"Output shape mismatch: {predicted_noise.shape}"
    print(f"  ✓ SimplifiedViTUNet forward: {images.shape} -> {predicted_noise.shape}")
    
    # Backward pass
    grad = model.backward(np.ones_like(predicted_noise))
    print(f"  ✓ SimplifiedViTUNet backward: computed gradients")
    
    # Count parameters
    params = model.parameters()
    total_params = sum(p.size for p in params.values())
    print(f"    Total parameters: {total_params:,}")


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    from src.unet_vit import SimplifiedViTUNet
    from src.diffusion_process import GaussianDiffusion
    from src.trainer import DiffusionTrainer
    from src.optimizers import Adam
    
    # Small model for testing
    batch_size = 2
    img_size, patch_size, channels = 32, 4, 3
    images = np.random.randn(batch_size, img_size, img_size, channels) * 0.5
    
    model = SimplifiedViTUNet(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=channels,
        embed_dim=32,
        depth=1,
        num_heads=2
    )
    
    diffusion = GaussianDiffusion(num_timesteps=50)
    optimizer = Adam(learning_rate=1e-3)
    trainer = DiffusionTrainer(model, diffusion, optimizer)
    
    # Single training step
    initial_loss = trainer.train_step(images)
    print(f"  ✓ Training step completed, loss: {initial_loss:.6f}")
    
    # Second step to see if loss changes
    second_loss = trainer.train_step(images)
    print(f"  ✓ Second step loss: {second_loss:.6f}")


def test_sampling():
    """Test sampling (generation)"""
    print("\nTesting sampling...")
    from src.unet_vit import SimplifiedViTUNet
    from src.diffusion_process import GaussianDiffusion
    from src.sampler import DDPMSampler
    
    # Very small model and few timesteps for quick test
    img_size, channels = 16, 3
    
    model = SimplifiedViTUNet(
        img_size=img_size,
        patch_size=4,
        in_channels=channels,
        embed_dim=32,
        depth=1,
        num_heads=2
    )
    
    diffusion = GaussianDiffusion(num_timesteps=10)  # Only 10 steps for testing
    sampler = DDPMSampler(diffusion)
    
    # Generate a single sample
    samples = sampler.sample(model, shape=(1, img_size, img_size, channels))
    print(f"  ✓ DDPM sampling: generated {samples.shape}")
    print(f"    Value range: [{samples.min():.2f}, {samples.max():.2f}]")


def test_advanced_features():
    """Test modern DiT additions: 2D RoPE, EDM Schedule, DPM-Solver++, EMA, FID, ModelExporter"""
    print("\nTesting advanced modules & modern DiT features...")
    import tempfile
    from src.rope import RotaryEmbedding2D
    from src.diffusion_process import EDMNoiseSchedule
    from src.sampler import DPMSolverPlusPlus
    from src.optimizers import EMAModelTracker
    from src.metrics import calculate_frechet_distance, calculate_inception_score
    from src.export import ModelExporter
    from src.unet_vit import SimplifiedViTUNet
    from src.diffusion_process import GaussianDiffusion
    
    # 1. RoPE-2D
    rope = RotaryEmbedding2D(dim=64)
    q = np.random.randn(2, 4, 16, 64)
    k = np.random.randn(2, 4, 16, 64)
    q_rot, k_rot = rope.apply_rotary_emb(q, k, 4, 4)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    print(f"  ✓ 2D Rotary Positional Embedding (RoPE): shape {q_rot.shape}")
    
    # 2. EDM Schedule
    edm = EDMNoiseSchedule()
    sigmas = edm.get_sigmas(num_steps=15)
    assert len(sigmas) == 16 and sigmas[-1] == 0.0
    print(f"  ✓ EDM Continuous Noise Schedule: {len(sigmas)} steps from σ_max={sigmas[0]} to σ_min={sigmas[-2]:.4f}")
    
    # 3. Fast DPM-Solver++
    model = SimplifiedViTUNet(img_size=16, patch_size=4, in_channels=3, embed_dim=32, depth=1, num_heads=2)
    diffusion = GaussianDiffusion(num_timesteps=20)
    dpm = DPMSolverPlusPlus(diffusion, order=2)
    samples_dpm = dpm.sample(model, shape=(1, 16, 16, 3), num_steps=5)
    assert samples_dpm.shape == (1, 16, 16, 3)
    print(f"  ✓ DPM-Solver++ (2nd Order Fast ODE): generated {samples_dpm.shape} in 5 steps")
    
    # 4. EMA Model Tracker
    ema = EMAModelTracker(model.parameters(), decay=0.999)
    ema.update(model.parameters(), step=10)
    assert len(ema.shadow_params) == len(model.parameters())
    print(f"  ✓ EMA Shadow Parameter Tracker: tracking {len(ema.shadow_params)} tensors")
    
    # 5. FID & Inception Score Metrics
    mu1, mu2 = np.zeros(32), np.ones(32) * 0.1
    sigma1, sigma2 = np.eye(32), np.eye(32)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    probs = np.ones((50, 10)) / 10.0
    is_mean, is_std = calculate_inception_score(probs, num_splits=5)
    print(f"  ✓ Generative Metrics: FID={fid:.4f}, IS={is_mean:.2f}±{is_std:.2f}")
    
    # 6. Model Exporter & SafeTensors
    tmp_path = os.path.join(tempfile.gettempdir(), 'test_vit_dit.npz')
    checksum = ModelExporter.save_checkpoint(tmp_path, model)
    meta = ModelExporter.load_checkpoint(tmp_path, model)
    print(f"  ✓ Checkpoint Serialization & SHA-256 Checksum: {checksum[:16]}...")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Vision Transformer Diffusion Model Components")
    print("=" * 60)
    
    np.random.seed(42)  # For reproducibility
    
    try:
        test_activations()
        test_layers()
        test_attention()
        test_transformer()
        test_diffusion()
        test_time_embedding()
        test_unet_vit()
        test_training_step()
        test_sampling()
        test_advanced_features()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        print("\nYou can now:")
        print("  1. Run 'python examples/train.py' to train a model")
        print("  2. Run 'python examples/generate.py' to generate images")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
