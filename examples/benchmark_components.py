"""
Vision Transformer Diffusion Component Benchmarks & Gradient Verification Suite

Performs:
1. High-precision numerical gradient checks via two-sided central finite differences
2. Latency, forward/backward throughput, and FLOPs profiling across all architectural blocks
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src
from src.layers import Linear, LayerNorm, PatchEmbedding
from src.attention import MultiHeadSelfAttention
from src.transformer import TransformerBlock
from src.unet_vit import SimplifiedViTUNet
from src.diffusion_process import GaussianDiffusion


def numerical_gradient_check(forward_fn, x, eps=1e-5):
    """
    Compute central finite-difference numerical gradient:
        grad_num = (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    grad_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]
        
        x[idx] = orig_val + eps
        loss_plus = np.sum(forward_fn(x))
        
        x[idx] = orig_val - eps
        loss_minus = np.sum(forward_fn(x))
        
        x[idx] = orig_val
        grad_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()
        
    return grad_num


def verify_gradients():
    print("=" * 70)
    print("1. Running Numerical Gradient Verification (Central Finite Differences)")
    print("=" * 70)
    
    # 1. Linear Layer Gradient Check
    linear = Linear(8, 4)
    x = np.random.randn(2, 8).astype(np.float64)
    linear.weight = linear.weight.astype(np.float64)
    if linear.use_bias:
        linear.bias = linear.bias.astype(np.float64)
    
    # Forward & Analytical Backward
    out = linear(x)
    grad_out = np.ones_like(out)
    grad_x_anal = linear.backward(grad_out)
    
    # Numerical
    grad_x_num = numerical_gradient_check(lambda inp: linear(inp), x)
    rel_error = np.linalg.norm(grad_x_anal - grad_x_num) / (np.linalg.norm(grad_x_anal) + np.linalg.norm(grad_x_num) + 1e-12)
    print(f"  [Linear Layer]        Relative Error: {rel_error:.2e} -> {'PASSED' if rel_error < 1e-4 else 'FAILED'}")

    # 2. LayerNorm Gradient Check
    ln = LayerNorm(8)
    x_ln = np.random.randn(2, 8).astype(np.float64)
    ln.gamma = ln.gamma.astype(np.float64)
    ln.beta = ln.beta.astype(np.float64)
    grad_out_ln = np.random.randn(2, 8).astype(np.float64)
    out_ln = ln(x_ln)
    grad_ln_anal = ln.backward(grad_out_ln)
    grad_ln_num = numerical_gradient_check(lambda inp: ln(inp) * grad_out_ln, x_ln)
    rel_error_ln = np.linalg.norm(grad_ln_anal - grad_ln_num) / (np.linalg.norm(grad_ln_anal) + np.linalg.norm(grad_ln_num) + 1e-12)
    print(f"  [Layer Normalization] Relative Error: {rel_error_ln:.2e} -> {'PASSED' if rel_error_ln < 1e-4 else 'FAILED'}")


def benchmark_components(num_iterations=20):
    print("\n" + "=" * 70)
    print("2. Computational Latency & Memory Profiling")
    print("=" * 70)
    print(f"{'Component':<30} | {'Forward (ms)':<14} | {'Backward (ms)':<14} | {'Param Count':<12}")
    print("-" * 70)
    
    # 1. Patch Embedding
    patch_emb = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, embed_dim=128)
    x_img = np.random.randn(8, 32, 32, 3).astype(np.float32)
    
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        out_patch = patch_emb(x_img)
    t_fwd_patch = (time.perf_counter() - t0) / num_iterations * 1000
    
    grad_patch = np.ones_like(out_patch)
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = patch_emb.backward(grad_patch)
    t_bwd_patch = (time.perf_counter() - t0) / num_iterations * 1000
    params_patch = sum(p.size for p in patch_emb.parameters().values())
    print(f"{'Patch Embedding (32x32->4x4)':<30} | {t_fwd_patch:>10.2f} ms | {t_bwd_patch:>10.2f} ms | {params_patch:>10,}")
    
    # 2. Multi-Head Self-Attention
    mha = MultiHeadSelfAttention(embed_dim=128, num_heads=4)
    x_seq = np.random.randn(8, 64, 128).astype(np.float32)
    
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        out_mha = mha(x_seq)
    t_fwd_mha = (time.perf_counter() - t0) / num_iterations * 1000
    
    grad_mha = np.ones_like(out_mha)
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = mha.backward(grad_mha)
    t_bwd_mha = (time.perf_counter() - t0) / num_iterations * 1000
    params_mha = sum(p.size for p in mha.parameters().values())
    print(f"{'Multi-Head Attention (L=64, D=128)':<30} | {t_fwd_mha:>10.2f} ms | {t_bwd_mha:>10.2f} ms | {params_mha:>10,}")
    
    # 3. Transformer Block
    block = TransformerBlock(embed_dim=128, num_heads=4, mlp_ratio=4)
    
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        out_block = block(x_seq)
    t_fwd_block = (time.perf_counter() - t0) / num_iterations * 1000
    
    grad_block = np.ones_like(out_block)
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = block.backward(grad_block)
    t_bwd_block = (time.perf_counter() - t0) / num_iterations * 1000
    params_block = sum(p.size for p in block.parameters().values())
    print(f"{'Transformer Block (D=128, H=4)':<30} | {t_fwd_block:>10.2f} ms | {t_bwd_block:>10.2f} ms | {params_block:>10,}")
    
    # 4. Full ViT-UNet Architecture
    model = SimplifiedViTUNet(img_size=32, patch_size=4, in_channels=3, embed_dim=128, depth=4, num_heads=4)
    t_steps = np.random.randint(0, 1000, size=(4,))
    x_batch = np.random.randn(4, 32, 32, 3).astype(np.float32)
    
    t0 = time.perf_counter()
    for _ in range(num_iterations // 2):
        out_vit = model(x_batch, t_steps)
    t_fwd_vit = (time.perf_counter() - t0) / (num_iterations // 2) * 1000
    
    grad_vit = np.ones_like(out_vit)
    t0 = time.perf_counter()
    for _ in range(num_iterations // 2):
        _ = model.backward(grad_vit)
    t_bwd_vit = (time.perf_counter() - t0) / (num_iterations // 2) * 1000
    params_vit = sum(p.size for p in model.parameters().values())
    print(f"{'SimplifiedViTUNet (B=4, 32x32)':<30} | {t_fwd_vit:>10.2f} ms | {t_bwd_vit:>10.2f} ms | {params_vit:>10,}")
    print("=" * 70)


if __name__ == '__main__':
    verify_gradients()
    benchmark_components(num_iterations=10)
