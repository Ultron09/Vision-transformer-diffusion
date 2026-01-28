"""
ViT-Based U-Net for Diffusion Models

This module implements a denoising network that combines:
1. Vision Transformer (ViT) architecture for self-attention
2. U-Net structure with skip connections
3. Time conditioning throughout the network

The network predicts noise ε added to an image at a given timestep.

Architecture Overview:
    Input (noisy image + time)
    -> Patch Embedding
    -> Encoder Path (ViT blocks with downsampling)
    -> Bottleneck (ViT blocks)
    -> Decoder Path (ViT blocks with upsampling + skip connections)
    -> Output (predicted noise)

For simplicity in a pure NumPy implementation, we use a simpler ViT-based
architecture without explicit downsampling/upsampling, focusing on the
educational aspects of combining ViT with diffusion models.
"""

import numpy as np
from .layers import Linear, LayerNorm, PatchEmbedding, Dropout
from .transformer import TransformerBlock
from .time_embedding import TimeEmbeddingMLP, AdaptiveLayerNorm
from . import initializers


class SimplifiedViTUNet:
    """
    Simplified ViT-based U-Net for Diffusion Models
    
    A more manageable architecture for educational purposes that still captures
    the key ideas of using Vision Transformers for diffusion models.
    
    Architecture:
    1. Patch embedding
    2. Time embedding injection
    3. Stack of ViT blocks with time conditioning
    4. Output projection to predict noise
    
    Note: This is a simplified version without explicit U-Net up/downsampling
    to keep the NumPy implementation focused on the core concepts.
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, out_channels=3,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, time_embed_dim=None):
        """
        Initialize Simplified ViT-UNet
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Patch size
            in_channels: Number of input channels
            out_channels: Number of output channels (usually same as in_channels)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            time_embed_dim: Time embedding dimension (default: 4 * embed_dim)
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Time embedding dimension
        time_embed_dim = time_embed_dim or 4 * embed_dim
        self.time_embed_dim = time_embed_dim
        
        # Patch embedding (for noisy image input)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Time embedding
        self.time_embed = TimeEmbeddingMLP(embed_dim, time_embed_dim)
        
        # Projection to inject time into patches
        self.time_to_embed = Linear(time_embed_dim, embed_dim)
        
        # Positional embedding
        from .layers import PositionalEmbedding
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim, learnable=True)
        
        # Dropout
        self.pos_dropout = Dropout(dropout)
        
        # Transformer blocks with time conditioning
        self.blocks = []
        for i in range(depth):
            block = TimeConditionedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                time_embed_dim=time_embed_dim
            )
            self.blocks.append(block)
        
        # Final layer norm
        self.norm = LayerNorm(embed_dim)
        
        # Output projection: patches back to image
        patch_dim = patch_size * patch_size * out_channels
        self.output_proj = Linear(embed_dim, patch_dim)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x, timesteps):
        """
        Forward pass
        
        Args:
            x: Noisy images (batch_size, H, W, C)
            timesteps: Timesteps (batch_size,)
            
        Returns:
            Predicted noise (batch_size, H, W, C)
        """
        batch_size, H, W, C = x.shape
        
        # 1. Embed patches
        x = self.patch_embed(x)  # (B, N, D)
        
        # 2. Get time embeddings
        time_embed = self.time_embed(timesteps)  # (B, time_embed_dim)
        
        # 3. Project time embedding and add to patches
        time_to_add = self.time_to_embed(time_embed)  # (B, D)
        x = x + time_to_add[:, np.newaxis, :]  # Broadcast across patches
        
        # 4. Add positional embeddings
        x = self.pos_embed(x)
        
        # 5. Dropout
        x = self.pos_dropout(x)
        
        # 6. Transformer blocks with time conditioning
        for block in self.blocks:
            x = block(x, time_embed)
        
        # 7. Final normalization
        x = self.norm(x)
        
        # 8. Project back to patch dimensions
        x = self.output_proj(x)  # (B, N, patch_dim)
        
        # 9. Reshape to image
        # (B, N, P*P*C) -> (B, H/P, W/P, P, P, C) -> (B, H/P, P, W/P, P, C) -> (B, H, W, C)
        num_patches_per_side = self.img_size // self.patch_size
        P = self.patch_size
        
        x = x.reshape(batch_size, num_patches_per_side, num_patches_per_side, P, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H/P, W/P, P, P, C)
        x = x.reshape(batch_size, H, W, C)
        
        # Cache
        self.cache = (timesteps, time_embed)
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient w.r.t. output (B, H, W, C)
            
        Returns:
            Gradient w.r.t. input (B, H, W, C)
        """
        batch_size, H, W, C = grad_output.shape
        num_patches_per_side = H // self.patch_size
        P = self.patch_size
        
        # Reshape gradient from image to patches
        grad = grad_output.reshape(batch_size, num_patches_per_side, P, num_patches_per_side, P, C)
        grad = grad.transpose(0, 1, 3, 2, 4, 5)
        grad = grad.reshape(batch_size, num_patches_per_side * num_patches_per_side, P * P * C)
        
        # Backprop through output projection
        grad = self.output_proj.backward(grad)
        
        # Backprop through final norm
        grad = self.norm.backward(grad)
        
        # Backprop through transformer blocks (reverse order)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        
        # Backprop through dropout
        grad = self.pos_dropout.backward(grad)
        
        # Backprop through positional embedding
        grad = self.pos_embed.backward(grad)
        
        # Backprop through time injection
        # grad_time_to_add = sum over N dimension
        grad_time_to_add = np.sum(grad, axis=1)  # (B, D)
        grad_time_embed = self.time_to_embed.backward(grad_time_to_add)
        
        # Backprop through time embedding
        _ = self.time_embed.backward(grad_time_embed)
        
        # Backprop through patch embedding
        grad = self.patch_embed.backward(grad)
        
        return grad
    
    def parameters(self):
        """Return all parameters"""
        params = {}
        
        # Patch embedding
        for k, v in self.patch_embed.parameters().items():
            params[f'patch_embed_{k}'] = v
        
        # Time embedding
        for k, v in self.time_embed.parameters().items():
            params[f'time_embed_{k}'] = v
        
        # Time to embed projection
        for k, v in self.time_to_embed.parameters().items():
            params[f'time_to_embed_{k}'] = v
        
        # Positional embedding
        for k, v in self.pos_embed.parameters().items():
            params[f'pos_embed_{k}'] = v
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            for k, v in block.parameters().items():
                params[f'block{i}_{k}'] = v
        
        # Final norm
        for k, v in self.norm.parameters().items():
            params[f'norm_{k}'] = v
        
        # Output projection
        for k, v in self.output_proj.parameters().items():
            params[f'output_proj_{k}'] = v
        
        return params
    
    def gradients(self):
        """Return all gradients"""
        grads = {}
        
        # Patch embedding
        for k, v in self.patch_embed.gradients().items():
            grads[f'patch_embed_{k}'] = v
        
        # Time embedding
        for k, v in self.time_embed.gradients().items():
            grads[f'time_embed_{k}'] = v
        
        # Time to embed projection
        for k, v in self.time_to_embed.gradients().items():
            grads[f'time_to_embed_{k}'] = v
        
        # Positional embedding
        for k, v in self.pos_embed.gradients().items():
            grads[f'pos_embed_{k}'] = v
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            for k, v in block.gradients().items():
                grads[f'block{i}_{k}'] = v
        
        # Final norm
        for k, v in self.norm.gradients().items():
            grads[f'norm_{k}'] = v
        
        # Output projection
        for k, v in self.output_proj.gradients().items():
            grads[f'output_proj_{k}'] = v
        
        return grads
    
    def __call__(self, x, timesteps):
        return self.forward(x, timesteps)


class TimeConditionedTransformerBlock:
    """
    Transformer Block with Time Conditioning
    
    Similar to standard transformer block, but uses time embeddings to modulate
    the layer normalization (Adaptive Layer Norm).
    
    Architecture:
        x = x + Attention(AdaLN(x, t))
        x = x + MLP(AdaLN(x, t))
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, 
                 time_embed_dim=None):
        """
        Initialize Time-Conditioned Transformer Block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            time_embed_dim: Time embedding dimension
        """
        from .attention import MultiHeadSelfAttention
        from .transformer import MLP
        
        time_embed_dim = time_embed_dim or 4 * embed_dim
        
        # Adaptive layer norms (conditioned on time)
        self.norm1 = AdaptiveLayerNorm(embed_dim, time_embed_dim)
        self.norm2 = AdaptiveLayerNorm(embed_dim, time_embed_dim)
        
        # Attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features=mlp_hidden_dim, dropout=dropout)
        
        self.cache = None
    
    def forward(self, x, time_embed):
        """
        Forward pass
        
        Args:
            x: Input (batch_size, seq_len, embed_dim)
            time_embed: Time embeddings (batch_size, time_embed_dim)
            
        Returns:
            Output (batch_size, seq_len, embed_dim)
        """
        # First block: Attention with time-conditioned norm
        residual1 = x
        x_norm1 = self.norm1(x, time_embed)
        attn_out = self.attn(x_norm1)
        x = residual1 + attn_out
        
        # Second block: MLP with time-conditioned norm
        residual2 = x
        x_norm2 = self.norm2(x, time_embed)
        mlp_out = self.mlp(x_norm2)
        x = residual2 + mlp_out
        
        self.cache = (residual1, x_norm1, attn_out, residual2, x_norm2, mlp_out, time_embed)
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        residual1, x_norm1, attn_out, residual2, x_norm2, mlp_out, time_embed = self.cache
        
        # Backprop through second residual
        grad_mlp_out = grad_output
        grad_residual2 = grad_output
        
        # Backprop through MLP
        grad_x_norm2 = self.mlp.backward(grad_mlp_out)
        
        # Backprop through adaptive norm 2
        grad_x_after_attn, grad_time_embed2 = self.norm2.backward(grad_x_norm2)
        grad_x_after_attn = grad_x_after_attn + grad_residual2
        
        # Backprop through first residual
        grad_attn_out = grad_x_after_attn
        grad_residual1 = grad_x_after_attn
        
        # Backprop through attention
        grad_x_norm1 = self.attn.backward(grad_attn_out)
        
        # Backprop through adaptive norm 1
        grad_input, grad_time_embed1 = self.norm1.backward(grad_x_norm1)
        grad_input = grad_input + grad_residual1
        
        # Note: Time embedding gradients are not backpropagated further
        # since time is a discrete input
        
        return grad_input
    
    def parameters(self):
        """Return parameters"""
        params = {}
        
        for k, v in self.norm1.parameters().items():
            params[f'norm1_{k}'] = v
        for k, v in self.attn.parameters().items():
            params[f'attn_{k}'] = v
        for k, v in self.norm2.parameters().items():
            params[f'norm2_{k}'] = v
        for k, v in self.mlp.parameters().items():
            params[f'mlp_{k}'] = v
        
        return params
    
    def gradients(self):
        """Return gradients"""
        grads = {}
        
        for k, v in self.norm1.gradients().items():
            grads[f'norm1_{k}'] = v
        for k, v in self.attn.gradients().items():
            grads[f'attn_{k}'] = v
        for k, v in self.norm2.gradients().items():
            grads[f'norm2_{k}'] = v
        for k, v in self.mlp.gradients().items():
            grads[f'mlp_{k}'] = v
        
        return grads
    
    def __call__(self, x, time_embed):
        return self.forward(x, time_embed)
