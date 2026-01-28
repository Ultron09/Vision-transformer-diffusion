"""
Transformer Blocks and Vision Transformer Architecture

This module implements:
1. MLP (Feed-Forward Network) used in transformers
2. TransformerBlock (attention + MLP with residual connections)
3. VisionTransformer (complete ViT architecture)

Architecture:
    Input Image -> Patch Embedding -> [Transformer Blocks] x L -> Output

Each Transformer Block:
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

Reference:
    Dosovitskiy et al., 2020: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import numpy as np
from .layers import Linear, LayerNorm, Dropout, PatchEmbedding, PositionalEmbedding
from .attention import MultiHeadSelfAttention
from .activations import GELU


class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Network)
    
    Two-layer MLP with GELU activation:
        MLP(x) = Linear2(GELU(Linear1(x)))
    
    Typically used with expansion factor (hidden_dim = expansion * input_dim).
    In ViT, expansion factor is usually 4.
    
    Architecture:
        Input (d) -> Linear (d → 4d) -> GELU -> Dropout -> Linear (4d → d) -> Dropout
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 dropout=0.1, activation='gelu'):
        """
        Initialize MLP
        
        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension (default: 4 * in_features)
            out_features: Output dimension (default: in_features)
            dropout: Dropout probability
            activation: Activation function ('gelu' or 'relu')
        """
        hidden_features = hidden_features or 4 * in_features
        out_features = out_features or in_features
        
        self.fc1 = Linear(in_features, hidden_features)
        self.fc2 = Linear(hidden_features, out_features)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Activation
        if activation == 'gelu':
            self.activation = GELU()
        else:
            from .activations import ReLU
            self.activation = ReLU()
        
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # First linear layer
        x = self.fc1(x)
        
        # Activation
        x = self.activation(x)
        
        # Dropout
        x = self.dropout1(x)
        
        # Second linear layer
        x = self.fc2(x)
        
        # Dropout
        x = self.dropout2(x)
        
        return x
    
    def backward(self, grad_output):
        """Backward pass"""
        # Backprop through second dropout
        grad = self.dropout2.backward(grad_output)
        
        # Backprop through second linear
        grad = self.fc2.backward(grad)
        
        # Backprop through first dropout
        grad = self.dropout1.backward(grad)
        
        # Backprop through activation
        grad = self.activation.backward(grad)
        
        # Backprop through first linear
        grad = self.fc1.backward(grad)
        
        return grad
    
    def parameters(self):
        """Return all parameters"""
        params = {}
        
        fc1_params = self.fc1.parameters()
        for key, val in fc1_params.items():
            params[f'fc1_{key}'] = val
        
        fc2_params = self.fc2.parameters()
        for key, val in fc2_params.items():
            params[f'fc2_{key}'] = val
        
        return params
    
    def gradients(self):
        """Return all gradients"""
        grads = {}
        
        fc1_grads = self.fc1.gradients()
        for key, val in fc1_grads.items():
            grads[f'fc1_{key}'] = val
        
        fc2_grads = self.fc2.gradients()
        for key, val in fc2_grads.items():
            grads[f'fc2_{key}'] = val
        
        return grads
    
    def train(self):
        """Set to training mode"""
        self.dropout1.train()
        self.dropout2.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.dropout1.eval()
        self.dropout2.eval()
    
    def __call__(self, x):
        return self.forward(x)


class TransformerBlock:
    """
    Transformer Encoder Block
    
    Standard transformer block with:
    - Multi-head self-attention with residual connection
    - Feed-forward network (MLP) with residual connection
    - Layer normalization (pre-norm architecture)
    
    Pre-Norm Architecture (used in ViT):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    This is more stable than post-norm during training.
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        """
        Initialize Transformer Block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate for MLP
            attn_dropout: Dropout rate for attention
        """
        self.embed_dim = embed_dim
        
        # Layer normalization
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=attn_dropout)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features=mlp_hidden_dim, dropout=dropout)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Save input for residual connection
        residual1 = x
        
        # Attention block with pre-norm
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1, mask=mask)
        x = residual1 + attn_output  # Residual connection
        
        # MLP block with pre-norm
        residual2 = x
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = residual2 + mlp_output  # Residual connection
        
        # Cache for backward pass
        self.cache = {
            'input': residual1,
            'x_norm1': x_norm1,
            'attn_output': attn_output,
            'residual2': residual2,
            'x_norm2': x_norm2,
            'mlp_output': mlp_output
        }
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass through transformer block
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        # Gradient through second residual connection
        grad_mlp_output = grad_output
        grad_residual2 = grad_output
        
        # Backprop through MLP
        grad_x_norm2 = self.mlp.backward(grad_mlp_output)
        
        # Backprop through second layer norm
        grad_x_after_attn = self.norm2.backward(grad_x_norm2)
        
        # Add gradient from residual
        grad_x_after_attn = grad_x_after_attn + grad_residual2
        
        # Gradient through first residual connection
        grad_attn_output = grad_x_after_attn
        grad_residual1 = grad_x_after_attn
        
        # Backprop through attention
        grad_x_norm1 = self.attn.backward(grad_attn_output)
        
        # Backprop through first layer norm
        grad_input = self.norm1.backward(grad_x_norm1)
        
        # Add gradient from residual
        grad_input = grad_input + grad_residual1
        
        return grad_input
    
    def parameters(self):
        """Return all parameters"""
        params = {}
        
        # Layer norm 1
        norm1_params = self.norm1.parameters()
        for key, val in norm1_params.items():
            params[f'norm1_{key}'] = val
        
        # Attention
        attn_params = self.attn.parameters()
        for key, val in attn_params.items():
            params[f'attn_{key}'] = val
        
        # Layer norm 2
        norm2_params = self.norm2.parameters()
        for key, val in norm2_params.items():
            params[f'norm2_{key}'] = val
        
        # MLP
        mlp_params = self.mlp.parameters()
        for key, val in mlp_params.items():
            params[f'mlp_{key}'] = val
        
        return params
    
    def gradients(self):
        """Return all gradients"""
        grads = {}
        
        # Layer norm 1
        norm1_grads = self.norm1.gradients()
        for key, val in norm1_grads.items():
            grads[f'norm1_{key}'] = val
        
        # Attention
        attn_grads = self.attn.gradients()
        for key, val in attn_grads.items():
            grads[f'attn_{key}'] = val
        
        # Layer norm 2
        norm2_grads = self.norm2.gradients()
        for key, val in norm2_grads.items():
            grads[f'norm2_{key}'] = val
        
        # MLP
        mlp_grads = self.mlp.gradients()
        for key, val in mlp_grads.items():
            grads[f'mlp_{key}'] = val
        
        return grads
    
    def train(self):
        """Set to training mode"""
        self.attn.train()
        self.mlp.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.attn.eval()
        self.mlp.eval()
    
    def __call__(self, x, mask=None):
        return self.forward(x, mask)


class VisionTransformer:
    """
    Vision Transformer (ViT) Architecture
    
    Complete ViT model for processing images:
    
    1. Patch Embedding: Split image into patches and embed
    2. Positional Embedding: Add position information
    3. Transformer Encoder: Stack of transformer blocks
    4. Output: Can be used for classification or as feature extractor
    
    For diffusion models, we'll use ViT as a feature extractor (no classification head).
    
    Architecture:
        Image (H×W×C) 
        -> Patch Embed (N×D) 
        -> + Pos Embed 
        -> [Transformer Block] × L 
        -> LayerNorm 
        -> Output (N×D)
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256,
                 depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        """
        Initialize Vision Transformer
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim, learnable=True)
        
        # Dropout after embeddings
        self.pos_dropout = Dropout(dropout)
        
        # Transformer blocks
        self.blocks = []
        for i in range(depth):
            block = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            self.blocks.append(block)
        
        # Final layer norm
        self.norm = LayerNorm(embed_dim)
        
        # Cache
        self.cache = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, height, width, channels)
            
        Returns:
            Output features (batch_size, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add positional embedding
        x = self.pos_embed(x)
        
        # Dropout
        x = self.pos_dropout(x)
        
        # Transformer blocks
        intermediate_outputs = []
        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Cache
        self.cache = intermediate_outputs
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass through entire ViT
        
        Args:
            grad_output: Gradient w.r.t. output
            
        Returns:
            Gradient w.r.t. input image
        """
        # Backprop through final layer norm
        grad = self.norm.backward(grad_output)
        
        # Backprop through transformer blocks (in reverse order)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        
        # Backprop through pos dropout
        grad = self.pos_dropout.backward(grad)
        
        # Backprop through positional embedding
        grad = self.pos_embed.backward(grad)
        
        # Backprop through patch embedding
        grad = self.patch_embed.backward(grad)
        
        return grad
    
    def parameters(self):
        """Return all parameters"""
        params = {}
        
        # Patch embedding
        patch_params = self.patch_embed.parameters()
        for key, val in patch_params.items():
            params[f'patch_embed_{key}'] = val
        
        # Positional embedding
        pos_params = self.pos_embed.parameters()
        for key, val in pos_params.items():
            params[f'pos_embed_{key}'] = val
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            block_params = block.parameters()
            for key, val in block_params.items():
                params[f'block{i}_{key}'] = val
        
        # Final norm
        norm_params = self.norm.parameters()
        for key, val in norm_params.items():
            params[f'norm_{key}'] = val
        
        return params
    
    def gradients(self):
        """Return all gradients"""
        grads = {}
        
        # Patch embedding
        patch_grads = self.patch_embed.gradients()
        for key, val in patch_grads.items():
            grads[f'patch_embed_{key}'] = val
        
        # Positional embedding
        pos_grads = self.pos_embed.gradients()
        for key, val in pos_grads.items():
            grads[f'pos_embed_{key}'] = val
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            block_grads = block.gradients()
            for key, val in block_grads.items():
                grads[f'block{i}_{key}'] = val
        
        # Final norm
        norm_grads = self.norm.gradients()
        for key, val in norm_grads.items():
            grads[f'norm_{key}'] = val
        
        return grads
    
    def train(self):
        """Set to training mode"""
        self.pos_dropout.train()
        for block in self.blocks:
            block.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.pos_dropout.eval()
        for block in self.blocks:
            block.eval()
    
    def __call__(self, x):
        return self.forward(x)
