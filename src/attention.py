"""
Multi-Head Self-Attention Mechanism

This module implements the core attention mechanism used in transformers,
specifically the scaled dot-product attention and multi-head attention.

Mathematical Background:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
    - Q: Query matrix (what we're looking for)
    - K: Key matrix (what we're comparing against)
    - V: Value matrix (what we return)
    - d_k: Dimension of keys (for scaling)

Multi-head attention runs multiple attention operations in parallel,
allowing the model to attend to different representation subspaces.

Reference:
    Vaswani et al., 2017: "Attention Is All You Need"
"""

import numpy as np
from .layers import Linear, Dropout
from .activations import Softmax


class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention Layer
    
    Performs self-attention with multiple attention heads in parallel:
    
    1. Project input to Q, K, V for each head
    2. Compute attention scores: scores = QK^T / √d_k
    3. Apply softmax to get attention weights
    4. Compute weighted sum: output = attention_weights @ V
    5. Concatenate heads and project
    
    Architecture:
        Input -> [Q, K, V projections] -> [Head 1, Head 2, ..., Head h] -> Concat -> Output projection
    
    Properties:
        - Parallelizable across heads
        - Captures different types of relationships
        - Permutation-invariant (hence need positional embeddings)
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        """
        Initialize Multi-Head Self-Attention
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in projections
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / √d_k for scaling
        
        # Query, Key, Value projections (combined for efficiency)
        self.qkv_projection = Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        # Output projection
        self.output_projection = Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout for attention weights
        self.attn_dropout = Dropout(dropout)
        
        # Softmax for attention weights
        self.softmax = Softmax(axis=-1)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head self-attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  True/1 values indicate positions to mask (set to -inf)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. Project to Q, K, V
        qkv = self.qkv_projection(x)  # (B, N, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # 2. Compute attention scores: QK^T / √d_k
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        # 3. Apply mask if provided
        if mask is not None:
            # Expand mask for num_heads: (B, 1, N, N)
            mask = mask[:, np.newaxis, :, :]
            # Set masked positions to large negative value
            attn_scores = np.where(mask, -1e9, attn_scores)
        
        # 4. Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores)  # (B, num_heads, N, N)
        
        # 5. Apply dropout to attention weights
        attn_weights_dropped = self.attn_dropout(attn_weights)
        
        # 6. Compute weighted sum of values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        attn_output = np.matmul(attn_weights_dropped, v)
        
        # 7. Reshape and concatenate heads
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        
        # 8. Final output projection
        output = self.output_projection(attn_output)
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'q': q,
            'k': k,
            'v': v,
            'attn_scores': attn_scores,
            'attn_weights': attn_weights,
            'attn_weights_dropped': attn_weights_dropped,
            'attn_output': attn_output,
            'mask': mask,
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass of multi-head self-attention
        
        This is complex due to multiple matrix multiplications and attention mechanism.
        We need to backpropagate through:
        1. Output projection
        2. Concatenation and reshaping
        3. Attention computation (softmax + matmul)
        4. QKV projections
        
        Args:
            grad_output: Gradient from next layer, shape (B, N, embed_dim)
            
        Returns:
            Gradient w.r.t. input, shape (B, N, embed_dim)
        """
        # Retrieve cached values
        x = self.cache['x']
        q = self.cache['q']
        k = self.cache['k']
        v = self.cache['v']
        attn_weights = self.cache['attn_weights']
        attn_weights_dropped = self.cache['attn_weights_dropped']
        batch_size = self.cache['batch_size']
        seq_len = self.cache['seq_len']
        
        # 1. Backprop through output projection
        grad_attn_output = self.output_projection.backward(grad_output)  # (B, N, embed_dim)
        
        # 2. Reshape for heads
        # (B, N, embed_dim) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        grad_attn_output = grad_attn_output.transpose(0, 2, 1, 3)
        
        # 3. Backprop through attention output = attn_weights @ v
        # grad_attn_weights_dropped = grad_attn_output @ v^T
        grad_attn_weights_dropped = np.matmul(grad_attn_output, v.transpose(0, 1, 3, 2))
        
        # grad_v = attn_weights^T @ grad_attn_output
        grad_v = np.matmul(attn_weights_dropped.transpose(0, 1, 3, 2), grad_attn_output)
        
        # 4. Backprop through dropout
        grad_attn_weights = self.attn_dropout.backward(grad_attn_weights_dropped)
        
        # 5. Backprop through softmax
        grad_attn_scores = self.softmax.backward(grad_attn_weights)
        
        # 6. Apply scaling
        grad_attn_scores = grad_attn_scores * self.scale
        
        # 7. Backprop through QK^T
        # attn_scores = q @ k^T
        # grad_q = grad_attn_scores @ k
        grad_q = np.matmul(grad_attn_scores, k)
        
        # grad_k = grad_attn_scores^T @ q
        grad_k = np.matmul(grad_attn_scores.transpose(0, 1, 3, 2), q)
        
        # 8. Reshape gradients for QKV
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, embed_dim)
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # 9. Concatenate QKV gradients
        grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)  # (B, N, 3*embed_dim)
        
        # 10. Backprop through QKV projection
        grad_input = self.qkv_projection.backward(grad_qkv)
        
        return grad_input
    
    def parameters(self):
        """Return dictionary of all parameters"""
        params = {}
        
        # QKV projection parameters
        qkv_params = self.qkv_projection.parameters()
        for key, val in qkv_params.items():
            params[f'qkv_{key}'] = val
        
        # Output projection parameters
        out_params = self.output_projection.parameters()
        for key, val in out_params.items():
            params[f'output_{key}'] = val
        
        return params
    
    def gradients(self):
        """Return dictionary of all gradients"""
        grads = {}
        
        # QKV projection gradients
        qkv_grads = self.qkv_projection.gradients()
        for key, val in qkv_grads.items():
            grads[f'qkv_{key}'] = val
        
        # Output projection gradients
        out_grads = self.output_projection.gradients()
        for key, val in out_grads.items():
            grads[f'output_{key}'] = val
        
        return grads
    
    def train(self):
        """Set to training mode"""
        self.attn_dropout.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.attn_dropout.eval()
    
    def __call__(self, x, mask=None):
        return self.forward(x, mask)


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention (single head)
    
    Lower-level attention mechanism used by MultiHeadSelfAttention.
    Provided for educational purposes and direct use if needed.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    
    def __init__(self, dropout=0.0):
        """
        Initialize Scaled Dot-Product Attention
        
        Args:
            dropout: Dropout probability for attention weights
        """
        self.dropout = Dropout(dropout)
        self.softmax = Softmax(axis=-1)
        self.cache = None
    
    def forward(self, q, k, v, mask=None):
        """
        Forward pass
        
        Args:
            q: Query matrix (..., seq_len_q, d_k)
            k: Key matrix (..., seq_len_k, d_k)
            v: Value matrix (..., seq_len_v, d_v)
            mask: Optional mask (..., seq_len_q, seq_len_k)
            
        Returns:
            Attention output (..., seq_len_q, d_v)
        """
        d_k = q.shape[-1]
        
        # Compute attention scores
        scores = np.matmul(q, k.transpose(*range(k.ndim - 2), -1, -2)) / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        # Softmax and dropout
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        output = np.matmul(attn_weights, v)
        
        self.cache = (q, k, v, attn_weights, mask)
        
        return output, attn_weights
    
    def __call__(self, q, k, v, mask=None):
        return self.forward(q, k, v, mask)
