# Vision Transformer Diffusion Model - Pure NumPy Implementation

An **educational-grade** implementation of a Generative Neural Network combining **Vision Transformers (ViT)** and **Diffusion Models** using **pure NumPy**. This project demonstrates how state-of-the-art generative AI works from first principles, with no PyTorch or TensorFlow dependencies.

## 🎯 Project Overview

This implementation showcases two cutting-edge architectures working together:

1. **Vision Transformers (ViT)**: Self-attention based architecture for processing images as sequences of patches
2. **Diffusion Models (DDPM)**: Modern generative models that learn to denoise images through iterative refinement

All components are built from scratch using only NumPy, making this an ideal educational resource for understanding:
- How transformers process visual data
- The mathematics behind diffusion models
- Backpropagation through complex architectures
- Training dynamics of generative models

## 📚 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Mathematical Background](#-mathematical-background)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Implementation Details](#-implementation-details)
- [Limitations](#-limitations)
- [Educational Resources](#-educational-resources)
- [References](#-references)

## ✨ Features

### Core Components

**Vision Transformer Components:**
- ✅ Patch embedding (image → patch sequences)
- ✅ Multi-head self-attention mechanism
- ✅ Position embeddings (learnable)
- ✅ Layer normalization
- ✅ Transformer encoder blocks
- ✅ MLP (feed-forward) layers

**Diffusion Model Components:**
- ✅ Forward diffusion process (noise scheduling)
- ✅ Reverse diffusion process (denoising)
- ✅ Noise prediction network (ViT-based)
- ✅ Beta schedules (linear, cosine)
- ✅ Time step embeddings
- ✅ DDPM and DDIM sampling

**Training Infrastructure:**
- ✅ Complete training loop
- ✅ Adam optimizer
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Checkpointing system
- ✅ Progress tracking

**Educational Features:**
- ✅ Detailed mathematical derivations in docstrings
- ✅ Comprehensive comments explaining each operation
- ✅ Modular, readable code structure
- ✅ Example training and generation scripts
- ✅ Visualization utilities

## 🏗️ Architecture

### Vision Transformer Diffusion Model

```
Input: Noisy Image (x_t) + Timestep (t)
│
├─► Patch Embedding
│   └─► Splits image into N patches, projects to embed_dim
│
├─► Time Embedding
│   └─► Sinusoidal encoding + MLP projection
│
├─► Positional Embedding
│   └─► Learnable position encodings
│
├─► Transformer Encoder Blocks (×L)
│   ├─► Adaptive Layer Norm (conditioned on time)
│   ├─► Multi-Head Self-Attention
│   ├─► Residual Connection
│   ├─► Adaptive Layer Norm (conditioned on time)
│   ├─► MLP (Feed-Forward)
│   └─► Residual Connection
│
├─► Final Layer Norm
│
└─► Output Projection
    └─► Project patches back to image space

Output: Predicted Noise (ε)
```

### Multi-Head Self-Attention

```
Input (x) → [Q, K, V Projections]
                    │
    ┌───────────────┼───────────────┐
    │               │               │
   Q_1             K_1             V_1
   Q_2             K_2             V_2
   ...             ...             ...
   Q_h             K_h             V_h
    │               │               │
    └──► Attention(Q, K, V) = softmax(QK^T / √d_k) V
                    │
            [Concatenate heads]
                    │
            [Output Projection]
                    │
                 Output
```

### Diffusion Process

**Forward Diffusion (Training):**
```
x_0 (clean) → x_1 → x_2 → ... → x_T (noise)

x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
where ε ~ N(0, I)
```

**Reverse Diffusion (Sampling):**
```
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (clean)

x_{t-1} = μ_θ(x_t, t) + σ_t * z
where μ_θ is computed from predicted noise ε_θ(x_t, t)
```

## 📦 Installation

### Requirements

- Python 3.7+
- NumPy
- (Optional) PIL/Pillow for image saving

```bash
# Clone the repository
git clone <repository-url>
cd numpy-gan

# Install dependencies
pip install numpy pillow

# Verify installation
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

### Project Structure

```
numpy-gan/
├── src/
│   ├── __init__.py
│   ├── activations.py          # GELU, SiLU, Softmax, ReLU
│   ├── initializers.py         # Weight initialization strategies
│   ├── optimizers.py           # Adam, SGD, LR schedulers
│   ├── layers.py               # Linear, LayerNorm, Dropout, Embeddings
│   ├── attention.py            # Multi-head self-attention
│   ├── transformer.py          # Transformer blocks, ViT
│   ├── time_embedding.py       # Time conditioning for diffusion
│   ├── diffusion_process.py    # Forward/reverse diffusion
│   ├── unet_vit.py            # ViT-based denoising network
│   ├── trainer.py              # Training infrastructure
│   ├── sampler.py              # DDPM/DDIM sampling
│   └── data_utils.py           # Data loading and preprocessing
├── examples/
│   ├── train.py                # Training example
│   └── generate.py             # Generation example
├── checkpoints/                # Saved model checkpoints
├── samples/                    # Generated samples
└── README.md                   # This file
```

## 🚀 Quick Start

### Training a Model

```bash
# Train on synthetic data (recommended for testing)
python examples/train.py

# This will:
# - Create a synthetic dataset (200 32x32 images)
# - Initialize a ViT-based diffusion model
# - Train for 5 epochs
# - Save checkpoints to checkpoints/
# - Generate samples to samples/
```

### Generating Images

```bash
# Generate images from trained model
python examples/generate.py --checkpoint checkpoints/model_final.npz --num_samples 16

# Use DDIM for faster sampling
python examples/generate.py --sampler ddim --num_steps 50

# Save progressive generation steps
python examples/generate.py --progressive --num_samples 4
```

### Custom Training

```python
import numpy as np
from src.unet_vit import SimplifiedViTUNet
from src.diffusion_process import GaussianDiffusion
from src.trainer import DiffusionTrainer
from src.data_utils import create_synthetic_dataset

# Create dataset
images = create_synthetic_dataset(num_samples=100, img_size=32, channels=3)

# Initialize model
model = SimplifiedViTUNet(
    img_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=128,
    depth=4,
    num_heads=4
)

# Initialize diffusion
diffusion = GaussianDiffusion(num_timesteps=100, beta_schedule='linear')

# Train
trainer = DiffusionTrainer(model, diffusion)
# ... training loop
```

## 📐 Mathematical Background

### Vision Transformers

**Patch Embedding:**

For an image of size H × W × C with patch size P:
- Number of patches: N = (H/P) × (W/P)
- Each patch: P × P × C flattened to dimension D

```
PatchEmbed(x) = Linear(Flatten(Patches(x)))
```

**Self-Attention:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q = xW_Q  (Query)
- K = xW_K  (Key)  
- V = xW_V  (Value)
- d_k = head dimension
```

**Multi-Head Attention:**

```
MultiHead(x) = Concat(head_1, ..., head_h)W_O

where head_i = Attention(xW_Q^i, xW_K^i, xW_V^i)
```

**Transformer Block:**

```
x' = x + MultiHeadAttention(LayerNorm(x))
x'' = x' + MLP(LayerNorm(x'))
```

### Diffusion Models

**Forward Process:**

The forward process gradually adds Gaussian noise to data:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t) I)

where:
- β_t: variance schedule
- α_t = 1 - β_t
- ᾱ_t = ∏_{i=1}^t α_i
```

**Reverse Process:**

The reverse process learns to denoise:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Training Objective:**

Simplified objective (predicting noise):

```
L_simple = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]

where:
- ε ~ N(0, I): random noise
- x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
- ε_θ: noise prediction network (our ViT model)
```

**Sampling:**

DDPM sampling (stochastic):

```
x_{t-1} = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t,t)) + σ_t z

where z ~ N(0, I)
```

DDIM sampling (deterministic):

```
x_{t-1} = √(ᾱ_{t-1}) pred_x_0 + √(1-ᾱ_{t-1}) ε_θ(x_t, t)

where pred_x_0 = (x_t - √(1-ᾱ_t) ε_θ(x_t,t)) / √(ᾱ_t)
```

### Time Conditioning

**Sinusoidal Time Embedding:**

```
PE(t, 2i) = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

**Adaptive Layer Normalization:**

```
AdaLN(x, t) = γ(t) ⊙ LayerNorm(x) + β(t)

where γ(t) and β(t) are learned from time embedding
```

## 💻 Implementation Details

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **GELU** | `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))` | Transformers |
| **SiLU/Swish** | `x * σ(x)` where `σ(x) = 1/(1+e^(-x))` | Diffusion models |
| **Softmax** | `exp(x_i) / Σexp(x_j)` | Attention weights |

### Weight Initialization

- **Xavier/Glorot**: For symmetric activations (tanh, sigmoid)
- **He**: For ReLU-like activations
- **Truncated Normal**: For position embeddings (ViT standard)

### Optimization

**Adam Optimizer:**
```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**Learning Rate Schedules:**
- Linear warmup
- Cosine annealing
- Step decay

### Gradient Computation

All layers implement:
- `forward(x)`: Compute output and cache inputs
- `backward(grad_output)`: Compute input gradients

Example (Linear layer):
```python
# Forward
output = x @ W + b

# Backward
grad_W = x.T @ grad_output
grad_b = sum(grad_output)
grad_x = grad_output @ W.T
```

## ⚠️ Limitations

### Performance

**This is an educational implementation, not a production system:**

- ⏱️ **Speed**: 100-1000x slower than GPU-accelerated frameworks
- 💾 **Memory**: Limited by RAM (no batching optimizations)
- 📊 **Scale**: Practical only for small images (32x32, 64x64)
- ⏳ **Training Time**: Hours for toy datasets vs. minutes on GPU

### Recommended Settings for NumPy

- **Image size**: 28x28 to 64x64
- **Batch size**: 4-16
- **Model size**: embed_dim ≤ 256, depth ≤ 6
- **Training samples**: 100-1000
- **Timesteps**: 50-200

### Not Implemented

- Mixed precision training
- Distributed training
- Advanced sampling (classifier guidance, inpainting)
- Gradient accumulation
- Advanced U-Net architecture (actual up/downsampling)

## 📖 Educational Resources

### Understanding the Code

1. **Start with basics** (`activations.py`, `initializers.py`)
   - See how activation functions work mathematically
   - Understand weight initialization strategies

2. **Build up to layers** (`layers.py`)
   - Linear transformations
   - Layer normalization
   - Patch embedding for images

3. **Master attention** (`attention.py`)
   - Scaled dot-product attention
   - Multi-head mechanism
   - Why attention works for sequences

4. **Grasp transformers** (`transformer.py`)
   - How blocks combine attention + MLP
   - Residual connections
   - Complete ViT architecture

5. **Learn diffusion** (`diffusion_process.py`)
   - Forward noise addition
   - Reverse denoising
   - Why this generates images

6. **Connect everything** (`unet_vit.py`)
   - Time conditioning
   - How ViT becomes a denoiser
   - Complete generative model

### Key Concepts to Understand

**Self-Attention:**
- Why images become patch sequences
- How patches attend to each other
- Multi-head parallelism benefits

**Diffusion Process:**
- Forward diffusion as gradual corruption
- Reverse diffusion as learned denoising
- Why predicting noise works

**Time Conditioning:**
- Why the model needs to know timestep
- How time embeddings work
- Adaptive normalization

**Training Dynamics:**
- Random timestep sampling
- Noise prediction objective
- Gradient flow through deep networks

## 📚 References

### Papers

1. **Vision Transformers**
   - Dosovitskiy et al., 2020: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. **Attention Mechanism**
   - Vaswani et al., 2017: "Attention Is All You Need"
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

3. **Diffusion Models**
   - Ho et al., 2020: "Denoising Diffusion Probabilistic Models"
   - [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

4. **DDIM Sampling**
   - Song et al., 2020: "Denoising Diffusion Implicit Models"
   - [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

5. **Layer Normalization**
   - Ba et al., 2016: "Layer Normalization"
   - [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

6. **Adam Optimizer**
   - Kingma & Ba, 2014: "Adam: A Method for Stochastic Optimization"
   - [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

### Blogs and Tutorials

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Lilian Weng
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) - Hugging Face

## 🎓 Learning Path

### Beginner Level

1. Review `activations.py` - understand GELU and SiLU
2. Study `layers.py` - see how Linear and LayerNorm work
3. Read `initializers.py` - learn weight initialization

### Intermediate Level

4. Deep dive into `attention.py` - master self-attention mechanism
5. Explore `transformer.py` - understand ViT architecture
6. Analyze `diffusion_process.py` - grasp forward/reverse diffusion

### Advanced Level

7. Study `unet_vit.py` - see how everything combines
8. Review `trainer.py` - understand training loop
9. Experiment with `sampler.py` - explore generation algorithms

### Exercises

1. **Modify attention**: Try different attention variants
2. **Change schedules**: Experiment with beta schedules
3. **Add features**: Implement classifier-free guidance
4. **Optimize code**: Profile and speed up bottlenecks
5. **Visualize**: Create attention heatmaps
6. **Compare**: Implement both pre-norm and post-norm

## 🤝 Contributing

This is an educational project. Contributions that improve:
- Code clarity and comments
- Mathematical explanations
- Documentation
- Educational examples
- Visualization tools

are welcome!

## 📄 License

MIT License - Feel free to use for educational purposes.

## 🙏 Acknowledgments

This implementation is inspired by:
- Original DDPM and ViT papers
- Various open-source implementations
- Educational resources from the ML community

Built with ❤️ for learning and education.

---

**Remember**: This is a learning tool. For production use, please use PyTorch, TensorFlow, or JAX with GPU acceleration!
