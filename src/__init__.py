"""
Vision Transformer Diffusion Model - Pure NumPy Implementation
Educational-grade implementation of ViT-based diffusion models.
"""

__version__ = "1.0.0"
__author__ = "Educational Project"

from . import activations
from . import initializers
from . import optimizers
from . import layers
from . import attention
from . import transformer
from . import diffusion_process
from . import time_embedding
from . import unet_vit
from . import trainer
from . import sampler
from . import data_utils

__all__ = [
    "activations",
    "initializers",
    "optimizers",
    "layers",
    "attention",
    "transformer",
    "diffusion_process",
    "time_embedding",
    "unet_vit",
    "trainer",
    "sampler",
    "data_utils",
]
