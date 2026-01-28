"""
Data Utilities for Diffusion Models

This module provides utilities for:
- Loading and preprocessing images
- Creating simple datasets
- Data augmentation (optional)
- Normalization and denormalization

For educational purposes, we provide simple dataset loaders.
Users can extend these for their own datasets.
"""

import numpy as np


def normalize_images(images, method='tanh'):
    """
    Normalize images to appropriate range for diffusion models
    
    Args:
        images: Images in range [0, 255] or [0, 1]
        method: Normalization method
            - 'tanh': Normalize to [-1, 1] (standard for diffusion models)
            - 'zero_one': Normalize to [0, 1]
            
    Returns:
        Normalized images
    """
    # Convert to float if needed
    images = images.astype(np.float32)
    
    # If in [0, 255], convert to [0, 1]
    if images.max() > 1.0:
        images = images / 255.0
    
    if method == 'tanh':
        # [0, 1] -> [-1, 1]
        return images * 2.0 - 1.0
    elif method == 'zero_one':
        return images
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_images(images, method='tanh', to_uint8=True):
    """
    Denormalize images back to [0, 255] for visualization
    
    Args:
        images: Normalized images
        method: Normalization method used
        to_uint8: Whether to convert to uint8
        
    Returns:
        Denormalized images
    """
    if method == 'tanh':
        # [-1, 1] -> [0, 1]
        images = (images + 1.0) / 2.0
    
    # Clip to valid range
    images = np.clip(images, 0, 1)
    
    if to_uint8:
        images = (images * 255).astype(np.uint8)
    
    return images


def create_synthetic_dataset(num_samples, img_size, pattern='random', channels=3):
    """
    Create synthetic dataset for testing
    
    Args:
        num_samples: Number of samples to create
        img_size: Image size (height and width)
        pattern: Type of pattern ('random', 'circles', 'squares', 'gradients')
        channels: Number of channels
        
    Returns:
        Array of synthetic images (num_samples, img_size, img_size, channels)
    """
    images = []
    
    for i in range(num_samples):
        if pattern == 'random':
            # Random noise
            img = np.random.rand(img_size, img_size, channels)
        
        elif pattern == 'circles':
            # Random circles
            img = np.zeros((img_size, img_size, channels))
            center_x = np.random.randint(img_size)
            center_y = np.random.randint(img_size)
            radius = np.random.randint(5, img_size // 3)
            
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            color = np.random.rand(channels)
            img[mask] = color
        
        elif pattern == 'squares':
            # Random squares
            img = np.zeros((img_size, img_size, channels))
            x1 = np.random.randint(0, img_size // 2)
            y1 = np.random.randint(0, img_size // 2)
            size = np.random.randint(5, img_size // 3)
            x2 = min(x1 + size, img_size)
            y2 = min(y1 + size, img_size)
            
            color = np.random.rand(channels)
            img[y1:y2, x1:x2] = color
        
        elif pattern == 'gradients':
            # Color gradients
            img = np.zeros((img_size, img_size, channels))
            for c in range(channels):
                gradient = np.linspace(0, 1, img_size)
                if np.random.rand() > 0.5:
                    # Horizontal gradient
                    img[:, :, c] = gradient[np.newaxis, :]
                else:
                    # Vertical gradient
                    img[:, :, c] = gradient[:, np.newaxis]
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        images.append(img)
    
    images = np.array(images, dtype=np.float32)
    
    # Normalize to [-1, 1]
    images = normalize_images(images, method='tanh')
    
    return images


def load_mnist_style_data(data_path=None, num_samples=1000, img_size=28):
    """
    Load MNIST-style data
    
    If data_path is None, creates synthetic MNIST-style digit images.
    
    Args:
        data_path: Path to data file (None for synthetic)
        num_samples: Number of samples (used if synthetic)
        img_size: Image size
        
    Returns:
        Images array (num_samples, img_size, img_size, 1)
    """
    if data_path is None:
        # Create synthetic digit-like patterns
        print(f"Creating {num_samples} synthetic MNIST-style images...")
        images = []
        
        for _ in range(num_samples):
            img = np.zeros((img_size, img_size))
            
            # Create random digit-like shapes
            num_shapes = np.random.randint(1, 4)
            for _ in range(num_shapes):
                center_x = np.random.randint(img_size)
                center_y = np.random.randint(img_size)
                radius = np.random.randint(3, img_size // 4)
                
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] = 1.0
            
            images.append(img[:, :, np.newaxis])  # Add channel dimension
        
        images = np.array(images, dtype=np.float32)
    else:
        # Load from file (user-provided)
        images = np.load(data_path)
        if images.ndim == 3:
            images = images[:, :, :, np.newaxis]
    
    # Normalize
    images = normalize_images(images, method='tanh')
    
    return images


def simple_augmentation(images, flip_horizontal=True, flip_vertical=False):
    """
    Simple data augmentation
    
    Args:
        images: Images array
        flip_horizontal: Whether to randomly flip horizontally
        flip_vertical: Whether to randomly flip vertically
        
    Returns:
        Augmented images
    """
    augmented = []
    
    for img in images:
        # Random horizontal flip
        if flip_horizontal and np.random.rand() > 0.5:
            img = np.flip(img, axis=1)
        
        # Random vertical flip
        if flip_vertical and np.random.rand() > 0.5:
            img = np.flip(img, axis=0)
        
        augmented.append(img.copy())
    
    return np.array(augmented)


def save_images_grid(images, save_path, nrow=8):
    """
    Save a grid of images
    
    Args:
        images: Images array (N, H, W, C)
        save_path: Path to save image
        nrow: Number of images per row
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available. Saving as numpy array instead.")
        np.save(save_path.replace('.png', '.npy'), images)
        return
    
    n_images = len(images)
    n_rows = (n_images + nrow - 1) // nrow
    
    # Denormalize
    images = denormalize_images(images, method='tanh', to_uint8=True)
    
    img_h, img_w, channels = images.shape[1:]
    
    # Create grid
    grid = np.zeros((n_rows * img_h, nrow * img_w, channels), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = img
    
    # Convert to PIL and save
    if channels == 1:
        grid = grid[:, :, 0]  # Remove channel dimension for grayscale
        pil_img = Image.fromarray(grid, mode='L')
    else:
        pil_img = Image.fromarray(grid, mode='RGB')
    
    pil_img.save(save_path)
    print(f"Saved image grid to {save_path}")


class SimpleDataset:
    """
    Simple dataset wrapper
    
    Wraps numpy arrays and provides iteration interface.
    """
    
    def __init__(self, images, transform=None):
        """
        Initialize dataset
        
        Args:
            images: Images array (N, H, W, C)
            transform: Optional transform function
        """
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def get_batch(self, batch_size, shuffle=True):
        """
        Get batches for training
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Yields:
            Batches of images
        """
        indices = np.arange(len(self.images))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.images), batch_size):
            end_idx = min(start_idx + batch_size, len(self.images))
            batch_indices = indices[start_idx:end_idx]
            
            batch = self.images[batch_indices]
            
            if self.transform is not None:
                batch = np.array([self.transform(img) for img in batch])
            
            yield batch
