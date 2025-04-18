"""
Augmentation strategies for FixMatch semi-supervised learning.
"""

import random
import numpy as np
import torch
from scipy import ndimage


class WeakAugmentation:
    """Weak augmentation for FixMatch."""
    
    def __init__(self, flip_prob=0.5):
        """
        Initialize the weak augmentation.
        
        Args:
            flip_prob: Probability of applying horizontal flip
        """
        self.flip_prob = flip_prob
    
    def __call__(self, x):
        """
        Apply weak augmentations to input data.
        
        Args:
            x: Input data (torch.Tensor or numpy.ndarray)
            
        Returns:
            Augmented data (torch.Tensor)
        """
        # Convert to numpy array for easier manipulation
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            if x.ndim == 3:
                x = np.flip(x, axis=2)
            elif x.ndim == 4:  # Batch of images
                x = np.flip(x, axis=3)
            else:
                x = np.flip(x, axis=1)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            x = torch.from_numpy(x).float()
        
        return x


class StrongAugmentation:
    """Strong augmentation for FixMatch."""
    
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, noise_prob=0.5, noise_std=0.1):
        """
        Initialize the strong augmentation.
        
        Args:
            flip_prob: Probability of applying horizontal flip
            rotate_prob: Probability of applying rotation
            noise_prob: Probability of applying Gaussian noise
            noise_std: Standard deviation of Gaussian noise
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
    
    def __call__(self, x):
        """
        Apply strong augmentations to input data.
        
        Args:
            x: Input data (torch.Tensor or numpy.ndarray)
            
        Returns:
            Augmented data (torch.Tensor)
        """
        # Convert to numpy array for easier manipulation
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        # Start with weak augmentations: random horizontal flip
        if random.random() < self.flip_prob:
            if x.ndim == 3:
                x = np.flip(x, axis=2)
            elif x.ndim == 4:  # Batch of images
                x = np.flip(x, axis=3)
            else:
                x = np.flip(x, axis=1)
        
        # Add stronger augmentations
        
        # 1. Random rotation
        if random.random() < self.rotate_prob:
            angle = random.choice([0, 90, 180, 270])
            
            if x.ndim == 4:  # Batch of images
                rotated = np.zeros_like(x)
                for b in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        rotated[b, c] = ndimage.rotate(x[b, c], angle, reshape=False)
                x = rotated
            elif x.ndim == 3:  # Single image with channels
                rotated = np.zeros_like(x)
                for c in range(x.shape[0]):
                    rotated[c] = ndimage.rotate(x[c], angle, reshape=False)
                x = rotated
            else:  # Single channel image
                x = ndimage.rotate(x, angle, reshape=False)
        
        # 2. Random Gaussian noise
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, x.shape)
            x = x + noise
            x = np.clip(x, 0, 1)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            x = torch.from_numpy(x).float()
        
        return x


def create_augmentation_pair():
    """
    Create a pair of weak and strong augmentation functions.
    
    Returns:
        Tuple of (WeakAugmentation, StrongAugmentation)
    """
    weak_aug = WeakAugmentation()
    strong_aug = StrongAugmentation()
    
    return weak_aug, strong_aug
