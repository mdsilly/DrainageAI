"""
Data augmentation for geospatial data.
"""

import numpy as np
import torch
import cv2
import random
from torchvision import transforms


class Augmentation:
    """Data augmentation for geospatial data."""
    
    def __init__(self, augmentation_types=None, p=0.5):
        """
        Initialize the augmentation.
        
        Args:
            augmentation_types: List of augmentation types to apply
            p: Probability of applying each augmentation
        """
        self.p = p
        
        # Default augmentation types
        if augmentation_types is None:
            self.augmentation_types = [
                'flip_horizontal',
                'flip_vertical',
                'rotate',
                'crop',
                'color_jitter',
                'gaussian_blur',
                'gaussian_noise'
            ]
        else:
            self.augmentation_types = augmentation_types
    
    def __call__(self, imagery, labels=None):
        """
        Apply augmentations to imagery and labels.
        
        Args:
            imagery: Input imagery (torch.Tensor)
            labels: Input labels (torch.Tensor, optional)
            
        Returns:
            Augmented imagery and labels
        """
        # Convert to numpy arrays for easier manipulation
        if isinstance(imagery, torch.Tensor):
            imagery = imagery.numpy()
        
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        # Apply augmentations
        for aug_type in self.augmentation_types:
            if random.random() < self.p:
                if aug_type == 'flip_horizontal':
                    imagery, labels = self.flip_horizontal(imagery, labels)
                elif aug_type == 'flip_vertical':
                    imagery, labels = self.flip_vertical(imagery, labels)
                elif aug_type == 'rotate':
                    imagery, labels = self.rotate(imagery, labels)
                elif aug_type == 'crop':
                    imagery, labels = self.random_crop(imagery, labels)
                elif aug_type == 'color_jitter':
                    imagery = self.color_jitter(imagery)
                elif aug_type == 'gaussian_blur':
                    imagery = self.gaussian_blur(imagery)
                elif aug_type == 'gaussian_noise':
                    imagery = self.gaussian_noise(imagery)
        
        # Convert back to torch tensors
        imagery = torch.from_numpy(imagery).float()
        
        if labels is not None:
            labels = torch.from_numpy(labels).float()
            return imagery, labels
        else:
            return imagery
    
    def flip_horizontal(self, imagery, labels=None):
        """
        Flip imagery and labels horizontally.
        
        Args:
            imagery: Input imagery (numpy array)
            labels: Input labels (numpy array, optional)
            
        Returns:
            Flipped imagery and labels
        """
        # Flip imagery
        if imagery.ndim == 3:
            imagery = np.flip(imagery, axis=2)
        else:
            imagery = np.flip(imagery, axis=1)
        
        # Flip labels if provided
        if labels is not None:
            if labels.ndim == 3:
                labels = np.flip(labels, axis=2)
            else:
                labels = np.flip(labels, axis=1)
        
        return imagery, labels
    
    def flip_vertical(self, imagery, labels=None):
        """
        Flip imagery and labels vertically.
        
        Args:
            imagery: Input imagery (numpy array)
            labels: Input labels (numpy array, optional)
            
        Returns:
            Flipped imagery and labels
        """
        # Flip imagery
        if imagery.ndim == 3:
            imagery = np.flip(imagery, axis=1)
        else:
            imagery = np.flip(imagery, axis=0)
        
        # Flip labels if provided
        if labels is not None:
            if labels.ndim == 3:
                labels = np.flip(labels, axis=1)
            else:
                labels = np.flip(labels, axis=0)
        
        return imagery, labels
    
    def rotate(self, imagery, labels=None, angle=None):
        """
        Rotate imagery and labels.
        
        Args:
            imagery: Input imagery (numpy array)
            labels: Input labels (numpy array, optional)
            angle: Rotation angle in degrees (if None, random angle)
            
        Returns:
            Rotated imagery and labels
        """
        # Choose random angle if not provided
        if angle is None:
            angle = random.choice([90, 180, 270])
        
        # Rotate imagery
        if imagery.ndim == 3:
            c, h, w = imagery.shape
            rotated = np.zeros_like(imagery)
            for i in range(c):
                rotated[i] = self._rotate_channel(imagery[i], angle)
        else:
            rotated = self._rotate_channel(imagery, angle)
        
        # Rotate labels if provided
        if labels is not None:
            if labels.ndim == 3:
                c, h, w = labels.shape
                rotated_labels = np.zeros_like(labels)
                for i in range(c):
                    rotated_labels[i] = self._rotate_channel(labels[i], angle)
            else:
                rotated_labels = self._rotate_channel(labels, angle)
        else:
            rotated_labels = None
        
        return rotated, rotated_labels
    
    def _rotate_channel(self, channel, angle):
        """
        Rotate a single channel.
        
        Args:
            channel: Input channel (numpy array)
            angle: Rotation angle in degrees
            
        Returns:
            Rotated channel
        """
        # Get dimensions
        h, w = channel.shape
        
        # Create rotation matrix
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(channel, matrix, (w, h))
        
        return rotated
    
    def random_crop(self, imagery, labels=None, crop_size=None):
        """
        Randomly crop imagery and labels.
        
        Args:
            imagery: Input imagery (numpy array)
            labels: Input labels (numpy array, optional)
            crop_size: Crop size (height, width) (if None, random size)
            
        Returns:
            Cropped imagery and labels
        """
        # Get dimensions
        if imagery.ndim == 3:
            c, h, w = imagery.shape
        else:
            h, w = imagery.shape
        
        # Choose random crop size if not provided
        if crop_size is None:
            crop_h = random.randint(int(0.7 * h), h)
            crop_w = random.randint(int(0.7 * w), w)
            crop_size = (crop_h, crop_w)
        else:
            crop_h, crop_w = crop_size
        
        # Choose random crop position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # Crop imagery
        if imagery.ndim == 3:
            cropped = imagery[:, top:top+crop_h, left:left+crop_w]
        else:
            cropped = imagery[top:top+crop_h, left:left+crop_w]
        
        # Crop labels if provided
        if labels is not None:
            if labels.ndim == 3:
                cropped_labels = labels[:, top:top+crop_h, left:left+crop_w]
            else:
                cropped_labels = labels[top:top+crop_h, left:left+crop_w]
        else:
            cropped_labels = None
        
        return cropped, cropped_labels
    
    def color_jitter(self, imagery, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        Apply color jittering to imagery.
        
        Args:
            imagery: Input imagery (numpy array)
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
            
        Returns:
            Color jittered imagery
        """
        # Only apply to RGB imagery
        if imagery.shape[0] != 3:
            return imagery
        
        # Convert to torch tensor for torchvision transforms
        tensor = torch.from_numpy(imagery).float()
        
        # Apply color jitter
        jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
        jittered = jitter(tensor)
        
        return jittered.numpy()
    
    def gaussian_blur(self, imagery, kernel_size=5, sigma=(0.1, 2.0)):
        """
        Apply Gaussian blur to imagery.
        
        Args:
            imagery: Input imagery (numpy array)
            kernel_size: Kernel size for Gaussian blur
            sigma: Range of sigma values for Gaussian blur
            
        Returns:
            Blurred imagery
        """
        # Choose random sigma
        sigma_value = random.uniform(sigma[0], sigma[1])
        
        # Apply Gaussian blur
        if imagery.ndim == 3:
            c, h, w = imagery.shape
            blurred = np.zeros_like(imagery)
            for i in range(c):
                blurred[i] = cv2.GaussianBlur(
                    imagery[i], (kernel_size, kernel_size), sigma_value
                )
        else:
            blurred = cv2.GaussianBlur(
                imagery, (kernel_size, kernel_size), sigma_value
            )
        
        return blurred
    
    def gaussian_noise(self, imagery, mean=0, std=0.1):
        """
        Add Gaussian noise to imagery.
        
        Args:
            imagery: Input imagery (numpy array)
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy imagery
        """
        # Generate noise
        noise = np.random.normal(mean, std, imagery.shape)
        
        # Add noise to imagery
        noisy = imagery + noise
        
        # Clip to valid range
        noisy = np.clip(noisy, 0, 1)
        
        return noisy
    
    def create_contrastive_pair(self, imagery):
        """
        Create a contrastive pair from imagery.
        
        Args:
            imagery: Input imagery (torch.Tensor)
            
        Returns:
            Tuple of two augmented views
        """
        # Convert to numpy array
        if isinstance(imagery, torch.Tensor):
            imagery = imagery.numpy()
        
        # Create first view with a subset of augmentations
        view1_augs = random.sample(self.augmentation_types, k=min(3, len(self.augmentation_types)))
        view1 = imagery.copy()
        
        for aug_type in view1_augs:
            if aug_type == 'flip_horizontal':
                view1, _ = self.flip_horizontal(view1)
            elif aug_type == 'flip_vertical':
                view1, _ = self.flip_vertical(view1)
            elif aug_type == 'rotate':
                view1, _ = self.rotate(view1)
            elif aug_type == 'crop':
                view1, _ = self.random_crop(view1)
            elif aug_type == 'color_jitter':
                view1 = self.color_jitter(view1)
            elif aug_type == 'gaussian_blur':
                view1 = self.gaussian_blur(view1)
            elif aug_type == 'gaussian_noise':
                view1 = self.gaussian_noise(view1)
        
        # Create second view with a different subset of augmentations
        view2_augs = random.sample(self.augmentation_types, k=min(3, len(self.augmentation_types)))
        view2 = imagery.copy()
        
        for aug_type in view2_augs:
            if aug_type == 'flip_horizontal':
                view2, _ = self.flip_horizontal(view2)
            elif aug_type == 'flip_vertical':
                view2, _ = self.flip_vertical(view2)
            elif aug_type == 'rotate':
                view2, _ = self.rotate(view2)
            elif aug_type == 'crop':
                view2, _ = self.random_crop(view2)
            elif aug_type == 'color_jitter':
                view2 = self.color_jitter(view2)
            elif aug_type == 'gaussian_blur':
                view2 = self.gaussian_blur(view2)
            elif aug_type == 'gaussian_noise':
                view2 = self.gaussian_noise(view2)
        
        # Convert back to torch tensors
        view1 = torch.from_numpy(view1).float()
        view2 = torch.from_numpy(view2).float()
        
        return view1, view2
