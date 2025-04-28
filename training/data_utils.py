"""
Data utilities for DrainageAI training.

This module provides utility functions for data loading and processing.
"""

import torch
import torch.nn.functional as F


def custom_collate(batch):
    """
    Custom collate function for DataLoader that handles images of different sizes.
    
    This function resizes all images in a batch to the same dimensions (using the
    dimensions of the first image in the batch) before stacking them.
    
    Args:
        batch: List of tuples from the dataset's __getitem__ method
        
    Returns:
        Batched tensors with consistent dimensions
    """
    # Determine target size from the first item in the batch
    # (use the dimensions of the first image in the batch)
    target_size = batch[0][0].shape[1:]
    
    processed_batch = []
    for item in batch:
        processed_item = []
        for tensor in item:
            if tensor is not None:
                # Resize tensor to target size if it's an image tensor
                if tensor.dim() == 3:  # Image tensor (C, H, W)
                    if tensor.shape[1:] != target_size:
                        tensor = F.interpolate(
                            tensor.unsqueeze(0),  # Add batch dimension
                            size=target_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # Remove batch dimension
                elif tensor.dim() == 2:  # Label tensor (H, W)
                    if tensor.shape != target_size:
                        tensor = F.interpolate(
                            tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                            size=target_size,
                            mode='nearest'
                        ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
            processed_item.append(tensor)
        processed_batch.append(tuple(processed_item))
    
    # Now stack the processed items using PyTorch's default collate function
    return torch.utils.data.dataloader.default_collate(processed_batch)
