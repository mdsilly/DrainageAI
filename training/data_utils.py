"""
Data utilities for FixMatch training.
"""

import os
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from preprocessing import DataLoader


def create_fixmatch_dataloaders(labeled_dir, unlabeled_dir, batch_size=4, unlabeled_batch_size=16):
    """
    Create data loaders for FixMatch training.
    
    Args:
        labeled_dir: Directory containing labeled data
        unlabeled_dir: Directory containing unlabeled data
        batch_size: Batch size for labeled data
        unlabeled_batch_size: Batch size for unlabeled data
        
    Returns:
        Tuple of (labeled_loader, unlabeled_loader)
    """
    # Create data loader
    data_loader = DataLoader(labeled_dir, batch_size=batch_size)
    
    # Find labeled data
    labeled_imagery_paths = data_loader.find_data_files(
        os.path.join(labeled_dir, "imagery"),
        [".tif", ".tiff"]
    )
    
    label_paths = data_loader.find_data_files(
        os.path.join(labeled_dir, "labels"),
        [".tif", ".tiff", ".shp"]
    )
    
    # Find unlabeled data
    unlabeled_imagery_paths = data_loader.find_data_files(
        os.path.join(unlabeled_dir, "imagery"),
        [".tif", ".tiff"]
    )
    
    # Create datasets
    labeled_dataset = data_loader.create_dataset(labeled_imagery_paths, label_paths)
    unlabeled_dataset = data_loader.create_dataset(unlabeled_imagery_paths)
    
    # Create data loaders
    labeled_loader = data_loader.create_dataloader(labeled_dataset, shuffle=True)
    
    # Create unlabeled data loader with different batch size
    unlabeled_loader = TorchDataLoader(
        unlabeled_dataset,
        batch_size=unlabeled_batch_size,
        shuffle=True,
        num_workers=data_loader.num_workers
    )
    
    return labeled_loader, unlabeled_loader


def create_validation_dataloader(val_dir, batch_size=4):
    """
    Create validation data loader.
    
    Args:
        val_dir: Directory containing validation data
        batch_size: Batch size for validation data
        
    Returns:
        Validation data loader
    """
    # Create data loader
    data_loader = DataLoader(val_dir, batch_size=batch_size)
    
    # Find validation data
    val_imagery_paths = data_loader.find_data_files(
        os.path.join(val_dir, "imagery"),
        [".tif", ".tiff"]
    )
    
    val_label_paths = data_loader.find_data_files(
        os.path.join(val_dir, "labels"),
        [".tif", ".tiff", ".shp"]
    )
    
    # Create dataset
    val_dataset = data_loader.create_dataset(val_imagery_paths, val_label_paths)
    
    # Create data loader
    val_loader = data_loader.create_dataloader(val_dataset, shuffle=False)
    
    return val_loader


def prepare_batch(batch, device=None):
    """
    Prepare batch for training.
    
    Args:
        batch: Batch from data loader
        device: Device to move data to
        
    Returns:
        Tuple of (imagery, labels)
    """
    imagery = batch['imagery']
    labels = batch.get('labels')
    
    if device is not None:
        imagery = imagery.to(device)
        if labels is not None:
            labels = labels.to(device)
    
    return imagery, labels
