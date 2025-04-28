"""
Training script for BYOL (Bootstrap Your Own Latent) model.

This script implements BYOL training with multi-view support for optical and SAR imagery.
It supports training with few or no labeled images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import random
from torchvision import transforms
import rasterio
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.byol_model import BYOLModel
from preprocessing.augmentation import Augmentation
from .data_utils import custom_collate


class MultiViewDataset(Dataset):
    """Dataset for multi-view BYOL training with optical and SAR imagery."""
    
    def __init__(self, optical_paths, sar_paths=None, label_paths=None, transform=None, target_size=(256, 256)):
        """
        Initialize the dataset.
        
        Args:
            optical_paths: List of paths to optical imagery
            sar_paths: List of paths to SAR imagery (optional)
            label_paths: List of paths to label masks (optional)
            transform: Data augmentation transforms
            target_size: Target size for resizing images (height, width)
        """
        self.optical_paths = optical_paths
        self.sar_paths = sar_paths if sar_paths is not None else [None] * len(optical_paths)
        self.label_paths = label_paths if label_paths is not None else [None] * len(optical_paths)
        self.transform = transform
        self.target_size = target_size
        
        # Ensure all lists have the same length
        assert len(self.optical_paths) == len(self.sar_paths) == len(self.label_paths), \
            "All path lists must have the same length"
    
    def __len__(self):
        return len(self.optical_paths)
    
    def __getitem__(self, idx):
        # Load optical imagery
        with rasterio.open(self.optical_paths[idx]) as src:
            optical = src.read()
        
        # Load SAR imagery if available
        sar = None
        if self.sar_paths[idx] is not None:
            with rasterio.open(self.sar_paths[idx]) as src:
                sar = src.read()
        
        # Load label if available
        label = None
        if self.label_paths[idx] is not None:
            with rasterio.open(self.label_paths[idx]) as src:
                label = src.read(1)  # Assume single band
                label = torch.from_numpy(label).float()
        
        # Convert to torch tensors
        optical = torch.from_numpy(optical).float()
        if sar is not None:
            sar = torch.from_numpy(sar).float()
        
        # Resize all images to the target size if specified
        if self.target_size is not None:
            optical = F.interpolate(optical.unsqueeze(0), size=self.target_size, 
                                   mode='bilinear', align_corners=False).squeeze(0)
            
            if sar is not None:
                sar = F.interpolate(sar.unsqueeze(0), size=self.target_size, 
                                   mode='bilinear', align_corners=False).squeeze(0)
            
            if label is not None:
                label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=self.target_size, 
                                     mode='nearest').squeeze(0).squeeze(0)
        
        # Apply transformations
        if self.transform:
            optical_view1 = self.transform(optical)
            optical_view2 = self.transform(optical)
            
            if sar is not None:
                sar_view = self.transform(sar)
            else:
                sar_view = None
        else:
            optical_view1 = optical
            optical_view2 = optical
            sar_view = sar
        
        # Return different data depending on what's available
        if label is not None:
            if sar_view is not None:
                return optical_view1, optical_view2, sar_view, label
            else:
                return optical_view1, optical_view2, label
        else:
            if sar_view is not None:
                return optical_view1, optical_view2, sar_view
            else:
                return optical_view1, optical_view2


class BYOLTrainer:
    """Trainer for BYOL model."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: BYOL model
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_byol(self, train_loader, optimizer, epochs=100):
        """
        Train the BYOL model.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            epochs: Number of training epochs
            
        Returns:
            Training losses
        """
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                # Move batch to device
                batch = [b.to(self.device) if b is not None else None for b in batch]
                
                # Extract views
                if len(batch) == 3:
                    optical_view1, optical_view2, sar_view = batch
                else:
                    optical_view1, optical_view2 = batch
                    sar_view = None
                
                # Compute BYOL loss
                loss, _ = self.model.byol_loss_multiview(optical_view1, optical_view2, sar_view)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network
                self.model.update_target_network()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
            
            # Calculate average loss
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def fine_tune(self, train_loader, val_loader=None, epochs=10, lr=0.0001):
        """
        Fine-tune the BYOL model on labeled data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training and validation metrics
        """
        # Set model to fine-tuning mode
        self.model.fine_tuned = True
        
        # Freeze encoder parameters
        for param in self.model.online_encoder.parameters():
            param.requires_grad = False
        
        # Only train the prediction head
        optimizer = optim.Adam(self.model.prediction_head.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        train_metrics = []
        val_metrics = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            all_preds = []
            all_labels = []
            
            pbar = tqdm(train_loader, desc=f"Fine-tuning {epoch+1}/{epochs}")
            for batch in pbar:
                # Move batch to device
                batch = [b.to(self.device) if b is not None else None for b in batch]
                
                # Extract data and labels
                if len(batch) == 4:
                    optical, _, sar, label = batch
                    # Combine optical and SAR
                    if sar is not None:
                        # Ensure optical and SAR have the same spatial dimensions
                        if optical.shape[2:] != sar.shape[2:]:
                            sar = torch.nn.functional.interpolate(
                                sar, size=optical.shape[2:], mode='bilinear', align_corners=False
                            )
                        # Concatenate along channel dimension
                        x = torch.cat([optical, sar], dim=1)
                    else:
                        x = optical
                else:
                    x, _, label = batch
                
                # Forward pass
                output = self.model(x)
                
                # Compute loss
                loss = criterion(output, label.unsqueeze(1))
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
                
                # Store predictions and labels for metrics
                preds = (output > 0.5).float().cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())
            
            # Calculate average loss and metrics
            avg_loss = train_loss / len(train_loader)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            accuracy = accuracy_score(all_labels, all_preds)
            
            train_metrics.append({
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Validation
            if val_loader is not None:
                val_metrics_epoch = self.evaluate(val_loader)
                val_metrics.append(val_metrics_epoch)
                
                print(f"Validation - Accuracy: {val_metrics_epoch['accuracy']:.4f}, "
                      f"Precision: {val_metrics_epoch['precision']:.4f}, "
                      f"Recall: {val_metrics_epoch['recall']:.4f}, "
                      f"F1: {val_metrics_epoch['f1']:.4f}")
        
        return train_metrics, val_metrics
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = [b.to(self.device) if b is not None else None for b in batch]
                
                # Extract data and labels
                if len(batch) == 4:
                    optical, _, sar, label = batch
                    # Combine optical and SAR
                    if sar is not None:
                        # Ensure optical and SAR have the same spatial dimensions
                        if optical.shape[2:] != sar.shape[2:]:
                            sar = torch.nn.functional.interpolate(
                                sar, size=optical.shape[2:], mode='bilinear', align_corners=False
                            )
                        # Concatenate along channel dimension
                        x = torch.cat([optical, sar], dim=1)
                    else:
                        x = optical
                else:
                    x, _, label = batch
                
                # Forward pass
                output = self.model(x)
                
                # Store predictions and labels for metrics
                preds = (output > 0.5).float().cpu().numpy()
                labels = label.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def find_data_files(directory, extensions):
    """
    Find all files with specified extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files


def train_byol_pipeline(
    optical_dir,
    output_dir,
    sar_dir=None,
    label_dir=None,
    num_labeled=5,
    pretrained=True,
    byol_epochs=100,
    finetune_epochs=10,
    batch_size=4,
    byol_lr=0.0001,
    finetune_lr=0.0001,
    val_split=0.2,
    seed=42,
    resize_method='dataset',  # 'dataset' or 'collate'
    target_size=(256, 256)
):
    """
    Complete pipeline for BYOL training and fine-tuning.
    
    Args:
        optical_dir: Directory containing optical imagery
        output_dir: Directory to save outputs
        sar_dir: Directory containing SAR imagery (optional)
        label_dir: Directory containing labels (optional)
        num_labeled: Number of labeled examples to use
        pretrained: Whether to use pretrained weights
        byol_epochs: Number of BYOL pretraining epochs
        finetune_epochs: Number of fine-tuning epochs
        batch_size: Batch size for training
        byol_lr: Learning rate for BYOL training
        finetune_lr: Learning rate for fine-tuning
        val_split: Validation split ratio
        seed: Random seed
        
    Returns:
        Trained model and metrics
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find data files
    optical_files = find_data_files(optical_dir, ['.tif', '.tiff'])
    sar_files = find_data_files(sar_dir, ['.tif', '.tiff']) if sar_dir else None
    label_files = find_data_files(label_dir, ['.tif', '.tiff']) if label_dir else None
    
    # Match SAR files to optical files if available
    if sar_files:
        # Simple matching by filename
        sar_dict = {os.path.basename(f).split('.')[0]: f for f in sar_files}
        matched_sar_files = []
        
        for opt_file in optical_files:
            base_name = os.path.basename(opt_file).split('.')[0]
            if base_name in sar_dict:
                matched_sar_files.append(sar_dict[base_name])
            else:
                matched_sar_files.append(None)
    else:
        matched_sar_files = [None] * len(optical_files)
    
    # Match label files to optical files if available
    if label_files:
        # Simple matching by filename
        label_dict = {os.path.basename(f).split('.')[0]: f for f in label_files}
        matched_label_files = []
        
        for opt_file in optical_files:
            base_name = os.path.basename(opt_file).split('.')[0]
            if base_name in label_dict:
                matched_label_files.append(label_dict[base_name])
            else:
                matched_label_files.append(None)
    else:
        matched_label_files = [None] * len(optical_files)
    
    # Count available labeled data
    labeled_indices = [i for i, label in enumerate(matched_label_files) if label is not None]
    print(f"Found {len(labeled_indices)} labeled examples")
    
    # Limit labeled data if requested
    if num_labeled > 0 and num_labeled < len(labeled_indices):
        # Randomly select subset of labeled data
        random.shuffle(labeled_indices)
        labeled_indices = labeled_indices[:num_labeled]
        print(f"Using {num_labeled} labeled examples")
    
    # Create augmentation transform
    transform = transforms.Compose([
        # Add your custom augmentations here
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    
    # Create dataset
    if resize_method == 'dataset':
        # Resize images in the dataset
        dataset = MultiViewDataset(
            optical_paths=optical_files,
            sar_paths=matched_sar_files,
            label_paths=matched_label_files,
            transform=transform,
            target_size=target_size
        )
        collate_fn = None
    else:
        # Use custom collate function to resize images during batching
        dataset = MultiViewDataset(
            optical_paths=optical_files,
            sar_paths=matched_sar_files,
            label_paths=matched_label_files,
            transform=transform,
            target_size=None  # Don't resize in the dataset
        )
        collate_fn = custom_collate
    
    # Split labeled data into train and validation
    if labeled_indices:
        num_val = max(1, int(len(labeled_indices) * val_split))
        val_indices = labeled_indices[:num_val]
        train_labeled_indices = labeled_indices[num_val:]
        
        # Create train and validation datasets
        train_labeled_dataset = Subset(dataset, train_labeled_indices)
        val_dataset = Subset(dataset, val_indices)
        
        # Create data loaders
        train_labeled_loader = DataLoader(
            train_labeled_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    else:
        train_labeled_loader = None
        val_loader = None
    
    # Create data loader for all data (for BYOL pretraining)
    train_all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    with_sar = any(sar is not None for sar in matched_sar_files)
    model = BYOLModel(pretrained=pretrained, with_sar=with_sar)
    
    # Initialize trainer
    trainer = BYOLTrainer(model)
    
    # BYOL pretraining
    print("Starting BYOL pretraining...")
    optimizer = optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.predictor.parameters()),
        lr=byol_lr
    )
    byol_losses = trainer.train_byol(train_all_loader, optimizer, epochs=byol_epochs)
    
    # Save BYOL pretrained model
    byol_model_path = os.path.join(output_dir, "byol_pretrained.pth")
    model.save(byol_model_path)
    print(f"BYOL pretrained model saved to {byol_model_path}")
    
    # Fine-tuning if labeled data is available
    if train_labeled_loader is not None:
        print("Starting fine-tuning...")
        train_metrics, val_metrics = trainer.fine_tune(
            train_labeled_loader, val_loader, epochs=finetune_epochs, lr=finetune_lr
        )
        
        # Save fine-tuned model
        finetuned_model_path = os.path.join(output_dir, "byol_finetuned.pth")
        model.save(finetuned_model_path)
        print(f"Fine-tuned model saved to {finetuned_model_path}")
        
        # Save metrics
        if val_metrics:
            print(f"Final validation metrics: "
                  f"Accuracy: {val_metrics[-1]['accuracy']:.4f}, "
                  f"Precision: {val_metrics[-1]['precision']:.4f}, "
                  f"Recall: {val_metrics[-1]['recall']:.4f}, "
                  f"F1: {val_metrics[-1]['f1']:.4f}")
    
    return model, {
        'byol_losses': byol_losses,
        'train_metrics': train_metrics if train_labeled_loader is not None else None,
        'val_metrics': val_metrics if val_loader is not None else None
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BYOL training pipeline")
    
    parser.add_argument("--optical-dir", required=True, help="Directory containing optical imagery")
    parser.add_argument("--sar-dir", help="Directory containing SAR imagery (optional)")
    parser.add_argument("--label-dir", help="Directory containing labels (optional)")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--num-labeled", type=int, default=5, help="Number of labeled examples to use")
    parser.add_argument("--byol-epochs", type=int, default=100, help="Number of BYOL pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--byol-lr", type=float, default=0.0001, help="Learning rate for BYOL training")
    parser.add_argument("--finetune-lr", type=float, default=0.0001, help="Learning rate for fine-tuning")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained weights")
    parser.add_argument("--resize-method", choices=["dataset", "collate"], default="dataset", 
                      help="Method to use for resizing images: 'dataset' (resize in dataset) or 'collate' (resize during batching)")
    parser.add_argument("--target-size", type=str, default="256,256", 
                      help="Target size for resizing images (height,width)")
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    train_byol_pipeline(
        optical_dir=args.optical_dir,
        sar_dir=args.sar_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        num_labeled=args.num_labeled,
        pretrained=not args.no_pretrained,
        byol_epochs=args.byol_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        byol_lr=args.byol_lr,
        finetune_lr=args.finetune_lr,
        val_split=args.val_split,
        seed=args.seed,
        resize_method=args.resize_method,
        target_size=target_size
    )
