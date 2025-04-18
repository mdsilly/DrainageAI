"""
Training script for FixMatch semi-supervised learning.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import SemiSupervisedModel
from preprocessing.fixmatch_augmentation import create_augmentation_pair
from training.data_utils import create_fixmatch_dataloaders, create_validation_dataloader, prepare_batch


def evaluate_model(model, val_loader, device=None):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Prepare batch
            imagery, labels = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(imagery)
            
            # Compute loss
            loss = F.binary_cross_entropy(outputs, labels)
            
            # Compute IoU
            predictions = (outputs > 0.5).float()
            intersection = torch.sum(predictions * labels)
            union = torch.sum(predictions) + torch.sum(labels) - intersection
            iou = intersection / (union + 1e-6)
            
            # Update metrics
            batch_size = imagery.size(0)
            total_loss += loss.item() * batch_size
            total_iou += iou.item() * batch_size
            total_samples += batch_size
    
    # Compute average metrics
    avg_loss = total_loss / total_samples
    avg_iou = total_iou / total_samples
    
    return {
        'loss': avg_loss,
        'iou': avg_iou
    }


def train_fixmatch(
    labeled_dir, 
    unlabeled_dir, 
    output_dir,
    val_dir=None,
    num_epochs=30,
    batch_size=4,
    unlabeled_batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    pretrained=True,
    device=None,
    log_interval=10,
    checkpoint_interval=5
):
    """
    Train a model using FixMatch approach.
    
    Args:
        labeled_dir: Directory containing labeled data
        unlabeled_dir: Directory containing unlabeled data
        output_dir: Directory to save model checkpoints and logs
        val_dir: Directory containing validation data (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size for labeled data
        unlabeled_batch_size: Batch size for unlabeled data
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        pretrained: Whether to use pretrained weights for encoder
        device: Device to run training on
        log_interval: Interval for logging training metrics
        checkpoint_interval: Interval for saving model checkpoints
        
    Returns:
        Trained model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Create data loaders
    labeled_loader, unlabeled_loader = create_fixmatch_dataloaders(
        labeled_dir, unlabeled_dir, batch_size, unlabeled_batch_size
    )
    
    # Create validation loader if validation directory is provided
    val_loader = None
    if val_dir is not None:
        val_loader = create_validation_dataloader(val_dir, batch_size)
    
    # Create model
    model = SemiSupervisedModel(pretrained=pretrained)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Create augmentations
    weak_aug, strong_aug = create_augmentation_pair()
    
    # Training loop
    best_val_iou = 0.0
    global_step = 0
    
    for epoch in range(num_epochs):
        # Initialize iterators
        unlabeled_iter = iter(unlabeled_loader)
        
        # Track metrics
        epoch_start_time = time.time()
        supervised_losses = []
        unsupervised_losses = []
        total_losses = []
        
        # Set model to training mode
        model.train()
        
        # Train on labeled data with FixMatch
        for batch_idx, labeled_batch in enumerate(labeled_loader):
            # Get labeled data
            labeled_images, labels = prepare_batch(labeled_batch, device)
            
            # Get unlabeled data
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            unlabeled_images, _ = prepare_batch(unlabeled_batch, device)
            
            # Compute FixMatch loss
            total_loss, sup_loss, unsup_loss = model.fixmatch_loss(
                labeled_images, labels, unlabeled_images, weak_aug, strong_aug
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            supervised_losses.append(sup_loss.item())
            unsupervised_losses.append(unsup_loss.item())
            total_losses.append(total_loss.item())
            
            # Log metrics
            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(labeled_loader)}")
                print(f"  Supervised Loss: {sup_loss.item():.4f}")
                print(f"  Unsupervised Loss: {unsup_loss.item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                
                # Log to tensorboard
                writer.add_scalar('train/supervised_loss', sup_loss.item(), global_step)
                writer.add_scalar('train/unsupervised_loss', unsup_loss.item(), global_step)
                writer.add_scalar('train/total_loss', total_loss.item(), global_step)
                writer.add_scalar('train/confidence_threshold', model.confidence_threshold, global_step)
            
            global_step += 1
        
        # Update learning rate
        scheduler.step()
        
        # Compute epoch metrics
        epoch_duration = time.time() - epoch_start_time
        epoch_supervised_loss = np.mean(supervised_losses)
        epoch_unsupervised_loss = np.mean(unsupervised_losses)
        epoch_total_loss = np.mean(total_losses)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s")
        print(f"  Supervised Loss: {epoch_supervised_loss:.4f}")
        print(f"  Unsupervised Loss: {epoch_unsupervised_loss:.4f}")
        print(f"  Total Loss: {epoch_total_loss:.4f}")
        
        # Log epoch metrics to tensorboard
        writer.add_scalar('epoch/supervised_loss', epoch_supervised_loss, epoch)
        writer.add_scalar('epoch/unsupervised_loss', epoch_unsupervised_loss, epoch)
        writer.add_scalar('epoch/total_loss', epoch_total_loss, epoch)
        writer.add_scalar('epoch/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        # Evaluate on validation set
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device)
            val_loss = val_metrics['loss']
            val_iou = val_metrics['iou']
            
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation IoU: {val_iou:.4f}")
            
            # Log validation metrics to tensorboard
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/iou', val_iou, epoch)
            
            # Save best model
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_iou': val_iou,
                    'val_loss': val_loss,
                }, os.path.join(output_dir, 'model_best.pt'))
                
                print(f"  New best model saved with IoU: {val_iou:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(output_dir, f'model_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pt'))
    
    # Close tensorboard writer
    writer.close()
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model using FixMatch approach')
    parser.add_argument('--labeled-dir', required=True, help='Directory containing labeled data')
    parser.add_argument('--unlabeled-dir', required=True, help='Directory containing unlabeled data')
    parser.add_argument('--output-dir', required=True, help='Directory to save model checkpoints and logs')
    parser.add_argument('--val-dir', help='Directory containing validation data')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for labeled data')
    parser.add_argument('--unlabeled-batch-size', type=int, default=16, help='Batch size for unlabeled data')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained weights for encoder')
    parser.add_argument('--device', help='Device to run training on (e.g., cuda:0, cpu)')
    parser.add_argument('--log-interval', type=int, default=10, help='Interval for logging training metrics')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Interval for saving model checkpoints')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    train_fixmatch(
        labeled_dir=args.labeled_dir,
        unlabeled_dir=args.unlabeled_dir,
        output_dir=args.output_dir,
        val_dir=args.val_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        unlabeled_batch_size=args.unlabeled_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        device=device,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval
    )
