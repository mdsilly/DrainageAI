"""
Example script for training a model using FixMatch approach and creating an ensemble.
"""

import os
import torch
from training import train_fixmatch, create_ensemble_with_semi, evaluate_ensemble, create_validation_dataloader


def main():
    """Main function."""
    # Set paths
    labeled_dir = 'data/labeled'
    unlabeled_dir = 'data/unlabeled'
    val_dir = 'data/validation'
    output_dir = 'data/models/fixmatch'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model using FixMatch
    print("Training model using FixMatch...")
    model = train_fixmatch(
        labeled_dir=labeled_dir,
        unlabeled_dir=unlabeled_dir,
        output_dir=output_dir,
        val_dir=val_dir,
        num_epochs=30,
        batch_size=4,
        unlabeled_batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        pretrained=True,
        device=device,
        log_interval=10,
        checkpoint_interval=5
    )
    
    # Create ensemble model
    print("Creating ensemble model...")
    semi_model_path = os.path.join(output_dir, 'model_best.pt')
    ensemble = create_ensemble_with_semi(semi_model_path)
    
    # Evaluate ensemble model
    print("Evaluating ensemble model...")
    val_loader = create_validation_dataloader(val_dir, batch_size=4)
    metrics = evaluate_ensemble(ensemble, val_loader, device)
    
    print(f"Ensemble model metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  IoU: {metrics['iou']:.4f}")
    
    # Save ensemble model
    print("Saving ensemble model...")
    torch.save(ensemble.state_dict(), os.path.join(output_dir, 'ensemble_model.pt'))
    
    print("Done!")


if __name__ == '__main__':
    main()
