"""
Utilities for creating and using ensemble models.
"""

import torch
from models import CNNModel, SemiSupervisedModel, EnsembleModel


def create_ensemble_with_semi(semi_model_path, weights=None):
    """
    Create an ensemble model that includes the semi-supervised model.
    
    Args:
        semi_model_path: Path to semi-supervised model weights
        weights: Optional weights for each model [cnn_weight, semi_weight]
                If None, equal weights will be used.
    
    Returns:
        Ensemble model
    """
    # Create individual models
    cnn_model = CNNModel(pretrained=True)
    semi_model = SemiSupervisedModel(pretrained=False)
    
    # Load semi-supervised model weights
    semi_model.load(semi_model_path)
    
    # Set weights
    if weights is None:
        weights = [0.5, 0.5]  # Equal weights for MVP
    
    # Create ensemble with just these two models for MVP
    ensemble = EnsembleModel(weights=weights)
    
    # Replace models in ensemble
    ensemble.models = torch.nn.ModuleList([cnn_model, semi_model])
    
    # Update forward method to handle only CNN and Semi models
    def forward_wrapper(self, data):
        # Process data through each model
        cnn_output = self.models[0](data['imagery'])
        semi_output = self.models[1](data['imagery'])
        
        # Combine outputs with fixed weights
        combined = (
            self.weights[0] * cnn_output +
            self.weights[1] * semi_output
        )
        
        return combined
    
    # Monkey patch the forward method
    import types
    ensemble.forward = types.MethodType(forward_wrapper, ensemble)
    
    return ensemble


def evaluate_ensemble(ensemble_model, val_loader, device=None):
    """
    Evaluate ensemble model on validation set.
    
    Args:
        ensemble_model: Ensemble model to evaluate
        val_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics
    """
    from training.train_fixmatch import evaluate_model
    
    # Prepare data for ensemble
    def prepare_ensemble_batch(batch, device=None):
        imagery = batch['imagery']
        labels = batch.get('labels')
        
        if device is not None:
            imagery = imagery.to(device)
            if labels is not None:
                labels = labels.to(device)
        
        return {'imagery': imagery}, labels
    
    # Set model to evaluation mode
    ensemble_model.eval()
    
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Prepare batch
            data, labels = prepare_ensemble_batch(batch, device)
            
            # Forward pass
            outputs = ensemble_model(data)
            
            # Compute loss
            loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
            
            # Compute IoU
            predictions = (outputs > 0.5).float()
            intersection = torch.sum(predictions * labels)
            union = torch.sum(predictions) + torch.sum(labels) - intersection
            iou = intersection / (union + 1e-6)
            
            # Update metrics
            batch_size = labels.size(0)
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
