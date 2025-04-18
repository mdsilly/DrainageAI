"""
Semi-supervised learning model for drainage pipe detection using FixMatch approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base_model import BaseModel


class SemiSupervisedModel(BaseModel):
    """Semi-supervised model using FixMatch approach."""
    
    def __init__(self, pretrained=True):
        super(SemiSupervisedModel, self).__init__()
        
        # Use ResNet-18 instead of ResNet-50 for faster training
        self.encoder = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Simpler prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(512, 256),  # ResNet-18 has 512 features
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Fixed confidence threshold
        self.confidence_threshold = 0.95
    
    def forward(self, x):
        """Forward pass of the model."""
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        return self.prediction_head(features)
    
    def fixmatch_loss(self, labeled_images, labels, unlabeled_images, weak_aug, strong_aug):
        """
        Compute FixMatch loss combining supervised and unsupervised components.
        
        Args:
            labeled_images: Batch of labeled images
            labels: Ground truth labels
            unlabeled_images: Batch of unlabeled images
            weak_aug: Weak augmentation function
            strong_aug: Strong augmentation function
            
        Returns:
            Total loss, supervised loss, unsupervised loss
        """
        # Supervised loss
        labeled_preds = self(labeled_images)
        supervised_loss = F.binary_cross_entropy(labeled_preds, labels)
        
        # Generate pseudo-labels
        with torch.no_grad():
            weak_aug_images = weak_aug(unlabeled_images)
            pseudo_outputs = self(weak_aug_images)
            # Only keep high-confidence predictions
            mask = (pseudo_outputs > self.confidence_threshold).float()
            pseudo_labels = (pseudo_outputs > 0.5).float()
        
        # Unsupervised loss
        strong_aug_images = strong_aug(unlabeled_images)
        strong_aug_preds = self(strong_aug_images)
        unsupervised_loss = (F.binary_cross_entropy(
            strong_aug_preds, pseudo_labels, reduction='none'
        ) * mask).mean()
        
        # Total loss (lambda=1 for simplicity)
        total_loss = supervised_loss + unsupervised_loss
        
        return total_loss, supervised_loss, unsupervised_loss
    
    def preprocess(self, data):
        """Preprocess input data before feeding to the model."""
        # For MVP, we'll assume data is already preprocessed
        return data
    
    def postprocess(self, output):
        """Postprocess model output to generate final predictions."""
        # Simple thresholding
        binary_mask = (output > 0.5).float()
        return binary_mask
    
    def predict(self, data):
        """Run inference on input data."""
        # Set model to evaluation mode
        self.eval()
        
        # Preprocess data
        x = self.preprocess(data)
        
        # Run forward pass
        with torch.no_grad():
            output = self.forward(x)
            
        # Postprocess output
        result = self.postprocess(output)
        
        return result
