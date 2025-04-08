"""
Self-supervised learning model for drainage pipe detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .base_model import BaseModel


class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, in_features, hidden_dim=512, out_dim=128):
        super(ContrastiveHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        x = self.projection(x)
        return F.normalize(x, dim=1)  # L2 normalization


class SelfSupervisedModel(BaseModel):
    """Self-supervised model for drainage pipe detection using contrastive learning."""
    
    def __init__(self, pretrained=True, fine_tuned=False):
        super(SelfSupervisedModel, self).__init__()
        
        # Load pretrained ResNet-50 as encoder
        self.encoder = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Projection head for contrastive learning
        self.projection_head = ContrastiveHead(2048)
        
        # For fine-tuning phase, add a prediction head
        self.fine_tuned = fine_tuned
        if fine_tuned:
            self.prediction_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        """Forward pass of the model."""
        # Get features from encoder
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        
        if self.training and not self.fine_tuned:
            # During self-supervised training, return projected features
            return self.projection_head(features)
        elif self.fine_tuned:
            # During fine-tuning or inference, return predictions
            return self.prediction_head(features)
        else:
            # During feature extraction, return features
            return features
    
    def contrastive_loss(self, z_i, z_j, temperature=0.5):
        """
        Compute NT-Xent loss for contrastive learning.
        
        Args:
            z_i: Projected features of first augmented view
            z_j: Projected features of second augmented view
            temperature: Temperature parameter
            
        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )
        
        # Remove diagonal (self-similarity)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Remove self-similarity from denominator
        mask = (~torch.eye(2 * batch_size, dtype=bool, device=z_i.device)).float()
        
        # Compute loss
        nominator = torch.exp(positives / temperature)
        denominator = mask * torch.exp(similarity_matrix / temperature)
        
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.mean(all_losses)
        
        return loss
    
    def preprocess(self, data):
        """Preprocess input data before feeding to the model."""
        # For self-supervised learning, we need to create two augmented views
        # of the same image for contrastive learning
        
        # For MVP, we'll assume data is already preprocessed
        # In a full implementation, this would include:
        # - Random cropping
        # - Random color jittering
        # - Random grayscale conversion
        # - Random Gaussian blur
        
        return data
    
    def postprocess(self, output):
        """Postprocess model output to generate final predictions."""
        # If in fine-tuned mode, output is already a probability map
        if self.fine_tuned:
            # Convert probability map to binary mask
            binary_mask = (output > 0.5).float()
            return binary_mask
        else:
            # If not fine-tuned, output is features
            # These would be used for downstream tasks
            return output
    
    def extract_features(self, x):
        """Extract features without projection head."""
        with torch.no_grad():
            features = self.encoder(x)
            features = torch.flatten(features, 1)
        return features
