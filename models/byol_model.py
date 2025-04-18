"""
BYOL (Bootstrap Your Own Latent) model for self-supervised learning with multi-view support.

This implementation supports multi-modal learning with optical and SAR imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from .base_model import BaseModel


class BYOLProjector(nn.Module):
    """Projection head for BYOL."""
    
    def __init__(self, in_features, hidden_dim=4096, out_dim=256):
        super(BYOLProjector, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.projection(x)


class BYOLPredictor(nn.Module):
    """Prediction head for BYOL."""
    
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256):
        super(BYOLPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.predictor(x)


class BYOLModel(BaseModel):
    """
    BYOL model for self-supervised learning with multi-view support.
    
    This implementation supports both optical and SAR imagery as different views
    of the same scene, enabling cross-modal learning.
    """
    
    def __init__(self, pretrained=True, with_sar=True, momentum=0.99):
        super(BYOLModel, self).__init__()
        
        # Online encoder
        self.online_encoder = models.resnet50(pretrained=pretrained)
        
        # Modify first layer if using SAR data
        if with_sar:
            # Get the first conv layer
            first_conv = self.online_encoder.conv1
            
            # Create a new conv layer with additional input channels for SAR
            new_conv = nn.Conv2d(
                in_channels=3 + 1,  # RGB + 1 SAR channel
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            
            # Copy weights from the original conv layer for the RGB channels
            with torch.no_grad():
                new_conv.weight[:, :3] = first_conv.weight
                
                # Initialize the SAR channel weights with small random values
                nn.init.kaiming_normal_(
                    new_conv.weight[:, 3:],
                    mode='fan_out',
                    nonlinearity='relu'
                )
            
            # Replace the first conv layer
            self.online_encoder.conv1 = new_conv
        
        # Remove the final fully connected layer
        self.online_encoder = nn.Sequential(*list(self.online_encoder.children())[:-1])
        
        # Online projector
        self.online_projector = BYOLProjector(2048)
        
        # Online predictor
        self.predictor = BYOLPredictor()
        
        # Target encoder and projector (initialized as copies of online networks)
        self.target_encoder = self._copy_encoder(self.online_encoder)
        self.target_projector = self._copy_weights(self.online_projector)
        
        # EMA update momentum
        self.momentum = momentum
        
        # For fine-tuning
        self.prediction_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.fine_tuned = False
        self.with_sar = with_sar
    
    def _copy_encoder(self, encoder):
        """Create a copy of the encoder with detached parameters."""
        target_encoder = copy.deepcopy(encoder)
        
        # Detach parameters to stop gradient flow
        for param in target_encoder.parameters():
            param.requires_grad = False
            
        return target_encoder
    
    def _copy_weights(self, model):
        """Create a copy of a model with detached parameters."""
        target_model = copy.deepcopy(model)
        
        # Detach parameters to stop gradient flow
        for param in target_model.parameters():
            param.requires_grad = False
            
        return target_model
    
    def forward(self, x):
        """Forward pass of the model."""
        if self.fine_tuned:
            # For fine-tuning or inference
            features = self.online_encoder(x)
            features = torch.flatten(features, 1)
            return self.prediction_head(features)
        else:
            # For feature extraction
            features = self.online_encoder(x)
            features = torch.flatten(features, 1)
            return features
    
    def byol_forward(self, x):
        """Forward pass through the BYOL pipeline."""
        # Online network forward pass
        online_features = self.online_encoder(x)
        online_features = torch.flatten(online_features, 1)
        online_proj = self.online_projector(online_features)
        online_pred = self.predictor(online_proj)
        
        # Target network forward pass (no gradients needed)
        with torch.no_grad():
            target_features = self.target_encoder(x)
            target_features = torch.flatten(target_features, 1)
            target_proj = self.target_projector(target_features)
        
        return online_pred, target_proj, online_features
    
    def byol_loss(self, view1, view2):
        """
        Compute BYOL loss between two views of the same image.
        
        Args:
            view1: First augmented view
            view2: Second augmented view
            
        Returns:
            Loss value, online features
        """
        # Process views through online and target networks
        online_pred1, target_proj2, online_feat1 = self.byol_forward(view1)
        online_pred2, target_proj1, online_feat2 = self.byol_forward(view2)
        
        # Normalize projections and predictions
        online_pred1 = F.normalize(online_pred1, dim=-1)
        online_pred2 = F.normalize(online_pred2, dim=-1)
        target_proj1 = F.normalize(target_proj1, dim=-1)
        target_proj2 = F.normalize(target_proj2, dim=-1)
        
        # Symmetric loss
        loss1 = 2 - 2 * (online_pred1 * target_proj2).sum(dim=-1).mean()
        loss2 = 2 - 2 * (online_pred2 * target_proj1).sum(dim=-1).mean()
        
        loss = (loss1 + loss2) / 2
        
        return loss, (online_feat1 + online_feat2) / 2
    
    def byol_loss_multiview(self, optical_view1, optical_view2, sar_view=None):
        """
        Compute BYOL loss with optical and SAR views.
        
        Args:
            optical_view1: First augmented optical view
            optical_view2: Second augmented optical view
            sar_view: SAR view (optional)
            
        Returns:
            Loss value, online features
        """
        # If no SAR view is provided, fall back to standard BYOL loss
        if sar_view is None or not self.with_sar:
            return self.byol_loss(optical_view1, optical_view2)
        
        # Process optical views
        online_pred_opt1, target_proj_opt2, online_feat_opt1 = self.byol_forward(optical_view1)
        online_pred_opt2, target_proj_opt1, online_feat_opt2 = self.byol_forward(optical_view2)
        
        # Process SAR view
        online_pred_sar, target_proj_sar, online_feat_sar = self.byol_forward(sar_view)
        
        # Normalize all projections and predictions
        online_pred_opt1 = F.normalize(online_pred_opt1, dim=-1)
        online_pred_opt2 = F.normalize(online_pred_opt2, dim=-1)
        online_pred_sar = F.normalize(online_pred_sar, dim=-1)
        target_proj_opt1 = F.normalize(target_proj_opt1, dim=-1)
        target_proj_opt2 = F.normalize(target_proj_opt2, dim=-1)
        target_proj_sar = F.normalize(target_proj_sar, dim=-1)
        
        # Cross-modal losses (optical predicts SAR and vice versa)
        loss_opt1_sar = 2 - 2 * (online_pred_opt1 * target_proj_sar).sum(dim=-1).mean()
        loss_opt2_sar = 2 - 2 * (online_pred_opt2 * target_proj_sar).sum(dim=-1).mean()
        loss_sar_opt1 = 2 - 2 * (online_pred_sar * target_proj_opt1).sum(dim=-1).mean()
        loss_sar_opt2 = 2 - 2 * (online_pred_sar * target_proj_opt2).sum(dim=-1).mean()
        
        # Same-modal losses (optical predicts optical)
        loss_opt1_opt2 = 2 - 2 * (online_pred_opt1 * target_proj_opt2).sum(dim=-1).mean()
        loss_opt2_opt1 = 2 - 2 * (online_pred_opt2 * target_proj_opt1).sum(dim=-1).mean()
        
        # Combine all losses
        loss = (loss_opt1_sar + loss_opt2_sar + loss_sar_opt1 + loss_sar_opt2 + loss_opt1_opt2 + loss_opt2_opt1) / 6
        
        # Average features for downstream tasks
        features = (online_feat_opt1 + online_feat_opt2 + online_feat_sar) / 3
        
        return loss, features
    
    def update_target_network(self):
        """Update target network using exponential moving average."""
        # Update encoder
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = self.momentum * target_params.data + \
                                (1 - self.momentum) * online_params.data
        
        # Update projector
        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = self.momentum * target_params.data + \
                                (1 - self.momentum) * online_params.data
    
    def preprocess(self, data):
        """Preprocess input data before feeding to the model."""
        # For MVP, we'll assume data is already preprocessed
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
            features = self.online_encoder(x)
            features = torch.flatten(features, 1)
        return features
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_projector': self.target_projector.state_dict(),
            'prediction_head': self.prediction_head.state_dict(),
            'fine_tuned': self.fine_tuned,
            'with_sar': self.with_sar,
            'momentum': self.momentum
        }, path)
        
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.target_projector.load_state_dict(checkpoint['target_projector'])
        self.prediction_head.load_state_dict(checkpoint['prediction_head'])
        self.fine_tuned = checkpoint['fine_tuned']
        self.with_sar = checkpoint.get('with_sar', False)
        self.momentum = checkpoint.get('momentum', 0.99)
