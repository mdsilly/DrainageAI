"""
Ensemble model that combines CNN, GNN, and self-supervised models for drainage pipe detection.
"""

import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel
from .cnn_model import CNNModel
from .gnn_model import GNNModel
from .ssl_model import SelfSupervisedModel


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines CNN, GNN, and self-supervised models
    for drainage pipe detection.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the ensemble model.
        
        Args:
            weights: Optional weights for each model [cnn_weight, gnn_weight, ssl_weight]
                    If None, equal weights will be used.
        """
        super(EnsembleModel, self).__init__()
        
        # Initialize individual models
        self.cnn_model = CNNModel(pretrained=True)
        self.gnn_model = GNNModel()
        self.ssl_model = SelfSupervisedModel(pretrained=True, fine_tuned=True)
        
        # Set ensemble weights
        if weights is None:
            self.weights = [1/3, 1/3, 1/3]  # Equal weights
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        # Attention mechanism for adaptive weighting (optional for MVP)
        self.use_attention = False
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            )
    
    def forward(self, data):
        """
        Forward pass of the ensemble model.
        
        Args:
            data: Dictionary containing inputs for each model:
                - 'imagery': Satellite imagery for CNN and SSL
                - 'graph_data': Graph representation for GNN
                
        Returns:
            Combined prediction
        """
        # Process data through each model
        cnn_output = self.cnn_model(data['imagery'])
        gnn_output = self.gnn_model(data['graph_data'])
        ssl_output = self.ssl_model(data['imagery'])
        
        # Combine outputs
        if self.use_attention:
            # Use attention mechanism to dynamically weight models
            # This is a simplified version - in practice, we would use
            # features from the data to determine weights
            confidence_scores = torch.tensor([
                torch.mean(cnn_output),
                torch.mean(gnn_output['node_probs']),
                torch.mean(ssl_output)
            ]).unsqueeze(0)
            
            weights = self.attention(confidence_scores).squeeze(0)
            combined = (
                weights[0] * cnn_output +
                weights[1] * self._format_gnn_output(gnn_output, cnn_output.shape) +
                weights[2] * ssl_output
            )
        else:
            # Use fixed weights
            combined = (
                self.weights[0] * cnn_output +
                self.weights[1] * self._format_gnn_output(gnn_output, cnn_output.shape) +
                self.weights[2] * ssl_output
            )
        
        return combined
    
    def _format_gnn_output(self, gnn_output, target_shape):
        """
        Format GNN output to match CNN output shape for combination.
        
        Args:
            gnn_output: Output from GNN model
            target_shape: Shape to match (from CNN output)
            
        Returns:
            Formatted GNN output
        """
        # This is a placeholder - in practice, we would need to convert
        # the graph representation to a spatial grid matching the CNN output
        # For MVP, we'll assume this conversion is handled elsewhere
        
        # Return a tensor of zeros matching the target shape
        return torch.zeros(target_shape, device=target_shape.device)
    
    def preprocess(self, data):
        """
        Preprocess input data for all models.
        
        Args:
            data: Raw input data
                
        Returns:
            Preprocessed data for each model
        """
        # Preprocess for CNN
        cnn_input = self.cnn_model.preprocess(data['imagery'])
        
        # Preprocess for GNN
        gnn_input = self.gnn_model.preprocess({
            'features': data['node_features'],
            'positions': data['node_positions'],
            'elevation': data.get('elevation')
        })
        
        # Preprocess for SSL
        ssl_input = self.ssl_model.preprocess(data['imagery'])
        
        return {
            'imagery': cnn_input,
            'graph_data': gnn_input
        }
    
    def postprocess(self, output):
        """
        Postprocess model output to generate final predictions.
        
        Args:
            output: Combined model output
                
        Returns:
            Final prediction
        """
        # Apply threshold to get binary mask
        binary_mask = (output > 0.5).float()
        
        # Additional postprocessing could include:
        # - Removing small isolated predictions
        # - Connecting nearby segments
        # - Ensuring network connectivity
        # - Vectorizing the results
        
        return binary_mask
    
    def predict(self, data):
        """
        Run inference on input data.
        
        Args:
            data: Raw input data
                
        Returns:
            Final prediction
        """
        # Set models to evaluation mode
        self.cnn_model.eval()
        self.gnn_model.eval()
        self.ssl_model.eval()
        
        # Preprocess data
        processed_data = self.preprocess(data)
        
        # Run forward pass
        with torch.no_grad():
            output = self.forward(processed_data)
            
        # Postprocess output
        result = self.postprocess(output)
        
        return result
