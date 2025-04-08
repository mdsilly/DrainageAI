"""
Base model class for DrainageAI models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Base class for all models in DrainageAI."""
    
    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def preprocess(self, data):
        """Preprocess input data before feeding to the model."""
        pass
    
    @abstractmethod
    def postprocess(self, output):
        """Postprocess model output to generate final predictions."""
        pass
    
    def save(self, path):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path))
        
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
