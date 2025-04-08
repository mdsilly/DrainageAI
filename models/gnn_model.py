"""
Graph Neural Network model for drainage pipe detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric.data import Data
import numpy as np
from .base_model import BaseModel


class DrainageGNN(nn.Module):
    """Graph Neural Network for drainage network modeling."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(DrainageGNN, self).__init__()
        
        # Use GraphSAGE as the base GNN architecture
        self.gnn = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
        )
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        """Forward pass of the GNN model."""
        x, edge_index = data.x, data.edge_index
        
        # Apply GNN to get node embeddings
        x = self.gnn(x, edge_index)
        
        # Predict node features (drainage probability)
        x = self.predictor(x)
        
        return x


class GNNModel(BaseModel):
    """GNN model for drainage pipe detection using graph representation."""
    
    def __init__(self, in_channels=64, hidden_channels=128, out_channels=1):
        super(GNNModel, self).__init__()
        
        self.model = DrainageGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )
        
    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)
    
    def preprocess(self, data):
        """
        Preprocess input data to create a graph representation.
        
        Args:
            data: Dictionary containing:
                - 'features': Node features (N x F)
                - 'positions': Node positions (N x 2)
                - 'elevation': Elevation data (optional)
        
        Returns:
            torch_geometric.data.Data: Graph representation
        """
        # Extract node features and positions
        features = data['features']
        positions = data['positions']
        
        # Convert to tensors if needed
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float)
        
        # Create edges based on proximity and elevation (if available)
        edge_index = self._create_edges(positions, data.get('elevation'))
        
        # Create PyTorch Geometric Data object
        graph_data = Data(x=features, edge_index=edge_index, pos=positions)
        
        return graph_data
    
    def _create_edges(self, positions, elevation=None, k=8):
        """
        Create edges between nodes based on proximity and elevation.
        
        Args:
            positions: Node positions (N x 2)
            elevation: Elevation data (optional)
            k: Number of nearest neighbors to connect
        
        Returns:
            torch.Tensor: Edge index tensor (2 x E)
        """
        # For MVP, use simple k-nearest neighbors approach
        n = positions.shape[0]
        edges = []
        
        # Compute pairwise distances
        for i in range(n):
            # Calculate Euclidean distances to all other nodes
            dists = torch.norm(positions - positions[i].unsqueeze(0), dim=1)
            
            # Find k nearest neighbors (excluding self)
            dists[i] = float('inf')  # Exclude self
            _, indices = torch.topk(dists, k=min(k, n-1), largest=False)
            
            # Create edges
            for j in indices:
                # If elevation data is available, only create edge if water can flow
                if elevation is not None:
                    # Water flows downhill
                    if elevation[i] <= elevation[j]:
                        continue
                
                edges.append([i, j.item()])
        
        # Convert to PyTorch edge format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def postprocess(self, output):
        """
        Postprocess model output to generate final predictions.
        
        Args:
            output: Model output (node probabilities)
        
        Returns:
            Dictionary containing:
                - 'node_probs': Node probabilities
                - 'edges': Connected edges that form the drainage network
        """
        # Threshold node probabilities
        node_mask = output > 0.5
        
        # Extract subgraph of predicted drainage nodes
        # This would be implemented based on the specific graph structure
        # For MVP, we'll return the node probabilities and let downstream
        # processing handle the network extraction
        
        return {
            'node_probs': output,
            'node_mask': node_mask
        }
