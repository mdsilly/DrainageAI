"""
Tests for DrainageAI models.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CNNModel, GNNModel, SelfSupervisedModel, EnsembleModel


class TestCNNModel(unittest.TestCase):
    """Tests for CNNModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CNNModel(pretrained=False)
        self.model.eval()
        
        # Create dummy input data
        self.input_data = torch.randn(1, 3, 224, 224)
    
    def test_forward(self):
        """Test forward pass."""
        with torch.no_grad():
            output = self.model(self.input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 1, 224, 224))
        
        # Check output range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_predict(self):
        """Test predict method."""
        with torch.no_grad():
            output = self.model.predict(self.input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 1, 224, 224))
        
        # Check output range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


class TestGNNModel(unittest.TestCase):
    """Tests for GNNModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = GNNModel()
        self.model.eval()
        
        # Create dummy input data
        from torch_geometric.data import Data
        
        # Create 10 nodes with 64 features each
        x = torch.randn(10, 64)
        
        # Create edges connecting each node to 3 random neighbors
        edge_index = []
        for i in range(10):
            for j in np.random.choice(10, 3, replace=False):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create positions
        pos = torch.randn(10, 2)
        
        # Create graph data
        self.graph_data = Data(x=x, edge_index=edge_index, pos=pos)
    
    def test_forward(self):
        """Test forward pass."""
        with torch.no_grad():
            output = self.model(self.graph_data)
        
        # Check output shape
        self.assertEqual(output.shape, (10, 1))
        
        # Check output range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_predict(self):
        """Test predict method."""
        # Create input data for predict method
        input_data = {
            'features': self.graph_data.x.numpy(),
            'positions': self.graph_data.pos.numpy()
        }
        
        with torch.no_grad():
            output = self.model.predict(input_data)
        
        # Check output type
        self.assertIsInstance(output, dict)
        self.assertIn('node_probs', output)
        self.assertIn('node_mask', output)


class TestSelfSupervisedModel(unittest.TestCase):
    """Tests for SelfSupervisedModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SelfSupervisedModel(pretrained=False, fine_tuned=True)
        self.model.eval()
        
        # Create dummy input data
        self.input_data = torch.randn(1, 3, 224, 224)
    
    def test_forward(self):
        """Test forward pass."""
        with torch.no_grad():
            output = self.model(self.input_data)
        
        # Check output shape (should be a single value per image in fine-tuned mode)
        self.assertEqual(output.shape, (1, 1))
        
        # Check output range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_predict(self):
        """Test predict method."""
        with torch.no_grad():
            output = self.model.predict(self.input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 1))
        
        # Check output range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


class TestEnsembleModel(unittest.TestCase):
    """Tests for EnsembleModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = EnsembleModel()
        self.model.eval()
        
        # Create dummy input data
        imagery = torch.randn(1, 3, 224, 224)
        
        # Create graph data
        from torch_geometric.data import Data
        
        # Create 10 nodes with 64 features each
        node_features = np.random.randn(10, 64)
        
        # Create positions
        node_positions = np.random.randn(10, 2)
        
        # Create input data
        self.input_data = {
            'imagery': imagery,
            'node_features': node_features,
            'node_positions': node_positions
        }
    
    def test_predict(self):
        """Test predict method."""
        with torch.no_grad():
            output = self.model.predict(self.input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 1, 224, 224))
        
        # Check output range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


if __name__ == '__main__':
    unittest.main()
