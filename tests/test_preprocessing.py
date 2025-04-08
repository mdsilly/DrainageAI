"""
Tests for DrainageAI preprocessing modules.
"""

import os
import sys
import unittest
import torch
import numpy as np
import rasterio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import DataLoader, ImageProcessor, GraphBuilder, Augmentation


class TestImageProcessor(unittest.TestCase):
    """Tests for ImageProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.image_processor = ImageProcessor(target_size=(256, 256))
        
        # Create dummy input data
        self.imagery = np.random.randint(0, 256, (3, 512, 512), dtype=np.uint8)
    
    def test_preprocess(self):
        """Test preprocess method."""
        # Preprocess imagery
        preprocessed = self.image_processor.preprocess(self.imagery)
        
        # Check output type
        self.assertIsInstance(preprocessed, torch.Tensor)
        
        # Check output shape
        self.assertEqual(preprocessed.shape, (3, 256, 256))
        
        # Check output range
        self.assertTrue(torch.all(preprocessed >= 0))
        self.assertTrue(torch.all(preprocessed <= 1))
    
    def test_resize(self):
        """Test resize method."""
        # Resize imagery
        resized = self.image_processor.resize(self.imagery, (256, 256))
        
        # Check output shape
        self.assertEqual(resized.shape, (3, 256, 256))
    
    def test_normalize_imagery(self):
        """Test normalize_imagery method."""
        # Normalize imagery
        normalized = self.image_processor.normalize_imagery(self.imagery)
        
        # Check output shape
        self.assertEqual(normalized.shape, self.imagery.shape)
        
        # Check output range
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))


class TestGraphBuilder(unittest.TestCase):
    """Tests for GraphBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph_builder = GraphBuilder(node_distance=10)
        
        # Create dummy input data
        self.imagery = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        self.elevation = np.random.rand(64, 64)
    
    def test_build_graph_from_raster(self):
        """Test build_graph_from_raster method."""
        # Build graph
        graph = self.graph_builder.build_graph_from_raster(
            self.imagery, self.elevation
        )
        
        # Check output type
        from torch_geometric.data import Data
        self.assertIsInstance(graph, Data)
        
        # Check graph properties
        self.assertIn('x', graph)
        self.assertIn('edge_index', graph)
        self.assertIn('pos', graph)
    
    def test_extract_nodes_from_raster(self):
        """Test _extract_nodes_from_raster method."""
        # Extract nodes
        positions, features = self.graph_builder._extract_nodes_from_raster(
            self.imagery, self.elevation
        )
        
        # Check output types
        self.assertIsInstance(positions, np.ndarray)
        self.assertIsInstance(features, np.ndarray)
        
        # Check output shapes
        self.assertEqual(positions.shape[1], 2)  # (N, 2)
        self.assertEqual(features.shape[1], 4)  # (N, 4) - 3 channels + elevation
    
    def test_build_edges(self):
        """Test _build_edges method."""
        # Create dummy positions
        positions = np.random.rand(10, 2)
        
        # Build edges
        edge_index, edge_attr = self.graph_builder._build_edges(positions)
        
        # Check output types
        self.assertIsInstance(edge_index, np.ndarray)
        
        # Check output shapes
        self.assertEqual(edge_index.shape[0], 2)  # (2, E)


class TestAugmentation(unittest.TestCase):
    """Tests for Augmentation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.augmentation = Augmentation()
        
        # Create dummy input data
        self.imagery = np.random.rand(3, 64, 64).astype(np.float32)
        self.labels = np.random.randint(0, 2, (1, 64, 64)).astype(np.float32)
    
    def test_call(self):
        """Test __call__ method."""
        # Apply augmentation
        augmented_imagery, augmented_labels = self.augmentation(
            self.imagery, self.labels
        )
        
        # Check output types
        self.assertIsInstance(augmented_imagery, torch.Tensor)
        self.assertIsInstance(augmented_labels, torch.Tensor)
        
        # Check output shapes
        self.assertEqual(augmented_imagery.shape, torch.Size([3, 64, 64]))
        self.assertEqual(augmented_labels.shape, torch.Size([1, 64, 64]))
    
    def test_create_contrastive_pair(self):
        """Test create_contrastive_pair method."""
        # Create contrastive pair
        view1, view2 = self.augmentation.create_contrastive_pair(self.imagery)
        
        # Check output types
        self.assertIsInstance(view1, torch.Tensor)
        self.assertIsInstance(view2, torch.Tensor)
        
        # Check output shapes
        self.assertEqual(view1.shape, torch.Size([3, 64, 64]))
        self.assertEqual(view2.shape, torch.Size([3, 64, 64]))
        
        # Check that views are different
        self.assertFalse(torch.allclose(view1, view2))


class TestDataLoader(unittest.TestCase):
    """Tests for DataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create data loader
        self.data_loader = DataLoader(str(self.test_dir), batch_size=2)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_create_dataset(self):
        """Test create_dataset method."""
        # Create dummy imagery paths
        imagery_paths = [str(self.test_dir / f'imagery_{i}.tif') for i in range(2)]
        
        # Create dummy label paths
        label_paths = [str(self.test_dir / f'label_{i}.tif') for i in range(2)]
        
        # Create dummy imagery and label files
        for i in range(2):
            # Create dummy imagery
            imagery = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
            
            # Create dummy label
            label = np.random.randint(0, 2, (1, 64, 64), dtype=np.uint8)
            
            # Save dummy imagery
            with rasterio.open(
                imagery_paths[i],
                'w',
                driver='GTiff',
                height=64,
                width=64,
                count=3,
                dtype=np.uint8
            ) as dst:
                dst.write(imagery)
            
            # Save dummy label
            with rasterio.open(
                label_paths[i],
                'w',
                driver='GTiff',
                height=64,
                width=64,
                count=1,
                dtype=np.uint8
            ) as dst:
                dst.write(label)
        
        # Create dataset
        dataset = self.data_loader.create_dataset(imagery_paths, label_paths)
        
        # Check dataset length
        self.assertEqual(len(dataset), 2)
        
        # Check dataset item
        item = dataset[0]
        self.assertIn('imagery', item)
        self.assertIn('labels', item)
        self.assertIn('meta', item)
    
    def test_find_data_files(self):
        """Test find_data_files method."""
        # Create dummy files
        for ext in ['.tif', '.shp', '.txt']:
            (self.test_dir / f'test{ext}').touch()
        
        # Find files
        files = self.data_loader.find_data_files(str(self.test_dir), ['.tif', '.shp'])
        
        # Check output
        self.assertEqual(len(files), 2)
        self.assertTrue(any(f.endswith('.tif') for f in files))
        self.assertTrue(any(f.endswith('.shp') for f in files))


if __name__ == '__main__':
    unittest.main()
