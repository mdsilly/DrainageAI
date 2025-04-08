"""
Graph builder for constructing graph representations from geospatial data.
"""

import numpy as np
import torch
from torch_geometric.data import Data
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree


class GraphBuilder:
    """Graph builder for constructing graph representations from geospatial data."""
    
    def __init__(self, node_distance=10, edge_features=True):
        """
        Initialize the graph builder.
        
        Args:
            node_distance: Distance threshold for connecting nodes
            edge_features: Whether to include edge features
        """
        self.node_distance = node_distance
        self.edge_features = edge_features
        
    def build_graph_from_raster(self, imagery, elevation=None, meta=None, threshold=0.5):
        """
        Build a graph representation from raster data.
        
        Args:
            imagery: Input imagery (numpy array)
            elevation: Elevation data (optional)
            meta: Metadata (optional)
            threshold: Threshold for node selection
            
        Returns:
            torch_geometric.data.Data: Graph representation
        """
        # Convert to numpy array if needed
        if isinstance(imagery, torch.Tensor):
            imagery = imagery.numpy()
        
        if elevation is not None and isinstance(elevation, torch.Tensor):
            elevation = elevation.numpy()
        
        # Extract node positions and features
        node_positions, node_features = self._extract_nodes_from_raster(
            imagery, elevation, threshold
        )
        
        # Build edges
        edge_index, edge_attr = self._build_edges(
            node_positions, elevation
        )
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(node_positions, dtype=torch.float)
        )
        
        if self.edge_features and edge_attr is not None:
            graph_data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return graph_data
    
    def build_graph_from_vector(self, vector_data, imagery=None, elevation=None):
        """
        Build a graph representation from vector data.
        
        Args:
            vector_data: Vector data (GeoDataFrame)
            imagery: Input imagery for node features (optional)
            elevation: Elevation data (optional)
            
        Returns:
            torch_geometric.data.Data: Graph representation
        """
        # Extract node positions and features
        node_positions, node_features = self._extract_nodes_from_vector(
            vector_data, imagery, elevation
        )
        
        # Build edges
        edge_index, edge_attr = self._build_edges(
            node_positions, elevation
        )
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(node_positions, dtype=torch.float)
        )
        
        if self.edge_features and edge_attr is not None:
            graph_data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return graph_data
    
    def _extract_nodes_from_raster(self, imagery, elevation=None, threshold=0.5):
        """
        Extract nodes from raster data.
        
        Args:
            imagery: Input imagery (numpy array)
            elevation: Elevation data (optional)
            threshold: Threshold for node selection
            
        Returns:
            Tuple of node positions and features
        """
        # For simplicity, we'll use a grid-based approach for the MVP
        # In practice, we would use more sophisticated methods
        
        # Get dimensions
        if imagery.ndim == 3:
            c, h, w = imagery.shape
        else:
            h, w = imagery.shape
            c = 1
            imagery = imagery.reshape(1, h, w)
        
        # Create a grid of positions
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        positions = np.stack([y.flatten(), x.flatten()], axis=1)
        
        # Extract features at each position
        features = []
        for i in range(len(positions)):
            y, x = positions[i]
            # Extract features from imagery
            pixel_features = imagery[:, y, x]
            features.append(pixel_features)
        
        features = np.array(features)
        
        # If elevation data is available, add it to features
        if elevation is not None:
            if elevation.ndim == 2:
                elev_features = elevation[positions[:, 0], positions[:, 1]]
                elev_features = elev_features.reshape(-1, 1)
            else:
                elev_features = elevation[0, positions[:, 0], positions[:, 1]]
                elev_features = elev_features.reshape(-1, 1)
            features = np.concatenate([features, elev_features], axis=1)
        
        # Filter nodes based on threshold
        # For MVP, we'll use a simple approach
        # In practice, we would use more sophisticated methods
        if c == 1:
            mask = features[:, 0] > threshold
        else:
            # Use average of all channels
            mask = np.mean(features[:, :c], axis=1) > threshold
        
        # Apply mask
        positions = positions[mask]
        features = features[mask]
        
        return positions, features
    
    def _extract_nodes_from_vector(self, vector_data, imagery=None, elevation=None):
        """
        Extract nodes from vector data.
        
        Args:
            vector_data: Vector data (GeoDataFrame)
            imagery: Input imagery for node features (optional)
            elevation: Elevation data (optional)
            
        Returns:
            Tuple of node positions and features
        """
        # Extract node positions from vector data
        positions = []
        
        # For each feature
        for _, feature in vector_data.iterrows():
            geom = feature.geometry
            
            # Extract positions based on geometry type
            if isinstance(geom, Point):
                positions.append([geom.y, geom.x])
            elif isinstance(geom, LineString):
                # Add points along the line
                for y, x in geom.coords:
                    positions.append([y, x])
        
        positions = np.array(positions)
        
        # Extract features
        features = []
        
        # If imagery is available, extract features from it
        if imagery is not None:
            if isinstance(imagery, str):
                with rasterio.open(imagery) as src:
                    img = src.read()
                    transform = src.transform
            else:
                img = imagery
                transform = None
            
            # Extract features at each position
            for pos in positions:
                y, x = pos
                
                # Convert to pixel coordinates if transform is available
                if transform is not None:
                    col, row = ~transform * (x, y)
                    col, row = int(col), int(row)
                else:
                    row, col = int(y), int(x)
                
                # Check if within bounds
                if 0 <= row < img.shape[1] and 0 <= col < img.shape[2]:
                    pixel_features = img[:, row, col]
                else:
                    pixel_features = np.zeros(img.shape[0])
                
                features.append(pixel_features)
        else:
            # If no imagery, use simple position-based features
            features = positions.copy()
        
        features = np.array(features)
        
        # If elevation data is available, add it to features
        if elevation is not None:
            if isinstance(elevation, str):
                with rasterio.open(elevation) as src:
                    elev = src.read(1)  # Assume single band
                    transform = src.transform
            else:
                elev = elevation
                transform = None
            
            # Extract elevation at each position
            elev_features = []
            for pos in positions:
                y, x = pos
                
                # Convert to pixel coordinates if transform is available
                if transform is not None:
                    col, row = ~transform * (x, y)
                    col, row = int(col), int(row)
                else:
                    row, col = int(y), int(x)
                
                # Check if within bounds
                if 0 <= row < elev.shape[0] and 0 <= col < elev.shape[1]:
                    elev_val = elev[row, col]
                else:
                    elev_val = 0
                
                elev_features.append(elev_val)
            
            elev_features = np.array(elev_features).reshape(-1, 1)
            features = np.concatenate([features, elev_features], axis=1)
        
        return positions, features
    
    def _build_edges(self, positions, elevation=None, k=8):
        """
        Build edges between nodes based on proximity and elevation.
        
        Args:
            positions: Node positions
            elevation: Elevation data (optional)
            k: Number of nearest neighbors to connect
            
        Returns:
            Tuple of edge index and edge attributes
        """
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(positions)
        
        # Find k nearest neighbors for each node
        distances, indices = tree.query(positions, k=min(k+1, len(positions)))
        
        # Create edges
        edges = []
        edge_features = []
        
        for i in range(len(positions)):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Skip if distance is greater than threshold
                if distance > self.node_distance:
                    continue
                
                # Check elevation if available
                if elevation is not None:
                    # Get elevation at both positions
                    if elevation.ndim == 2:
                        elev_i = elevation[int(positions[i][0]), int(positions[i][1])]
                        elev_j = elevation[int(positions[neighbor_idx][0]), int(positions[neighbor_idx][1])]
                    else:
                        elev_i = elevation[0, int(positions[i][0]), int(positions[i][1])]
                        elev_j = elevation[0, int(positions[neighbor_idx][0]), int(positions[neighbor_idx][1])]
                    
                    # Water flows downhill
                    if elev_i <= elev_j:
                        continue
                    
                    # Add edge feature (elevation difference)
                    if self.edge_features:
                        edge_features.append([distance, elev_i - elev_j])
                else:
                    # Add edge feature (distance only)
                    if self.edge_features:
                        edge_features.append([distance])
                
                # Add edge
                edges.append([i, neighbor_idx])
        
        # Convert to numpy arrays
        edge_index = np.array(edges).T  # Shape: [2, num_edges]
        
        if self.edge_features and len(edge_features) > 0:
            edge_attr = np.array(edge_features)
        else:
            edge_attr = None
        
        return edge_index, edge_attr
