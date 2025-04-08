"""
Data loader for geospatial data.
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


class GeospatialDataset(Dataset):
    """Dataset for geospatial data."""
    
    def __init__(self, imagery_paths, label_paths=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            imagery_paths: List of paths to imagery files
            label_paths: List of paths to label files (optional)
            transform: Transforms to apply to the data
        """
        self.imagery_paths = imagery_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.imagery_paths)
    
    def __getitem__(self, idx):
        # Load imagery
        with rasterio.open(self.imagery_paths[idx]) as src:
            # Read all bands
            imagery = src.read()
            # Get metadata
            meta = src.meta
        
        # Convert to torch tensor and normalize
        imagery = torch.from_numpy(imagery).float()
        imagery = imagery / 255.0  # Simple normalization
        
        # Load labels if available
        if self.label_paths is not None:
            # For vector data (e.g., shapefiles)
            if self.label_paths[idx].endswith('.shp'):
                labels = self._load_vector_labels(self.label_paths[idx], meta)
            # For raster data
            else:
                with rasterio.open(self.label_paths[idx]) as src:
                    labels = src.read(1)  # Assume single band for labels
                labels = torch.from_numpy(labels).float()
        else:
            labels = None
        
        # Apply transforms if available
        if self.transform:
            imagery, labels = self.transform(imagery, labels)
        
        # Return data
        if labels is not None:
            return {'imagery': imagery, 'labels': labels, 'meta': meta}
        else:
            return {'imagery': imagery, 'meta': meta}
    
    def _load_vector_labels(self, path, meta):
        """
        Load vector labels and convert to raster mask.
        
        Args:
            path: Path to vector file
            meta: Metadata from imagery
            
        Returns:
            Raster mask
        """
        # Load vector data
        gdf = gpd.read_file(path)
        
        # Create empty mask
        mask = np.zeros((meta['height'], meta['width']), dtype=np.float32)
        
        # Rasterize vectors
        # This is a simplified version - in practice, we would use
        # rasterio.features.rasterize or similar
        
        # For each feature
        for _, feature in gdf.iterrows():
            geom = feature.geometry
            
            # Convert to pixel coordinates
            if isinstance(geom, LineString):
                # Convert line to pixels
                pixels = []
                for x, y in geom.coords:
                    # Convert to pixel coordinates
                    row, col = ~meta['transform'] * (x, y)
                    pixels.append((int(row), int(col)))
                
                # Draw line on mask
                for i in range(len(pixels) - 1):
                    x0, y0 = pixels[i]
                    x1, y1 = pixels[i + 1]
                    # Use Bresenham's line algorithm
                    # This is a simplified version
                    dx = abs(x1 - x0)
                    dy = abs(y1 - y0)
                    sx = 1 if x0 < x1 else -1
                    sy = 1 if y0 < y1 else -1
                    err = dx - dy
                    
                    while x0 != x1 or y0 != y1:
                        if 0 <= x0 < mask.shape[1] and 0 <= y0 < mask.shape[0]:
                            mask[y0, x0] = 1
                        e2 = 2 * err
                        if e2 > -dy:
                            err -= dy
                            x0 += sx
                        if e2 < dx:
                            err += dx
                            y0 += sy
        
        return torch.from_numpy(mask).float()


class DataLoader:
    """Data loader for geospatial data."""
    
    def __init__(self, data_dir, batch_size=4, num_workers=4):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def load_imagery(self, paths):
        """
        Load imagery from paths.
        
        Args:
            paths: List of paths to imagery files
            
        Returns:
            List of loaded imagery
        """
        imagery = []
        for path in paths:
            with rasterio.open(path) as src:
                img = src.read()
                meta = src.meta
                imagery.append({'data': img, 'meta': meta})
        return imagery
    
    def load_labels(self, paths):
        """
        Load labels from paths.
        
        Args:
            paths: List of paths to label files
            
        Returns:
            List of loaded labels
        """
        labels = []
        for path in paths:
            # For vector data (e.g., shapefiles)
            if path.endswith('.shp'):
                gdf = gpd.read_file(path)
                labels.append({'data': gdf, 'type': 'vector'})
            # For raster data
            else:
                with rasterio.open(path) as src:
                    label = src.read(1)  # Assume single band for labels
                    meta = src.meta
                    labels.append({'data': label, 'meta': meta, 'type': 'raster'})
        return labels
    
    def create_dataset(self, imagery_paths, label_paths=None, transform=None):
        """
        Create a dataset from paths.
        
        Args:
            imagery_paths: List of paths to imagery files
            label_paths: List of paths to label files (optional)
            transform: Transforms to apply to the data
            
        Returns:
            GeospatialDataset
        """
        return GeospatialDataset(imagery_paths, label_paths, transform)
    
    def create_dataloader(self, dataset, shuffle=True):
        """
        Create a data loader from dataset.
        
        Args:
            dataset: GeospatialDataset
            shuffle: Whether to shuffle the data
            
        Returns:
            torch.utils.data.DataLoader
        """
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
    
    def find_data_files(self, directory, extensions):
        """
        Find files with specified extensions in directory.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to include
            
        Returns:
            List of file paths
        """
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
        return files
