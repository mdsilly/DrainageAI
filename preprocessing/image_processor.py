"""
Image processor for satellite imagery and SAR data.
"""

import numpy as np
import cv2
import torch
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from scipy.ndimage import median_filter


class ImageProcessor:
    """Image processor for satellite imagery."""
    
    def __init__(self, target_size=(512, 512), normalize=True):
        """
        Initialize the image processor.
        
        Args:
            target_size: Target size for resizing (height, width)
            normalize: Whether to normalize the imagery
        """
        self.target_size = target_size
        self.normalize = normalize
        
    def preprocess(self, imagery, meta=None):
        """
        Preprocess imagery.
        
        Args:
            imagery: Input imagery (numpy array or path to file)
            meta: Metadata (optional)
            
        Returns:
            Preprocessed imagery
        """
        # Load imagery if path is provided
        if isinstance(imagery, str):
            with rasterio.open(imagery) as src:
                imagery = src.read()
                meta = src.meta
        
        # Convert to numpy array if needed
        if isinstance(imagery, torch.Tensor):
            imagery = imagery.numpy()
        
        # Resize to target size
        imagery = self.resize(imagery, self.target_size)
        
        # Normalize if requested
        if self.normalize:
            imagery = self.normalize_imagery(imagery)
        
        # Convert to torch tensor
        imagery = torch.from_numpy(imagery).float()
        
        return imagery
    
    def resize(self, imagery, target_size):
        """
        Resize imagery to target size.
        
        Args:
            imagery: Input imagery (numpy array)
            target_size: Target size (height, width)
            
        Returns:
            Resized imagery
        """
        # Get current dimensions
        if imagery.ndim == 3:
            c, h, w = imagery.shape
            
            # Resize each channel
            resized = np.zeros((c, target_size[0], target_size[1]), dtype=imagery.dtype)
            for i in range(c):
                resized[i] = cv2.resize(imagery[i], (target_size[1], target_size[0]))
        else:
            # Single channel
            resized = cv2.resize(imagery, (target_size[1], target_size[0]))
        
        return resized
    
    def normalize_imagery(self, imagery):
        """
        Normalize imagery.
        
        Args:
            imagery: Input imagery (numpy array)
            
        Returns:
            Normalized imagery
        """
        # Simple min-max normalization
        if imagery.ndim == 3:
            # Normalize each channel separately
            normalized = np.zeros_like(imagery, dtype=np.float32)
            for i in range(imagery.shape[0]):
                channel = imagery[i].astype(np.float32)
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val > min_val:
                    normalized[i] = (channel - min_val) / (max_val - min_val)
                else:
                    normalized[i] = channel
        else:
            # Single channel
            imagery = imagery.astype(np.float32)
            min_val = np.min(imagery)
            max_val = np.max(imagery)
            if max_val > min_val:
                normalized = (imagery - min_val) / (max_val - min_val)
            else:
                normalized = imagery
        
        return normalized
    
    def reproject_imagery(self, imagery, src_crs, dst_crs, meta=None):
        """
        Reproject imagery to a different coordinate reference system.
        
        Args:
            imagery: Input imagery (numpy array)
            src_crs: Source coordinate reference system
            dst_crs: Destination coordinate reference system
            meta: Metadata (optional)
            
        Returns:
            Reprojected imagery and updated metadata
        """
        # This is a simplified version - in practice, we would use
        # rasterio.warp.reproject for proper reprojection
        
        # Create a temporary rasterio dataset
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=imagery.shape[1],
                width=imagery.shape[2],
                count=imagery.shape[0],
                dtype=imagery.dtype,
                crs=src_crs,
                transform=meta['transform'] if meta else None
            ) as src:
                # Write the input imagery
                src.write(imagery)
                
                # Calculate the transform for the output
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, 
                    *src.bounds
                )
                
                # Create the destination array
                dst_imagery = np.zeros(
                    (imagery.shape[0], dst_height, dst_width),
                    dtype=imagery.dtype
                )
                
                # Create updated metadata
                dst_meta = src.meta.copy()
                dst_meta.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })
                
                # Reproject each band
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=dst_imagery[i-1],
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )
        
        return dst_imagery, dst_meta
    
    def extract_patches(self, imagery, patch_size, stride):
        """
        Extract patches from imagery.
        
        Args:
            imagery: Input imagery (numpy array)
            patch_size: Patch size (height, width)
            stride: Stride for patch extraction
            
        Returns:
            List of patches
        """
        # Get dimensions
        if imagery.ndim == 3:
            c, h, w = imagery.shape
        else:
            h, w = imagery.shape
            c = 1
            imagery = imagery.reshape(1, h, w)
        
        # Calculate number of patches
        n_h = (h - patch_size[0]) // stride + 1
        n_w = (w - patch_size[1]) // stride + 1
        
        # Extract patches
        patches = []
        for i in range(n_h):
            for j in range(n_w):
                # Extract patch
                y = i * stride
                x = j * stride
                patch = imagery[:, y:y+patch_size[0], x:x+patch_size[1]]
                patches.append(patch)
        
        return patches
    
    def stitch_patches(self, patches, original_size, patch_size, stride):
        """
        Stitch patches back into a full image.
        
        Args:
            patches: List of patches
            original_size: Original image size (height, width)
            patch_size: Patch size (height, width)
            stride: Stride used for patch extraction
            
        Returns:
            Stitched image
        """
        # Get dimensions
        c = patches[0].shape[0]
        h, w = original_size
        
        # Create output image
        output = np.zeros((c, h, w), dtype=patches[0].dtype)
        counts = np.zeros((c, h, w), dtype=np.int32)
        
        # Calculate number of patches
        n_h = (h - patch_size[0]) // stride + 1
        n_w = (w - patch_size[1]) // stride + 1
        
        # Stitch patches
        patch_idx = 0
        for i in range(n_h):
            for j in range(n_w):
                # Get patch position
                y = i * stride
                x = j * stride
                
                # Add patch to output
                output[:, y:y+patch_size[0], x:x+patch_size[1]] += patches[patch_idx]
                counts[:, y:y+patch_size[0], x:x+patch_size[1]] += 1
                
                patch_idx += 1
        
        # Average overlapping regions
        output = output / np.maximum(counts, 1)
        
        return output
    
    def preprocess_sar(self, sar_imagery, meta=None):
        """
        Preprocess SAR imagery.
        
        Args:
            sar_imagery: SAR imagery (numpy array or path to file)
            meta: Metadata (optional)
            
        Returns:
            Preprocessed SAR imagery
        """
        # Load imagery if path is provided
        if isinstance(sar_imagery, str):
            with rasterio.open(sar_imagery) as src:
                sar_imagery = src.read()
                meta = src.meta
        
        # Convert to numpy array if needed
        if isinstance(sar_imagery, torch.Tensor):
            sar_imagery = sar_imagery.numpy()
        
        # Apply speckle filtering (basic median filter for quick implementation)
        filtered_sar = np.zeros_like(sar_imagery)
        for i in range(sar_imagery.shape[0]):
            filtered_sar[i] = median_filter(sar_imagery[i], size=3)
        
        # Resize to target size
        filtered_sar = self.resize(filtered_sar, self.target_size)
        
        # Normalize
        if self.normalize:
            filtered_sar = self.normalize_imagery(filtered_sar)
        
        # Convert to torch tensor
        filtered_sar = torch.from_numpy(filtered_sar).float()
        
        return filtered_sar
    
    def preprocess_combined(self, optical_imagery, sar_imagery=None, meta=None):
        """
        Preprocess both optical and SAR imagery.
        
        Args:
            optical_imagery: Optical imagery (numpy array or path to file)
            sar_imagery: SAR imagery (numpy array or path to file)
            meta: Metadata (optional)
            
        Returns:
            Combined preprocessed imagery
        """
        # Preprocess optical imagery
        processed_optical = self.preprocess(optical_imagery, meta)
        
        # If no SAR imagery, return optical only
        if sar_imagery is None:
            return processed_optical
        
        # Preprocess SAR imagery
        processed_sar = self.preprocess_sar(sar_imagery, meta)
        
        # Combine optical and SAR
        combined = torch.cat([processed_optical, processed_sar], dim=0)
        
        return combined
    
    def calculate_sar_indices(self, sar_imagery, optical_imagery=None):
        """
        Calculate SAR-specific indices for drainage detection.
        
        Args:
            sar_imagery: SAR imagery (VV and VH polarizations)
            optical_imagery: Optional optical imagery for combined indices
            
        Returns:
            Dictionary of SAR indices
        """
        # Ensure numpy arrays
        if isinstance(sar_imagery, torch.Tensor):
            sar_imagery = sar_imagery.numpy()
        
        if optical_imagery is not None and isinstance(optical_imagery, torch.Tensor):
            optical_imagery = optical_imagery.numpy()
        
        # Extract polarizations (assuming VV is first band, VH is second)
        if sar_imagery.shape[0] >= 2:
            vv = sar_imagery[0]
            vh = sar_imagery[1]
        else:
            # If only one band, assume it's VV
            vv = sar_imagery[0]
            vh = None
        
        indices = {}
        
        # Calculate polarization ratio (sensitive to soil moisture)
        if vh is not None:
            pol_ratio = vv / (vh + 1e-8)
            indices['pol_ratio'] = pol_ratio
            
            # Calculate polarization difference (highlights linear features)
            pol_diff = vv - vh
            indices['pol_diff'] = pol_diff
        
        # If optical imagery is available, calculate combined indices
        if optical_imagery is not None and optical_imagery.shape[0] >= 4:
            # Extract NIR band (assuming band 4 is NIR)
            nir = optical_imagery[3]
            
            # Combined SAR-optical index for moisture
            combined_moisture = (nir - vv) / (nir + vv + 1e-8)
            indices['combined_moisture'] = combined_moisture
        
        return indices
