"""
Calculate spectral indices from multispectral imagery.
"""

import numpy as np
import rasterio

def calculate_indices(args):
    """
    Calculate spectral indices from multispectral imagery.
    
    Args:
        args: Command line arguments
    """
    print(f"Calculating spectral indices for {args.imagery}...")
    
    # Parse indices to calculate
    indices_to_calculate = args.indices.lower().split(",")
    print(f"Indices to calculate: {', '.join(indices_to_calculate)}")
    
    # Load imagery
    with rasterio.open(args.imagery) as src:
        # Read specific bands
        red = src.read(args.red_band)
        nir = src.read(args.nir_band)
        
        # Read optional bands if needed
        green = src.read(args.green_band) if "ndwi" in indices_to_calculate else None
        swir = src.read(args.swir_band) if "ndmi" in indices_to_calculate else None
        
        # Get metadata
        meta = src.meta.copy()
    
    # Calculate indices
    calculated_indices = []
    band_names = []
    
    # Calculate NDVI
    if "ndvi" in indices_to_calculate:
        print("Calculating NDVI...")
        ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
        calculated_indices.append(ndvi)
        band_names.append("NDVI")
    
    # Calculate NDMI
    if "ndmi" in indices_to_calculate and swir is not None:
        print("Calculating NDMI...")
        ndmi = (nir - swir) / (nir + swir + 1e-8)
        calculated_indices.append(ndmi)
        band_names.append("NDMI")
    
    # Calculate MSAVI2
    if "msavi2" in indices_to_calculate:
        print("Calculating MSAVI2...")
        msavi2 = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
        calculated_indices.append(msavi2)
        band_names.append("MSAVI2")
    
    # Calculate NDWI
    if "ndwi" in indices_to_calculate and green is not None:
        print("Calculating NDWI...")
        ndwi = (green - nir) / (green + nir + 1e-8)
        calculated_indices.append(ndwi)
        band_names.append("NDWI")
    
    # Calculate SAVI
    if "savi" in indices_to_calculate:
        print("Calculating SAVI...")
        savi = (nir - red) * (1 + args.l_param) / (nir + red + args.l_param + 1e-8)
        calculated_indices.append(savi)
        band_names.append("SAVI")
    
    # Stack indices
    if not calculated_indices:
        print("No indices were calculated. Please check your input parameters.")
        return
    
    indices_stack = np.stack(calculated_indices)
    
    # Update metadata for output
    meta.update({
        'count': len(calculated_indices),
        'dtype': 'float32'
    })
    
    # Save as GeoTIFF
    with rasterio.open(args.output, 'w', **meta) as dst:
        for i, (index, name) in enumerate(zip(calculated_indices, band_names), 1):
            dst.write(index.astype(np.float32), i)
            dst.set_band_description(i, name)
    
    print(f"Spectral indices calculation completed. Results saved to {args.output}")
