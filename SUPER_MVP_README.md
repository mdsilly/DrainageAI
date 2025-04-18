# DrainageAI Super-MVP

This is the super-minimal viable product (MVP) for the DrainageAI system, designed to demonstrate the core functionality of detecting agricultural drainage pipes using satellite imagery and spectral indices.

## Overview

The DrainageAI Super-MVP includes:

1. **Spectral Indices Calculation**: Tools to calculate NDVI, NDMI, and MSAVI2 from multispectral imagery
2. **CNN-based Detection**: A convolutional neural network that uses both imagery and spectral indices
3. **Vectorization**: Tools to convert raster detection results to vector format
4. **GIS Integration**: Support for both ArcGIS and QGIS

## Quick Start

### Prerequisites

- Python 3.9+
- PyTorch
- GDAL/Rasterio
- Scikit-image
- Geopandas
- ArcGIS Pro or QGIS (for visualization)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DrainageAI.git
   cd DrainageAI
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Super-MVP Workflow

The easiest way to run the complete workflow is using the provided example script:

```bash
python examples/super_mvp_workflow.py --imagery path/to/imagery.tif --output path/to/output/directory
```

This script will:
1. Calculate spectral indices (NDVI, NDMI, MSAVI2)
2. Detect drainage pipes using the CNN model
3. Vectorize the results for use in GIS software

### Command-Line Interface

You can also run individual steps using the main CLI:

```bash
# Calculate spectral indices
python main.py indices --imagery path/to/imagery.tif --output path/to/indices.tif

# Detect drainage pipes
python main.py detect --imagery path/to/imagery.tif --indices path/to/indices.tif --output path/to/detection.tif --model cnn

# Vectorize results
python main.py vectorize --input path/to/detection.tif --output path/to/drainage_lines.shp
```

## GIS Integration

### ArcGIS Integration

1. Open ArcGIS Pro
2. Add the `arcgis_extension` directory as a folder connection in the Catalog pane
3. Add the `DrainageAI.pyt` toolbox to your project
4. Use the tools from the toolbox:
   - Calculate Spectral Indices
   - Detect Drainage Pipes
   - Vectorize Drainage Results

You can also use the Python script in `arcgis_extension/example_workflow.py` to automate the workflow.

### QGIS Integration

1. Install the MCP Plugin for QGIS
2. Copy the `qgis_extension/claude_desktop_config.json` file to the appropriate location
3. Start the DrainageAI MCP server: `python main.py server`
4. Use the DrainageAI tools from the MCP toolbar in QGIS

## Data Requirements

For best results, use multispectral imagery with the following bands:
- Red (typically band 3)
- Near Infrared (NIR, typically band 4)
- Shortwave Infrared (SWIR, typically band 5)
- Green (typically band 2)

Recommended sources:
- Sentinel-2 (10m resolution)
- Landsat 8/9 (30m resolution)
- Commercial high-resolution imagery (e.g., Planet, Maxar)

## Limitations

This Super-MVP has several limitations:
- Limited accuracy compared to the full implementation
- No GNN or ensemble model integration yet
- Basic post-processing only
- Limited validation capabilities

## Next Steps

After testing the Super-MVP, consider:
1. Training on your own data
2. Implementing the full ensemble model
3. Adding more sophisticated post-processing
4. Integrating with other data sources (e.g., LiDAR, SAR)

## Tutorials

For more detailed examples, see the Jupyter notebooks in the `notebooks` directory:
- `drainage_detection_tutorial.ipynb`: Basic tutorial for drainage pipe detection
- `spectral_indices_tutorial.ipynb`: Tutorial for using spectral indices
- `fixmatch_tutorial.ipynb`: Tutorial for semi-supervised learning with FixMatch
