# DrainageAI

An AI-powered system for detecting agricultural drainage pipes using satellite imagery, LIDAR data, and machine learning techniques.

## Project Overview

DrainageAI uses an ensemble of deep learning models to identify subsurface drainage systems in farmland. The system integrates with GIS platforms (QGIS and ArcGIS) through the Model Context Protocol (MCP) to provide a user-friendly interface for visualization and analysis.

## Features

- Multi-modal data processing (satellite imagery, LIDAR, SAR)
- Ensemble model architecture (CNN, GNN, Self-Supervised Learning)
- Semi-supervised learning with FixMatch for limited labeled data
- Spectral indices integration (NDVI, NDMI, MSAVI2) for enhanced detection
- QGIS and ArcGIS integration
- Visualization and export tools

## Project Structure

```
DrainageAI/
├── data/                  # Sample data for development
├── models/                # Model definitions and weights
├── preprocessing/         # Data preparation scripts
├── training/              # Training utilities and scripts
├── mcp_server/            # MCP server implementation
│   └── drainage_server.py # Main server code
├── qgis_extension/        # QGIS plugin extensions
├── arcgis_extension/      # ArcGIS Python Toolbox
├── notebooks/             # Jupyter notebooks for exploration
├── examples/              # Example scripts
├── tests/                 # Test scripts
└── README.md              # Documentation
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- PyTorch
- GDAL/Rasterio
- QGIS Desktop 3.22+
- MCP Plugin for QGIS

### Installation

#### Option 1: Using pip

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

#### Option 2: Using conda (Recommended)

Conda is recommended for easier installation of geospatial dependencies like GDAL.

1. Clone this repository
2. Create a conda environment:
   ```bash
   conda env create -f environment.yaml
   ```
3. Activate the environment:
   ```bash
   conda activate drainageai
   ```

### Google Colab Integration

For running DrainageAI with GPU acceleration without local setup:

1. Upload the `notebooks/drainageai_colab_demo.ipynb` notebook to Google Colab
2. Follow the instructions in the notebook to run the complete workflow
3. See `notebooks/colab_integration_README.md` for detailed instructions

## Usage

[Usage instructions will be added as the project develops]

## Development

[Development guidelines will be added as the project develops]

## License

[License information]
