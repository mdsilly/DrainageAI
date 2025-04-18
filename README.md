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

1. Choose one of the following notebooks to upload to Google Colab:
   - `notebooks/drainageai_colab_demo.ipynb` - Standard workflow
   - `notebooks/drainageai_colab_demo_byol.ipynb` - BYOL workflow with SAR integration
   - `notebooks/drainageai_colab_demo_unlabeled.ipynb` - Workflow optimized for unlabeled data only
2. Follow the instructions in the notebook to run the complete workflow
3. See `notebooks/colab_integration_README.md` for detailed instructions

## Usage

### Basic Workflow

The basic workflow for using DrainageAI consists of the following steps:

1. **Calculate spectral indices** from multispectral imagery
2. **Detect drainage pipes** using the trained model
3. **Vectorize the results** for use in GIS software

You can run the complete workflow using the super-MVP example script:

```bash
python examples/super_mvp_workflow.py --imagery <path_to_imagery> --output <output_directory>
```

### SAR Integration

DrainageAI supports Synthetic Aperture Radar (SAR) data integration for improved drainage pipe detection, especially in challenging conditions:

```bash
python examples/sar_integration_example.py --imagery <path_to_optical_imagery> --sar <path_to_sar_imagery> --output <output_directory>
```

Benefits of SAR integration:
- Weather-independent detection (works through cloud cover)
- Improved performance when no recent rainfall has occurred
- Enhanced detection of subsurface drainage features
- Reduced false positives from surface water features

### Command Line Interface

DrainageAI provides a command-line interface for individual operations:

```bash
# Calculate spectral indices
python main.py indices --imagery <path_to_imagery> --output <output_path>

# Detect drainage pipes
python main.py detect --imagery <path_to_imagery> --indices <path_to_indices> --output <output_path>

# Detect with SAR data
python main.py detect --imagery <path_to_imagery> --indices <path_to_indices> --sar <path_to_sar> --output <output_path>

# Vectorize results
python main.py vectorize --input <path_to_detection_results> --output <output_path>
```

For more examples and detailed usage instructions, see the `examples` directory.

## Development

[Development guidelines will be added as the project develops]

## License

[License information]
