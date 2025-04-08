# DrainageAI

An AI-powered system for detecting agricultural drainage pipes using satellite imagery, LIDAR data, and machine learning techniques.

## Project Overview

DrainageAI uses an ensemble of deep learning models to identify subsurface drainage systems in farmland. The system integrates with QGIS through the Model Context Protocol (MCP) to provide a user-friendly interface for visualization and analysis.

## Features

- Multi-modal data processing (satellite imagery, LIDAR, SAR)
- Ensemble model architecture (CNN, GNN, Self-Supervised Learning)
- QGIS integration via MCP
- Visualization and export tools

## Project Structure

```
DrainageAI/
├── data/                  # Sample data for development
├── models/                # Model definitions and weights
├── preprocessing/         # Data preparation scripts
├── mcp_server/            # MCP server implementation
│   ├── drainage_server.py # Main server code
│   └── tools/             # Tool implementations
├── qgis_extension/        # QGIS plugin extensions
├── notebooks/             # Jupyter notebooks for exploration
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

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

[Usage instructions will be added as the project develops]

## Development

[Development guidelines will be added as the project develops]

## License

[License information]
