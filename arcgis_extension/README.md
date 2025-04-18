# DrainageAI ArcGIS Extension

This extension integrates DrainageAI with ArcGIS Pro through a Python Toolbox, allowing you to detect drainage pipes directly from the ArcGIS Pro interface.

## Installation

### Prerequisites

- ArcGIS Pro 2.8+
- Python 3.9+
- DrainageAI package installed
- Spatial Analyst extension for ArcGIS Pro

### Setup

1. Clone or download the DrainageAI repository
2. Add the `arcgis_extension` directory to your ArcGIS Pro project
3. In ArcGIS Pro, open the Catalog pane
4. Navigate to Toolboxes
5. Right-click and select "Add Toolbox"
6. Browse to the `arcgis_extension` directory and select `DrainageAI.pyt`

## Usage

The DrainageAI toolbox provides three main tools:

### 1. Calculate Spectral Indices

This tool calculates spectral indices from multispectral imagery, which can enhance the detection of drainage pipes.

**Parameters:**
- **Input Multispectral Imagery**: The input multispectral raster dataset
- **Output Indices Raster**: The output raster dataset to store the calculated indices
- **Indices to Calculate**: Select which indices to calculate (NDVI, NDMI, MSAVI2, NDWI, SAVI)
- **Red Band Number**: The band number for the red band (default: 3)
- **NIR Band Number**: The band number for the near-infrared band (default: 4)
- **SWIR Band Number**: The band number for the shortwave infrared band (default: 5)
- **Green Band Number**: The band number for the green band (default: 2)
- **L Parameter**: The soil adjustment factor for SAVI (default: 0.5)

### 2. Detect Drainage Pipes

This tool uses the DrainageAI model to detect drainage pipes in agricultural fields.

**Parameters:**
- **Input Imagery**: The input satellite or aerial imagery
- **Input Spectral Indices**: Optional spectral indices calculated from the first tool
- **Input Elevation**: Optional elevation data
- **Output Drainage Raster**: The output raster dataset to store the detection results
- **Confidence Threshold**: Threshold for detection confidence (0-1, default: 0.5)
- **Model Type**: The type of model to use (CNN or Ensemble)
- **Model Path**: Optional path to a pre-trained model

### 3. Vectorize Drainage Results

This tool converts the raster detection results to vector format for easier analysis and editing.

**Parameters:**
- **Input Drainage Raster**: The raster dataset containing drainage detection results
- **Output Drainage Lines**: The output feature class to store the vectorized drainage lines
- **Threshold Value**: The threshold value for converting to binary (default: 0.5)
- **Simplify Tolerance**: The tolerance for simplifying the resulting lines (default: 1.0)

## Workflow Example

1. **Prepare your data**:
   - Obtain multispectral imagery of the area of interest
   - Optionally, obtain elevation data for the same area

2. **Calculate spectral indices**:
   - Use the "Calculate Spectral Indices" tool
   - Select appropriate indices (NDMI, NDVI, and MSAVI2 recommended)
   - Specify the correct band numbers for your imagery

3. **Detect drainage pipes**:
   - Use the "Detect Drainage Pipes" tool
   - Input your imagery and the calculated indices
   - Set an appropriate confidence threshold

4. **Vectorize results**:
   - Use the "Vectorize Drainage Results" tool
   - Convert the detection raster to vector lines
   - Adjust the simplification tolerance as needed

5. **Analyze and edit**:
   - Use ArcGIS Pro's editing tools to refine the results
   - Add attributes to the drainage lines as needed
   - Perform spatial analysis with other datasets

## Troubleshooting

If you encounter issues with the tools:

1. Check that the DrainageAI package is installed correctly
2. Verify that the Spatial Analyst extension is available
3. Ensure that your input data has the correct projection and format
4. Check the ArcGIS Pro Python window for detailed error messages

## Development

To modify or extend the DrainageAI ArcGIS extension:

1. Edit the `DrainageAI.pyt` file to add new tools or modify existing ones
2. Update the tool parameters and execution code as needed
3. Restart ArcGIS Pro to apply changes

## License

See the LICENSE file in the root directory of the DrainageAI repository.
