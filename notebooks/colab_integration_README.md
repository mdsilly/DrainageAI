# DrainageAI Google Colab Integration

This document provides instructions for using DrainageAI with Google Colab to leverage GPU acceleration for drainage pipe detection.

## Overview

Google Colab provides free access to GPU resources, which can significantly speed up the DrainageAI workflow. The `drainageai_colab_demo.ipynb` notebook in this directory provides a complete implementation of the DrainageAI super-MVP workflow in Google Colab.

## Setup Instructions

### 1. Open Google Colab

- Go to [colab.research.google.com](https://colab.research.google.com)
- Sign in with your Google account

### 2. Upload the Notebook

**Option 1: Upload from your computer**
- Click "File" → "Upload notebook"
- Select the `drainageai_colab_demo.ipynb` file from your computer

**Option 2: Upload from GitHub**
- Click "File" → "Open notebook"
- Select the "GitHub" tab
- Enter your GitHub repository URL
- Select the `notebooks/drainageai_colab_demo.ipynb` file

### 3. Enable GPU Acceleration

- Click "Runtime" → "Change runtime type"
- Set "Hardware accelerator" to "GPU"
- Click "Save"

### 4. Run the Notebook

- You can run each cell individually by clicking the play button next to each cell
- Or run all cells with "Runtime" → "Run all"
- Follow the instructions in the notebook for uploading your imagery and downloading results

## Using the Notebook for Your Demo

### Before the Demo

1. **Test the notebook**: Run through the entire notebook with a small test image to ensure everything works correctly
2. **Prepare your imagery**: Have your multispectral imagery ready in GeoTIFF format
3. **Check GPU availability**: Make sure you have GPU access by running the first few cells

### During the Demo

1. **Explain each step**: The notebook is organized into clear steps with explanations
2. **Show visualizations**: The notebook includes visualizations of the spectral indices and detection results
3. **Download results**: At the end of the notebook, you can download the results for use in ArcGIS or QGIS

### After Processing

1. **Load results in ArcGIS/QGIS**: Import the downloaded shapefiles and rasters into your GIS software
2. **Analyze results**: Use the GIS tools to analyze the detected drainage pipes
3. **Compare with ground truth**: If available, compare the results with known drainage pipe locations

## Troubleshooting

### Common Issues

1. **Upload size limits**: Files over 100MB need to be uploaded to Google Drive first
   - Solution: Use Google Drive mounting in Colab (`from google.colab import drive; drive.mount('/content/drive')`)

2. **Memory errors**: Processing large images may cause out-of-memory errors
   - Solution: Reduce image size or use a smaller subset of the image

3. **Session timeouts**: Colab sessions disconnect after 90 minutes of inactivity
   - Solution: Keep the browser tab active or use browser extensions to prevent timeout

4. **Package installation failures**: Some packages may fail to install
   - Solution: Try installing packages one by one or use alternative versions

### Getting Help

If you encounter issues with the notebook:

1. Check the error messages in the cell outputs
2. Refer to the DrainageAI documentation
3. Check the Google Colab FAQ at [research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html)

## Limitations

- **Session length**: Colab sessions are limited to 12 hours
- **Resource allocation**: GPU resources are shared and may vary in availability
- **Storage**: Limited to 15GB (sufficient for most demos)
- **No direct GIS integration**: Results must be downloaded and loaded into GIS software separately

## Tips for Optimal Performance

1. **Use small test images**: Start with small images (1000×1000 pixels) for quick testing
2. **Pre-process imagery**: Consider pre-processing your imagery to reduce size and improve quality
3. **Save intermediate results**: Download indices and detection results separately
4. **Use appropriate thresholds**: Adjust the detection threshold based on your imagery and requirements
5. **Try different models**: Compare results from different models (CNN, semi-supervised, ensemble)

## Next Steps

After successfully running the demo in Colab, consider:

1. **Training on your own data**: Use the training scripts to train models on your own labeled data
2. **Implementing the full ensemble model**: Explore the full DrainageAI capabilities
3. **Setting up local GPU processing**: For production use, consider setting up local GPU processing
