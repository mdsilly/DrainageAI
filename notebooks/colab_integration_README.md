# DrainageAI Google Colab Integration

This document provides instructions for using DrainageAI with Google Colab to leverage GPU acceleration for drainage pipe detection.

## Overview

Google Colab provides free access to GPU resources, which can significantly speed up the DrainageAI workflow. Three notebooks are provided in this directory:

1. `drainageai_colab_demo.ipynb` - The standard DrainageAI super-MVP workflow
2. `drainageai_colab_demo_byol.ipynb` - Enhanced workflow with BYOL self-supervised learning for few-shot learning and SAR data integration
3. `drainageai_colab_demo_unlabeled.ipynb` - Workflow optimized for unlabeled data only, with optional steps for labeled data and SAR integration

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

## BYOL Workflow for Few-Shot Learning

The `drainageai_colab_demo_byol.ipynb` notebook provides an enhanced workflow that uses BYOL (Bootstrap Your Own Latent) self-supervised learning to work with very few labeled examples.

### Key Features

1. **Self-supervised pretraining**: BYOL learns from unlabeled data without requiring any labels
2. **Few-shot learning**: Fine-tune with as few as 5 labeled examples
3. **SAR data integration**: Combine optical and SAR imagery for improved detection
4. **Multi-view learning**: Learn from different views of the same scene

### When to Use BYOL

The BYOL workflow is particularly useful in these scenarios:

- You have very limited labeled data (less than 10 labeled examples)
- You have access to unlabeled data from the same domain
- You need to detect drainage pipes in challenging conditions (cloud cover, no recent rainfall)
- You have access to both optical and SAR imagery

### BYOL Workflow Steps

1. **Upload unlabeled and labeled data**: Provide unlabeled imagery for pretraining and a small set of labeled examples for fine-tuning
2. **BYOL pretraining**: Train the model on unlabeled data to learn general representations
3. **Fine-tuning**: Fine-tune the pretrained model on a small set of labeled examples
4. **Inference**: Run inference on new imagery
5. **SAR integration**: Optionally integrate SAR data for improved detection

## Unlabeled Data Workflow

The `drainageai_colab_demo_unlabeled.ipynb` notebook is specifically designed for scenarios where you only have unlabeled data available, with optional steps for incorporating labeled data or SAR imagery if they become available later.

### Key Features

1. **Focus on unlabeled data**: Optimized workflow for scenarios with no labeled examples
2. **Interactive decision points**: User prompts to determine whether to include labeled data or SAR
3. **Modular approach**: Run only the parts of the workflow that are relevant to your data
4. **Flexible pipeline**: Start with unlabeled data and add labeled data or SAR when available

### When to Use the Unlabeled Data Workflow

This workflow is ideal in these scenarios:

- You have no labeled drainage pipe examples yet
- You're starting a new project and want to establish a baseline
- You plan to collect labeled data in the future
- You want to explore what's possible with only unlabeled data

### Unlabeled Data Workflow Steps

1. **Upload unlabeled data**: Provide unlabeled imagery for pretraining
2. **BYOL pretraining**: Train the model on unlabeled data to learn general representations
3. **Inference with pretrained model**: Run inference using only the pretrained model
4. **Optional labeled data integration**: Add labeled data for fine-tuning if available
5. **Optional SAR integration**: Add SAR data if available

### Benefits of Starting with Unlabeled Data

- **No annotation required**: Get started without the time-consuming process of creating labeled examples
- **Establish baseline performance**: See what's possible with self-supervised learning alone
- **Incremental improvement**: Add labeled data or SAR later to improve results
- **Efficient resource use**: Focus annotation efforts on the most valuable examples

## Next Steps

After successfully running the demo in Colab, consider:

1. **Training on your own data**: Use the training scripts to train models on your own labeled data
2. **Implementing the full ensemble model**: Explore the full DrainageAI capabilities
3. **Using the BYOL approach**: Try the BYOL workflow if you have limited labeled data
4. **Setting up local GPU processing**: For production use, consider setting up local GPU processing
