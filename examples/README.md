# DrainageAI Examples

This directory contains example scripts for using the DrainageAI system.

## Available Examples

### Super-MVP Workflow

The `super_mvp_workflow.py` script demonstrates the complete workflow for the DrainageAI super-MVP:

```bash
python super_mvp_workflow.py --imagery <path_to_imagery> --output <output_directory>
```

This script:
1. Calculates spectral indices from multispectral imagery
2. Detects drainage pipes using the CNN model with spectral indices
3. Vectorizes the results for use in GIS software

### SAR Integration Example

The `sar_integration_example.py` script demonstrates how to use SAR data with DrainageAI for improved drainage pipe detection:

```bash
python sar_integration_example.py --imagery <path_to_optical_imagery> --sar <path_to_sar_imagery> --output <output_directory>
```

This script:
1. Calculates spectral indices from optical imagery
2. Optionally generates visualizations of SAR indices
3. Detects drainage pipes using both optical and SAR data
4. Vectorizes the results for use in GIS software

Benefits of SAR integration:
- Improved detection in cloudy conditions
- Better performance when no recent rainfall has occurred
- Enhanced detection of subsurface drainage features
- Reduced false positives from surface water features

### BYOL MVP Workflow

The `byol_mvp_workflow.py` script demonstrates how to use BYOL (Bootstrap Your Own Latent) self-supervised learning with few or no labeled images:

```bash
# Full pipeline with few labeled images
python byol_mvp_workflow.py --optical-dir data/imagery --sar-dir data/sar --label-dir data/labels --output-dir results --num-labeled 5

# Inference only
python byol_mvp_workflow.py --inference-only --model-path results/byol_finetuned.pth --test-image data/test/image.tif --test-sar data/test/sar.tif --output-dir results
```

This script:
1. Performs BYOL pretraining on unlabeled data (both optical and SAR)
2. Fine-tunes with very few labeled examples (as few as 5)
3. Runs inference and evaluates results
4. Vectorizes the results for use in GIS software

Benefits of BYOL approach:
- Works with extremely limited labeled data
- Leverages unlabeled data effectively
- Integrates optical and SAR data in a multi-view framework
- Provides robust features even with varied weather conditions

### FixMatch Training Example

The `train_fixmatch_example.py` script demonstrates how to train a model using the FixMatch semi-supervised learning approach and create an ensemble model.

```bash
python train_fixmatch_example.py
```

This script:
1. Trains a semi-supervised model using FixMatch
2. Creates an ensemble model that combines the CNN and semi-supervised models
3. Evaluates the ensemble model on a validation set

## Data Directory Structure

The examples expect the following data directory structure:

```
data/
├── labeled/
│   ├── imagery/
│   │   └── *.tif
│   └── labels/
│       └── *.tif
├── unlabeled/
│   └── imagery/
│       └── *.tif
└── validation/
    ├── imagery/
    │   └── *.tif
    └── labels/
        └── *.tif
```

## Creating Your Own Examples

To create your own examples, you can use the existing examples as a starting point. The key components to include are:

1. Import the necessary modules from the DrainageAI package
2. Set up the data directories and create data loaders
3. Create and train the model
4. Evaluate the model on a validation set
5. Save the model for later use

## Jupyter Notebooks

For more interactive examples, see the Jupyter notebooks in the `notebooks` directory:

- `drainage_detection_tutorial.ipynb`: Basic tutorial for drainage pipe detection
- `fixmatch_tutorial.ipynb`: Tutorial for FixMatch semi-supervised learning
