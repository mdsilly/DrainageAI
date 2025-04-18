# DrainageAI Examples

This directory contains example scripts for using the DrainageAI system.

## Available Examples

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
