#!/bin/bash
# Setup script for DrainageAI demo environment
# This script sets up the conda environment and prepares for the demo

# Print colored messages
print_green() {
    echo -e "\e[32m$1\e[0m"
}

print_blue() {
    echo -e "\e[34m$1\e[0m"
}

print_yellow() {
    echo -e "\e[33m$1\e[0m"
}

print_red() {
    echo -e "\e[31m$1\e[0m"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_red "Conda is not installed. Please install Miniconda or Anaconda first."
    print_yellow "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_blue "=== Setting up DrainageAI Demo Environment ==="

# Create conda environment
print_green "Creating conda environment from environment.yaml..."
conda env create -f environment.yaml

# Check if environment creation was successful
if [ $? -ne 0 ]; then
    print_red "Failed to create conda environment. Please check the error messages above."
    exit 1
fi

print_green "Conda environment 'drainageai' created successfully!"

# Activate the environment
print_yellow "To activate the environment, run:"
echo "conda activate drainageai"

# Create data directories
print_green "Creating data directories..."
mkdir -p data/labeled/imagery data/labeled/labels data/unlabeled/imagery data/validation/imagery data/validation/labels

# Check if demo data exists
if [ -d "demo_data" ]; then
    print_green "Demo data found. Copying to data directory..."
    cp -r demo_data/* data/
fi

print_blue "=== Setup Complete ==="
print_yellow "Next steps:"
echo "1. Activate the environment: conda activate drainageai"
echo "2. For local execution: python examples/super_mvp_workflow.py --imagery <path_to_imagery> --output results"
echo "3. For Google Colab: Upload notebooks/drainageai_colab_demo.ipynb to Google Colab"
echo "4. For more information, see README.md and SUPER_MVP_README.md"

print_blue "=== Optional: Prepare Demo Imagery ==="
print_yellow "If you have large imagery files, you can create smaller subsets for the demo:"
echo "python scripts/prepare_demo_imagery.py --input <path_to_large_image> --output data/demo_image.tif --size 1000"

print_green "Happy drainage detection!"
