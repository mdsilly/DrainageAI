"""
Convert Grayscale Images to RGB

This script converts single-channel grayscale images to 3-channel RGB format
for compatibility with models that expect RGB input.

Usage:
    python scripts/convert_grayscale_to_rgb.py --input-dir data/unlabeled --output-dir data/unlabeled_rgb
"""

import os
import argparse
import numpy as np
from pathlib import Path
import rasterio
from rasterio.plot import reshape_as_image
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert grayscale images to RGB")
    
    parser.add_argument("--input-dir", required=True, help="Directory containing grayscale images")
    parser.add_argument("--output-dir", required=True, help="Directory to save RGB images")
    parser.add_argument("--pattern", default=".tif", help="File pattern to match (default: .tif)")
    
    return parser.parse_args()


def convert_grayscale_to_rgb(input_path, output_path):
    """
    Convert a single-channel grayscale image to 3-channel RGB.
    
    Args:
        input_path: Path to input grayscale image
        output_path: Path to save RGB image
    
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        with rasterio.open(input_path) as src:
            # Read the data
            data = src.read()
            profile = src.profile.copy()
            
            # Check if it's already multi-channel
            if data.shape[0] >= 3:
                print(f"Image {input_path} already has {data.shape[0]} channels, skipping.")
                return False
            
            # Create 3-channel image by duplicating the grayscale channel
            if data.shape[0] == 1:
                rgb_data = np.repeat(data, 3, axis=0)
            else:
                # If it has 2 channels, add a third one
                zeros = np.zeros_like(data[0:1])
                rgb_data = np.concatenate([data, zeros], axis=0)
            
            # Update profile for RGB output
            profile.update(count=3)
            
            # Write the RGB image
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(rgb_data)
            
            return True
    
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all images in input directory
    input_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if args.pattern in file:
                input_files.append(os.path.join(root, file))
    
    print(f"Found {len(input_files)} images to convert.")
    
    # Convert each image
    converted = 0
    for input_file in tqdm(input_files):
        # Create relative path for output
        rel_path = os.path.relpath(input_file, args.input_dir)
        output_file = os.path.join(args.output_dir, rel_path)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert the image
        if convert_grayscale_to_rgb(input_file, output_file):
            converted += 1
    
    print(f"Converted {converted} images to RGB format.")
    print(f"Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
