#!/usr/bin/env python
"""
Prepare imagery for DrainageAI demo.

This script creates a smaller subset of a larger multispectral image to make it
more manageable for the DrainageAI demo, especially when using Google Colab.

Usage:
    python prepare_demo_imagery.py --input large_image.tif --output demo_image.tif --size 1000
"""

import os
import argparse
import rasterio
from rasterio.windows import Window


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare imagery for DrainageAI demo")
    
    parser.add_argument("--input", required=True, help="Path to input imagery file")
    parser.add_argument("--output", required=True, help="Path to output imagery file")
    parser.add_argument("--size", type=int, default=1000, help="Size of output image (width and height)")
    parser.add_argument("--offset-x", type=int, default=0, help="X offset from top-left corner")
    parser.add_argument("--offset-y", type=int, default=0, help="Y offset from top-left corner")
    
    return parser.parse_args()


def create_subset(input_path, output_path, size, offset_x=0, offset_y=0):
    """
    Create a subset of a larger image.
    
    Args:
        input_path: Path to input imagery file
        output_path: Path to output imagery file
        size: Size of output image (width and height)
        offset_x: X offset from top-left corner
        offset_y: Y offset from top-left corner
    """
    with rasterio.open(input_path) as src:
        # Get metadata
        meta = src.meta.copy()
        
        # Check if the requested window is valid
        if offset_x + size > src.width or offset_y + size > src.height:
            print(f"Warning: Requested window exceeds image dimensions ({src.width}x{src.height})")
            # Adjust size to fit within image
            size_x = min(size, src.width - offset_x)
            size_y = min(size, src.height - offset_y)
            print(f"Adjusting window size to {size_x}x{size_y}")
        else:
            size_x = size
            size_y = size
        
        # Define the window
        window = Window(offset_x, offset_y, size_x, size_y)
        
        # Read the data
        data = src.read(window=window)
        
        # Update metadata
        meta.update({
            'height': size_y,
            'width': size_x,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        # Write the output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)
    
    print(f"Subset created successfully: {output_path}")
    print(f"Dimensions: {size_x}x{size_y}")
    print(f"Bands: {data.shape[0]}")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subset
    create_subset(args.input, args.output, args.size, args.offset_x, args.offset_y)


if __name__ == "__main__":
    main()
