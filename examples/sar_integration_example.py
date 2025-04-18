"""
SAR Integration Example

This script demonstrates how to use SAR data with DrainageAI for improved drainage pipe detection.
SAR data can help detect drainage pipes even when optical imagery is limited by cloud cover or
when there hasn't been recent rainfall to create visible moisture patterns.

Usage:
    python sar_integration_example.py --imagery <path_to_optical_imagery> --sar <path_to_sar_imagery> --output <output_directory>
"""

import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing.image_processor import ImageProcessor
import rasterio
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DrainageAI SAR Integration Example")
    
    parser.add_argument("--imagery", required=True, help="Path to optical imagery file")
    parser.add_argument("--sar", required=True, help="Path to SAR imagery file")
    parser.add_argument("--output", required=True, help="Directory to save output files")
    parser.add_argument("--model", default="cnn", choices=["cnn"], 
                        help="Model type to use (currently only CNN supports SAR)")
    parser.add_argument("--weights", help="Path to model weights file (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Confidence threshold for detection (0-1)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualization of SAR indices")
    
    return parser.parse_args()


def run_command(command):
    """Run a command and print its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def visualize_sar_indices(optical_path, sar_path, output_dir):
    """
    Generate visualization of SAR indices.
    
    Args:
        optical_path: Path to optical imagery
        sar_path: Path to SAR imagery
        output_dir: Directory to save visualizations
    """
    print("\n=== Generating SAR Index Visualizations ===\n")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load imagery
    with rasterio.open(optical_path) as src:
        optical = src.read()
        meta = src.meta
    
    with rasterio.open(sar_path) as src:
        sar = src.read()
    
    # Create image processor
    processor = ImageProcessor()
    
    # Calculate SAR indices
    indices = processor.calculate_sar_indices(sar, optical)
    
    # Create figure
    fig, axes = plt.subplots(1, len(indices) + 2, figsize=(4 * (len(indices) + 2), 4))
    
    # Plot optical imagery (RGB)
    if optical.shape[0] >= 3:
        # Get RGB bands (assuming bands are in order R, G, B)
        rgb = np.stack([
            optical[0] / optical[0].max(),
            optical[1] / optical[1].max(),
            optical[2] / optical[2].max()
        ], axis=-1)
        
        axes[0].imshow(rgb)
        axes[0].set_title("Optical (RGB)")
    else:
        axes[0].imshow(optical[0], cmap='gray')
        axes[0].set_title("Optical")
    
    # Plot SAR imagery
    if sar.shape[0] >= 2:
        # For dual-pol SAR, show VV polarization
        axes[1].imshow(sar[0], cmap='gray')
        axes[1].set_title("SAR (VV)")
    else:
        axes[1].imshow(sar[0], cmap='gray')
        axes[1].set_title("SAR")
    
    # Plot SAR indices
    for i, (name, index) in enumerate(indices.items()):
        im = axes[i + 2].imshow(index, cmap='viridis')
        axes[i + 2].set_title(f"SAR Index: {name}")
        plt.colorbar(im, ax=axes[i + 2])
    
    # Remove axis ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save figure
    plt.tight_layout()
    fig_path = output_dir / "sar_indices_visualization.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {fig_path}")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Calculate spectral indices for optical imagery
    print("\n=== Step 1: Calculate Spectral Indices ===\n")
    
    indices_path = output_dir / f"indices_{timestamp}.tif"
    
    indices_command = (
        f"python {parent_dir}/main.py indices "
        f"--imagery {args.imagery} "
        f"--output {indices_path} "
        f"--indices ndvi,ndmi,msavi2"
    )
    
    run_command(indices_command)
    
    # Step 2: Generate SAR index visualizations if requested
    if args.visualize:
        visualize_sar_indices(args.imagery, args.sar, output_dir)
    
    # Step 3: Detect drainage pipes using both optical and SAR data
    print("\n=== Step 3: Detect Drainage Pipes with SAR Integration ===\n")
    
    detection_path = output_dir / f"drainage_sar_{timestamp}.tif"
    
    detect_command = (
        f"python {parent_dir}/main.py detect "
        f"--imagery {args.imagery} "
        f"--indices {indices_path} "
        f"--sar {args.sar} "
        f"--output {detection_path} "
        f"--model {args.model} "
        f"--threshold {args.threshold}"
    )
    
    if args.weights:
        detect_command += f" --weights {args.weights}"
    
    run_command(detect_command)
    
    # Step 4: Vectorize results
    print("\n=== Step 4: Vectorize Results ===\n")
    
    vector_path = output_dir / f"drainage_lines_sar_{timestamp}.shp"
    
    vectorize_command = (
        f"python {parent_dir}/main.py vectorize "
        f"--input {detection_path} "
        f"--output {vector_path} "
        f"--simplify 1.0"
    )
    
    run_command(vectorize_command)
    
    # Print summary
    print("\n=== SAR Integration Workflow Complete ===\n")
    print("Output files:")
    print(f"  Spectral Indices: {indices_path}")
    print(f"  Drainage Detection: {detection_path}")
    print(f"  Drainage Lines: {vector_path}")
    
    if args.visualize:
        print(f"  SAR Visualization: {output_dir / 'sar_indices_visualization.png'}")
    
    print("\nThese files can now be loaded into ArcGIS or QGIS for visualization and analysis.")
    print("\nBenefits of SAR Integration:")
    print("  - Improved detection in cloudy conditions")
    print("  - Better performance when no recent rainfall has occurred")
    print("  - Enhanced detection of subsurface drainage features")
    print("  - Reduced false positives from surface water features")


if __name__ == "__main__":
    main()
