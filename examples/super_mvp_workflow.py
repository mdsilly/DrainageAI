"""
Super-MVP Workflow Example

This script demonstrates the complete workflow for the DrainageAI super-MVP:
1. Calculate spectral indices from multispectral imagery
2. Detect drainage pipes using the CNN model with spectral indices
3. Vectorize the results for use in GIS software

Usage:
    python super_mvp_workflow.py --imagery <path_to_imagery> --output <output_directory>
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DrainageAI Super-MVP Workflow")
    
    parser.add_argument("--imagery", required=True, help="Path to multispectral imagery file")
    parser.add_argument("--output", required=True, help="Directory to save output files")
    parser.add_argument("--model", default="cnn", choices=["cnn", "semi", "ensemble"], help="Model type to use")
    parser.add_argument("--weights", help="Path to model weights file (optional)")
    parser.add_argument("--red-band", type=int, default=3, help="Band number for red (default: 3)")
    parser.add_argument("--nir-band", type=int, default=4, help="Band number for NIR (default: 4)")
    parser.add_argument("--swir-band", type=int, default=5, help="Band number for SWIR (default: 5)")
    parser.add_argument("--green-band", type=int, default=2, help="Band number for green (default: 2)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection (0-1)")
    parser.add_argument("--simplify", type=float, default=1.0, help="Tolerance for line simplification")
    
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


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Calculate spectral indices
    print("\n=== Step 1: Calculate Spectral Indices ===\n")
    
    indices_path = output_dir / f"indices_{timestamp}.tif"
    
    indices_command = (
        f"python {parent_dir}/main.py indices "
        f"--imagery {args.imagery} "
        f"--output {indices_path} "
        f"--indices ndvi,ndmi,msavi2 "
        f"--red-band {args.red_band} "
        f"--nir-band {args.nir_band} "
        f"--swir-band {args.swir_band} "
        f"--green-band {args.green_band}"
    )
    
    run_command(indices_command)
    
    # Step 2: Detect drainage pipes
    print("\n=== Step 2: Detect Drainage Pipes ===\n")
    
    detection_path = output_dir / f"drainage_{timestamp}.tif"
    
    detect_command = (
        f"python {parent_dir}/main.py detect "
        f"--imagery {args.imagery} "
        f"--indices {indices_path} "
        f"--output {detection_path} "
        f"--model {args.model} "
        f"--threshold {args.threshold}"
    )
    
    if args.weights:
        detect_command += f" --weights {args.weights}"
    
    run_command(detect_command)
    
    # Step 3: Vectorize results
    print("\n=== Step 3: Vectorize Results ===\n")
    
    vector_path = output_dir / f"drainage_lines_{timestamp}.shp"
    
    vectorize_command = (
        f"python {parent_dir}/main.py vectorize "
        f"--input {detection_path} "
        f"--output {vector_path} "
        f"--simplify {args.simplify}"
    )
    
    run_command(vectorize_command)
    
    # Print summary
    print("\n=== Workflow Complete ===\n")
    print("Output files:")
    print(f"  Spectral Indices: {indices_path}")
    print(f"  Drainage Detection: {detection_path}")
    print(f"  Drainage Lines: {vector_path}")
    print("\nThese files can now be loaded into ArcGIS or QGIS for visualization and analysis.")


if __name__ == "__main__":
    main()
