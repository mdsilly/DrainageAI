"""
BYOL MVP Workflow Example

This script demonstrates the complete workflow for using BYOL (Bootstrap Your Own Latent)
with few or no labeled images for drainage pipe detection:

1. BYOL pretraining on unlabeled data (both optical and SAR)
2. Fine-tuning with very few labeled examples
3. Inference and evaluation

Usage:
    python byol_mvp_workflow.py --optical-dir <path_to_optical_imagery> --sar-dir <path_to_sar_imagery> --output-dir <output_directory>
"""

import os
import sys
import argparse
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import rasterio
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.byol_model import BYOLModel
from training.train_byol import train_byol_pipeline


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BYOL MVP Workflow for DrainageAI")
    
    parser.add_argument("--optical-dir", required=True, help="Directory containing optical imagery")
    parser.add_argument("--sar-dir", help="Directory containing SAR imagery (optional)")
    parser.add_argument("--label-dir", help="Directory containing labels (optional)")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--num-labeled", type=int, default=5, help="Number of labeled examples to use")
    parser.add_argument("--byol-epochs", type=int, default=50, help="Number of BYOL pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--inference-only", action="store_true", help="Skip training and run inference only")
    parser.add_argument("--model-path", help="Path to pretrained model for inference")
    parser.add_argument("--test-image", help="Path to test image for inference")
    parser.add_argument("--test-sar", help="Path to test SAR image for inference")
    
    return parser.parse_args()


def visualize_results(image_path, sar_path, prediction_path, output_path):
    """
    Visualize the detection results.
    
    Args:
        image_path: Path to optical image
        sar_path: Path to SAR image (optional)
        prediction_path: Path to prediction mask
        output_path: Path to save visualization
    """
    # Load optical image
    with rasterio.open(image_path) as src:
        optical = src.read()
        transform = src.transform
        crs = src.crs
    
    # Load SAR image if available
    sar = None
    if sar_path:
        with rasterio.open(sar_path) as src:
            sar = src.read()
    
    # Load prediction
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3 if sar is not None else 2, figsize=(15, 5))
    
    # Plot optical image (RGB)
    if optical.shape[0] >= 3:
        # Get RGB bands (assuming bands are in order R, G, B)
        rgb = np.stack([
            optical[0] / optical[0].max(),
            optical[1] / optical[1].max(),
            optical[2] / optical[2].max()
        ], axis=-1)
        
        axes[0].imshow(rgb)
        axes[0].set_title("Optical Image")
    else:
        axes[0].imshow(optical[0], cmap='gray')
        axes[0].set_title("Optical Image")
    
    # Plot SAR image if available
    if sar is not None:
        axes[1].imshow(sar[0], cmap='gray')
        axes[1].set_title("SAR Image")
        pred_idx = 2
    else:
        pred_idx = 1
    
    # Plot prediction
    axes[pred_idx].imshow(prediction, cmap='hot')
    axes[pred_idx].set_title("Drainage Prediction")
    
    # Remove axis ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def inference(model_path, optical_path, sar_path=None, output_dir=None):
    """
    Run inference on a test image.
    
    Args:
        model_path: Path to trained model
        optical_path: Path to optical image
        sar_path: Path to SAR image (optional)
        output_dir: Directory to save outputs
        
    Returns:
        Path to prediction mask
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(optical_path)
    
    # Load model
    model = BYOLModel()
    model.load(model_path)
    model.fine_tuned = True  # Ensure model is in fine-tuned mode
    model.eval()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load optical image
    with rasterio.open(optical_path) as src:
        optical = src.read()
        meta = src.meta.copy()
    
    # Load SAR image if available
    sar = None
    if sar_path:
        with rasterio.open(sar_path) as src:
            sar = src.read()
    
    # Convert to torch tensors
    optical = torch.from_numpy(optical).float().unsqueeze(0).to(device)
    
    # Combine with SAR if available
    if sar is not None and model.with_sar:
        sar = torch.from_numpy(sar).float().unsqueeze(0).to(device)
        
        # Ensure optical and SAR have the same spatial dimensions
        if optical.shape[2:] != sar.shape[2:]:
            sar = torch.nn.functional.interpolate(
                sar, size=optical.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Concatenate along channel dimension
        x = torch.cat([optical, sar], dim=1)
    else:
        x = optical
    
    # Run inference
    with torch.no_grad():
        prediction = model(x)
        binary_mask = (prediction > 0.5).float()
    
    # Convert to numpy
    binary_mask = binary_mask.cpu().numpy()[0, 0]
    
    # Save prediction
    output_path = os.path.join(output_dir, f"prediction_{Path(optical_path).stem}.tif")
    
    # Update metadata for output
    meta.update({
        'count': 1,
        'dtype': 'float32'
    })
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(binary_mask.astype(np.float32), 1)
    
    print(f"Prediction saved to {output_path}")
    
    return output_path


def evaluate_prediction(prediction_path, label_path):
    """
    Evaluate prediction against ground truth.
    
    Args:
        prediction_path: Path to prediction mask
        label_path: Path to ground truth mask
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load prediction
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    
    # Load ground truth
    with rasterio.open(label_path) as src:
        label = src.read(1)
    
    # Flatten arrays
    prediction = prediction.flatten()
    label = label.flatten()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, prediction, average='binary', zero_division=0
    )
    accuracy = accuracy_score(label, prediction)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"Evaluation metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return metrics


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not args.inference_only:
        # Step 1: BYOL pretraining and fine-tuning
        print("\n=== Step 1: BYOL Pretraining and Fine-tuning ===\n")
        
        model, metrics = train_byol_pipeline(
            optical_dir=args.optical_dir,
            sar_dir=args.sar_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            num_labeled=args.num_labeled,
            byol_epochs=args.byol_epochs,
            finetune_epochs=args.finetune_epochs
        )
        
        # Use the fine-tuned model for inference if available
        model_path = os.path.join(args.output_dir, "byol_finetuned.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.output_dir, "byol_pretrained.pth")
    else:
        # Use provided model path for inference
        model_path = args.model_path
        if not model_path:
            print("Error: Must provide --model-path when using --inference-only")
            sys.exit(1)
    
    # Step 2: Run inference on test image
    print("\n=== Step 2: Run Inference ===\n")
    
    # Use provided test image or find one in the optical directory
    test_image = args.test_image
    test_sar = args.test_sar
    
    if not test_image and args.optical_dir:
        # Find first image in optical directory
        for root, _, files in os.walk(args.optical_dir):
            for file in files:
                if file.endswith(('.tif', '.tiff')):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("Error: No test image found. Please provide --test-image")
        sys.exit(1)
    
    # Find matching SAR image if not provided
    if not test_sar and args.sar_dir:
        test_base = os.path.basename(test_image).split('.')[0]
        for root, _, files in os.walk(args.sar_dir):
            for file in files:
                if file.startswith(test_base) and file.endswith(('.tif', '.tiff')):
                    test_sar = os.path.join(root, file)
                    break
            if test_sar:
                break
    
    # Run inference
    prediction_path = inference(
        model_path=model_path,
        optical_path=test_image,
        sar_path=test_sar,
        output_dir=args.output_dir
    )
    
    # Step 3: Visualize results
    print("\n=== Step 3: Visualize Results ===\n")
    
    visualization_path = os.path.join(args.output_dir, f"visualization_{timestamp}.png")
    visualize_results(
        image_path=test_image,
        sar_path=test_sar,
        prediction_path=prediction_path,
        output_path=visualization_path
    )
    
    # Step 4: Evaluate if ground truth is available
    if args.label_dir:
        print("\n=== Step 4: Evaluate Results ===\n")
        
        # Find matching label
        test_base = os.path.basename(test_image).split('.')[0]
        test_label = None
        
        for root, _, files in os.walk(args.label_dir):
            for file in files:
                if file.startswith(test_base) and file.endswith(('.tif', '.tiff')):
                    test_label = os.path.join(root, file)
                    break
            if test_label:
                break
        
        if test_label:
            metrics = evaluate_prediction(prediction_path, test_label)
        else:
            print("No matching label found for evaluation")
    
    # Step 5: Vectorize results
    print("\n=== Step 5: Vectorize Results ===\n")
    
    vector_path = os.path.join(args.output_dir, f"drainage_lines_{timestamp}.shp")
    
    vectorize_command = (
        f"python {parent_dir}/main.py vectorize "
        f"--input {prediction_path} "
        f"--output {vector_path} "
        f"--simplify 1.0"
    )
    
    run_command(vectorize_command)
    
    # Print summary
    print("\n=== Workflow Complete ===\n")
    print("Output files:")
    print(f"  BYOL Model: {model_path}")
    print(f"  Prediction: {prediction_path}")
    print(f"  Visualization: {visualization_path}")
    print(f"  Vector Lines: {vector_path}")
    
    print("\nThese files can now be loaded into ArcGIS or QGIS for visualization and analysis.")
    print("\nExample command for full pipeline with few labeled images:")
    print(f"python examples/byol_mvp_workflow.py --optical-dir data/imagery --sar-dir data/sar --label-dir data/labels --output-dir results --num-labeled 5 --byol-epochs 50 --finetune-epochs 10")
    
    print("\nExample command for inference only:")
    print(f"python examples/byol_mvp_workflow.py --inference-only --model-path results/byol_finetuned.pth --test-image data/test/image.tif --test-sar data/test/sar.tif --output-dir results")


if __name__ == "__main__":
    main()
