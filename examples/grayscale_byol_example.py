"""
Grayscale BYOL Example

This script demonstrates how to use the GrayscaleBYOLModel for training and inference
with grayscale (single-channel) images.

Usage:
    python examples/grayscale_byol_example.py --optical-dir data/unlabeled --output-dir results
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import rasterio

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.grayscale_byol_model import GrayscaleBYOLModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Grayscale BYOL Example")
    
    parser.add_argument("--optical-dir", required=True, help="Directory containing optical imagery")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--test-image", help="Path to test image for inference")
    
    return parser.parse_args()


def find_images(directory, extensions=['.tif', '.tiff']):
    """Find all images with specified extensions in a directory."""
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                images.append(os.path.join(root, file))
    return images


def check_image_channels(image_path):
    """Check the number of channels in an image."""
    with rasterio.open(image_path) as src:
        num_channels = src.count
    return num_channels


def load_image(image_path):
    """Load an image as a PyTorch tensor."""
    with rasterio.open(image_path) as src:
        data = src.read()
        # Convert to torch tensor and add batch dimension
        tensor = torch.from_numpy(data).float().unsqueeze(0)
    return tensor


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all images in the input directory
    images = find_images(args.optical_dir)
    if not images:
        print(f"No images found in {args.optical_dir}")
        return
    
    print(f"Found {len(images)} images")
    
    # Check the number of channels in the first image
    num_channels = check_image_channels(images[0])
    print(f"First image has {num_channels} channel(s)")
    
    # Initialize the grayscale-compatible BYOL model
    model = GrayscaleBYOLModel(pretrained=True, with_sar=False)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Simple training loop (for demonstration purposes)
    # In a real application, you would use a proper DataLoader and training loop
    print("\n=== Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        
        # Process images in batches
        for i in range(0, len(images), args.batch_size):
            batch_images = images[i:i+args.batch_size]
            
            # Load images
            views1 = []
            views2 = []
            
            for img_path in batch_images:
                # Load image
                img = load_image(img_path)
                
                # Create two views (in a real application, you would apply different augmentations)
                views1.append(img)
                views2.append(img)
            
            # Stack images into batches
            if views1:
                view1_batch = torch.cat(views1, dim=0).to(device)
                view2_batch = torch.cat(views2, dim=0).to(device)
                
                # Compute BYOL loss
                loss, _ = model.byol_loss(view1_batch, view2_batch)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network
                model.update_target_network()
                
                total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / (len(images) / args.batch_size)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, "grayscale_byol_model.pth")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Run inference if a test image is provided
    if args.test_image:
        print("\n=== Inference ===")
        
        # Load test image
        test_img = load_image(args.test_image).to(device)
        
        # Run inference
        with torch.no_grad():
            features = model(test_img)
        
        print(f"Extracted features shape: {features.shape}")
        
        # Save features (for demonstration purposes)
        features_path = os.path.join(args.output_dir, "features.npy")
        np.save(features_path, features.cpu().numpy())
        print(f"Features saved to {features_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
