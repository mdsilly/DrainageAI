"""
Main script for DrainageAI.
"""

import os
import sys
import argparse
import torch
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path

from models import EnsembleModel, CNNModel, GNNModel, SelfSupervisedModel
from preprocessing import DataLoader, ImageProcessor, GraphBuilder, Augmentation
from mcp_server import DrainageServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DrainageAI: AI-powered drainage pipe detection")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect drainage pipes in satellite imagery")
    detect_parser.add_argument("--imagery", required=True, help="Path to satellite imagery file")
    detect_parser.add_argument("--elevation", help="Path to elevation data file (optional)")
    detect_parser.add_argument("--output", required=True, help="Path to save detection results")
    detect_parser.add_argument("--model", default="ensemble", choices=["ensemble", "cnn", "gnn", "ssl"], help="Model type to use")
    detect_parser.add_argument("--weights", help="Path to model weights file (optional)")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection (0-1)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a DrainageAI model")
    train_parser.add_argument("--data", required=True, help="Path to training data directory")
    train_parser.add_argument("--model", default="ensemble", choices=["ensemble", "cnn", "gnn", "ssl"], help="Model type to train")
    train_parser.add_argument("--output", required=True, help="Path to save trained model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training")
    
    # Vectorize command
    vectorize_parser = subparsers.add_parser("vectorize", help="Convert raster detection results to vector format")
    vectorize_parser.add_argument("--input", required=True, help="Path to raster detection results")
    vectorize_parser.add_argument("--output", required=True, help="Path to save vector results")
    vectorize_parser.add_argument("--simplify", type=float, default=1.0, help="Tolerance for line simplification")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the DrainageAI MCP server")
    
    return parser.parse_args()


def detect(args):
    """
    Detect drainage pipes in satellite imagery.
    
    Args:
        args: Command line arguments
    """
    print(f"Detecting drainage pipes in {args.imagery}...")
    
    # Load model
    model = load_model(args.model, args.weights)
    
    # Load imagery
    with rasterio.open(args.imagery) as src:
        imagery = src.read()
        meta = src.meta
    
    # Load elevation data if provided
    elevation = None
    if args.elevation:
        with rasterio.open(args.elevation) as src:
            elevation = src.read(1)  # Assume single band
    
    # Preprocess data
    image_processor = ImageProcessor()
    preprocessed_imagery = image_processor.preprocess(imagery)
    
    # Create graph representation if needed
    graph_builder = GraphBuilder()
    graph_data = None
    
    if elevation is not None:
        # Extract node features and positions
        node_positions, node_features = graph_builder._extract_nodes_from_raster(
            imagery, elevation
        )
        
        # Create input data for model
        input_data = {
            "imagery": preprocessed_imagery,
            "node_features": node_features,
            "node_positions": node_positions,
            "elevation": elevation
        }
    else:
        # Create input data for model (CNN only)
        input_data = {
            "imagery": preprocessed_imagery
        }
    
    # Run inference
    with torch.no_grad():
        result = model.predict(input_data)
    
    # Apply confidence threshold
    binary_result = (result > args.threshold).float()
    
    # Save results
    if isinstance(binary_result, torch.Tensor):
        binary_result = binary_result.numpy()
    
    # Save as GeoTIFF
    with rasterio.open(
        args.output,
        "w",
        driver="GTiff",
        height=binary_result.shape[1],
        width=binary_result.shape[2],
        count=1,
        dtype=binary_result.dtype,
        crs=meta["crs"],
        transform=meta["transform"]
    ) as dst:
        dst.write(binary_result[0], 1)
    
    print(f"Detection completed. Results saved to {args.output}")


def train(args):
    """
    Train a DrainageAI model.
    
    Args:
        args: Command line arguments
    """
    print(f"Training {args.model} model on data from {args.data}...")
    
    # Create data loader
    data_loader = DataLoader(args.data, batch_size=args.batch_size)
    
    # Find training data
    imagery_paths = data_loader.find_data_files(
        os.path.join(args.data, "imagery"),
        [".tif", ".tiff"]
    )
    
    label_paths = data_loader.find_data_files(
        os.path.join(args.data, "labels"),
        [".tif", ".tiff", ".shp"]
    )
    
    # Create dataset
    augmentation = Augmentation()
    dataset = data_loader.create_dataset(imagery_paths, label_paths, augmentation)
    
    # Create data loader
    train_loader = data_loader.create_dataloader(dataset, shuffle=True)
    
    # Create model
    model = create_model(args.model)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create loss function
    criterion = torch.nn.BCELoss()
    
    # Train model
    for epoch in range(args.epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Get data
            imagery = batch["imagery"]
            labels = batch["labels"]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(imagery)
            
            # Calculate loss
            loss = criterion(output, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss/len(train_loader):.4f}")
    
    # Save model
    model.save(args.output)
    
    print(f"Training completed. Model saved to {args.output}")


def vectorize(args):
    """
    Convert raster detection results to vector format.
    
    Args:
        args: Command line arguments
    """
    print(f"Vectorizing detection results from {args.input}...")
    
    # Load raster results
    with rasterio.open(args.input) as src:
        raster = src.read(1)  # Assume single band
        transform = src.transform
        crs = src.crs
    
    # Vectorize results
    from skimage.morphology import skeletonize
    skeleton = skeletonize(raster > 0)
    
    # Convert to vector lines
    lines = []
    
    # Find all skeleton pixels
    skeleton_pixels = np.column_stack(np.where(skeleton > 0))
    
    # Group pixels into lines
    if len(skeleton_pixels) > 0:
        current_line = [skeleton_pixels[0]]
        for i in range(1, len(skeleton_pixels)):
            # Check if pixel is adjacent to the last pixel in the line
            last_pixel = current_line[-1]
            pixel = skeleton_pixels[i]
            
            if (abs(pixel[0] - last_pixel[0]) <= 1 and
                abs(pixel[1] - last_pixel[1]) <= 1):
                # Adjacent pixel, add to current line
                current_line.append(pixel)
            else:
                # Not adjacent, start a new line
                if len(current_line) > 1:
                    # Convert pixel coordinates to world coordinates
                    coords = []
                    for p in current_line:
                        # Convert pixel coordinates to world coordinates
                        x, y = transform * (p[1], p[0])
                        coords.append((x, y))
                    
                    # Create line
                    line = LineString(coords)
                    
                    # Simplify line
                    line = line.simplify(args.simplify)
                    
                    # Add to lines
                    lines.append(line)
                
                # Start a new line
                current_line = [pixel]
        
        # Add the last line
        if len(current_line) > 1:
            # Convert pixel coordinates to world coordinates
            coords = []
            for p in current_line:
                # Convert pixel coordinates to world coordinates
                x, y = transform * (p[1], p[0])
                coords.append((x, y))
            
            # Create line
            line = LineString(coords)
            
            # Simplify line
            line = line.simplify(args.simplify)
            
            # Add to lines
            lines.append(line)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"geometry": lines},
        crs=crs
    )
    
    # Save to file
    gdf.to_file(args.output)
    
    print(f"Vectorization completed. Results saved to {args.output}")


def server(args):
    """
    Run the DrainageAI MCP server.
    
    Args:
        args: Command line arguments
    """
    print("Starting DrainageAI MCP server...")
    
    # Create server
    server = DrainageServer()
    
    # Run server
    server.run()


def create_model(model_type):
    """
    Create a DrainageAI model.
    
    Args:
        model_type: Type of model to create
    
    Returns:
        DrainageAI model
    """
    if model_type == "ensemble":
        return EnsembleModel()
    elif model_type == "cnn":
        return CNNModel()
    elif model_type == "gnn":
        return GNNModel()
    elif model_type == "ssl":
        return SelfSupervisedModel(fine_tuned=True)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def load_model(model_type, weights_path=None):
    """
    Load a DrainageAI model.
    
    Args:
        model_type: Type of model to load
        weights_path: Path to model weights file (optional)
    
    Returns:
        DrainageAI model
    """
    # Create model
    model = create_model(model_type)
    
    # Load weights if provided
    if weights_path:
        model.load(weights_path)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Run command
    if args.command == "detect":
        detect(args)
    elif args.command == "train":
        train(args)
    elif args.command == "vectorize":
        vectorize(args)
    elif args.command == "server":
        server(args)
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == "__main__":
    main()
