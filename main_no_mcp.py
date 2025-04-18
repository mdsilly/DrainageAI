"""
Simplified main script for DrainageAI without MCP dependencies.
"""

import os
import sys
import argparse
import torch
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString

from models import EnsembleModel, CNNModel, GNNModel, SelfSupervisedModel, SemiSupervisedModel, BYOLModel, GrayscaleBYOLModel
from preprocessing import DataLoader, ImageProcessor, GraphBuilder, Augmentation
from preprocessing.fixmatch_augmentation import WeakAugmentation, StrongAugmentation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DrainageAI: AI-powered drainage pipe detection")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect drainage pipes in satellite imagery")
    detect_parser.add_argument("--imagery", required=True, help="Path to satellite imagery file")
    detect_parser.add_argument("--elevation", help="Path to elevation data file (optional)")
    detect_parser.add_argument("--indices", help="Path to spectral indices file (optional)")
    detect_parser.add_argument("--sar", help="Path to SAR imagery file (optional)")
    detect_parser.add_argument("--output", required=True, help="Path to save detection results")
    detect_parser.add_argument("--model", default="ensemble", 
                              choices=["ensemble", "cnn", "gnn", "ssl", "semi", "byol", "grayscale-byol"], 
                              help="Model type to use")
    detect_parser.add_argument("--weights", help="Path to model weights file (optional)")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection (0-1)")
    
    # Calculate indices command
    indices_parser = subparsers.add_parser("indices", help="Calculate spectral indices from multispectral imagery")
    indices_parser.add_argument("--imagery", required=True, help="Path to multispectral imagery file")
    indices_parser.add_argument("--output", required=True, help="Path to save indices as a multi-band raster")
    indices_parser.add_argument("--indices", default="ndvi,ndmi,msavi2", help="Comma-separated list of indices to calculate (ndvi,ndmi,msavi2,ndwi,savi)")
    indices_parser.add_argument("--red-band", type=int, default=3, help="Band number for red (default: 3)")
    indices_parser.add_argument("--nir-band", type=int, default=4, help="Band number for NIR (default: 4)")
    indices_parser.add_argument("--swir-band", type=int, default=5, help="Band number for SWIR (default: 5)")
    indices_parser.add_argument("--green-band", type=int, default=2, help="Band number for green (default: 2)")
    indices_parser.add_argument("--l-param", type=float, default=0.5, help="L parameter for SAVI (default: 0.5)")
    
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
    
    return parser.parse_args()


def detect(args):
    """
    Detect drainage pipes in satellite imagery.
    
    Args:
        args: Command line arguments
    """
    print(f"Detecting drainage pipes in {args.imagery}...")
    
    # Check if SAR data is provided
    has_sar = args.sar is not None
    
    # Load model with SAR support if needed
    model = load_model(args.model, args.weights, with_sar=has_sar)
    
    # Load imagery
    with rasterio.open(args.imagery) as src:
        imagery = src.read()
        meta = src.meta
    
    # Load elevation data if provided
    elevation = None
    if args.elevation:
        with rasterio.open(args.elevation) as src:
            elevation = src.read(1)  # Assume single band
    
    # Load spectral indices if provided
    indices = None
    if args.indices:
        with rasterio.open(args.indices) as src:
            indices = src.read()
            print(f"Loaded spectral indices with {indices.shape[0]} bands")
    
    # Load SAR data if provided
    sar_data = None
    if args.sar:
        with rasterio.open(args.sar) as src:
            sar_data = src.read()
            print(f"Loaded SAR data with {sar_data.shape[0]} bands")
    
    # Preprocess data
    image_processor = ImageProcessor()
    
    # Use combined preprocessing if SAR data is available
    if has_sar:
        preprocessed_imagery = image_processor.preprocess_combined(imagery, sar_data)
        print(f"Combined optical and SAR data with {preprocessed_imagery.shape[0]} bands")
    else:
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
        # Create input data for model
        input_data = {
            "imagery": preprocessed_imagery
        }
    
    # Add spectral indices if available
    if indices is not None:
        input_data["indices"] = torch.from_numpy(indices).float()
    
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
    elif model_type == "semi":
        return SemiSupervisedModel(pretrained=True)
    elif model_type == "byol":
        return BYOLModel(fine_tuned=True)
    elif model_type == "grayscale-byol":
        return GrayscaleBYOLModel(fine_tuned=True)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def load_model(model_type, weights_path=None, with_sar=False):
    """
    Load a DrainageAI model.
    
    Args:
        model_type: Type of model to load
        weights_path: Path to model weights file (optional)
        with_sar: Whether to enable SAR support in the model
    
    Returns:
        DrainageAI model
    """
    # Create model with SAR support if needed
    if model_type == "cnn" and with_sar:
        model = CNNModel(with_sar=True)
    elif model_type == "byol" and with_sar:
        model = BYOLModel(with_sar=True)
    elif model_type == "grayscale-byol":
        model = GrayscaleBYOLModel(with_sar=with_sar)
    else:
        # For other model types, SAR support is not implemented in this quick integration
        if with_sar and model_type not in ["cnn", "byol", "grayscale-byol"]:
            print(f"Warning: SAR support is only implemented for CNN, BYOL, and Grayscale-BYOL models in this integration. Using standard {model_type} model.")
        model = create_model(model_type)
    
    # Load weights if provided
    if weights_path:
        model.load(weights_path)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def calculate_indices(args):
    """
    Calculate spectral indices from multispectral imagery.
    
    Args:
        args: Command line arguments
    """
    print(f"Calculating spectral indices for {args.imagery}...")
    
    # Parse indices to calculate
    indices_to_calculate = args.indices.lower().split(",")
    print(f"Indices to calculate: {', '.join(indices_to_calculate)}")
    
    # Load imagery
    with rasterio.open(args.imagery) as src:
        # Read specific bands
        red = src.read(args.red_band)
        nir = src.read(args.nir_band)
        
        # Read optional bands if needed
        green = src.read(args.green_band) if "ndwi" in indices_to_calculate else None
        swir = src.read(args.swir_band) if "ndmi" in indices_to_calculate else None
        
        # Get metadata
        meta = src.meta.copy()
    
    # Calculate indices
    calculated_indices = []
    band_names = []
    
    # Calculate NDVI
    if "ndvi" in indices_to_calculate:
        print("Calculating NDVI...")
        ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
        calculated_indices.append(ndvi)
        band_names.append("NDVI")
    
    # Calculate NDMI
    if "ndmi" in indices_to_calculate and swir is not None:
        print("Calculating NDMI...")
        ndmi = (nir - swir) / (nir + swir + 1e-8)
        calculated_indices.append(ndmi)
        band_names.append("NDMI")
    
    # Calculate MSAVI2
    if "msavi2" in indices_to_calculate:
        print("Calculating MSAVI2...")
        msavi2 = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
        calculated_indices.append(msavi2)
        band_names.append("MSAVI2")
    
    # Calculate NDWI
    if "ndwi" in indices_to_calculate and green is not None:
        print("Calculating NDWI...")
        ndwi = (green - nir) / (green + nir + 1e-8)
        calculated_indices.append(ndwi)
        band_names.append("NDWI")
    
    # Calculate SAVI
    if "savi" in indices_to_calculate:
        print("Calculating SAVI...")
        savi = (nir - red) * (1 + args.l_param) / (nir + red + args.l_param + 1e-8)
        calculated_indices.append(savi)
        band_names.append("SAVI")
    
    # Stack indices
    if not calculated_indices:
        print("No indices were calculated. Please check your input parameters.")
        return
    
    indices_stack = np.stack(calculated_indices)
    
    # Update metadata for output
    meta.update({
        'count': len(calculated_indices),
        'dtype': 'float32'
    })
    
    # Save as GeoTIFF
    with rasterio.open(args.output, 'w', **meta) as dst:
        for i, (index, name) in enumerate(zip(calculated_indices, band_names), 1):
            dst.write(index.astype(np.float32), i)
            dst.set_band_description(i, name)
    
    print(f"Spectral indices calculation completed. Results saved to {args.output}")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Run command
    if args.command == "detect":
        detect(args)
    elif args.command == "indices":
        calculate_indices(args)
    elif args.command == "train":
        train(args)
    elif args.command == "vectorize":
        vectorize(args)
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == "__main__":
    main()
