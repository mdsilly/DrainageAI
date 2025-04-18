"""
Example workflow for using the DrainageAI ArcGIS extension programmatically.

This script demonstrates how to use the DrainageAI tools in a Python script
outside of the ArcGIS Pro interface.
"""

import os
import arcpy
import tempfile
import datetime

# Import the DrainageAI toolbox
script_dir = os.path.dirname(os.path.abspath(__file__))
toolbox_path = os.path.join(script_dir, "DrainageAI.pyt")
arcpy.ImportToolbox(toolbox_path)


def run_drainage_detection_workflow(
    input_imagery,
    output_directory,
    calculate_indices=True,
    detect_drainage=True,
    vectorize_results=True,
    red_band=3,
    nir_band=4,
    swir_band=5,
    green_band=2,
    indices=["NDVI", "NDMI", "MSAVI2"],
    confidence_threshold=0.5,
    model_type="CNN",
    model_path=None,
    simplify_tolerance=1.0
):
    """
    Run the complete DrainageAI workflow.
    
    Args:
        input_imagery: Path to input multispectral imagery
        output_directory: Directory to save outputs
        calculate_indices: Whether to calculate spectral indices
        detect_drainage: Whether to detect drainage pipes
        vectorize_results: Whether to vectorize results
        red_band: Red band number
        nir_band: NIR band number
        swir_band: SWIR band number
        green_band: Green band number
        indices: List of indices to calculate
        confidence_threshold: Confidence threshold for detection
        model_type: Model type (CNN or Ensemble)
        model_path: Path to model weights (optional)
        simplify_tolerance: Tolerance for line simplification
    
    Returns:
        Dictionary with paths to output files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize output paths
    outputs = {
        "indices_raster": None,
        "drainage_raster": None,
        "drainage_vector": None
    }
    
    try:
        # Step 1: Calculate spectral indices
        if calculate_indices:
            print("Calculating spectral indices...")
            indices_raster = os.path.join(output_directory, f"indices_{timestamp}.tif")
            
            # Convert indices list to semicolon-separated string
            indices_str = ";".join(indices)
            
            # Run the Calculate Spectral Indices tool
            arcpy.DrainageAITools.CalculateSpectralIndices(
                input_imagery,
                indices_raster,
                indices_str,
                red_band,
                nir_band,
                swir_band,
                green_band,
                0.5  # L parameter for SAVI
            )
            
            outputs["indices_raster"] = indices_raster
            print(f"Indices calculated and saved to: {indices_raster}")
        
        # Step 2: Detect drainage pipes
        if detect_drainage:
            print("Detecting drainage pipes...")
            drainage_raster = os.path.join(output_directory, f"drainage_{timestamp}.tif")
            
            # Run the Detect Drainage Pipes tool
            arcpy.DrainageAITools.DetectDrainage(
                input_imagery,
                outputs["indices_raster"] if calculate_indices else None,
                None,  # No elevation data for simplicity
                drainage_raster,
                confidence_threshold,
                model_type,
                model_path
            )
            
            outputs["drainage_raster"] = drainage_raster
            print(f"Drainage detection completed and saved to: {drainage_raster}")
        
        # Step 3: Vectorize results
        if vectorize_results and outputs["drainage_raster"]:
            print("Vectorizing drainage results...")
            drainage_vector = os.path.join(output_directory, f"drainage_lines_{timestamp}.shp")
            
            # Run the Vectorize Drainage Results tool
            arcpy.DrainageAITools.VectorizeResults(
                outputs["drainage_raster"],
                drainage_vector,
                confidence_threshold,
                simplify_tolerance
            )
            
            outputs["drainage_vector"] = drainage_vector
            print(f"Vectorization completed and saved to: {drainage_vector}")
        
        print("Workflow completed successfully!")
        return outputs
    
    except Exception as e:
        print(f"Error in workflow: {str(e)}")
        arcpy.AddError(str(e))
        raise


if __name__ == "__main__":
    # Example usage
    # Replace these paths with actual data paths
    input_imagery = r"C:\Data\sample_imagery.tif"
    output_directory = r"C:\Data\drainage_results"
    
    # Run the workflow
    results = run_drainage_detection_workflow(
        input_imagery=input_imagery,
        output_directory=output_directory,
        calculate_indices=True,
        detect_drainage=True,
        vectorize_results=True,
        red_band=3,
        nir_band=4,
        swir_band=5,
        green_band=2,
        indices=["NDVI", "NDMI", "MSAVI2"],
        confidence_threshold=0.5,
        model_type="CNN",
        simplify_tolerance=1.0
    )
    
    print("\nOutput files:")
    for key, path in results.items():
        if path:
            print(f"  {key}: {path}")
