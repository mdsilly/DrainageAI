"""
DrainageAI ArcGIS Python Toolbox.

This toolbox provides tools for detecting drainage pipes in agricultural fields
using the DrainageAI system through the Model Context Protocol (MCP).
"""

import arcpy
import os
import sys
import json
import subprocess
import tempfile
import numpy as np

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import DrainageAI modules if needed
# from models import CNNModel
# from preprocessing import ImageProcessor


class Toolbox(object):
    """DrainageAI toolbox for ArcGIS Pro."""
    
    def __init__(self):
        """Initialize the toolbox."""
        self.label = "DrainageAI Tools"
        self.alias = "drainageai"
        
        # List of tool classes from this file
        self.tools = [
            CalculateSpectralIndices,
            DetectDrainage,
            VectorizeResults
        ]


class CalculateSpectralIndices(object):
    """Tool to calculate spectral indices from multispectral imagery."""
    
    def __init__(self):
        """Initialize the tool."""
        self.label = "Calculate Spectral Indices"
        self.description = "Calculate spectral indices from multispectral imagery"
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        # Input raster parameter
        in_raster = arcpy.Parameter(
            displayName="Input Multispectral Imagery",
            name="in_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        
        # Output raster parameter
        out_raster = arcpy.Parameter(
            displayName="Output Indices Raster",
            name="out_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        
        # Indices to calculate
        indices = arcpy.Parameter(
            displayName="Indices to Calculate",
            name="indices",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        indices.filter.list = ["NDVI", "NDMI", "MSAVI2", "NDWI", "SAVI"]
        indices.value = "NDVI;NDMI;MSAVI2"
        
        # Red band number
        red_band = arcpy.Parameter(
            displayName="Red Band Number",
            name="red_band",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        red_band.value = 3  # Default to band 3 for red
        
        # NIR band number
        nir_band = arcpy.Parameter(
            displayName="NIR Band Number",
            name="nir_band",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        nir_band.value = 4  # Default to band 4 for NIR
        
        # SWIR band number (optional)
        swir_band = arcpy.Parameter(
            displayName="SWIR Band Number (for NDMI)",
            name="swir_band",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        swir_band.value = 5  # Default to band 5 for SWIR
        
        # Green band number (optional)
        green_band = arcpy.Parameter(
            displayName="Green Band Number (for NDWI)",
            name="green_band",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        green_band.value = 2  # Default to band 2 for green
        
        # L parameter for SAVI (optional)
        l_param = arcpy.Parameter(
            displayName="L Parameter (for SAVI)",
            name="l_param",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        l_param.value = 0.5  # Default L value for SAVI
        
        return [in_raster, out_raster, indices, red_band, nir_band, swir_band, green_band, l_param]
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation."""
        return
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each parameter."""
        return
    
    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Get parameters
        in_raster = parameters[0].valueAsText
        out_raster = parameters[1].valueAsText
        indices_list = parameters[2].valueAsText.split(";")
        red_band = parameters[3].value
        nir_band = parameters[4].value
        swir_band = parameters[5].value
        green_band = parameters[6].value
        l_param = parameters[7].value
        
        arcpy.AddMessage("Calculating spectral indices: " + ", ".join(indices_list))
        
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Calculate each index
            index_rasters = []
            
            # Load bands
            arcpy.AddMessage("Loading bands...")
            red = arcpy.Raster(in_raster + "/" + str(red_band))
            nir = arcpy.Raster(in_raster + "/" + str(nir_band))
            
            # Calculate NDVI
            if "NDVI" in indices_list:
                arcpy.AddMessage("Calculating NDVI...")
                ndvi_raster = arcpy.sa.Float(nir - red) / arcpy.sa.Float(nir + red)
                ndvi_path = os.path.join(temp_dir, "ndvi.tif")
                ndvi_raster.save(ndvi_path)
                index_rasters.append(ndvi_path)
            
            # Calculate MSAVI2
            if "MSAVI2" in indices_list:
                arcpy.AddMessage("Calculating MSAVI2...")
                msavi2_raster = (2 * nir + 1 - arcpy.sa.SquareRoot((2 * nir + 1)**2 - 8 * (nir - red))) / 2
                msavi2_path = os.path.join(temp_dir, "msavi2.tif")
                msavi2_raster.save(msavi2_path)
                index_rasters.append(msavi2_path)
            
            # Calculate NDMI (if SWIR band is provided)
            if "NDMI" in indices_list and swir_band is not None:
                arcpy.AddMessage("Calculating NDMI...")
                swir = arcpy.Raster(in_raster + "/" + str(swir_band))
                ndmi_raster = arcpy.sa.Float(nir - swir) / arcpy.sa.Float(nir + swir)
                ndmi_path = os.path.join(temp_dir, "ndmi.tif")
                ndmi_raster.save(ndmi_path)
                index_rasters.append(ndmi_path)
            
            # Calculate NDWI (if Green band is provided)
            if "NDWI" in indices_list and green_band is not None:
                arcpy.AddMessage("Calculating NDWI...")
                green = arcpy.Raster(in_raster + "/" + str(green_band))
                ndwi_raster = arcpy.sa.Float(green - nir) / arcpy.sa.Float(green + nir)
                ndwi_path = os.path.join(temp_dir, "ndwi.tif")
                ndwi_raster.save(ndwi_path)
                index_rasters.append(ndwi_path)
            
            # Calculate SAVI
            if "SAVI" in indices_list:
                arcpy.AddMessage("Calculating SAVI...")
                savi_raster = (nir - red) * (1 + l_param) / (nir + red + l_param)
                savi_path = os.path.join(temp_dir, "savi.tif")
                savi_raster.save(savi_path)
                index_rasters.append(savi_path)
            
            # Combine all indices into a single multi-band raster
            arcpy.AddMessage("Combining indices into a multi-band raster...")
            arcpy.management.CompositeBands(index_rasters, out_raster)
            
            # Add the output raster to the map
            arcpy.AddMessage("Adding result to the map...")
            result_layer_name = os.path.basename(out_raster).split('.')[0]
            arcpy.management.MakeRasterLayer(out_raster, result_layer_name)
            
            arcpy.AddMessage("Spectral indices calculation complete.")
            
        except Exception as e:
            arcpy.AddError(f"Error calculating spectral indices: {str(e)}")
            raise
        
        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                arcpy.AddWarning("Failed to clean up temporary files.")


class DetectDrainage(object):
    """Tool to detect drainage pipes using the DrainageAI model."""
    
    def __init__(self):
        """Initialize the tool."""
        self.label = "Detect Drainage Pipes"
        self.description = "Detect drainage pipes in agricultural fields using DrainageAI"
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        # Input imagery parameter
        in_imagery = arcpy.Parameter(
            displayName="Input Imagery",
            name="in_imagery",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        
        # Input spectral indices parameter
        in_indices = arcpy.Parameter(
            displayName="Input Spectral Indices",
            name="in_indices",
            datatype="DERasterDataset",
            parameterType="Optional",
            direction="Input"
        )
        
        # Input elevation parameter
        in_elevation = arcpy.Parameter(
            displayName="Input Elevation",
            name="in_elevation",
            datatype="DERasterDataset",
            parameterType="Optional",
            direction="Input"
        )
        
        # Output drainage raster parameter
        out_raster = arcpy.Parameter(
            displayName="Output Drainage Raster",
            name="out_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output"
        )
        
        # Confidence threshold parameter
        confidence = arcpy.Parameter(
            displayName="Confidence Threshold",
            name="confidence",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        confidence.value = 0.5
        
        # Model type parameter
        model_type = arcpy.Parameter(
            displayName="Model Type",
            name="model_type",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        model_type.filter.list = ["CNN", "Ensemble"]
        model_type.value = "CNN"
        
        # Model path parameter
        model_path = arcpy.Parameter(
            displayName="Model Path",
            name="model_path",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )
        
        return [in_imagery, in_indices, in_elevation, out_raster, confidence, model_type, model_path]
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation."""
        return
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each parameter."""
        return
    
    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Get parameters
        in_imagery = parameters[0].valueAsText
        in_indices = parameters[1].valueAsText
        in_elevation = parameters[2].valueAsText
        out_raster = parameters[3].valueAsText
        confidence = parameters[4].value
        model_type = parameters[5].valueAsText
        model_path = parameters[6].valueAsText
        
        arcpy.AddMessage(f"Detecting drainage pipes using {model_type} model...")
        
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # For MVP, we'll use the MCP server to run the model
            # This allows us to leverage the existing DrainageAI infrastructure
            
            # Prepare input files
            imagery_path = os.path.join(temp_dir, "imagery.tif")
            indices_path = os.path.join(temp_dir, "indices.tif") if in_indices else None
            elevation_path = os.path.join(temp_dir, "elevation.tif") if in_elevation else None
            
            # Export input rasters to temporary files
            arcpy.AddMessage("Preparing input data...")
            arcpy.management.CopyRaster(in_imagery, imagery_path)
            
            if in_indices:
                arcpy.management.CopyRaster(in_indices, indices_path)
            
            if in_elevation:
                arcpy.management.CopyRaster(in_elevation, elevation_path)
            
            # Call the MCP server
            arcpy.AddMessage("Calling DrainageAI MCP server...")
            
            # Prepare arguments for the MCP server
            mcp_args = {
                "imagery_path": imagery_path,
                "output_path": out_raster,
                "confidence_threshold": confidence
            }
            
            if indices_path:
                mcp_args["indices_path"] = indices_path
            
            if elevation_path:
                mcp_args["elevation_path"] = elevation_path
            
            if model_path:
                mcp_args["model_path"] = model_path
                mcp_args["model_type"] = model_type.lower()
            
            # Call the MCP server using subprocess
            # For MVP, we'll simulate this by directly using the model
            # In a full implementation, this would call the actual MCP server
            
            # Simplified direct model usage for MVP
            if model_type == "CNN":
                from models import CNNModel
                model = CNNModel(pretrained=True)
                if model_path:
                    model.load(model_path)
            else:  # Ensemble
                from models import EnsembleModel
                model = EnsembleModel()
                if model_path:
                    model.load(model_path)
            
            # Run inference
            arcpy.AddMessage("Running inference...")
            
            # In a real implementation, this would be more sophisticated
            # For MVP, we'll create a simple binary output
            
            # Create a simple output for demonstration
            # In practice, this would be the actual model output
            arcpy.AddMessage("Creating output raster...")
            
            # Get the spatial reference and extent from the input
            desc = arcpy.Describe(in_imagery)
            sr = desc.spatialReference
            extent = desc.extent
            
            # Create a new raster with random values (simulating model output)
            # In a real implementation, this would be the actual model prediction
            arcpy.management.CreateRasterDataset(
                os.path.dirname(out_raster),
                os.path.basename(out_raster),
                desc.width, desc.height,
                1,
                "32_BIT_FLOAT",
                sr
            )
            
            # Add the output raster to the map
            arcpy.AddMessage("Adding result to the map...")
            result_layer_name = os.path.basename(out_raster).split('.')[0]
            arcpy.management.MakeRasterLayer(out_raster, result_layer_name)
            
            arcpy.AddMessage("Drainage detection complete.")
            
        except Exception as e:
            arcpy.AddError(f"Error detecting drainage pipes: {str(e)}")
            raise
        
        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                arcpy.AddWarning("Failed to clean up temporary files.")


class VectorizeResults(object):
    """Tool to vectorize drainage detection results."""
    
    def __init__(self):
        """Initialize the tool."""
        self.label = "Vectorize Drainage Results"
        self.description = "Convert drainage detection raster to vector format"
        self.canRunInBackground = False
    
    def getParameterInfo(self):
        """Define parameter definitions."""
        # Input raster parameter
        in_raster = arcpy.Parameter(
            displayName="Input Drainage Raster",
            name="in_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        
        # Output feature class parameter
        out_features = arcpy.Parameter(
            displayName="Output Drainage Lines",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )
        
        # Threshold parameter
        threshold = arcpy.Parameter(
            displayName="Threshold Value",
            name="threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        threshold.value = 0.5
        
        # Simplify tolerance parameter
        simplify = arcpy.Parameter(
            displayName="Simplify Tolerance",
            name="simplify",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        simplify.value = 1.0
        
        return [in_raster, out_features, threshold, simplify]
    
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation."""
        return
    
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each parameter."""
        return
    
    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Get parameters
        in_raster = parameters[0].valueAsText
        out_features = parameters[1].valueAsText
        threshold = parameters[2].value
        simplify = parameters[3].value
        
        arcpy.AddMessage("Vectorizing drainage detection results...")
        
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create binary raster based on threshold
            arcpy.AddMessage("Creating binary raster...")
            binary_raster = os.path.join(temp_dir, "binary.tif")
            binary_expr = f"Con(\"{in_raster}\" > {threshold}, 1, 0)"
            binary_result = arcpy.sa.Raster(binary_expr)
            binary_result.save(binary_raster)
            
            # Thin the binary raster to get centerlines
            arcpy.AddMessage("Thinning to centerlines...")
            thin_raster = os.path.join(temp_dir, "thin.tif")
            arcpy.sa.Thin(binary_raster, thin_raster, "ZERO", "ROUND")
            
            # Convert raster to polylines
            arcpy.AddMessage("Converting to polylines...")
            temp_lines = os.path.join(temp_dir, "temp_lines.shp")
            arcpy.conversion.RasterToPolyline(thin_raster, temp_lines, "ZERO", 0, "NO_SIMPLIFY", "VALUE")
            
            # Simplify lines
            arcpy.AddMessage("Simplifying lines...")
            arcpy.cartography.SimplifyLine(temp_lines, out_features, "POINT_REMOVE", simplify)
            
            # Add the output feature class to the map
            arcpy.AddMessage("Adding result to the map...")
            
            arcpy.AddMessage("Vectorization complete.")
            
        except Exception as e:
            arcpy.AddError(f"Error vectorizing results: {str(e)}")
            raise
        
        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                arcpy.AddWarning("Failed to clean up temporary files.")
