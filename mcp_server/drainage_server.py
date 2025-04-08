"""
MCP server for DrainageAI integration with QGIS.
"""

import os
import sys
import json
import torch
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString
from modelcontextprotocol.sdk.server import Server
from modelcontextprotocol.sdk.server.stdio import StdioServerTransport
from modelcontextprotocol.sdk.types import (
    CallToolRequestSchema,
    ErrorCode,
    ListResourcesRequestSchema,
    ListResourceTemplatesRequestSchema,
    ListToolsRequestSchema,
    McpError,
    ReadResourceRequestSchema,
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EnsembleModel
from preprocessing import DataLoader, ImageProcessor, GraphBuilder


class DrainageServer:
    """MCP server for DrainageAI integration with QGIS."""
    
    def __init__(self):
        """Initialize the DrainageAI MCP server."""
        self.server = Server(
            {
                "name": "drainage-ai-server",
                "version": "0.1.0",
            },
            {
                "capabilities": {
                    "resources": {},
                    "tools": {},
                }
            }
        )
        
        # Initialize model
        self.model = None
        
        # Set up request handlers
        self.setup_resource_handlers()
        self.setup_tool_handlers()
        
        # Error handling
        self.server.onerror = lambda error: print(f"[MCP Error] {error}", file=sys.stderr)
        
        # Signal handling
        import signal
        signal.signal(signal.SIGINT, self._handle_sigint)
    
    def _handle_sigint(self, sig, frame):
        """Handle SIGINT signal."""
        print("Shutting down DrainageAI MCP server...", file=sys.stderr)
        self.server.close()
        sys.exit(0)
    
    def setup_resource_handlers(self):
        """Set up resource handlers."""
        self.server.setRequestHandler(
            ListResourcesRequestSchema,
            self.handle_list_resources
        )
        
        self.server.setRequestHandler(
            ListResourceTemplatesRequestSchema,
            self.handle_list_resource_templates
        )
        
        self.server.setRequestHandler(
            ReadResourceRequestSchema,
            self.handle_read_resource
        )
    
    def setup_tool_handlers(self):
        """Set up tool handlers."""
        self.server.setRequestHandler(
            ListToolsRequestSchema,
            self.handle_list_tools
        )
        
        self.server.setRequestHandler(
            CallToolRequestSchema,
            self.handle_call_tool
        )
    
    def handle_list_resources(self, request):
        """Handle list resources request."""
        return {
            "resources": [
                {
                    "uri": "drainage://model/info",
                    "name": "DrainageAI Model Information",
                    "mimeType": "application/json",
                    "description": "Information about the current DrainageAI model"
                }
            ]
        }
    
    def handle_list_resource_templates(self, request):
        """Handle list resource templates request."""
        return {
            "resourceTemplates": [
                {
                    "uriTemplate": "drainage://results/{id}",
                    "name": "DrainageAI Detection Results",
                    "mimeType": "application/json",
                    "description": "Results of a drainage detection run"
                }
            ]
        }
    
    def handle_read_resource(self, request):
        """Handle read resource request."""
        uri = request.params.uri
        
        # Handle model info resource
        if uri == "drainage://model/info":
            model_info = self._get_model_info()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(model_info, indent=2)
                    }
                ]
            }
        
        # Handle results resource
        if uri.startswith("drainage://results/"):
            result_id = uri.split("/")[-1]
            result_path = os.path.join("data", "results", f"{result_id}.json")
            
            if not os.path.exists(result_path):
                raise McpError(
                    ErrorCode.NotFound,
                    f"Result with ID {result_id} not found"
                )
            
            with open(result_path, "r") as f:
                result_data = json.load(f)
            
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(result_data, indent=2)
                    }
                ]
            }
        
        raise McpError(
            ErrorCode.InvalidRequest,
            f"Invalid resource URI: {uri}"
        )
    
    def handle_list_tools(self, request):
        """Handle list tools request."""
        return {
            "tools": [
                {
                    "name": "detect_drainage",
                    "description": "Detect drainage pipes in satellite imagery",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "imagery_path": {
                                "type": "string",
                                "description": "Path to satellite imagery file"
                            },
                            "elevation_path": {
                                "type": "string",
                                "description": "Path to elevation data file (optional)"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path to save detection results"
                            },
                            "confidence_threshold": {
                                "type": "number",
                                "description": "Confidence threshold for detection (0-1)",
                                "default": 0.5
                            }
                        },
                        "required": ["imagery_path", "output_path"]
                    }
                },
                {
                    "name": "load_model",
                    "description": "Load a DrainageAI model",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model_path": {
                                "type": "string",
                                "description": "Path to model weights file"
                            },
                            "model_type": {
                                "type": "string",
                                "description": "Type of model to load",
                                "enum": ["ensemble", "cnn", "gnn", "ssl"],
                                "default": "ensemble"
                            }
                        },
                        "required": ["model_path"]
                    }
                },
                {
                    "name": "vectorize_results",
                    "description": "Convert raster detection results to vector format",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "input_path": {
                                "type": "string",
                                "description": "Path to raster detection results"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path to save vector results"
                            },
                            "simplify_tolerance": {
                                "type": "number",
                                "description": "Tolerance for line simplification",
                                "default": 1.0
                            }
                        },
                        "required": ["input_path", "output_path"]
                    }
                }
            ]
        }
    
    def handle_call_tool(self, request):
        """Handle call tool request."""
        tool_name = request.params.name
        args = request.params.arguments
        
        if tool_name == "detect_drainage":
            return self._detect_drainage(args)
        elif tool_name == "load_model":
            return self._load_model(args)
        elif tool_name == "vectorize_results":
            return self._vectorize_results(args)
        else:
            raise McpError(
                ErrorCode.MethodNotFound,
                f"Unknown tool: {tool_name}"
            )
    
    def _detect_drainage(self, args):
        """
        Detect drainage pipes in satellite imagery.
        
        Args:
            args: Tool arguments
                - imagery_path: Path to satellite imagery file
                - elevation_path: Path to elevation data file (optional)
                - output_path: Path to save detection results
                - confidence_threshold: Confidence threshold for detection
        
        Returns:
            Tool result
        """
        # Validate arguments
        if "imagery_path" not in args:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: imagery_path"
            )
        
        if "output_path" not in args:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: output_path"
            )
        
        # Load model if not already loaded
        if self.model is None:
            self._load_default_model()
        
        # Load imagery
        try:
            with rasterio.open(args["imagery_path"]) as src:
                imagery = src.read()
                meta = src.meta
        except Exception as e:
            raise McpError(
                ErrorCode.InternalError,
                f"Failed to load imagery: {str(e)}"
            )
        
        # Load elevation data if provided
        elevation = None
        if "elevation_path" in args and args["elevation_path"]:
            try:
                with rasterio.open(args["elevation_path"]) as src:
                    elevation = src.read(1)  # Assume single band
            except Exception as e:
                raise McpError(
                    ErrorCode.InternalError,
                    f"Failed to load elevation data: {str(e)}"
                )
        
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
        try:
            with torch.no_grad():
                result = self.model.predict(input_data)
        except Exception as e:
            raise McpError(
                ErrorCode.InternalError,
                f"Failed to run inference: {str(e)}"
            )
        
        # Apply confidence threshold
        confidence_threshold = args.get("confidence_threshold", 0.5)
        binary_result = (result > confidence_threshold).float()
        
        # Save results
        try:
            # Convert to numpy array
            if isinstance(binary_result, torch.Tensor):
                binary_result = binary_result.numpy()
            
            # Save as GeoTIFF
            with rasterio.open(
                args["output_path"],
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
        except Exception as e:
            raise McpError(
                ErrorCode.InternalError,
                f"Failed to save results: {str(e)}"
            )
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Drainage detection completed. Results saved to {args['output_path']}"
                }
            ]
        }
    
    def _load_model(self, args):
        """
        Load a DrainageAI model.
        
        Args:
            args: Tool arguments
                - model_path: Path to model weights file
                - model_type: Type of model to load
        
        Returns:
            Tool result
        """
        # Validate arguments
        if "model_path" not in args:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: model_path"
            )
        
        model_type = args.get("model_type", "ensemble")
        
        # Load model
        try:
            if model_type == "ensemble":
                from models import EnsembleModel
                self.model = EnsembleModel()
            elif model_type == "cnn":
                from models import CNNModel
                self.model = CNNModel()
            elif model_type == "gnn":
                from models import GNNModel
                self.model = GNNModel()
            elif model_type == "ssl":
                from models import SelfSupervisedModel
                self.model = SelfSupervisedModel(fine_tuned=True)
            else:
                raise McpError(
                    ErrorCode.InvalidParams,
                    f"Invalid model type: {model_type}"
                )
            
            self.model.load(args["model_path"])
            self.model.eval()
        except Exception as e:
            raise McpError(
                ErrorCode.InternalError,
                f"Failed to load model: {str(e)}"
            )
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Model loaded successfully: {model_type} from {args['model_path']}"
                }
            ]
        }
    
    def _vectorize_results(self, args):
        """
        Convert raster detection results to vector format.
        
        Args:
            args: Tool arguments
                - input_path: Path to raster detection results
                - output_path: Path to save vector results
                - simplify_tolerance: Tolerance for line simplification
        
        Returns:
            Tool result
        """
        # Validate arguments
        if "input_path" not in args:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: input_path"
            )
        
        if "output_path" not in args:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: output_path"
            )
        
        simplify_tolerance = args.get("simplify_tolerance", 1.0)
        
        # Load raster results
        try:
            with rasterio.open(args["input_path"]) as src:
                raster = src.read(1)  # Assume single band
                transform = src.transform
                crs = src.crs
        except Exception as e:
            raise McpError(
                ErrorCode.InternalError,
                f"Failed to load raster results: {str(e)}"
            )
        
        # Vectorize results
        try:
            # Use skeletonize to get centerlines
            from skimage.morphology import skeletonize
            skeleton = skeletonize(raster > 0)
            
            # Convert to vector lines
            lines = []
            
            # This is a simplified approach - in practice, we would use
            # more sophisticated methods to extract connected lines
            
            # For MVP, we'll use a simple approach to extract lines
            # by scanning the skeleton image
            
            # Find all skeleton pixels
            skeleton_pixels = np.column_stack(np.where(skeleton > 0))
            
            # Group pixels into lines
            # This is a very simplified approach
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
                            line = line.simplify(simplify_tolerance)
                            
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
                    line = line.simplify(simplify_tolerance)
                    
                    # Add to lines
                    lines.append(line)
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {"geometry": lines},
                crs=crs
            )
            
            # Save to file
            gdf.to_file(args["output_path"])
        except Exception as e:
            raise McpError(
                ErrorCode.InternalError,
                f"Failed to vectorize results: {str(e)}"
            )
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Vectorization completed. Results saved to {args['output_path']}"
                }
            ]
        }
    
    def _load_default_model(self):
        """Load the default model."""
        # For MVP, we'll create a new model instance
        # In practice, we would load pre-trained weights
        self.model = EnsembleModel()
        self.model.eval()
    
    def _get_model_info(self):
        """Get information about the current model."""
        if self.model is None:
            return {
                "status": "not_loaded",
                "message": "No model is currently loaded"
            }
        
        return {
            "status": "loaded",
            "type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
    
    def run(self):
        """Run the MCP server."""
        transport = StdioServerTransport()
        self.server.connect(transport)
        print("DrainageAI MCP server running on stdio", file=sys.stderr)


if __name__ == "__main__":
    server = DrainageServer()
    server.run()
