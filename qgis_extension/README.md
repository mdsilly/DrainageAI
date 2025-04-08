# DrainageAI QGIS Extension

This extension integrates DrainageAI with QGIS through the Model Context Protocol (MCP), allowing you to detect drainage pipes directly from the QGIS interface.

## Installation

### Prerequisites

- QGIS Desktop 3.22+
- Python 3.9+
- DrainageAI package installed
- MCP Plugin for QGIS installed

### Setup

1. Install the MCP Plugin for QGIS:
   - Open QGIS
   - Go to Plugins -> Manage and Install Plugins
   - Search for "MCP" or "Model Context Protocol"
   - Install the plugin

2. Configure the DrainageAI MCP server:
   - Copy the `claude_desktop_config.json` file to the appropriate location:
     - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\qgis_mcp`
     - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/qgis_mcp`
     - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/qgis_mcp`

3. Restart QGIS

## Usage

1. Start the DrainageAI MCP server:
   - Open QGIS
   - Go to Plugins -> MCP -> QGIS MCP
   - Click "Start Server"

2. Use the DrainageAI tools:
   - The DrainageAI tools will be available in the MCP toolbar
   - Click on the tool you want to use
   - Follow the prompts to provide input data and parameters
   - The results will be loaded as new layers in QGIS

## Available Tools

- **Detect Drainage**: Detect drainage pipes in satellite imagery
- **Load Model**: Load a DrainageAI model
- **Vectorize Results**: Convert raster detection results to vector format

## Troubleshooting

If you encounter issues with the MCP server connection:

1. Check that the DrainageAI package is installed correctly
2. Verify that the MCP plugin is installed and enabled in QGIS
3. Ensure that the configuration file is in the correct location
4. Check the QGIS log for error messages (View -> Panels -> Log Messages)

## Development

To modify or extend the DrainageAI QGIS extension:

1. Edit the `drainage_server.py` file to add new tools or resources
2. Update the configuration file as needed
3. Restart the MCP server to apply changes
