# DrainageAI Super-MVP Demonstration Script

This script provides a step-by-step guide for demonstrating the DrainageAI Super-MVP to stakeholders.

## Preparation

Before the demonstration:

1. Ensure all dependencies are installed
2. Prepare sample multispectral imagery (Sentinel-2 or Landsat recommended)
3. Have ArcGIS Pro or QGIS installed and ready
4. Clone the repository and navigate to the project directory

## Introduction (2 minutes)

"Today I'm going to demonstrate our DrainageAI system, which uses artificial intelligence to detect subsurface drainage pipes in agricultural fields using satellite imagery. This is a super-minimal viable product that demonstrates the core functionality and workflow."

## Problem Statement (2 minutes)

"Subsurface drainage systems are critical infrastructure for agriculture, but their locations are often poorly documented. Detecting these systems is important for:
- Precision agriculture and field management
- Water quality monitoring and conservation
- Infrastructure planning and maintenance
- Environmental impact assessment

Traditional methods of mapping drainage systems are time-consuming and expensive, involving ground-penetrating radar or physical excavation."

## Solution Overview (3 minutes)

"DrainageAI addresses this challenge by using:
1. Multispectral satellite imagery, which is widely available
2. Spectral indices that highlight patterns related to soil moisture and vegetation health
3. Deep learning models trained to recognize the subtle signatures of drainage pipes
4. GIS integration for easy visualization and analysis

Our approach is non-invasive, scalable, and can work with publicly available satellite data."

## Technical Demonstration (10 minutes)

### Step 1: Calculate Spectral Indices

"First, we'll calculate spectral indices from our multispectral imagery. These indices highlight patterns that might not be visible in the original imagery."

```bash
python examples/super_mvp_workflow.py --imagery sample_data/sentinel2_image.tif --output demo_results --step indices
```

"We're calculating three key indices:
- NDVI (Normalized Difference Vegetation Index): Highlights vegetation patterns
- NDMI (Normalized Difference Moisture Index): Highlights soil moisture patterns
- MSAVI2 (Modified Soil Adjusted Vegetation Index): Provides better sensitivity to sparse vegetation"

[Show the resulting indices images in ArcGIS/QGIS]

### Step 2: Detect Drainage Pipes

"Now we'll use our CNN model to detect drainage pipes. This model has been trained to recognize the patterns associated with drainage systems using both the original imagery and the spectral indices we just calculated."

```bash
python examples/super_mvp_workflow.py --imagery sample_data/sentinel2_image.tif --output demo_results --step detect --model cnn
```

"The model produces a probability map where higher values indicate a greater likelihood of drainage pipes."

[Show the detection results in ArcGIS/QGIS]

### Step 3: Vectorize Results

"Finally, we'll convert the raster detection results to vector format, which is more useful for GIS analysis and integration with other data sources."

```bash
python examples/super_mvp_workflow.py --imagery sample_data/sentinel2_image.tif --output demo_results --step vectorize
```

"The vectorization process includes:
1. Thresholding the probability map
2. Skeletonizing to get centerlines
3. Converting to vector lines
4. Simplifying the geometry for cleaner results"

[Show the vectorized results in ArcGIS/QGIS]

### Step 4: GIS Integration

"One of the key features of DrainageAI is its integration with GIS software. We support both ArcGIS Pro and QGIS."

#### ArcGIS Demo

"In ArcGIS Pro, we've created a Python Toolbox that provides a user-friendly interface for our tools."

[Open ArcGIS Pro and demonstrate the toolbox]

"The toolbox includes tools for:
1. Calculating spectral indices
2. Detecting drainage pipes
3. Vectorizing results

This makes it easy for GIS analysts to incorporate DrainageAI into their existing workflows."

#### QGIS Demo (if time permits)

"We also support QGIS through the Model Context Protocol, which provides a similar set of tools."

[Open QGIS and demonstrate the MCP tools]

## Results and Validation (3 minutes)

"Let's look at the results of our detection compared to known drainage systems."

[Show a comparison between detected drainage lines and ground truth data, if available]

"While the super-MVP has some limitations in terms of accuracy, it demonstrates the potential of our approach. The full implementation will include:
- Ensemble models that combine multiple detection approaches
- More sophisticated post-processing
- Integration with additional data sources like LiDAR and SAR"

## Next Steps and Roadmap (2 minutes)

"Our next steps include:
1. Refining the models with more training data
2. Implementing the full ensemble architecture
3. Adding temporal analysis to leverage multi-date imagery
4. Developing more advanced post-processing techniques

We're also exploring partnerships with drainage contractors and agricultural service providers to validate and improve our system."

## Q&A (5-10 minutes)

"I'd be happy to answer any questions about the technology, our approach, or potential applications."

## Backup Demonstration

If the live demonstration encounters technical issues, have screenshots or videos prepared showing:
1. The original imagery
2. Calculated spectral indices
3. Detection results
4. Vectorized drainage lines
5. GIS integration

This ensures you can still effectively communicate the capabilities of the system even if technical issues arise.
