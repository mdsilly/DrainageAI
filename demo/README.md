# DrainageAI Demo Materials

This directory contains materials for demonstrating the DrainageAI system to stakeholders, including presentation slides, scripts, and diagrams.

## Contents

- `presentation_slides.md`: Markdown-based presentation slides
- `presentation_script.md`: Detailed script for presenting the DrainageAI system
- `workflow_diagram.md`: Mermaid diagram showing the complete DrainageAI workflow
- `fixmatch_diagram.md`: Mermaid diagram explaining the FixMatch semi-supervised learning approach

## Using These Materials

### Presentation Preparation

1. Convert the Mermaid diagrams to images:
   - Use an online Mermaid renderer like [Mermaid Live Editor](https://mermaid.live/)
   - Save the rendered diagrams as PNG or SVG files
   - Place them in this directory with the same names as referenced in the slides

2. Create a presentation from the slides:
   - Option 1: Use a Markdown presentation tool like [Marp](https://marp.app/) or [Reveal.js](https://revealjs.com/) to convert `presentation_slides.md` to a presentation
   - Option 2: Manually create slides in PowerPoint or Google Slides using the content from `presentation_slides.md`

3. Review the presentation script:
   - Familiarize yourself with the talking points in `presentation_script.md`
   - Customize as needed for your specific audience

### Demo Preparation

1. Ensure you have sample data ready:
   - Multispectral imagery (Sentinel-2 or Landsat recommended)
   - Pre-trained models (if available)
   - Sample output files (for backup)

2. Test the complete workflow before the presentation:
   ```bash
   python examples/super_mvp_workflow.py --imagery sample_data/sentinel2_image.tif --output demo_results
   ```

3. Prepare GIS software:
   - Install ArcGIS Pro or QGIS
   - Set up the DrainageAI toolbox or MCP plugin
   - Test loading and visualizing the results

## Customization

Feel free to customize these materials for your specific audience:

- For technical audiences: Focus on the model architecture, spectral indices, and technical implementation
- For agricultural stakeholders: Emphasize practical benefits, accuracy, and ease of use
- For potential partners: Highlight the roadmap and opportunities for collaboration

## Additional Resources

- Full documentation: See the main `README.md` and `SUPER_MVP_README.md` files
- Code examples: See the `examples` directory
- Tutorials: See the `notebooks` directory
