# DrainageAI Scripts

This directory contains utility scripts for the DrainageAI project.

## prepare_demo_imagery.py

This script helps prepare imagery for the DrainageAI demo by creating a smaller subset of a larger multispectral image. This is particularly useful when using Google Colab, which has upload size limitations.

### Usage

```bash
python prepare_demo_imagery.py --input large_image.tif --output demo_image.tif --size 1000
```

### Parameters

- `--input`: Path to the input imagery file (required)
- `--output`: Path to the output imagery file (required)
- `--size`: Size of the output image in pixels (width and height, default: 1000)
- `--offset-x`: X offset from the top-left corner (default: 0)
- `--offset-y`: Y offset from the top-left corner (default: 0)

### Examples

1. Create a 1000×1000 pixel subset from the top-left corner:
   ```bash
   python prepare_demo_imagery.py --input sentinel2_image.tif --output demo_image.tif --size 1000
   ```

2. Create a 500×500 pixel subset with an offset:
   ```bash
   python prepare_demo_imagery.py --input sentinel2_image.tif --output demo_image.tif --size 500 --offset-x 1000 --offset-y 1000
   ```

3. Process multiple images with different offsets to find areas with drainage pipes:
   ```bash
   python prepare_demo_imagery.py --input sentinel2_image.tif --output demo_image_1.tif --size 1000 --offset-x 0 --offset-y 0
   python prepare_demo_imagery.py --input sentinel2_image.tif --output demo_image_2.tif --size 1000 --offset-x 1000 --offset-y 0
   python prepare_demo_imagery.py --input sentinel2_image.tif --output demo_image_3.tif --size 1000 --offset-x 0 --offset-y 1000
   python prepare_demo_imagery.py --input sentinel2_image.tif --output demo_image_4.tif --size 1000 --offset-x 1000 --offset-y 1000
   ```

### Requirements

- rasterio

### Notes

- The script will automatically adjust the window size if it exceeds the image dimensions.
- The output image will have the same number of bands and data type as the input image.
- The script preserves the geospatial metadata, so the output image can be used in GIS software.
