import geopandas as gpd

# Load the GeoJSON (replace with your path)
gdf = gpd.read_file("fieldBoundaries.geojson")

# Force CRS to EPSG:4326 if it's currently CRS84
# STEP 2: Reproject to EPSG:4326 for GEE
gdf = gdf.to_crs("EPSG:4326")

# Export to shapefile
gdf.to_file("fieldBoundaries.shp", driver="ESRI Shapefile")

# Zip the shapefile components
import zipfile
with zipfile.ZipFile("fieldBoundaries.zip", "w") as zipf:
    for ext in [".shp", ".shx", ".dbf", ".prj"]:
        zipf.write(f"fieldBoundaries{ext}")