import geopandas as gpd
import pandas as pd
from shapely import wkb
import pyarrow.parquet as pq

# Load parquet (assumes it contains geometry in WKT or geometry column)
parq = pq.ParquetFile("france_eurocrops_2018_fiboa.parquet")

batch = parq.read_row_group(0)  # You can also use read_row_groups
df = batch.to_pandas().head(100)

#df = pd.read_parquet("your_file.parquet")
print(df.columns)
print(df['geometry'].iloc[0])

df['geometry'] = df['geometry'].apply(wkb.loads)

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs("EPSG:2154", inplace=True)

# Set CRS if not set
# gdf = gdf.set_crs("EPSG:4326")

# Save as GeoJSON
gdf.to_file("fieldBoundaries.geojson", driver="GeoJSON")