import time

import geopandas as gpd
import momepy

# Load data
path = momepy.datasets.get_path("bubenec")
buildings = gpd.read_file(path, layer="buildings")
streets = gpd.read_file(path, layer="streets")

# Generate enclosures
enclosures = momepy.enclosures(streets)

# Use streets as inner_barriers for testing
inner_barriers = streets

# Time the enclosed_tessellation without inner_barriers
start = time.time()
tess_no_inner = momepy.enclosed_tessellation(buildings, enclosures)
end = time.time()
time_no_inner = end - start

# Time the enclosed_tessellation with inner_barriers
start = time.time()
tess_with_inner = momepy.enclosed_tessellation(
    buildings, enclosures, inner_barriers=inner_barriers
)
end = time.time()
time_with_inner = end - start

print(f"Time without inner_barriers: {time_no_inner} seconds")
print(f"Time with inner_barriers: {time_with_inner} seconds")
print(f"Number of tessellation cells without: {len(tess_no_inner)}")
print(f"Number of tessellation cells with: {len(tess_with_inner)}")
