import momepy as mm
import geopandas as gpd
import statistics
from tqdm import tqdm

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181204/blg.shp")
streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181204/str.shp")

tessellation = mm.tessellation(buildings)

buildings, tessellation, blocks = mm.blocks(tessellation, streets, buildings, 'bID', 'uID')

buildings, tessellation = mm.get_network_id(buildings, streets, tessellation, 'uID', 'nID')

# from here
block_ids = buildings['bID'].tolist()

unique_block_ids = list(set(block_ids))

cells = {}
areas = {}

for id in unique_block_ids:
    unique_IDs = len(set(buildings.loc[buildings['bID'] == id]['uID'].tolist()))
    cells[id] = unique_IDs
    areas[id] = blocks.loc[blocks['bID'] == id].iloc[0]['geometry'].area

for index, row in tqdm(buildings.iterrows(), total=buildings.shape[0]):
    buildings.loc[index, 'gran'] = 10000 * cells[row["bID"]] / areas[row["bID"]]

buildings.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181204/blg2.shp")
