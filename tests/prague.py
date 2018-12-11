import momepy as mm
import geopandas as gpd
from timeit import default_timer as timer
from tqdm import tqdm

start = timer()

buildings = gpd.read_file('files/prg_buildings.shp')
streets = gpd.read_file('files/prg_street_network.shp')
cells = gpd.read_file('files/mm_g_tesselation.shp')


fixed = mm.snap_street_network_edge(streets, buildings, cells, 20, 70)

buildings, cells, blocks = mm.blocks(cells, fixed, buildings, 'bID', 'uID')

granularity = mm.block_density(buildings, 'granula', blocks, 'bID', 'uID')

to_blocks = granularity[['bID', 'granula']]

cleaned = to_blocks.drop_duplicates(subset='bID')

for index, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
    value = cleaned[cleaned['bID'] == row['bID']].iloc[0]['granula']
    blocks.loc[index, 'granula'] = value


granularity.to_file('files/mm2_buildings.shp')
cells.to_file('files/mm2_g_tesselation.shp')
blocks.to_file('files/mm2_g_blocks.shp')
