import momepy as mm
import geopandas as gpd
from timeit import default_timer as timer

start = timer()

buildings = 'files/mm_buildings.shp'
streets = 'files/mm_streets.shp'
cells = 'files/mm_g_tesselation.shp'
blocks = 'files/mm_g_blocks.shp'
edges = 'files/mm_g_edges.shp'

# load
print('Loading file', buildings)
blg = gpd.read_file(buildings)  # load file into geopandas
print('Shapefile loaded.')

print('Generating unique ID for buildings.')
mm.unique_id(blg)
blg.to_file(buildings)

tess_start = timer()

print('Generating tessellation.')
mm.tessellation(blg, cells)

tess_end = timer()
print('Tesselation finished in', tess_end - tess_start)

print('Loading file', streets)
str = gpd.read_file(streets)  # load file into geopandas
print('Shapefile loaded.')

print('Loading file', buildings)
blg = gpd.read_file(buildings)  # load file into geopandas
print('Shapefile loaded.')

print('Loading file', cells)
tess = gpd.read_file(cells)
print('Shapefile loaded.')

print('Fixing street network...')
fixed_streets = mm.snap_street_network_edge(str, blg, tess, 20, 70)

blo_start = timer()

print('Generating blocks.')
mm.blocks(tess, fixed_streets, blg, id_name='bID', unique_id='uID', cells_to=cells, buildings_to=buildings, blocks_to=blocks)

blo_end = timer()
print('Blocks finished in', blo_end - blo_start)

print('Loading file', cells)
tess = gpd.read_file(cells)
print('Shapefile loaded.')

print('Loading file', buildings)
blg = gpd.read_file(buildings)  # load file into geopandas
print('Shapefile loaded.')

print('Generating unique ID for street network.')
mm.unique_id(str, id_name='nID')
str.to_file(streets)

edg_start = timer()
print('Generating street edges.')
mm.street_edges(blg, str, tess, street_name_column='MKN_1', unique_id_column='uID', block_id_column='bID', network_id_column='nID',
                tesselation_to=cells, buildings_to=buildings, save_to=edges)
edg_end = timer()
print('Street edges finished in', edg_end - edg_start)
end = timer()
print('Script finished in', end - start)
