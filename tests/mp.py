import momepy as mm
import geopandas as gpd
from timeit import default_timer as timer

start = timer()

buildings_path = 'files/prg_buildings.shp'
tessellation_path = 'files/mm_g_tesselation.shp'
streets_path = 'files/prg_street_network.shp'
blocks_path = 'files/mm_g_blocks.shp'
edges_path = 'files/mm_g_edges.shp'

buildings_path_bid = 'files/mm_buildings_bID.shp'
tessellation_path_bid = 'files/mm_g_tesselation_bID.shp'

buildings_path_all = 'files/mm_buildings_all.shp'
tessellation_path_all = 'files/mm_g_tesselation_all.shp'


tess_start = timer()
print('Generating tessellation.')

buildings = gpd.read_file(buildings_path)

mm.tessellation(buildings, tessellation_path)

tess_end = timer()
print('Tesselation finished in', tess_end - tess_start)

blo_start = timer()
print('Generating blocks.')

streets = gpd.read_file(streets_path)
tessellation = gpd.read_file(tessellation_path)
fixed_streets = mm.snap_street_network_edge(streets, buildings, tessellation, 20, 70)
mm.blocks(tessellation, fixed_streets, buildings, 'bID', 'uID', tessellation_path_bid, buildings_path_bid, blocks_path)

blo_end = timer()
print('Blocks finished in', blo_end - blo_start)
edg_start = timer()
print('Generating street edges.')

buildings_bid = gpd.read_file(buildings_path_bid)
tessellation_bid = gpd.read_file(tessellation_path_bid)

mm.street_edges(buildings_bid, streets, tessellation_bid, 'MKN_1', 'uID', 'bID', 'nID', tessellation_path_all, buildings_path_all, edges_path)
edg_end = timer()
print('Street edges finished in', edg_end - edg_start)
end = timer()
print('Script finished in', end - start)
