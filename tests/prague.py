import momepy as mm
import geopandas as gpd

buildings = gpd.read_file('files/prg_buildings.shp')
streets = gpd.read_file('files/prg_street_network.shp')

tess = mm.tessellation(buildings)

street_snap = mm.snap_street_network_edge(streets, buildings, tess, 20, 70)
blg2, tess2, blocks = mm.blocks(tess, street_snap, buildings, 'bID', 'uID')

blg2.to_file('files/prg_g_buildings_bID.shp')
tess2.to_file('files/prg_g_tessellation_bID.shp')
blocks.to_file('files/prg_g_blocks_bID.shp')
