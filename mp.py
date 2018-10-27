import momepy as mm
import geopandas as gpd
# import pandas as pd
# import pysal
# from tqdm import tqdm  # progress bar
#
# cells = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7_voro_single2.shp"
buildings = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"
objects = gpd.read_file(buildings)
objects.columns
mm.moran_i_local(buildings, 'pdbFlA', column_name='moran')
#
# objects.columns
# look_for = gpd.read_file(buildings)
# look_for.columns
# look_for = look_for[['uID', 'pdbAre']]
# look_for
# objects = objects.merge(look_for, on='uID')
# objects.columns

voronoi10 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_10.shp"
voronoi15 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_15.shp"
voronoi20 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_20.shp"
voronoi25 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_25.shp"
voronoi30 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_30.shp"
voronoi40 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_40.shp"
voronoi50 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_50.shp"
voronoi60 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_60.shp"
voronoi70 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_70.shp"
voronoi80 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_80.shp"
voronoi90 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_90.shp"
voronoi100 = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/voronoi_100.shp"
plots = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Plots/Zurich_cadastre.shp"
buildings = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Buildings/Buildings_clean_id.shp"

# buildings = gpd.read_file(buildings)
#
# buildings_clean = buildings[['uID', 'blgarea']]
# buildings_merged = buildings_clean.groupby(['uID']).sum()
#
# test = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/test/voronoi_10.shp"
# tests = [test]
files = [voronoi10, voronoi20, voronoi15, voronoi25, voronoi30, voronoi40, voronoi50, voronoi60, voronoi70, voronoi80, voronoi90, voronoi100, plots]
# index = 1


for f in files:
    print('Loading file', f)
    objects = gpd.read_file(f)  # load file into geopandas
    print('Shapefile loaded.')

    mm.orientation(objects, column_name='ptbOri')

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(f)
    print('File saved.')

# mm.unique_id(plots, clear=True)
# mm.gethead(path)
tessel = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Cells/prg_tesselation50_IDs.shp'
bl = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Buildings/prg_buildings_IDs.shp'
mm.unique_id('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Street Network/prg_street_network.shp', id_name='nID')
blg = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Buildings/prg_buildings_values_bID.shp')
tess = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Cells/prg_tesselation50_values_bID.shp')
str = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Street Network/prg_street_network.shp')
save_path = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Street Edges/prg_edges_IDs.shp'
mm.street_edges(blg, str, tess, 'MKN_1', 'uID', 'bID', 'nID', tessel, bl, save_path)
#
# #
# objects = gpd.read_file(path)
# del objects['vicFre']
# objects
# objects.to_file(path)

cells = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/tess.shp')
streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/str.shp')
buildings = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/blg.shp')
cells_to = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/tessPY.shp'
buildings_to = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/blgPY.shp'
blocks_to = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/blocksPY.shp'

mm.blocks(cells, streets, buildings, 'bID', 'uID', cells_to, buildings_to, blocks_to)

from timeit import default_timer as timer
start = timer()
mm.tessellation(buildings, '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/p7/tess_enlarged2.shp')
end = timer()
print(end - start)


import momepy as mm
import geopandas as gpd
from timeit import default_timer as timer

start = timer()

buildings = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/blg.shp'
streets = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Prague/Street Network/prg_street_network.shp'
cells = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/tess_enlarged2.shp'
blocks = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/mm_g_blocks_tod.shp'
edges = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/mm_g_edges_tod.shp'
# p7 = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7-6.shp"

# load
print('Loading file', buildings)
blg = gpd.read_file(buildings)  # load file into geopandas
print('Shapefile loaded.')

print('Loading file', streets)
str = gpd.read_file(streets)  # load file into geopandas
print('Shapefile loaded.')

print('Loading file', cells)
tess = gpd.read_file(cells)

blo_start = timer()

print('Generating blocks.')
mm.blocks(tess, str, blg, id_name='bID', unique_id='uID', cells_to=cells, buildings_to=buildings, blocks_to=blocks)

blo_end = timer()
print('Blocks finished in', blo_end - blo_start)

print('Loading file', cells)
tess = gpd.read_file(cells)

edg_start = timer()
print('Generating street edges.')
mm.street_edges(blg, str, tess, street_name_column='MKN_1', unique_id_column='uID', block_id_column='bID', network_id_column='nID',
                tesselation_to=cells, buildings_to=buildings, save_to=edges)
edg_end = timer()
print('Edges finished in', edg_end - edg_start)
