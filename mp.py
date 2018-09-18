import momepy as mm
import geopandas as gpd
import pandas as pd
import pysal
# from tqdm import tqdm  # progress bar
#
# cells = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7_voro_single2.shp"
# buildings = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"
#
#
# mm.building_centroid_corners(buildings)
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

#
# #
# objects = gpd.read_file(path)
# del objects['vicFre']
# objects
# objects.to_file(path)
