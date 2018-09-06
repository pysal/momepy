import momepy as mm
import geopandas as gpd
import pysal
# from tqdm import tqdm  # progress bar

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

buildings = gpd.read_file(buildings)

buildings_clean = buildings[['uID', 'blgarea']]
buildings_merged = buildings_clean.groupby(['uID']).sum()

test = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/test/voronoi_10.shp"
tests = [test]
files = [voronoi10, voronoi20, voronoi15, voronoi25, voronoi30, voronoi40, voronoi50, voronoi60, voronoi70, voronoi80, voronoi90, voronoi100, plots]

objects = gpd.read_file(test)

Wrook = pysal.weights.Rook.from_dataframe(objects, silent_island_warning=True)  # weight matrix 'rook'
y = objects[['car']]
moran = pysal.Moran(y, Wrook)
moran.I
moran.EI
moran.p_norm

for f in files:
    print('Loading file', f)
    objects = gpd.read_file(f)  # load file into geopandas
    print('Shapefile loaded.')

    mm.moran_i_local(objects, source='car', column_name='car-moran')

    # save dataframe back to file
    print('Saving file.')
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
