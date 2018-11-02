import momepy as mm
import geopandas as gpd
from timeit import default_timer as timer


buildings = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/blg5.shp'
# buildings = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/blg.shp'
streets = 'files/vinohrady_streets.shp'
cells = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Vinohrady/tess_time.shp'
# cells = '/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/tess_time.shp'
blocks = 'files/vinohrady_g_blocks.shp'
edges = 'files/vinohrady_g_edges.shp'

# load
print('Loading file', buildings)
blg = gpd.read_file(buildings)  # load file into geopandas
print('Shapefile loaded.')

tess_start = timer()

print('Generating tessellation.')
mm.tessellation(blg, cells)

tess_end = timer()
print('Tesselation finished in', tess_end - tess_start)
