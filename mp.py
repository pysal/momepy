import momepy as mm
import geopandas as gpd

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Prague/p7_voro_single.shp"

mm.cell_frequency(path)
mm.gethead(path)


#
objects = gpd.read_file(path)
del objects['vicFre']
objects
objects.to_file(path)
