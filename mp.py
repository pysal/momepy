import momepy as mm
import geopandas as gpd

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"

mm.building_convexeity(path)
mm.gethead(path)

#
objects = gpd.read_file(path)
del objects['coix']
objects.columns
objects.to_file(path)
