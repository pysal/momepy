import momepy as mm
import geopandas as gpd

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"

mm.building_courtyard_index(path)
mm.gethead(path)

#
objects = gpd.read_file(path)
del objects['coix']
objects.head
objects.to_file(path)
