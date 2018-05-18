import momepy as mm
import geopandas as gpd

path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"

mm.b(path)
mm.gethead(path)



objects = gpd.read_file(path)
del objects['volFal2']
objects.columns
objects.to_file(path)
