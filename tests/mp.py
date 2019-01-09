import momepy as mm
import geopandas as gpd

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/buildings.shp")
blgs = mm.unique_id(buildings)

tessellation = mm.tessellation(blgs)

tessellation.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/tessellation.shp")
blgs.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/buildings_uID.shp")
