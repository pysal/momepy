import momepy as mm
import geopandas as gpd

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/buildings_uID.shp")
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/tessellation.shp")

buildings = mm.area(buildings, 'area')
tessellation = mm.area(tessellation, 't_area')
tessellation = mm.covered_area_ratio(tessellation, buildings, 'car', 't_area', 'area')

tessellation.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/tessellation.shp")
# blgs.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Botswana/buildings_uID.shp")
