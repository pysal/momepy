import momepy as mm
import geopandas as gpd

streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_street_shape.shp")
buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_shape.shp")

profile = mm.street_profile(streets, buildings, height_column='pdbHei')
streets['width'], streets['height'], streets['profile'] = profile

streets.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_street_shape.shp")
