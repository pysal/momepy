import geopandas as gpd
import momepy as mm
import pandas as pd

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190123/blg.shp")
streets = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190123/str.shp")

streets['widths_only'] = mm.street_profile(streets, buildings)
profile = mm.street_profile(streets, buildings, height_column='pdbHei')
streets['wid'], streets['hei'], streets['prof'] = profile
