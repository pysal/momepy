import momepy as mm
import geopandas as gpd

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/181217/blg_mmp.shp")
buildings.drop(['column_nam', 'gran', 'column_n_1'], axis=1, inplace=True)
buildings.columns
buildings['area'] = mm.area(buildings)
buildings['cc'] = mm.centroid_corners(buildings)
buildings['circom'] = mm.circular_compactness(buildings, 'area')
buildings['conv'] = mm.convexeity(buildings, 'area')
buildings['court'] = mm.courtyards(buildings, 'bID')
buildings['freq'] = mm.frequency(buildings, buildings)
buildings['peri'] = mm.perimeter(buildings)
buildings['sharedfix2'] = mm.shared_walls_ratio(buildings, 'peri', 'uID')

buildings.plot(column='shared2', legend=True)
buildings.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190110/blg_mmp.shp")
