import momepy as mm
import geopandas as gpd
from tqdm import tqdm
import pandas as pd

# S - building - dimension
# buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy190122/prg_g_buildings_bID.shp")
#
# buildings['area'] = mm.area(buildings)
# buildings['fl_area'] = mm.floor_area(buildings, 'area', 'pdbHei', area_calculated=True)
# buildings['volume'] = mm.volume(buildings, 'area', 'pdbHei', area_calculated=True)
# buildings['perimet'] = mm.perimeter(buildings)
# buildings['courtyard'] = mm.courtyard_area(buildings, 'area', area_calculated=True)
#
#
# def bboxwl(objects):
#     from shapely.geometry import Point
#     # define empty list for results
#     results_list_w = []
#     results_list_l = []
#     area = []
#     perimeter = []
#     print('Calculating bbox dimensions...')
#
#     for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
#         bbox = row['geometry'].minimum_rotated_rectangle
#         area.append(bbox.area)
#         perimeter.append(bbox.length)
#         a = Point(bbox.exterior.coords[0]).distance(Point(bbox.exterior.coords[1]))
#         b = Point(bbox.exterior.coords[1]).distance(Point(bbox.exterior.coords[2]))
#         if a > b:
#             results_list_l.append(a)
#             results_list_w.append(b)
#         else:
#             results_list_l.append(b)
#             results_list_w.append(a)
#
#     widths = pd.Series(results_list_w)
#     lenghts = pd.Series(results_list_l)
#     areas = pd.Series(area)
#     perimeters = pd.Series(perimeter)
#     return widths, lenghts, areas, perimeters
#
# buildings['bboxwid'], buildings['bboxlen'], buildings['bboxarea'], buildings['bboxper'] = bboxwl(buildings)
#
#
# def circle_dims(objects):
#     results_list = []
#     print('Calculating circle...')
#     for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
#         results_list.append((mm.make_circle(list(row['geometry'].convex_hull.exterior.coords)))[2])
#     series = pd.Series(results_list)
#
#     print('Circle calculated.')
#     return series
#
# buildings['circle_r'] = circle_dims(buildings)
#
#
# def chull(objects):
#     results_list_a = []
#     results_list_p = []
#     print('Calculating bbox dimensions...')
#
#     for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
#         chull = row['geometry'].convex_hull
#         results_list_a.append(chull.area)
#         results_list_p.append(chull.length)
#     areas = pd.Series(results_list_a)
#     peris = pd.Series(results_list_p)
#     return areas, peris
#
# buildings['ch_area'], buildings['ch_perim'] = chull(buildings)
#
# buildings.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_dimension.shp")


# S - building - shape
# buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_dimension.shp")
#
# buildings['formf'] = mm.form_factor(buildings, 'area', 'volume')
# buildings['fractal'] = mm.fractal_dimension(buildings, 'area', 'perimet')
# buildings['vfr'] = mm.volume_facade_ratio(buildings, 'volume', 'perimet', 'pdbHei')
# buildings['circom'] = mm.circular_compactness(buildings, 'area')
# buildings['squcom'] = mm.square_compactness(buildings, 'area', 'perimet')
# buildings['convex'] = mm.convexeity(buildings, 'area')
# buildings['corners'] = mm.corners(buildings)
# buildings['lal'] = mm.longest_axis_length(buildings)
# buildings['shape'] = mm.shape_index(buildings, 'area', 'lal')
# buildings['squ'] = mm.squareness(buildings)
# buildings['eri'] = mm.equivalent_rectangular_index(buildings, 'area', 'perimet')
# buildings['elo'] = mm.elongation(buildings)
# buildings['ccd'] = mm.centroid_corners(buildings)
#
# buildings.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_shape.shp")


# S - building - distribution
# buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_shape.shp")
# streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Clean data (wID)/Street Network/prg_street_network.shp')
# tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dist.shp")
#
# buildings['orient'] = mm.orientation(buildings)
# buildings['s_align'] = mm.street_alignment(buildings, streets, 'orient', 'nID')
# buildings['c_align'] = mm.cell_alignment(buildings, tessellation, 'orient', 'orient', 'uID')
#
# buildings.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_dist.shp")


# PART IS MISSING, CHECK COLUNMS

# L - buildings
buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/m_building_dist.shp")

buildings['bl_court'] = mm.courtyards(buildings, 'bID')

buildings.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/l_building.shp")
