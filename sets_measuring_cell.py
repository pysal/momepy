import momepy as mm
import geopandas as gpd
from tqdm import tqdm
import pandas as pd

# S - building - dimension
# tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy190122/prg_g_tessellation_bID.shp")
#
# tessellation['area'] = mm.area(tessellation)
# tessellation['lal'] = mm.longest_axis_length(tessellation)
# tessellation['perimet'] = mm.perimeter(tessellation)
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
# tessellation['bboxwid'], tessellation['bboxlen'], tessellation['bboxarea'], tessellation['bboxper'] = bboxwl(tessellation)
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
# tessellation['circle_r'] = circle_dims(tessellation)
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
# tessellation['ch_area'], tessellation['ch_perim'] = chull(tessellation)
#
# tessellation.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dimension.shp")
#
#
# # # S - tessellation - shape
# # tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dimension.shp")
#
# tessellation['fractal'] = mm.fractal_dimension(tessellation, 'area', 'perimet')
# tessellation['circom'] = mm.circular_compactness(tessellation, 'area')
# tessellation['squcom'] = mm.square_compactness(tessellation, 'area', 'perimet')
# tessellation['convex'] = mm.convexeity(tessellation, 'area')
# tessellation['shape'] = mm.shape_index(tessellation, 'area', 'lal')
# tessellation['eri'] = mm.equivalent_rectangular_index(tessellation, 'area', 'perimet')
# tessellation['elo'] = mm.elongation(tessellation)
#
# tessellation.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_shape.shp")

# S - tessellation - distribution
# tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_shape.shp")
# streets = gpd.read_file('/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Clean data (wID)/Street Network/prg_street_network.shp')
#
# tessellation['orient'] = mm.orientation(tessellation)
# tessellation['s_align'] = mm.street_alignment(tessellation, streets, 'orient', 'nID')
#
# tessellation.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dist.shp")

# S - tessellation - distribution
tessellation = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_dist.shp")
buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_dist.shp")

tessellation['car'] = mm.covered_area_ratio(tessellation, buildings, 'area', 'area')
tessellation['far'] = mm.floor_area_ratio(tessellation, buildings, 'area', 'fl_area')

tessellation.to_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_tessellation_int.shp")
