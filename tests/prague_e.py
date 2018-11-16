import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # progress bar
from osgeo import ogr
from shapely.wkt import loads
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import MultiPoint, Point, Polygon, LineString, MultiPolygon
import shapely.ops
import operator
import osmnx as ox


buildings = gpd.read_file('files/prg_buildings.shp')
save_tessellation = 'files/prg_tess.shp'
unique_id = 'uID'
cut_buffer = 50

objects = buildings.copy()

from timeit import default_timer as timer
tqdm.pandas()
start = timer()
start_ = timer()
# buffer geometry to resolve shared walls
print('Bufferring geometry...')
objects['geometry'] = objects.geometry.progress_apply(lambda g: g.buffer(-0.5, cap_style=2, join_style=2))
print('Done in', timer() - start, 'seconds')
start = timer()
print('Simplifying geometry...')
# simplify geometry before Voronoi
objects['geometry'] = objects.simplify(0.25, preserve_topology=True)


# densify geometry before Voronoi tesselation
def densify(geom):
    wkt = geom.wkt  # shapely Polygon to wkt
    geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
    geom.Segmentize(2)  # densify geometry by 2 metres
    wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
    new = loads(wkt2)  # wkt to shapely Polygon
    return new

start = timer()
print('Densifying geometry...')
objects['geometry'] = objects['geometry'].progress_map(densify)


# resolve multipart polygons, singlepart are needed
def multi2single(gpdf):
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T] * len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly

objects = multi2single(objects)
objects.to_file('files/BEFORE_POINTS.shp')
#
# def pointize(geom):
#     multipoint = []
#     if geom.boundary.type is 'MultiLineString':
#         for line in geom.boundary:
#             arr = line.coords.xy
#             for p in range(len(arr[0])):
#                 point = (arr[0][p], arr[1][p])
#                 multipoint.append(point)
#     elif geom.boundary.type is 'LineString':
#         arr = geom.boundary.coords.xy
#         for p in range(len(arr[0])):
#             point = (arr[0][p], arr[1][p])
#             multipoint.append(point)
#     else:
#         raise Exception('Boundary type is {}'.format(geom.boundary.type))
#     new = MultiPoint(list(set(multipoint)))
#     return new
#
# objects['geometry'] = objects['geometry'].progress_map(pointize)
#
# # spatial join
# objects = objects.dropna()



# PROBLEM - MULTIPOINT OBJECTS CONTAINS DIFFERENT GEOMETRY TYPES
