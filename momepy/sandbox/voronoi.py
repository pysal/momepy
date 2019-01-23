import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # progress bar
import math
from rtree import index
from osgeo import ogr
from shapely.wkt import loads
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import *
import shapely.ops
import osmnx as ox
import operator

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190122/blg.shp")

unique_id = 'uID'
cut_buffer = 10


bounds = buildings['geometry'].bounds
centre_x = -(bounds['maxx'].max() + bounds['minx'].min()) / 2
centre_y = -(bounds['maxy'].max() + bounds['miny'].min()) / 2
objects = buildings.copy()
objects['geometry'] = objects['geometry'].translate(xoff=centre_x, yoff=centre_y)

# objects = buildings.to_crs(reprojected_crs)

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
obj_simple = objects.copy()

# generate built_up area around buildings to resolve the edge
print('Done in', timer() - start, 'seconds')
start = timer()
print('Preparing buffer zone for edge resolving (buffering)...')
obj_simple['geometry'] = obj_simple.buffer(cut_buffer)
print('Done in', timer() - start, 'seconds')
start = timer()
print('Preparing buffer zone for edge resolving (dissolving)...')
obj_simple['diss'] = 0
built_up = obj_simple.dissolve(by='diss')
print('Done in', timer() - start, 'seconds')
start = timer()
print('Preparing buffer zone for edge resolving (convex hull)...')
hull = built_up.convex_hull.buffer(cut_buffer)
print('Done in', timer() - start, 'seconds')

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

print('Done in', timer() - start, 'seconds')
start = timer()
print('Converting multipart geometry to singlepart...')
objects = multi2single(objects)

print('Done in', timer() - start, 'seconds')
start = timer()
print('Generating input point array...')

list_points = []
for idx, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    poly_ext = row['geometry'].boundary
    if poly_ext is not None:
        if poly_ext.type is 'MultiLineString':
            for line in poly_ext:
                point_coords = line.coords
                row_array = np.array(point_coords).tolist()
                for i in range(len(row_array)):
                    list_points.append(row_array[i])
        elif poly_ext.type is 'LineString':
            point_coords = poly_ext.coords
            row_array = np.array(point_coords).tolist()
            for i in range(len(row_array)):
                list_points.append(row_array[i])
        else:
            raise Exception('Boundary type is {}'.format(poly_ext.type))

# add hull
point_coords = hull[0].boundary.coords
row_array = np.array(point_coords).tolist()
for i in range(len(row_array)):
    list_points.append(row_array[i])

voronoi_points = np.array(list_points)

# make voronoi diagram
print('Done in', timer() - start, 'seconds')
start = timer()
print('Generating Voronoi diagram...')
voronoi_diagram = Voronoi(voronoi_points)
# generate lines from scipy voronoi output
print('Done in', timer() - start, 'seconds')
start = timer()
print('Generating Voronoi ridges...')
lines = [LineString(voronoi_diagram.vertices[line]) for line in voronoi_diagram.ridge_vertices if -1 not in line]

# generate dataframe with polygons clipped by built_up
print('Done in', timer() - start, 'seconds')
start = timer()
print('Generating Voronoi geometry...')
result = pd.DataFrame({'geometry':
                      [poly for poly in shapely.ops.polygonize(lines)]})

# generate geoDataFrame of Voronoi polygons
print('Done in', timer() - start, 'seconds')
start = timer()
print('Generating GeoDataFrame of Voronoi polygons...')
voronoi_polygons = gpd.GeoDataFrame(result, geometry='geometry')
voronoi_polygons = voronoi_polygons.loc[voronoi_polygons['geometry'].length < 1000000]
# set crs
voronoi_polygons.crs = objects.crs
print('Done in', timer() - start, 'seconds')

start = timer()
print('Generating MultiPoint geometry...')

def pointize(geom):
    multipoint = []
    if geom.boundary.type is 'MultiLineString':
        for line in geom.boundary:
            arr = line.coords.xy
            for p in range(len(arr[0])):
                point = (arr[0][p], arr[1][p])
                multipoint.append(point)
    elif geom.boundary.type is 'LineString':
        arr = geom.boundary.coords.xy
        for p in range(len(arr[0])):
            point = (arr[0][p], arr[1][p])
            multipoint.append(point)
    else:
        raise Exception('Boundary type is {}'.format(geom.boundary.type))
    new = MultiPoint(list(set(multipoint)))
    return new

objects_none = objects[objects['geometry'].notnull()]
objects_none['geometry'] = objects_none['geometry'].progress_map(pointize)
print('Done in', timer() - start, 'seconds')
start = timer()
print('Spatial join of MultiPoint geometry and Voronoi polygons...')
# spatial join
objects_none = objects_none.dropna(subset=['geometry'])
voronoi_with_id = gpd.sjoin(voronoi_polygons, objects_none, how='left')
voronoi_with_id.crs = objects.crs

# resolve thise cells which were not joined spatially (again, due to unprecision caused by scipy Voronoi function)
# select those unjoined
unjoined = voronoi_with_id[voronoi_with_id[unique_id].isnull()]
print('Done in', timer() - start, 'seconds')
if len(unjoined.index) > 0:
    start = timer()
    print('Fixing unjoined geometry:', len(unjoined.index), 'problems...')
    # for each polygon, find neighbours, measure boundary and set uID to the most neighbouring one
    print(' Building R-tree...')
    join_index = voronoi_with_id.sindex
    print(' Done in', timer() - start, 'seconds')
    start = timer()
    for idx, row in tqdm(unjoined.iterrows(), total=unjoined.shape[0]):
        neighbors = list(join_index.intersection(row.geometry.bounds))  # find neigbours
        neighbors_ids = []
        for n in neighbors:
            neighbors_ids.append(voronoi_with_id.iloc[n][unique_id])
        neighbors_ids = [x for x in neighbors_ids if str(x) != 'nan']  # remove polygon itself

        global boundaries
        boundaries = {}
        for i in neighbors_ids:
            subset = voronoi_with_id.loc[voronoi_with_id[unique_id] == i]['geometry']
            l = 0
            for s in subset:
                l = l + row.geometry.intersection(s).length
            boundaries[i] = l

        voronoi_with_id.loc[idx, unique_id] = max(boundaries.items(), key=operator.itemgetter(1))[0]

    unjoined2 = voronoi_with_id[voronoi_with_id[unique_id].isnull()]
    if len(unjoined2.index) is not 0:
        raise Exception('Some geometry remained unfinxed: {} problems'.format(len(unjoined2.index)))
    # dissolve polygons by unique_id
    print('Done in', timer() - start, 'seconds')
start = timer()
print('Dissolving Voronoi polygons...')
voronoi_with_id['geometry'] = voronoi_with_id.buffer(0)
voronoi_plots = voronoi_with_id.dissolve(by=unique_id)
voronoi_plots[unique_id] = voronoi_plots.index.astype('float')  # save unique id to column from index

# cut infinity of voronoi by set buffer (thanks for script to Geoff Boeing)
print('Done in', timer() - start, 'seconds')
start = timer()
print('Preparing buffer zone for edge resolving (quadrat cut)...')
geometry = built_up['geometry'].iloc[0].boundary
# quadrat_width is in the units the geometry is in
geometry_cut = ox.quadrat_cut_geometry(geometry, quadrat_width=100)

# build the r-tree index
print('Done in', timer() - start, 'seconds')
start = timer()
print('Building R-tree...')
sindex = voronoi_plots.sindex
# find the points that intersect with each subpolygon and add them to points_within_geometry
to_cut = pd.DataFrame()
for poly in geometry_cut:
    # find approximate matches with r-tree, then precise matches from those approximate ones
    possible_matches_index = list(sindex.intersection(poly.bounds))
    possible_matches = voronoi_plots.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(poly)]
    to_cut = to_cut.append(precise_matches)

# delete duplicates
to_cut = to_cut.drop_duplicates(subset=[unique_id])
subselection = list(to_cut.index)
print('Done in', timer() - start, 'seconds')
start = timer()
print('Cutting...')
for idx, row in tqdm(voronoi_plots.loc[subselection].iterrows(), total=voronoi_plots.loc[subselection].shape[0]):
    intersection = row.geometry.intersection(built_up['geometry'].iloc[0])
    if intersection.type == 'MultiPolygon':
        areas = {}
        for p in range(len(intersection)):
            area = intersection[p].area
            areas[p] = area
        maximal = max(areas.items(), key=operator.itemgetter(1))[0]
        voronoi_plots.loc[idx, 'geometry'] = intersection[maximal]
    elif intersection.type == 'GeometryCollection':
        for geom in list(intersection.geoms):
                if geom.type != 'Polygon':
                    pass
                else:
                    voronoi_plots.loc[idx, 'geometry'] = geom
    else:
        voronoi_plots.loc[idx, 'geometry'] = intersection

voronoi_plots = voronoi_plots.drop(['index_right'], axis=1)

voronoi_plots['geometry'] = voronoi_plots['geometry'].translate(xoff=-centre_x, yoff=-centre_y)

morphological_tessellation = voronoi_plots

ids_original = list(buildings[unique_id])
ids_generated = list(morphological_tessellation[unique_id])
if len(ids_original) != len(ids_generated):
    import warnings
    diff = set(ids_original).difference(ids_generated)
    warnings.warn("Tessellation does not fully match buildings. {len} element(s) collapsed "
                  "during generation - unique_id: {i}".format(len=len(diff), i=diff))

uids = morphological_tessellation[morphological_tessellation.geometry.type == 'MultiPolygon'][unique_id]
if len(uids) is not 0:
    import warnings
    warnings.warn('Tessellation contains MultiPolygon elements. Initial objects should be edited. '
                  'unique_id of affected elements: {}'.format(list(uids)))

print('Done in', timer() - start, 'seconds')
print('Done. Tessellation finished in', timer() - start_, 'seconds.')
# return morphological_tessellation
