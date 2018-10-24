import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # progress bar
import math
from rtree import index
from osgeo import ogr
from shapely.wkt import loads
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon, LineString, MultiPolygon
import shapely.ops
import shapefile
import shutil


path = "/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/P7/p7-6.shp"
unique_id = 'uID'

print('Loading file.')
objects = gpd.read_file(path)  # load file into geopandas
print('Shapefile loaded.')

from timeit import default_timer as timer

start = timer()


# buffer geometry to resolve shared walls
print('Bufferring geometry...', timer() - start)
objects['geometry'] = objects.geometry.apply(lambda g: g.buffer(-0.5, cap_style=2, join_style=2))

print('Simplifying geometry...', timer() - start)
# simplify geometry before Voronoi
objects['geometry'] = objects.simplify(0.25, preserve_topology=True)
obj_simple = objects.copy()

# densify geometry before Voronoi tesselation
def densify(geom):
    wkt = geom.wkt  # shapely Polygon to wkt
    geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
    geom.Segmentize(2)  # densify geometry by 2 metres
    wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
    new = loads(wkt2)  # wkt to shapely Polygon
    return new

print('Densifying geometry...', timer() - start)
objects['geometry'] = objects['geometry'].map(densify)

# define new numpy.array
voronoi_points = np.empty([1, 2])

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

print('Converting multipart geometry to singlepart...', timer() - start)
objects = multi2single(objects)

print('Generating input point array...', timer() - start)
# fill array with all points from densified geometry
for idx, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    poly_ext = row['geometry'].exterior
    if poly_ext is not None:
        point_coords = poly_ext.coords
        row_array = np.array(point_coords)
        voronoi_points = np.concatenate((voronoi_points, row_array))
        # it might be faster to use python list.append(a) and then l = np.array(l)

# delete initial row of array to keep only points from geometry
voronoi_points = voronoi_points[1:]
# make voronoi diagram
print('Generating Voronoi diagram...', timer() - start)
voronoi_diagram = Voronoi(voronoi_points)
# generate lines from scipy voronoi output
print('Generating Voronoi ridges...', timer() - start)
lines = [LineString(voronoi_diagram.vertices[line]) for line in voronoi_diagram.ridge_vertices if -1 not in line]

# generate dataframe with polygons clipped by built_up
print('Generating Voronoi geometry...', timer() - start)
result = pd.DataFrame({'geometry':
                      [poly for poly in shapely.ops.polygonize(lines)]})

# generate geoDataFrame of Voronoi polygons
print('Generating GeoDataFrame of Voronoi polygons...', timer() - start)
voronoi_polygons = gpd.GeoDataFrame(result, geometry='geometry')

print('Saving to temporary file...', timer() - start)

# set crs
voronoi_polygons.crs = objects.crs
# make temporary directory
os.mkdir('tempDir')
# save temp file to tempDir
objects.to_file('tempDir/temp_file.shp')
# read temp file to shapefile
sf = shapefile.Reader('tempDir/temp_file.shp')

# convert geometry to points
print('Generating MultiPoint geometry...', timer() - start)
newType = shapefile.MULTIPOINT
w = shapefile.Writer(newType)
w._shapes.extend(sf.shapes())
for s in w.shapes():
    s.shapeType = newType
w.fields = list(sf.fields)
w.records.extend(sf.records())
w.save('tempDir/temp_file.shp')

# load points to GDF
points = gpd.read_file('tempDir/temp_file.shp')

# delete tempDir
# print('Cleaning temporary files...')
# shutil.rmtree('tempDir')

# set CRS
points.crs = objects.crs

# buffer points to capture unprecision caused by scipy Voronoi function
print('Buffering MultiPoint geometry...', timer() - start)
points = points.dropna()
points['geometry'] = points.buffer(0.25)
# join attributes from buildings to new voronoi cells
print('Spatial join of MultiPoint geometry and Voronoi polygons...', timer() - start)
# spatial join is super slow, it might be better to try rtree
start = timer()
voronoi_with_id = gpd.sjoin(voronoi_polygons, points, how='left', op='intersects')
print(timer() - start)
voronoi_with_id.crs = objects.crs

# resolve thise cells which were not joined spatially (again, due to unprecision caused by scipy Voronoi function)
# select those unjoined
unjoined = voronoi_with_id[voronoi_with_id['uID'].isnull()]
len(unjoined.index)
print('Fixing unjoined geometry:', len(unjoined.index), 'problems...', timer() - start)
# for each polygon, find neighbours, measure boundary and set uID to the most neighbouring one
start = timer()
for idx, row in unjoined.iterrows():
    neighbors = voronoi_with_id[~voronoi_with_id.geometry.disjoint(row.geometry)][unique_id].tolist()  # find neigbours
    neighbors = [x for x in neighbors if str(x) != 'nan']  # remove polygon itself

    import operator
    global boundaries
    boundaries = {}
    for i in neighbors:
        subset = voronoi_with_id.loc[voronoi_with_id[unique_id] == i]['geometry']
        l = 0
        for s in subset:
            l = l + row.geometry.intersection(s).length
        boundaries[i] = l

    voronoi_with_id.loc[idx, unique_id] = max(boundaries.items(), key=operator.itemgetter(1))[0]
print(timer() - start)
# dissolve polygons by unique_id
print('Dissolving Voronoi polygons...', timer() - start)
voronoi_plots = voronoi_with_id.dissolve(by=unique_id)
voronoi_plots[unique_id] = voronoi_plots.index  # save unique id to column from index

# generate built_up area around buildings to resolve the edge
print('Preparing buffer zone for edge resolving (buffering)...', timer() - start)
obj_simple['geometry'] = obj_simple.buffer(cut_buffer)
print('Preparing buffer zone for edge resolving (dissolving)...', timer() - start)
obj_simple['diss'] = 0
built_up = obj_simple.dissolve(by='diss')

# cut infinity of voronoi by set buffer (thanks for script to Geoff Boeing)
import osmnx as ox
print('Preparing buffer zone for edge resolving (quadrat cut)...', timer() - start)
geometry = built_up['geometry'].iloc[0].boundary
# quadrat_width is in the units the geometry is in
geometry_cut = ox.quadrat_cut_geometry(geometry, quadrat_width=100)

# build the r-tree index
print('Building R-tree...', timer() - start)
sindex = voronoi_plots.sindex
# find the points that intersect with each subpolygon and add them to points_within_geometry
to_cut = pd.DataFrame()
for poly in geometry_cut:
    # find approximate matches with r-tree, then precise matches from those approximate ones
    possible_matches_index = list(sindex.intersection(poly.bounds))
    possible_matches_index
    possible_matches = voronoi_plots.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(poly)]
    to_cut = to_cut.append(precise_matches)

# delete duplicates
to_cut = to_cut.drop_duplicates(subset=[unique_id])
subselection = list(to_cut.index)
voronoi_plots.to_file('tempDir/voro_pl.shp')
voronoi_plots.loc[subselection].to_file('tempDir/voro_subs.shp')
print('Cutting...', timer() - start)
for idx, row in voronoi_plots.loc[subselection].iterrows():
    wkt = row.geometry.intersection(built_up['geometry'].iloc[0]).wkt
    new = loads(wkt)
    voronoi_plots.loc[idx, 'geometry'] = new

voronoi_plots = voronoi_plots.drop(['index_right'], axis=1)
voronoi_plots[unique_id] = voronoi_plots[unique_id].astype('float')
print('Saving morphological tessellation to', save_tessellation, timer() - start)

voronoi_plots.to_file(save_tessellation)
print('Done.', timer() - start)
