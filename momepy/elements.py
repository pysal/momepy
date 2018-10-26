#!/usr/bin/env python
# -*- coding: utf-8 -*-

# elements.py
# generating derived elements (street edge, block)
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

'''
clean_buildings():

Clean building geometry

Delete building with zero height (to avoid division by 0)
'''


def clean_buildings(path, height_column):
    print('Loading file...')
    objects = gpd.read_file(path)  # load file into geopandas
    print('Shapefile loaded.')

    objects = objects[objects[height_column] > 0]
    print('Zero height buildings ommited.')

    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')

'''
clean_null():

Clean null geometry

Delete rows of GeoDataFrame with null geometry.
'''


def clean_null(path):
    print('Loading file...')
    objects = gpd.read_file(path)
    print('Shapefile loaded.')

    objects_none = objects[objects['geometry'].notnull()]  # filter nulls
    # save dataframe back to file
    print('Saving file...')
    objects_none.to_file(path)
    print('File saved.')

'''
unique_id():

Add an attribute with unique ID to each row of GeoDataFrame.

Optional: Delete all the columns except ID and geometry (set clear to True)
          To keep some of the columns and delete rest, pass them to keep.
          Typically we want to keep building height and get rid of the rest.
'''


def unique_id(path, clear=False, keep=None, id_name='uID'):
    print('Loading file...')
    objects = gpd.read_file(path)
    print('Shapefile loaded.')

    objects[id_name] = None
    objects[id_name] = objects[id_name].astype('float')
    id = 1
    for idx, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        objects.loc[idx, id_name] = id
        id = id + 1

    cols = objects.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    if clear is False:
        objects = objects[cols]
    else:
        if keep is None:
            objects = objects.iloc[:, [-2, -1]]
        else:
            keep_col = objects.columns.get_loc(keep)
            objects = objects.iloc[:, [-2, keep_col, -1]]

    # objects['uID'] = objects['uID'].astype('int16') - it is making weird errors
    # save dataframe back to file
    print('Saving file...')
    objects.to_file(path)
    print('File saved.')


'''
tesselation():

Generate tessellation ADD DESCRIPTION
FIX TEMPDIR DELETION
MAKE IT FASTER - RTREE IN SPATIAL JOINS AND INTERSECTIONS
FIX SAW GEOMETRY BY CUT OUT JOIN (VIA RTREE)

Optional:
'''


def tessellation(buildings, save_tessellation, unique_id='uID', cut_buffer=50):
    objects = buildings

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

    # densify geometry before Voronoi tesselation
    def densify(geom):
        wkt = geom.wkt  # shapely Polygon to wkt
        geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
        geom.Segmentize(2)  # densify geometry by 2 metres
        wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
        new = loads(wkt2)  # wkt to shapely Polygon
        return new

    print('Done in', timer() - start, 'seconds')
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
    # define new numpy.array
    # voronoi_points = np.empty([1, 2])
    # # fill array with all points from densified geometry
    # for idx, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    #     poly_ext = row['geometry'].exterior
    #     if poly_ext is not None:
    #         point_coords = poly_ext.coords
    #         row_array = np.array(point_coords)
    #         voronoi_points = np.concatenate((voronoi_points, row_array))
    #         # it might be faster to use python list.append(a) and then l = np.array(l)
    #
    # # delete initial row of array to keep only points from geometry
    # voronoi_points = voronoi_points[1:]

    list_points = []
    for idx, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        poly_ext = row['geometry'].exterior
        if poly_ext is not None:
            point_coords = poly_ext.coords
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

    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Saving to temporary file...')

    # set crs
    voronoi_polygons.crs = objects.crs
    # make temporary directory
    os.mkdir('tempDir')
    # save temp file to tempDir
    objects.to_file('tempDir/temp_file.shp')
    # read temp file to shapefile
    sf = shapefile.Reader('tempDir/temp_file.shp')

    # convert geometry to points
    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Generating MultiPoint geometry...')
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
    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Cleaning temporary files...')
    shutil.rmtree('tempDir')

    # set CRS
    points.crs = objects.crs

    # buffer points to capture unprecision caused by scipy Voronoi function
    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Buffering MultiPoint geometry...')
    points = points.dropna()
    # points['geometry'] = points.buffer(0.25)
    # join attributes from buildings to new voronoi cells
    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Spatial join of MultiPoint geometry and Voronoi polygons...')
    # spatial join is super slow, it might be better to try rtree
    voronoi_with_id = gpd.sjoin(voronoi_polygons, points, how='left')
    voronoi_with_id.crs = objects.crs

    # resolve thise cells which were not joined spatially (again, due to unprecision caused by scipy Voronoi function)
    # select those unjoined
    unjoined = voronoi_with_id[voronoi_with_id[unique_id].isnull()]
    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Fixing unjoined geometry:', len(unjoined.index), 'problems...')
    # for each polygon, find neighbours, measure boundary and set uID to the most neighbouring one
    join_index = voronoi_with_id.sindex
    for idx, row in tqdm(unjoined.iterrows(), total=unjoined.shape[0]):
        neighbors = list(join_index.intersection(row.geometry.bounds))  # find neigbours
        neighbors_ids = []
        for n in neighbors:
            neighbors_ids.append(voronoi_with_id.iloc[n][unique_id])
        neighbors_ids = [x for x in neighbors_ids if str(x) != 'nan']  # remove polygon itself

        import operator
        global boundaries
        boundaries = {}
        for i in neighbors_ids:
            subset = voronoi_with_id.loc[voronoi_with_id[unique_id] == i]['geometry']
            l = 0
            for s in subset:
                l = l + row.geometry.intersection(s).length
            boundaries[i] = l

        voronoi_with_id.loc[idx, unique_id] = max(boundaries.items(), key=operator.itemgetter(1))[0]
    # dissolve polygons by unique_id
    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Dissolving Voronoi polygons...')
    voronoi_with_id['geometry'] = voronoi_with_id.buffer(0)
    voronoi_plots = voronoi_with_id.dissolve(by=unique_id)
    voronoi_plots[unique_id] = voronoi_plots.index.astype('float')  # save unique id to column from index

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

    # cut infinity of voronoi by set buffer (thanks for script to Geoff Boeing)
    import osmnx as ox
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
        wkt = row.geometry.intersection(built_up['geometry'].iloc[0]).wkt
        new = loads(wkt)
        voronoi_plots.loc[idx, 'geometry'] = new

    voronoi_plots = voronoi_plots.drop(['index_right'], axis=1)

    print('Done in', timer() - start, 'seconds')
    start = timer()
    print('Saving morphological tessellation to', save_tessellation)
    voronoi_plots.to_file(save_tessellation)
    print('Done in', timer() - start, 'seconds')
    print('Done. Tessellation finished in', timer() - start_, 'seconds.')

'''
blocks():

Generate blocks based on buildings, tesselation and street network
Adds bID to buildings and tesselation.


Optional:
'''


def blocks(cells, streets, buildings, id_name, unique_id, cells_to, buildings_to, blocks_to):
    print('Dissolving tesselation...')
    cells['diss'] = 0
    built_up = cells.dissolve(by='diss')

    print('Buffering streets...')
    street_buff = streets.copy()
    street_buff['geometry'] = streets.buffer(0.1)

    print('Dissolving streets...')
    street_cut = street_buff.unary_union

    print('Defining street-based blocks...')
    street_blocks = built_up['geometry'].difference(street_cut)

    blocks_gdf = gpd.GeoDataFrame(street_blocks)

    blocks_gdf = blocks_gdf.rename(columns={0: 'geometry'}).set_geometry('geometry')

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

    print('Multipart to singlepart...')
    blocks_single = multi2single(blocks_gdf)

    print('Defining block ID...')  # street based
    blocks_single[id_name] = None
    blocks_single[id_name] = blocks_single[id_name].astype('float')
    id = 1
    for idx, row in tqdm(blocks_single.iterrows(), total=blocks_single.shape[0]):
        blocks_single.loc[idx, id_name] = id
        id = id + 1

    print('Generating centroids...')
    buildings_c = buildings.copy()
    buildings_c['geometry'] = buildings_c.centroid  # make centroids
    blocks_single.crs = buildings.crs

    print('Spatial join...')
    centroids_tempID = gpd.sjoin(buildings_c, blocks_single, how='inner', op='intersects')

    tempID_to_uID = centroids_tempID[[unique_id, id_name]]

    print('Attribute join (tesselation)...')
    cells = cells.merge(tempID_to_uID, on=unique_id)
    cells = cells.drop(['diss'], axis=1)

    print('Generating blocks...')
    blocks = cells.dissolve(by=id_name)
    cells = cells.drop([id_name], axis=1)

    print('Multipart to singlepart...')
    blocks = multi2single(blocks)

    blocks['geometry'] = blocks.exterior

    id = 1
    for idx, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
        blocks.loc[idx, id_name] = id
        id = id + 1
        blocks.loc[idx, 'geometry'] = Polygon(row['geometry'])

    blocks_save = blocks[[id_name, 'geometry']]
    blocks_save['geometry'] = blocks_save.buffer(0.000000001)

    centroids_w_bl_ID2 = gpd.sjoin(buildings_c, blocks_save, how='inner', op='intersects')
    bl_ID_to_uID = centroids_w_bl_ID2[[unique_id, id_name]]

    print('Attribute join (buildings)...')
    buildings = buildings.merge(bl_ID_to_uID, on=unique_id)

    print('Attribute join (tesselation)...')
    cells = cells.merge(bl_ID_to_uID, on=unique_id)

    print('Saving buildings to', buildings_to)
    buildings.to_file(buildings_to)

    print('Saving tesselation to', cells_to)
    cells.to_file(cells_to)

    print('Saving blocks to', blocks_to)
    blocks_save.to_file(blocks_to)

    print('Done')


'''
street_edges():

Generate street edges based on buildings, blocks, tesselation and street network with street names
Adds nID and eID to buildings and tesselation.

    buildings = gdf of buildings (with unique id)
    streets = gdf of street network (with street names and unique network segment id)
    tesselation = gdf of tesselation (with unique id and block id)
    street_name_column = column with street names
    unique_id_column = column with unique ids
    block_id_column = column with block ids
    network_id_column = column with network ids
    tesselation_to = path to save tesselation with nID, eID
    buildings_to = path to save buildings with nID, eID
    save_to = path to save street edges

Optional:
'''


def street_edges(buildings, streets, tesselation, street_name_column,
                 unique_id_column, block_id_column, network_id_column,
                 tesselation_to, buildings_to, save_to):
    INFTY = 1000000000000
    MIN_SIZE = 100
    # MIN_SIZE should be a vaule such that if you build a box centered in each
    # point with edges of size 2*MIN_SIZE, you know a priori that at least one
    # segment is intersected with the box. Otherwise, you could get an inexact
    # solution, there is an exception checking this, though.

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_distance(apoint, segment):
        a = apoint
        b, c = segment
        # t = <a-b, c-b>/|c-b|**2
        # because p(a) = t*(c-b)+b is the ortogonal projection of vector a
        # over the rectline that includes the points b and c.
        t = (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1])
        t = t / ((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)
        # Only if t 0 <= t <= 1 the projection is in the interior of
        # segment b-c, and it is the point that minimize the distance
        # (by pythagoras theorem).
        if 0 < t < 1:
            pcoords = (t * (c[0] - b[0]) + b[0], t * (c[1] - b[1]) + b[1])
            dmin = distance(a, pcoords)
            return pcoords, dmin
        elif t <= 0:
            return b, distance(a, b)
        elif 1 <= t:
            return c, distance(a, c)

    def get_rtree(lines):
        def generate_items():
            sindx = 0
            for lid, nid, l in tqdm(lines, total=len(lines)):
                for i in range(len(l) - 1):
                    a, b = l[i]
                    c, d = l[i + 1]
                    segment = ((a, b), (c, d))
                    box = (min(a, c), min(b, d), max(a, c), max(b, d))
                    # box = left, bottom, right, top
                    yield (sindx, box, (lid, segment, nid))
                    sindx += 1
        return index.Index(generate_items())

    def get_solution(idx, points):
        result = {}
        for p in tqdm(points, total=len(points)):
            pbox = (p[0] - MIN_SIZE, p[1] - MIN_SIZE, p[0] + MIN_SIZE, p[1] + MIN_SIZE)
            hits = idx.intersection(pbox, objects='raw')
            d = INFTY
            s = None
            for h in hits:
                nearest_p, new_d = get_distance(p, h[1])
                if d >= new_d:
                    d = new_d
                    # s = (h[0], h[1], nearest_p, new_d)
                    s = (h[0], h[1], h[-1])
            result[p] = s
            if s is None:
                result[p] = (0, 0)

            # some checking you could remove after you adjust the constants
            # if s is None:
            #     raise Warning("It seems INFTY is not big enough. Point was not attached to street. It might be too far.", p)

            # pboxpol = ((pbox[0], pbox[1]), (pbox[2], pbox[1]),
            #            (pbox[2], pbox[3]), (pbox[0], pbox[3]))
            # if not Polygon(pboxpol).intersects(LineString(s[1])):
            #     msg = "It seems MIN_SIZE is not big enough. "
            #     msg += "You could get inexact solutions if remove this exception."
            #     raise Exception(msg)

        return result

    print('Generating centroids...')
    buildings_c = buildings.copy()
    buildings_c['geometry'] = buildings_c.centroid  # make centroids

    print('Generating list of points...')
    # make points list for input
    centroid_list = []
    for idx, row in tqdm(buildings_c.iterrows(), total=buildings_c.shape[0]):
        centroid_list = centroid_list + list(row['geometry'].coords)

    print('Generating list of lines...')
    # make streets list for input
    street_list = []
    for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
        street_list.append((row[street_name_column], row[network_id_column], list(row['geometry'].coords)))
    print('Generating rtree...')
    idx = get_rtree(street_list)

    print('Snapping...')
    solutions = get_solution(idx, centroid_list)

    print('Forming DataFrame...')
    df = pd.DataFrame.from_dict(solutions, orient='index', columns=['street', 'unused', network_id_column])  # solutions dict to df
    df['point'] = df.index  # point to column
    df = df.reset_index()
    df['idx'] = df.index
    buildings_c['idx'] = buildings_c.index

    print('Joining DataFrames...')
    joined = buildings_c.merge(df, on='idx')
    print('Cleaning DataFrames...')
    cleaned = joined[[unique_id_column, 'street', network_id_column]]

    print('Merging with tesselation...')
    tesselation = tesselation.merge(cleaned, on=unique_id_column)

    print('Defining merge ID...')
    for idx, row in tqdm(tesselation.iterrows(), total=tesselation.shape[0]):
        tesselation.loc[idx, 'mergeID'] = str(row['street']) + str(row[block_id_column])

    print('Dissolving...')
    edges = tesselation.dissolve(by='mergeID')

    # multipart geometry to singlepart
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

    edges_single = multi2single(edges)
    edges_single['geometry'] = edges_single.exterior
    print('Generating unique edge ID...')
    id = 1
    for idx, row in tqdm(edges_single.iterrows(), total=edges_single.shape[0]):
        edges_single.loc[idx, 'eID'] = id
        id = id + 1
        edges_single.loc[idx, 'geometry'] = Polygon(row['geometry'])

    print('Cleaning edges...')
    edges_clean = edges_single[['geometry', 'eID', block_id_column]]
    edges_clean['geometry'] = edges_clean.buffer(0.000000001)

    print('Saving street edges to', save_to)
    edges_clean.to_file(save_to)

    print('Cleaning tesselation...')
    tesselation = tesselation.drop(['street', 'mergeID'], axis=1)

    print('Tesselation spatial join [1/3]...')
    tess_centroid = tesselation.copy()
    tess_centroid['geometry'] = tess_centroid.centroid

    edg_join = edges_clean.drop(['bID'], axis=1)

    print('Tesselation spatial join [2/3]...')
    tess_with_eID = gpd.sjoin(tess_centroid, edg_join, how='inner', op='intersects')
    tess_with_eID = tess_with_eID[['uID', 'eID']]

    print('Tesselation spatial join [3/3]...')
    tesselation = tesselation.merge(tess_with_eID, on='uID')

    print('Saving tesselation to', tesselation_to)
    tesselation.to_file(tesselation_to)

    print('Buildings attribute join...')
    # attribute join cell -> building
    tess_nid_eid = tesselation[['uID', 'eID', 'nID']]

    buildings = buildings.merge(tess_nid_eid, on='uID')

    print('Saving buildings to', buildings_to)
    buildings.to_file(buildings_to)

    print('Done.')
