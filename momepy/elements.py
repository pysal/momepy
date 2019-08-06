#!/usr/bin/env python
# -*- coding: utf-8 -*-

# elements.py
# generating derived elements (street edge, block)
import geopandas as gpd
import pandas as pd
from tqdm import tqdm  # progress bar
import math
from osgeo import ogr
from shapely.wkt import loads
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import shapely
import operator
from libpysal.weights import Queen


def buffered_limit(gdf, buffer=100):
    """
    Define limit for tessellation as a buffer around buildings.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing building footprints
    buffer : float
        buffer around buildings limiting the extend of tessellation

    Returns
    -------
    MultiPolygon
        MultiPolygon or Polygon defining the study area

    Examples
    --------
    >>> limit = mm.buffered_limit(buildings_df)
    >>> type(limit)
    shapely.geometry.polygon.Polygon

    """
    study_area = gdf.copy()
    study_area['geometry'] = study_area.buffer(buffer)
    study_area['diss'] = 1
    built_up_df = study_area.dissolve(by='diss')
    built_up = built_up_df.geometry[1]
    return built_up


def _get_centre(gdf):
    """
    Returns centre coords of gdf.
    """
    bounds = gdf['geometry'].bounds
    centre_x = (bounds['maxx'].max() + bounds['minx'].min()) / 2
    centre_y = (bounds['maxy'].max() + bounds['miny'].min()) / 2
    return centre_x, centre_y


# densify geometry before Voronoi tesselation
def _densify(geom, segment):
    """
    Returns densified geoemtry with segments no longer than `segment`.
    """
    poly = geom
    wkt = geom.wkt  # shapely Polygon to wkt
    geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
    geom.Segmentize(segment)  # densify geometry by 2 metres
    geom.CloseRings()  # fix for GDAL 2.4.1 bug
    wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
    try:
        new = loads(wkt2)  # wkt to shapely Polygon
        return new
    except:
        return poly


def _point_array(objects, unique_id):
    """
    Returns lists of points and ids based on geometry and unique_id.
    """
    points = []
    ids = []
    for idx, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        poly_ext = row['geometry'].boundary
        if poly_ext is not None:
            if poly_ext.type == 'MultiLineString':
                for line in poly_ext:
                    point_coords = line.coords
                    row_array = np.array(point_coords).tolist()
                    for i in range(len(row_array)):
                        points.append(row_array[i])
                        ids.append(row[unique_id])
            elif poly_ext.type == 'LineString':
                point_coords = poly_ext.coords
                row_array = np.array(point_coords).tolist()
                for i in range(len(row_array)):
                    points.append(row_array[i])
                    ids.append(row[unique_id])
            else:
                raise Exception('Boundary type is {}'.format(poly_ext.type))
    return points, ids


def _regions(voronoi_diagram, unique_id, ids, crs):
    """
    Generate GeoDataFrame of Voronoi regions from scipy.spatial.Voronoi.
    """
    # generate DataFrame of results
    regions = pd.DataFrame()
    regions[unique_id] = ids  # add unique id
    regions['region'] = voronoi_diagram.point_region  # add region id for each point

    # add vertices of each polygon
    vertices = []
    for region in regions.region:
        vertices.append(voronoi_diagram.regions[region])
    regions['vertices'] = vertices

    # convert vertices to Polygons
    polygons = []
    for region in tqdm(regions.vertices, desc='Vertices to Polygons'):
        if -1 not in region:
            polygons.append(Polygon(voronoi_diagram.vertices[region]))
        else:
            polygons.append(None)
    # save polygons as geometry column
    regions['geometry'] = polygons

    # generate GeoDataFrame
    regions_gdf = gpd.GeoDataFrame(regions.dropna(), geometry='geometry')
    regions_gdf = regions_gdf.loc[regions_gdf['geometry'].length < 1000000]  # delete errors
    regions_gdf = regions_gdf.loc[regions_gdf[unique_id] != -1]  # delete hull-based cells
    regions_gdf.crs = crs
    return regions_gdf


def _split_lines(polygon, distance):
    """Split polygon into GeoSeries of lines no longer than `distance`."""
    list_points = []
    current_dist = distance  # set the current distance to place the point

    boundary = polygon.boundary  # make shapely MultiLineString object
    if boundary.type == 'LineString':
        line_length = boundary.length  # get the total length of the line
        while current_dist < line_length:  # while the current cumulative distance is less than the total length of the line
            list_points.append(boundary.interpolate(current_dist))  # use interpolate and increase the current distance
            current_dist += distance
    elif boundary.type == 'MultiLineString':
        for ls in boundary:
            line_length = ls.length  # get the total length of the line
            while current_dist < line_length:  # while the current cumulative distance is less than the total length of the line
                list_points.append(ls.interpolate(current_dist))  # use interpolate and increase the current distance
                current_dist += distance

    cutted = shapely.ops.split(boundary, shapely.geometry.MultiPoint(list_points).buffer(0.001))
    return cutted


def _cut(tessellation, limit, unique_id):
    """
    Cut tessellation by the limit (Multi)Polygon.

    ADD: add option to delete everything outside of limit. Now it keeps it.
    """
    print('Preparing limit for edge resolving...')
    geometry_cut = _split_lines(limit, 100)

    print('Building R-tree...')
    sindex = tessellation.sindex
    # find the points that intersect with each subpolygon and add them to points_within_geometry
    print('Identifying edge cells...')
    to_cut = pd.DataFrame()
    for poly in tqdm(geometry_cut, total=(len(geometry_cut))):
        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(sindex.intersection(poly.bounds))
        possible_matches = tessellation.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]
        to_cut = to_cut.append(precise_matches)

    # delete duplicates
    to_cut = to_cut.drop_duplicates(subset=[unique_id])
    subselection = list(to_cut.index)

    print('Cutting...')
    for idx, row in tqdm(tessellation.loc[subselection].iterrows(), total=tessellation.loc[subselection].shape[0]):
        intersection = row.geometry.intersection(limit)
        if intersection.type == 'MultiPolygon':
            areas = {}
            for p in range(len(intersection)):
                area = intersection[p].area
                areas[p] = area
            maximal = max(areas.items(), key=operator.itemgetter(1))[0]
            tessellation.loc[idx, 'geometry'] = intersection[maximal]
        elif intersection.type == 'GeometryCollection':
            for geom in list(intersection.geoms):
                if geom.type != 'Polygon':
                    pass
                else:
                    tessellation.loc[idx, 'geometry'] = geom
        else:
            tessellation.loc[idx, 'geometry'] = intersection
    return tessellation, sindex


def _check_result(tesselation, orig_gdf, unique_id):
    """
    Check whether result of tessellation matches buildings and contains only Polygons.
    """
    # check against input layer
    ids_original = list(orig_gdf[unique_id])
    ids_generated = list(tesselation[unique_id])
    if len(ids_original) != len(ids_generated):
        import warnings
        diff = set(ids_original).difference(ids_generated)
        warnings.warn("Tessellation does not fully match buildings. {len} element(s) collapsed "
                      "during generation - unique_id: {i}".format(len=len(diff), i=diff))

    # check MultiPolygons - usually caused by error in input geometry
    uids = tesselation[tesselation.geometry.type == 'MultiPolygon'][unique_id]
    if len(uids) > 0:
        import warnings
        warnings.warn('Tessellation contains MultiPolygon elements. Initial objects should be edited. '
                      'unique_id of affected elements: {}'.format(list(uids)))


def _queen_corners(tessellation, sensitivity, sindex):
    """
    Experimental: Fix unprecise corners.
    """
    changes = {}
    qid = 0

    for ix, row in tqdm(tessellation.iterrows(), total=tessellation.shape[0]):
        corners = []
        change = []

        cell = row.geometry
        coords = cell.exterior.coords
        for i in coords:
            point = Point(i)
            possible_matches_index = list(sindex.intersection(point.bounds))
            possible_matches = tessellation.iloc[possible_matches_index]
            precise_matches = sum(possible_matches.intersects(point))
            if precise_matches > 2:
                corners.append(point)

        if len(corners) > 2:
            for c in range(len(corners)):
                next_c = c + 1
                if c == (len(corners) - 1):
                    next_c = 0
                if corners[c].distance(corners[next_c]) < sensitivity:
                    change.append([corners[c], corners[next_c]])
        elif len(corners) == 2:
            if corners[0].distance(corners[1]) > 0:
                if corners[0].distance(corners[1]) < sensitivity:
                    change.append([corners[0], corners[1]])

        if change:
            for points in change:
                x_new = np.mean([points[0].x, points[1].x])
                y_new = np.mean([points[0].y, points[1].y])
                new = [(x_new, y_new), id]
                changes[(points[0].x, points[0].y)] = new
                changes[(points[1].x, points[1].y)] = new
                qid = qid + 1

    for ix, row in tqdm(tessellation.iterrows(), total=tessellation.shape[0]):
        cell = row.geometry
        coords = list(cell.exterior.coords)

        moves = {}
        for x in coords:
            if x in changes.keys():
                moves[coords.index(x)] = changes[x]
        keys = list(moves.keys())
        delete_points = []
        for move in range(len(keys)):
            if move < len(keys) - 1:
                if moves[keys[move]][1] == moves[keys[move + 1]][1] and keys[move + 1] - keys[move] < 5:
                    delete_points = delete_points + (coords[keys[move]:keys[move + 1]])
                    # change the code above to have if based on distance not number

        newcoords = [changes[x][0] if x in changes.keys() else x for x in coords]
        for coord in newcoords:
            if coord in delete_points:
                newcoords.remove(coord)
        if coords != newcoords:
            if not cell.interiors:
                # newgeom = Polygon(newcoords).buffer(0)
                be = Polygon(newcoords).exterior
                mls = be.intersection(be)
                if len(list(shapely.ops.polygonize(mls))) > 1:
                    newgeom = MultiPolygon(shapely.ops.polygonize(mls))
                    geoms = []
                    for g in range(len(newgeom)):
                        geoms.append(newgeom[g].area)
                    newgeom = newgeom[geoms.index(max(geoms))]
                else:
                    newgeom = list(shapely.ops.polygonize(mls))[0]
            else:
                newgeom = Polygon(newcoords, holes=cell.interiors)
            tessellation.loc[ix, 'geometry'] = newgeom
    return tessellation


def tessellation(gdf, unique_id, limit, shrink=0.4, segment=0.5, queen_corners=False, sensitivity=2):
    """
    Generate morphological tessellation around given buildings.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing building footprints
    unique_id : str
        name of the column with unique id
    limit : MultiPolygon or Polygon
        MultiPolygon or Polygon defining the study area limiting tessellation (otherwise it could go to infinity).
    shrink : float (default 0.4)
        distance for negative buffer to generate space between adjacent buildings.
    segment : float (default 0.5)
        maximum distance between points on Polygon after discretisation

    Returns
    -------
    GeoDataFrame
        GeoDataFrame of morphological tessellation with the unique id based on original buildings.

    Examples
    --------
    >>> tessellation_df = mm.tessellation(buildings_df, 'uID', limit=mm.buffered_limit(buildings_df))
    Bufferring geometry...
    Converting multipart geometry to singlepart...
    Densifying geometry...
    Generating input point array...
    100%|██████████| 144/144 [00:00<00:00, 376.15it/s]
    Generating Voronoi diagram...
    Generating GeoDataFrame...
    Vertices to Polygons: 100%|██████████| 33059/33059 [00:01<00:00, 31532.72it/s]
    Dissolving Voronoi polygons...
    Preparing buffer zone for edge resolving...
    Building R-tree...
    100%|██████████| 42/42 [00:00<00:00, 752.54it/s]
    Cutting...
    >>> tessellation_df2.head()
        uID	geometry
    0	1	POLYGON ((1603586.677274485 6464344.667944215,...
    1	2	POLYGON ((1603048.399497852 6464176.180701573,...
    2	3	POLYGON ((1603071.342637536 6464158.863329805,...
    3	4	POLYGON ((1603055.834005827 6464093.614718676,...
    4	5	POLYGON ((1603106.417554705 6464130.215958447,...

    Notes
    -------
    queen_corners and sensitivity are currently experimental only and can cause errors.
    """
    objects = gdf.copy()

    centre = _get_centre(objects)
    objects['geometry'] = objects['geometry'].translate(xoff=-centre[0], yoff=-centre[1])

    print('Bufferring geometry...')
    objects['geometry'] = objects.geometry.apply(lambda g: g.buffer(-shrink, cap_style=2, join_style=2))

    print('Converting multipart geometry to singlepart...')
    objects = objects.explode()
    objects.reset_index(inplace=True, drop=True)

    print('Densifying geometry...')
    objects['geometry'] = objects['geometry'].apply(_densify, segment=segment)

    print('Generating input point array...')
    points, ids = _point_array(objects, unique_id)

    # add convex hull buffered large distance to eliminate infinity issues
    series = gpd.GeoSeries(limit, crs=gdf.crs).translate(xoff=-centre[0], yoff=-centre[1])
    hull = series.geometry[0].convex_hull.buffer(300)
    hull = _densify(hull, 20)
    hull_array = np.array(hull.boundary.coords).tolist()
    for i in range(len(hull_array)):
        points.append(hull_array[i])
        ids.append(-1)

    print('Generating Voronoi diagram...')
    voronoi_diagram = Voronoi(np.array(points))

    print('Generating GeoDataFrame...')
    regions_gdf = _regions(voronoi_diagram, unique_id, ids, crs=gdf.crs)

    print('Dissolving Voronoi polygons...')
    morphological_tessellation = regions_gdf[[unique_id, 'geometry']].dissolve(by=unique_id, as_index=False)

    morphological_tessellation['geometry'] = morphological_tessellation['geometry'].translate(xoff=centre[0], yoff=centre[1])

    morphological_tessellation, sindex = _cut(morphological_tessellation, limit, unique_id)

    if queen_corners is True:
        morphological_tessellation = _queen_corners(morphological_tessellation, sensitivity, sindex)

    _check_result(morphological_tessellation, gdf, unique_id=unique_id)

    return morphological_tessellation


def blocks(tessellation, edges, buildings, id_name, unique_id):
    """
    Generate blocks based on buildings, tesselation and street network

    Adds bID to buildings and tesselation.

    Parameters
    ----------
    tessellation : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    edges : GeoDataFrame
        GeoDataFrame containing street network
    buildings : GeoDataFrame
        GeoDataFrame containing buildings
    id_name : str
        name of the unique blocks id column to be generated
    unique_id : str
        name of the column with unique id. If there is none, it could be generated by unique_id().
        This should be the same for cells and buildings, id's should match.

    Returns
    -------
    blocks, buildings_ID, cells_ID : tuple

    blocks : GeoDataFrame
        GeoDataFrame containing generated blocks
    buildings_ID : Series
        Series derived from buildings with block ID
    cells_ID : Series
        Series derived from morphological tessellation with block ID

    Examples
    --------
    >>> blocks, buildings_df['blockID'], tessellation_df['blockID'] = mm.blocks(tessellation_df, streets_df, buildings_df, 'bID', 'uID')
    Buffering streets...
    Generating spatial index...
    Difference...
    Defining adjacency...
    Defining street-based blocks...
    Defining block ID...
    Generating centroids...
    Spatial join...
    Attribute join (tesselation)...
    Generating blocks...
    Multipart to singlepart...
    Attribute join (buildings)...
    Attribute join (tesselation)...
    >>> blocks.head()
        bID	geometry
    0	1.0	POLYGON ((1603560.078648818 6464202.366899694,...
    1	2.0	POLYGON ((1603457.225976106 6464299.454696888,...
    2	3.0	POLYGON ((1603056.595487018 6464093.903488506,...
    3	4.0	POLYGON ((1603260.943782872 6464141.327631323,...
    4	5.0	POLYGON ((1603183.399594798 6463966.109982309,...

    """

    cells_copy = tessellation.copy()
    cells_copy = cells_copy[[unique_id, 'geometry']]

    print('Buffering streets...')
    street_buff = edges.copy()
    street_buff['geometry'] = street_buff.buffer(0.1)

    print('Generating spatial index...')
    streets_index = street_buff.sindex

    print('Difference...')
    cells_geom = cells_copy.geometry
    new_geom = []

    for ix, cell in tqdm(cells_geom.iteritems(), total=cells_geom.shape[0]):
        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(streets_index.intersection(cell.bounds))
        possible_matches = street_buff.iloc[possible_matches_index]
        new_geom.append(cell.difference(possible_matches.geometry.unary_union))

    single_geom = []
    print('Defining adjacency...')
    for p in new_geom:
        if p.type == 'MultiPolygon':
            for polygon in p:
                single_geom.append(polygon)
        else:
            single_geom.append(p)

    blocks_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(single_geom))
    spatial_weights = Queen.from_dataframe(blocks_gdf, silence_warnings=True)

    patches = {}
    jID = 1
    for idx, row in tqdm(blocks_gdf.iterrows(), total=blocks_gdf.shape[0]):

        # if the id is already present in courtyards, continue (avoid repetition)
        if idx in patches:
            continue
        else:
            to_join = [idx]  # list of indices which should be joined together
            neighbours = []  # list of neighbours
            weights = spatial_weights.neighbors[idx]  # neighbours from spatial weights
            for w in weights:
                neighbours.append(w)  # make a list from weigths

            for n in neighbours:
                while n not in to_join:  # until there is some neighbour which is not in to_join
                    to_join.append(n)
                    weights = spatial_weights.neighbors[n]
                    for w in weights:
                        neighbours.append(w)  # extend neighbours by neighbours of neighbours :)
            for b in to_join:
                patches[b] = jID  # fill dict with values
            jID = jID + 1

    blocks_gdf['patch'] = blocks_gdf.index.map(patches)

    print('Defining street-based blocks...')
    blocks_single = blocks_gdf.dissolve(by='patch')
    blocks_single.crs = buildings.crs

    blocks_single['geometry'] = blocks_single.buffer(0.1)

    print('Defining block ID...')  # street based
    blocks_single[id_name] = range(len(blocks_single))

    print('Generating centroids...')
    buildings_c = buildings.copy()
    buildings_c['geometry'] = buildings_c.representative_point()  # make points

    print('Spatial join...')
    centroids_tempID = gpd.sjoin(buildings_c, blocks_single, how='left', op='intersects')

    tempID_to_uID = centroids_tempID[[unique_id, id_name]]

    print('Attribute join (tesselation)...')
    cells_copy = cells_copy.merge(tempID_to_uID, on=unique_id, how='left')

    print('Generating blocks...')
    blocks = cells_copy.dissolve(by=id_name)

    print('Multipart to singlepart...')
    blocks = blocks.explode()
    blocks.reset_index(inplace=True, drop=True)

    blocks['geometry'] = blocks.exterior
    blocks[id_name] = range(len(blocks))

    for idx, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
        blocks.loc[idx, 'geometry'] = Polygon(row['geometry'])

    # if polygon is within another one, delete it
    sindex = blocks.sindex
    for idx, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
        possible_matches = list(sindex.intersection(row.geometry.bounds))
        possible_matches.remove(idx)
        possible = blocks.iloc[possible_matches]

        for idx2, row2 in possible.iterrows():
            if row['geometry'].within(row2['geometry']):
                blocks.loc[idx, 'delete'] = 1

    if 'delete' in blocks.columns:
        blocks = blocks.drop(list(blocks.loc[blocks['delete'] == 1].index))

    blocks_save = blocks[[id_name, 'geometry']]

    centroids_w_bl_ID2 = gpd.sjoin(buildings_c, blocks_save, how='left', op='intersects')
    bl_ID_to_uID = centroids_w_bl_ID2[[unique_id, id_name]]

    print('Attribute join (buildings)...')
    buildings_m = buildings[[unique_id]].merge(bl_ID_to_uID, on=unique_id, how='left')

    print('Attribute join (tesselation)...')
    cells_m = tessellation[[unique_id]].merge(bl_ID_to_uID, on=unique_id, how='left')

    return (blocks_save, buildings_m[id_name], cells_m[id_name])


def get_network_id(left, right, unique_id, network_id, min_size=100):
    """
    Snap each element (preferably building) to the closest street network segment, saves its id.

    Adds network ID to elements.

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing objects to snap
    right : GeoDataFrame
        GeoDataFrame containing street network with unique network ID.
        If there is none, it could be generated by :py:func:`momepy.elements.unique_id`.
    unique_id : str, list, np.array, pd.Series (default None)
        the name of the left dataframe column, np.array, or pd.Series with unique id
    network_id : str, list, np.array, pd.Series (default None)
        the name of the streets dataframe column, np.array, or pd.Series with network unique id.
    min_size : int (default 100)
        min_size should be a vaule such that if you build a box centered in each
        building centroid with edges of size `2*min_size`, you know a priori that at least one
        segment is intersected with the box.

    Returns
    -------
    elements_nID : Series
        Series containing network ID for elements

    Examples
    --------
    >>> buildings_df['nID'] = momepy.get_network_id(buildings_df, streets_df, 'uID', 'nID')
    Generating centroids...
    Generating rtree...
    Snapping: 100%|██████████| 144/144 [00:00<00:00, 2718.98it/s]
    >>> buildings_df['nID'][0]
    1
    """
    INFTY = 1000000000000
    MIN_SIZE = min_size
    # MIN_SIZE should be a vaule such that if you build a box centered in each
    # point with edges of size 2*MIN_SIZE, you know a priori that at least one
    # segment is intersected with the box. Otherwise, you could get an inexact
    # solution, there is an exception checking this, though.
    left = left.copy()
    right = right.copy()
    
    if not isinstance(unique_id, str):
        left['mm_uid'] = unique_id
        unique_id = 'mm_uid'
    if not isinstance(network_id, str):
        right['mm_nid'] = network_id
        network_id = 'mm_nid'

    print('Generating centroids...')
    buildings_c = left.copy()

    buildings_c['geometry'] = buildings_c.centroid  # make centroids

    print('Generating rtree...')
    idx = right.sindex

    result = []
    for ix, r in tqdm(buildings_c.iterrows(), total=buildings_c.shape[0], desc='Snapping'):
        p = r.geometry
        pbox = (p.x - MIN_SIZE, p.y - MIN_SIZE, p.x + MIN_SIZE, p.y + MIN_SIZE)
        hits = list(idx.intersection(pbox))
        d = INFTY
        nid = None
        for h in hits:
            new_d = p.distance(right.geometry.loc[h])
            if d >= new_d:
                d = new_d
                nid = right[network_id].loc[h]
        if nid is None:
            result.append(np.nan)
        else:
            result.append(nid)

    series = pd.Series(result)

    if series.isnull().any():
        import warnings
        warnings.warn('Some objects were not attached to the network. '
                      'Set larger min_size. {} affected elements'.format(sum(series.isnull())))
    return series


def get_node_id(objects, nodes, edges, node_id, edge_id):
    """
    Snap each building to closest street network node on the closest network edge.

    Adds node ID to objects (preferably buildings). Gets ID of edge (:py:func:`momepy.get_node_id`)
    , and determines which of its end points is closer to building centroid.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to snap
    nodes : GeoDataFrame
        GeoDataFrame containing street nodes with unique node ID.
        If there is none, it could be generated by :py:func:`momepy.unique_id`.
    edges : GeoDataFrame
        GeoDataFrame containing street edges with unique edge ID and IDs of start
        and end points of each segment. Start and endpoints are default outcome of :py:func:`momepy.nx_to_gdf`.
    node_id : str, list, np.array, pd.Series (default None)
        the name of the nodes dataframe column, np.array, or pd.Series with unique id

    Returns
    -------
    node_ids : Series
        Series containing node ID for objects

    """
    nodes = nodes.copy()
    if not isinstance(node_id, str):
        nodes['mm_noid'] = node_id
        node_id = 'mm_noid'

    results_list = []
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        if np.isnan(row[edge_id]):

            results_list.append(np.nan)
        else:
            centroid = row.geometry.centroid
            edge = edges.loc[edges[edge_id] == row[edge_id]].iloc[0]
            startID = edge.node_start
            start = nodes.loc[nodes[node_id] == startID].iloc[0].geometry
            sd = centroid.distance(start)
            endID = edge.node_end
            end = nodes.loc[nodes[node_id] == endID].iloc[0].geometry
            ed = centroid.distance(end)
            if sd > ed:
                results_list.append(endID)
            else:
                results_list.append(startID)

    series = pd.Series(results_list, index=objects.index)
    return series
