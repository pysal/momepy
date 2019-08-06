#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd
import libpysal
from shapely.geometry import Point, LineString
import networkx as nx
import numpy as np
from tqdm import tqdm
import operator
import shapely
import math

from .shape import circular_compactness


def unique_id(objects):
    """
    Add an attribute with unique ID to each row of GeoDataFrame.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.

    """
    series = range(len(objects))
    return series


def Queen_higher(k, geodataframe=None, weights=None, ids=None):
    """
    Generate spatial weights based on Queen contiguity of order k

    Pass either geoDataFrame or weights. If both are passed, weights is used.

    Parameters
    ----------
    k : int
        order of contiguity
    geodataframe : GeoDataFrame
        GeoDataFrame containing objects to analyse. Index has to be consecutive range 0:x.
        Otherwise, spatial weights will not match objects.
    weights : libpysal.weights
        libpysal.weights of order 1

    Returns
    -------
    libpysal.weights
        libpysal.weights object

    Examples
    --------
    >>> first_order = libpysal.weights.Queen.from_dataframe(geodataframe)
    >>> first_order.mean_neighbors
    5.848032564450475
    >>> fourth_order = Queen_higher(k=4, geodataframe=geodataframe)
    >>> fourth.mean_neighbors
    85.73188602442333

    """
    if weights is not None:
        first_order = weights
    elif geodataframe is not None:
        if not all(geodataframe.index == range(len(geodataframe))):
            raise ValueError('Index is not consecutive range 0:x, spatial weights will not match objects.')
        first_order = libpysal.weights.Queen.from_dataframe(geodataframe, ids=ids)
    else:
        raise Warning('GeoDataFrame of spatial weights must be given.')

    joined = first_order
    for i in list(range(2, k + 1)):
        i_order = libpysal.weights.higher_order(first_order, k=i)
        joined = libpysal.weights.w_union(joined, i_order)
    return joined


def gdf_to_nx(gdf_network, length='mm_len'):
    """
    Convert LineString GeoDataFrame to networkx.Graph

    Parameters
    ----------
    gdf_network : GeoDataFrame
        GeoDataFrame containing objects to convert
    length : str, optional
        name of attribute of segment length (geographical) which will be saved to graph

    Returns
    -------
    networkx.Graph
        Graph

    """
    # generate graph from GeoDataFrame of LineStrings
    net = nx.Graph()
    net.graph['crs'] = gdf_network.crs
    gdf_network[length] = gdf_network.geometry.length
    fields = list(gdf_network.columns)

    for index, row in gdf_network.iterrows():
        first = row.geometry.coords[0]
        last = row.geometry.coords[-1]

        data = [row[f] for f in fields]
        attributes = dict(zip(fields, data))
        net.add_edge(first, last, **attributes)

    return net


def nx_to_gdf(net, nodes=True, edges=True, spatial_weights=False, nodeID='nodeID'):
    """
    Convert networkx.Graph to LineString GeoDataFrame and Point GeoDataFrame

    Parameters
    ----------
    net : networkx.Graph
        networkx.Graph
    nodes : bool
        export nodes gdf
    edges : bool
        export edges gdf
    spatial_weights : bool
        export libpysal spatial weights for nodes
    nodeID : str
        name of node ID column to be generated

    Returns
    -------
    GeoDataFrame
        Selected gdf or tuple of both gdf or tuple of gdfs and weights

    """
    # generate nodes and edges geodataframes from graph
    if nodes is True:
        nid = 1
        for n in net:
            net.nodes[n][nodeID] = nid
            nid += 1
        node_xy, node_data = zip(*net.nodes(data=True))
        gdf_nodes = gpd.GeoDataFrame(list(node_data), geometry=[Point(i, j) for i, j in node_xy])
        gdf_nodes.crs = net.graph['crs']
        if spatial_weights is True:
            W = libpysal.weights.W.from_networkx(net)

    if edges is True:
        starts, ends, edge_data = zip(*net.edges(data=True))
        if nodes is True:
            node_start = []
            node_end = []
            for s in starts:
                node_start.append(net.node[s][nodeID])
            for e in ends:
                node_end.append(net.node[e][nodeID])
        gdf_edges = gpd.GeoDataFrame(list(edge_data))
        if nodes is True:
            gdf_edges['node_start'] = node_start
            gdf_edges['node_end'] = node_end
        gdf_edges.crs = net.graph['crs']

    if nodes is True and edges is True:
        if spatial_weights is True:
            return gdf_nodes, gdf_edges, W
        return gdf_nodes, gdf_edges
    elif nodes is True and edges is False:
        if spatial_weights is True:
            return gdf_nodes, W
        return gdf_nodes
    return gdf_edges


def limit_range(vals, rng):
    """
    Extract values within selected range

    Parameters
    ----------
    vals : array

    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.
    Returns
    -------
    array
        limited array
    """
    if len(vals) > 2:
        limited = []
        rng = sorted(rng)
        lower = np.percentile(vals, rng[0], interpolation='nearest')
        higher = np.percentile(vals, rng[1], interpolation='nearest')
        for x in vals:
            if x >= lower and x <= higher:
                limited.append(x)
        return limited
    return vals


def preprocess(buildings, size=30, compactness=True, islands=True):
    """
    Preprocesses building geometry to eliminate additional structures being single features.

    Certain data providers (e.g. Ordnance Survey in GB) do not provide building geometry
    as one feature, but divided into different features depending their level (if they are
    on ground floor or not - passages, overhangs). Ideally, these features should share
    one building ID on which they could be dissolved. If this is not the case, series of
    steps needs to be done to minimize errors in morphological analysis.

    This script attempts to preprocess such geometry based on several condidions:
    If feature area is smaller than set size it will be a) deleted if it does not
    touch any other feature; b) will be joined to feature with which it shares the
    longest boundary. If feature is fully within other feature, these will be joined.
    If feature's circular compactness (:py:func:`momepy.shape.circular compactness`)
    is < 0.2, it will be joined to feature with which it shares the longest boundary.
    Function does two loops through.


    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        geopandas.GeoDataFrame containing building layer
    size : float (default 30)
        maximum area of feature to be considered as additional structure. Set to
        None if not wanted.
    compactness : bool (default True)
        if True, function will resolve additional structures identified based on
        their circular compactness.
    islands : bool (default True)
        if True, function will resolve additional structures which are fully within
        other structures (share 100% of exterior boundary).

    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing preprocessed geometry
    """
    blg = buildings.copy()
    blg = blg.explode()
    blg.reset_index(drop=True, inplace=True)
    for l in range(0, 2):
        print('Loop', l + 1, 'out of 2.')
        blg.reset_index(inplace=True, drop=True)
        blg['mm_uid'] = range(len(blg))
        sw = libpysal.weights.contiguity.Rook.from_dataframe(blg, silence_warnings=True)
        blg['neighbors'] = sw.neighbors
        blg['neighbors'] = blg['neighbors'].map(sw.neighbors)
        blg['n_count'] = blg.apply(lambda row: len(row.neighbors), axis=1)
        blg['circu'] = circular_compactness(blg)

        # idetify those smaller than x with only one neighbor and attaches it to it.
        join = {}
        delete = []

        for idx, row in tqdm(blg.iterrows(), total=blg.shape[0], desc='Identifying changes'):
            if size:
                if row.geometry.area < size:
                    if row.n_count == 1:
                        uid = blg.iloc[row.neighbors[0]].mm_uid

                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]
                    elif row.n_count > 1:
                        shares = {}
                        for n in row.neighbors:
                            shares[n] = row.geometry.intersection(blg.at[n, 'geometry']).length
                        maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                        uid = blg.loc[maximal].mm_uid
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]
                    else:
                        delete.append(idx)
            if compactness:
                if row.circu < 0.2:
                    if row.n_count == 1:
                        uid = blg.iloc[row.neighbors[0]].mm_uid
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]
                    elif row.n_count > 1:
                        shares = {}
                        for n in row.neighbors:
                            shares[n] = row.geometry.intersection(blg.at[n, 'geometry']).length
                        maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                        uid = blg.loc[maximal].mm_uid
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]

            if islands:
                if row.n_count == 1:
                    shared = row.geometry.intersection(blg.at[row.neighbors[0], 'geometry']).length
                    if shared == row.geometry.exterior.length:
                        uid = blg.iloc[row.neighbors[0]].mm_uid
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]

        for key in tqdm(join, total=len(join), desc='Changing geometry'):
            selection = blg[blg['mm_uid'] == key]
            if not selection.empty:
                geoms = [selection.iloc[0].geometry]

                for j in join[key]:
                    subset = blg[blg['mm_uid'] == j]
                    if not subset.empty:
                        geoms.append(blg[blg['mm_uid'] == j].iloc[0].geometry)
                        blg.drop(blg[blg['mm_uid'] == j].index[0], inplace=True)
                new_geom = shapely.ops.unary_union(geoms)
                blg.loc[blg.loc[blg['mm_uid'] == key].index[0], 'geometry'] = new_geom

        blg.drop(delete, inplace=True)
    return blg[buildings.columns]


def network_false_nodes(gdf):
    """
    Check topology of street network and eliminate nodes of degree 2 by joining
    affected edges.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containg edge representation of street network.
    Returns
    -------
    gdf : GeoDataFrame
    """
    streets = gdf.copy().explode()
    streets.reset_index(inplace=True)
    sindex = streets.sindex

    false_points = []
    print('Identifying false points...')
    for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
        line = row['geometry']
        l_coords = list(line.coords)
        # network_w = network.drop(idx, axis=0)['geometry']  # ensure that it wont intersect itself
        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        # find out whether ends of the line are connected or not
        possible_first_index = list(sindex.intersection(start.bounds))
        possible_first_matches = streets.iloc[possible_first_index]
        possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
        real_first_matches = possible_first_matches_clean[possible_first_matches_clean.intersects(start)]

        possible_second_index = list(sindex.intersection(end.bounds))
        possible_second_matches = streets.iloc[possible_second_index]
        possible_second_matches_clean = possible_second_matches.drop(idx, axis=0)
        real_second_matches = possible_second_matches_clean[possible_second_matches_clean.intersects(end)]

        if len(real_first_matches) == 1:
            false_points.append(start)
        if len(real_second_matches) == 1:
            false_points.append(end)

    false_xy = []
    for p in false_points:
        false_xy.append([p.x, p.y])

    false_xy_unique = [list(x) for x in set(tuple(x) for x in false_xy)]

    false_unique = []
    for p in false_xy_unique:
        false_unique.append(Point(p[0], p[1]))

    geoms = streets.geometry

    print('Merging segments...')
    for point in tqdm(false_unique):
        matches = list(geoms[geoms.intersects(point)].index)
        idx = max(geoms.index) + 1
        try:
            multiline = geoms[matches[0]].union(geoms[matches[1]])
            linestring = shapely.ops.linemerge(multiline)
            geoms = geoms.append(gpd.GeoSeries(linestring, index=[idx]))
            geoms = geoms.drop(matches)
        except IndexError:
            import warnings
            warnings.warn('An exception during merging occured. Lines at point [{x}, {y}] were not merged.'.format(x=point.x, y=point.y))

    geoms_gdf = gpd.GeoDataFrame(geometry=geoms)
    geoms_gdf.crs = streets.crs
    streets = geoms_gdf
    return streets


def snap_street_network_edge(edges, buildings, tolerance_street, tessellation=None, tolerance_edge=None):
    """
    Fix street network before performing blocks()

    Extends unjoined ends of street segments to join with other segmets or tessellation boundary.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing street network
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
    tolerance_street : float
        tolerance in snapping to street network (by how much could be street segment extended).
    tessellation : GeoDataFrame (default None)
        GeoDataFrame containing morphological tessellation
    tolerance_edge : float (default None)
        tolerance in snapping to edge of tessellated area (by how much could be street segment extended).

    Returns
    -------
    GeoDataFrame
        GeoDataFrame of extended street network.

    """
    # extrapolating function - makes line as a extrapolation of existing with set length (tolerance)
    def getExtrapoledLine(p1, p2, tolerance):
        """
        Creates a line extrapoled in p1->p2 direction.
        """
        EXTRAPOL_RATIO = tolerance  # length of a line
        a = p2

        # defining new point based on the vector between existing points
        if p1[0] >= p2[0] and p1[1] >= p2[1]:
            b = (p2[0] - EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] - EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        elif p1[0] <= p2[0] and p1[1] >= p2[1]:
            b = (p2[0] + EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] - EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        elif p1[0] <= p2[0] and p1[1] <= p2[1]:
            b = (p2[0] + EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] + EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        else:
            b = (p2[0] - EXTRAPOL_RATIO * math.cos(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))),
                 p2[1] + EXTRAPOL_RATIO * math.sin(math.atan(math.fabs(p1[1] - p2[1] + 0.000001) / math.fabs(p1[0] - p2[0] + 0.000001))))
        return LineString([a, b])

    # function extending line to closest object within set distance
    def extend_line(tolerance, idx):
        """
        Extends a line geometry withing GeoDataFrame to snap on itself withing tolerance.
        """
        if Point(l_coords[-2]).distance(Point(l_coords[-1])) <= 0.001:
            if len(l_coords) > 2:
                extra = l_coords[-3:-1]
            else:
                return False
        else:
            extra = l_coords[-2:]
        extrapolation = getExtrapoledLine(*extra, tolerance=tolerance)  # we use the last two points

        possible_intersections_index = list(sindex.intersection(extrapolation.bounds))
        possible_intersections_lines = network.iloc[possible_intersections_index]
        possible_intersections_clean = possible_intersections_lines.drop(idx, axis=0)
        possible_intersections = possible_intersections_clean.intersection(extrapolation)

        if possible_intersections.any():

            true_int = []
            for one in list(possible_intersections.index):
                if possible_intersections[one].type == 'Point':
                    true_int.append(possible_intersections[one])
                elif possible_intersections[one].type == 'MultiPoint':
                    true_int.append(possible_intersections[one][0])
                    true_int.append(possible_intersections[one][1])

            if len(true_int) >= 1:
                if len(true_int) > 1:
                    distances = {}
                    ix = 0
                    for p in true_int:
                        distance = p.distance(Point(l_coords[-1]))
                        distances[ix] = distance
                        ix = ix + 1
                    minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                    new_point_coords = true_int[minimal].coords[0]
                else:
                    new_point_coords = true_int[0].coords[0]

                l_coords.append(new_point_coords)
                new_extended_line = LineString(l_coords)

                # check whether the line goes through buildings. if so, ignore it
                possible_buildings_index = list(bindex.intersection(new_extended_line.bounds))
                possible_buildings = buildings.iloc[possible_buildings_index]
                possible_intersections = possible_buildings.intersection(new_extended_line)

                if possible_intersections.any():
                    pass
                else:
                    network.loc[idx, 'geometry'] = new_extended_line
        else:
            return False

    # function extending line to closest object within set distance to edge defined by tessellation
    def extend_line_edge(tolerance, idx):
        """
        Extends a line geometry withing GeoDataFrame to snap on the boundary of tessellation withing tolerance.
        """
        if Point(l_coords[-2]).distance(Point(l_coords[-1])) <= 0.001:
            if len(l_coords) > 2:
                extra = l_coords[-3:-1]
            else:
                return False
        else:
            extra = l_coords[-2:]
        extrapolation = getExtrapoledLine(*extra, tolerance)  # we use the last two points

        # possible_intersections_index = list(qindex.intersection(extrapolation.bounds))
        # possible_intersections_lines = geometry_cut.iloc[possible_intersections_index]
        possible_intersections = geometry.intersection(extrapolation)

        if possible_intersections.type != 'GeometryCollection':

            true_int = []

            if possible_intersections.type == 'Point':
                true_int.append(possible_intersections)
            elif possible_intersections.type == 'MultiPoint':
                true_int.append(possible_intersections[0])
                true_int.append(possible_intersections[1])

            if len(true_int) >= 1:
                if len(true_int) > 1:
                    distances = {}
                    ix = 0
                    for p in true_int:
                        distance = p.distance(Point(l_coords[-1]))
                        distances[ix] = distance
                        ix = ix + 1
                    minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                    new_point_coords = true_int[minimal].coords[0]
                else:
                    new_point_coords = true_int[0].coords[0]

                l_coords.append(new_point_coords)
                new_extended_line = LineString(l_coords)

                # check whether the line goes through buildings. if so, ignore it
                possible_buildings_index = list(bindex.intersection(new_extended_line.bounds))
                possible_buildings = buildings.iloc[possible_buildings_index]
                possible_intersections = possible_buildings.intersection(new_extended_line)

                if possible_intersections.any():
                    pass
                else:
                    network.loc[idx, 'geometry'] = new_extended_line

    network = edges.copy()
    # generating spatial index (rtree)
    print('Building R-tree for network...')
    sindex = network.sindex
    print('Building R-tree for buildings...')
    bindex = buildings.sindex
    if tessellation is not None:
        print('Dissolving tesselation...')
        geometry = tessellation.geometry.unary_union.boundary

    print('Snapping...')
    # iterating over each street segment
    for idx, row in tqdm(network.iterrows(), total=network.shape[0]):

        line = row['geometry']
        l_coords = list(line.coords)
        # network_w = network.drop(idx, axis=0)['geometry']  # ensure that it wont intersect itself
        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        # find out whether ends of the line are connected or not
        possible_first_index = list(sindex.intersection(start.bounds))
        possible_first_matches = network.iloc[possible_first_index]
        possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
        first = possible_first_matches_clean.intersects(start).any()

        possible_second_index = list(sindex.intersection(end.bounds))
        possible_second_matches = network.iloc[possible_second_index]
        possible_second_matches_clean = possible_second_matches.drop(idx, axis=0)
        second = possible_second_matches_clean.intersects(end).any()

        # both ends connected, do nothing
        if first and second:
            continue
        # start connected, extend  end
        elif first and not second:
            if extend_line(tolerance_street, idx) is False:
                if tessellation is not None:
                    extend_line_edge(tolerance_edge, idx)
        # end connected, extend start
        elif not first and second:
            l_coords.reverse()
            if extend_line(tolerance_street, idx) is False:
                if tessellation is not None:
                    extend_line_edge(tolerance_edge, idx)
        # unconnected, extend both ends
        elif not first and not second:
            if extend_line(tolerance_street, idx) is False:
                if tessellation is not None:
                    extend_line_edge(tolerance_edge, idx)
            l_coords.reverse()
            if extend_line(tolerance_street, idx) is False:
                if tessellation is not None:
                    extend_line_edge(tolerance_edge, idx)
        else:
            print('Something went wrong.')

    return network
