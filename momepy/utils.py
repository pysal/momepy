#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd
import libpysal
from shapely.geometry import Point
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import operator
import shapely

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


def Queen_higher(k, geodataframe=None, weights=None):
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
        first_order = libpysal.weights.Queen.from_dataframe(geodataframe)
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


def nx_to_gdf(net, nodes=True, edges=True, spatial_weights=False):
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

    Returns
    -------
    GeoDataFrame
        Selected gdf or tuple of both gdf or tuple of gdfs and weights

    """
    # generate nodes and edges geodataframes from graph
    if nodes is True:
        node_xy, node_data = zip(*net.nodes(data=True))
        gdf_nodes = gpd.GeoDataFrame(list(node_data), geometry=[Point(i, j) for i, j in node_xy])
        gdf_nodes.crs = net.graph['crs']
        if spatial_weights is True:
            W = libpysal.weights.W.from_networkx(net)

    if edges is True:
        starts, ends, edge_data = zip(*net.edges(data=True))
        gdf_edges = gpd.GeoDataFrame(list(edge_data))
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


def multi2single(gdf):
    """
    Convert multi-part geometry of GeoDataFrame to single-part geometries.

    Parameters
    ----------
    gpdf : geopandas.GeoDataFrame
        geopandas.GeoDataFrame

    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing only single-part geometry

    """
    if gdf.iloc[0].geometry.type in ['Polygon', 'MultiPolygon']:
        gdf_singlepoly = gdf[gdf.geometry.type == 'Polygon']
        gdf_multipoly = gdf[gdf.geometry.type == 'MultiPolygon']
    elif gdf.iloc[0].geometry.type in ['LineString', 'MultiLineString']:
        gdf_singlepoly = gdf[gdf.geometry.type == 'LineString']
        gdf_multipoly = gdf[gdf.geometry.type == 'MultiLineString']
    elif gdf.iloc[0].geometry.type in ['Point', 'MultiPoint']:
        gdf_singlepoly = gdf[gdf.geometry.type == 'Point']
        gdf_multipoly = gdf[gdf.geometry.type == 'MultiPoint']
    else:
        raise ValueError('Unsupported geometry type:', gdf.iloc[0].geometry.type)

    for i, row in gdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gdf_multipoly.crs).T] * len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        gdf_singlepoly = pd.concat([gdf_singlepoly, df])

    gdf_singlepoly.reset_index(inplace=True, drop=True)
    return gdf_singlepoly


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
        lower = np.nanpercentile(vals, rng[0])
        higher = np.nanpercentile(vals, rng[1])
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
    for l in range(0, 2):
        print('Loop', l + 1, 'out of 2.')
        blg.reset_index(inplace=True)
        blg['uID'] = range(len(blg))
        sw = libpysal.weights.contiguity.Rook.from_dataframe(blg, silence_warnings=True)
        blg['neighbors'] = sw.neighbors
        blg['neighbors'] = blg['neighbors'].map(sw.neighbors)
        blg['n_count'] = blg.apply(lambda row: len(row.neighbors), axis=1)
        blg['circu'] = circular_compactness(blg)
        # blg['con1'] = None
        # blg.loc[delete, 'con1'] = 'island'

        # idetify those smaller than x with only one neighbor and attaches it to it.
        join = {}
        delete = []

        for idx, row in tqdm(blg.iterrows(), total=blg.shape[0], desc='Identifying changes'):
            if size:
                if row.geometry.area < size:
                    if row.n_count == 1:
                        uid = blg.iloc[row.neighbors[0]].uID

                        if uid in join:
                            existing = join[uid]
                            existing.append(row.uID)
                            join[uid] = existing
                        else:
                            join[uid] = [row.uID]
                    elif row.n_count > 1:
                        shares = {}
                        for n in row.neighbors:
                            shares[n] = row.geometry.intersection(blg.at[n, 'geometry']).length
                        maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                        uid = blg.loc[maximal].uID
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.uID)
                            join[uid] = existing
                        else:
                            join[uid] = [row.uID]
                    else:
                        delete.append(idx)
            if compactness:
                if row.circu < 0.2:
                    if row.n_count == 1:
                        uid = blg.iloc[row.neighbors[0]].uID
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.uID)
                            join[uid] = existing
                        else:
                            join[uid] = [row.uID]
                    elif row.n_count > 1:
                        shares = {}
                        for n in row.neighbors:
                            shares[n] = row.geometry.intersection(blg.at[n, 'geometry']).length
                        maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                        uid = blg.loc[maximal].uID
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.uID)
                            join[uid] = existing
                        else:
                            join[uid] = [row.uID]

            if islands:
                if row.n_count == 1:
                    shared = row.geometry.intersection(blg.at[row.neighbors[0], 'geometry']).length
                    if shared == row.geometry.exterior.length:
                        uid = blg.iloc[row.neighbors[0]].uID
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.uID)
                            join[uid] = existing
                        else:
                            join[uid] = [row.uID]

        for key in tqdm(join, total=len(join), desc='Changing geometry'):
            selection = blg[blg['uID'] == key]
            if not selection.empty:
                geoms = [selection.iloc[0].geometry]

                for j in join[key]:
                    subset = blg[blg['uID'] == j]
                    if not subset.empty:
                        geoms.append(blg[blg['uID'] == j].iloc[0].geometry)
                        blg.drop(blg[blg['uID'] == j].index[0], inplace=True)
                new_geom = shapely.ops.unary_union(geoms)
                blg.loc[blg.loc[blg['uID'] == key].index[0], 'geometry'] = new_geom

        blg.drop(delete, inplace=True)
    blg.drop(['neighbors', 'n_count', 'circu', 'uID'], axis=1, inplace=True)
    return blg


def network_false_nodes(streets):
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
