#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd
import libpysal
from shapely.geometry import Point
import networkx as nx
import pandas as pd
import numpy as np


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
    series = pd.Series(range(len(objects)))
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


def gdf_to_nx(gdf_network):
    """
    Convert LineString GeoDataFrame to networkx.Graph

    Parameters
    ----------
    gdf_network : GeoDataFrame
        GeoDataFrame containing objects to convert

    Returns
    -------
    networkx.Graph
        Graph

    """
    # generate graph from GeoDataFrame of LineStrings
    net = nx.Graph()
    net.graph['crs'] = gdf_network.crs
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


def multi2single(gpdf):
    """
    Convert MultiPolygon geometry of GeoDataFrame to Polygon geometries.

    Parameters
    ----------
    gpdf : geopandas.GeoDataFrame
        geopandas.GeoDataFrame

    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing only Polygons

    """
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T] * len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly


def limit_range(vals, mode):
    """
    Extract values within selected mode ('id', 'iq')

    Parameters
    ----------
    vals : array

    mode : str
        'iq' for interquartile range, 'id' for interdecile range

    Returns
    -------
    array
        limited array
    """
    limited = []
    if mode == 'iq':
        min = 25
        max = 75
    elif mode == 'id':
        min = 10
        max = 90
    lower = np.percentile(vals, min)
    higher = np.percentile(vals, max)
    for x in vals:
        if x >= lower and x <= higher:
            limited.append(x)
    return limited
