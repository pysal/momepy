#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd
import libpysal
from shapely.geometry import Point
import networkx as nx
import pandas as pd


def _clean_buildings(objects, height_column):
    """
    Clean building geometry.

    Delete building with zero height (to avoid division by 0). Be careful, this might
    negatively affect your analysis.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    height_column : str
        name of the column of objects gdf where is stored height value

    Returns
    -------
    GeoDataFrame
    """

    objects = objects[objects[height_column] > 0]
    print('Zero height buildings ommited.')
    return objects


def _clean_null(objects):
    """
    Clean null geometry.

    Delete rows of GeoDataFrame with null geometry.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    GeoDataFrame
    """
    objects_none = objects[objects['geometry'].notnull()]  # filter nulls
    return objects_none


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
        GeoDataFrame containing objects to analyse
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


def _multi2single(gpdf):
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T] * len(Series_geometries), ignore_index=True)
        df['geometry'] = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly
