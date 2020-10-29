#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np

from shapely.geometry import Point


__all__ = [
    "unique_id",
    "gdf_to_nx",
    "nx_to_gdf",
    "limit_range",
]


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


def _angle(a, b, c):
    """
    Measure angle between a-b, b-c. In radians.
    Helper for gdf_to_nx.
    """
    ba = [aa - bb for aa, bb in zip(a, b)]
    bc = [cc - bb for cc, bb in zip(c, b)]
    nba = math.sqrt(sum((x ** 2.0 for x in ba)))
    ba = [x / nba for x in ba]
    nbc = math.sqrt(sum((x ** 2.0 for x in bc)))
    bc = [x / nbc for x in bc]
    scal = sum((aa * bb for aa, bb in zip(ba, bc)))
    angle = math.acos(round(scal, 10))
    return angle


def _generate_primal(G, gdf_network, fields):
    """
    Generate primal graph.
    Helper for gdf_to_nx.
    """
    G.graph["approach"] = "primal"
    key = 0
    for index, row in gdf_network.iterrows():
        first = row.geometry.coords[0]
        last = row.geometry.coords[-1]

        data = [row[f] for f in fields]
        attributes = dict(zip(fields, data))
        G.add_edge(first, last, key=key, **attributes)
        key += 1


def _generate_dual(G, gdf_network, fields):
    """
    Generate dual graph
    Helper for gdf_to_nx.
    """
    G.graph["approach"] = "dual"
    sw = libpysal.weights.Queen.from_dataframe(gdf_network)
    gdf_network["mm_cent"] = gdf_network.geometry.centroid

    for i, (index, row) in enumerate(gdf_network.iterrows()):
        centroid = (row.mm_cent.x, row.mm_cent.y)
        data = [row[f] for f in fields]
        attributes = dict(zip(fields, data))
        G.add_node(centroid, **attributes)

        if sw.cardinalities[i] > 0:
            for n in sw.neighbors[i]:
                start = centroid
                end = list(gdf_network.iloc[n]["mm_cent"].coords)[0]
                p0 = row.geometry.coords[0]
                p1 = row.geometry.coords[-1]
                p2 = gdf_network.iloc[n]["geometry"].coords[0]
                p3 = gdf_network.iloc[n]["geometry"].coords[-1]
                points = [p0, p1, p2, p3]
                shared = [x for x in points if points.count(x) > 1]
                if shared:  # fix for non-planar graph
                    remaining = [e for e in points if e not in [shared[0]]]
                    if len(remaining) == 2:
                        angle = _angle(remaining[0], shared[0], remaining[1])
                        G.add_edge(start, end, key=0, angle=angle)


def gdf_to_nx(gdf_network, approach="primal", length="mm_len"):
    """
    Convert LineString GeoDataFrame to networkx.MultiGraph

    Parameters
    ----------
    gdf_network : GeoDataFrame
        GeoDataFrame containing objects to convert
    approach : str, default 'primal'
        Decide wheter genereate ``'primal'`` or ``'dual'`` graph.
    length : str, default mm_len
        name of attribute of segment length (geographical) which will be saved to graph

    Returns
    -------
    networkx.Graph
        Graph

    """
    gdf_network = gdf_network.copy()
    if "key" in gdf_network.columns:
        gdf_network.rename(columns={"key": "__key"}, inplace=True)
    # generate graph from GeoDataFrame of LineStrings
    net = nx.MultiGraph()
    net.graph["crs"] = gdf_network.crs
    gdf_network[length] = gdf_network.geometry.length
    fields = list(gdf_network.columns)

    if approach == "primal":
        _generate_primal(net, gdf_network, fields)

    elif approach == "dual":
        _generate_dual(net, gdf_network, fields)

    else:
        raise ValueError(
            "Approach {} is not supported. Use 'primal' or 'dual'.".format(approach)
        )

    return net


def _points_to_gdf(net):
    """
    Generate point gdf from nodes.
    Helper for nx_to_gdf.
    """
    node_xy, node_data = zip(*net.nodes(data=True))
    if isinstance(node_xy[0], int) and "x" in node_data[0].keys():
        geometry = [Point(data["x"], data["y"]) for data in node_data]  # osmnx graph
    else:
        geometry = [Point(*p) for p in node_xy]
    gdf_nodes = gpd.GeoDataFrame(list(node_data), geometry=geometry)
    if "crs" in net.graph.keys():
        gdf_nodes.crs = net.graph["crs"]
    return gdf_nodes


def _lines_to_gdf(net, lines, points, nodeID):
    """
    Generate linestring gdf from edges.
    Helper for nx_to_gdf.
    """
    starts, ends, edge_data = zip(*net.edges(data=True))
    if lines is True:
        node_start = []
        node_end = []
        for s in starts:
            node_start.append(net.nodes[s][nodeID])
        for e in ends:
            node_end.append(net.nodes[e][nodeID])
    gdf_edges = gpd.GeoDataFrame(list(edge_data))
    if points is True:
        gdf_edges["node_start"] = node_start
        gdf_edges["node_end"] = node_end
    if "crs" in net.graph.keys():
        gdf_edges.crs = net.graph["crs"]
    return gdf_edges


def _primal_to_gdf(net, points, lines, spatial_weights, nodeID):
    """
    Generate gdf(s) from primal network.
    Helper for nx_to_gdf.
    """
    if points is True:
        gdf_nodes = _points_to_gdf(net)

        if spatial_weights is True:
            W = libpysal.weights.W.from_networkx(net)
            W.transform = "b"

    if lines is True:
        gdf_edges = _lines_to_gdf(net, lines, points, nodeID)

    if points is True and lines is True:
        if spatial_weights is True:
            return gdf_nodes, gdf_edges, W
        return gdf_nodes, gdf_edges
    if points is True and lines is False:
        if spatial_weights is True:
            return gdf_nodes, W
        return gdf_nodes
    return gdf_edges


def _dual_to_gdf(net):
    """
    Generate linestring gdf from dual network.
    Helper for nx_to_gdf.
    """
    starts, edge_data = zip(*net.nodes(data=True))
    gdf_edges = gpd.GeoDataFrame(list(edge_data))
    gdf_edges.crs = net.graph["crs"]
    return gdf_edges


def nx_to_gdf(net, points=True, lines=True, spatial_weights=False, nodeID="nodeID"):
    """
    Convert ``networkx.Graph`` to LineString GeoDataFrame and Point GeoDataFrame

    Parameters
    ----------
    net : networkx.Graph
        ``networkx.Graph``
    points : bool
        export point-based gdf representing intersections
    lines : bool
        export line-based gdf representing streets
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
    primal = None
    if "approach" in net.graph.keys():
        if net.graph["approach"] == "primal":
            primal = True
        elif net.graph["approach"] == "dual":
            return _dual_to_gdf(net)
        else:
            raise ValueError(
                "Approach {} is not supported. Use 'primal' or 'dual'.".format(
                    net.graph["approach"]
                )
            )

    if not primal:
        import warnings

        warnings.warn("Approach is not set. Defaulting to 'primal'.")

    nid = 1
    for n in net:
        net.nodes[n][nodeID] = nid
        nid += 1
    return _primal_to_gdf(
        net, points=points, lines=lines, spatial_weights=spatial_weights, nodeID=nodeID
    )


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
        rng = sorted(rng)
        lower = np.percentile(vals, rng[0], interpolation="nearest")
        higher = np.percentile(vals, rng[1], interpolation="nearest")
        limited = [x for x in vals if x >= lower and x <= higher]
        return np.array(limited)
    return vals


def _azimuth(point1, point2):
    """azimuth between 2 shapely points (interval 0 - 180)"""
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180
