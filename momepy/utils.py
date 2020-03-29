#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import operator

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from tqdm import tqdm

from .shape import CircularCompactness

__all__ = [
    "unique_id",
    "sw_high",
    "gdf_to_nx",
    "nx_to_gdf",
    "limit_range",
    "preprocess",
    "network_false_nodes",
    "snap_street_network_edge",
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


def sw_high(k, gdf=None, weights=None, ids=None, contiguity="queen", silent=True):
    """
    Generate spatial weights based on Queen or Rook contiguity of order k.

    Adjacent are all features within <= k steps. Pass either gdf or weights.
    If both are passed, weights is used. If weights are passed, contiguity is
    ignored and high order spatial weights based on `weights` are computed.

    Parameters
    ----------
    k : int
        order of contiguity
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse. Index has to be consecutive range 0:x.
        Otherwise, spatial weights will not match objects.
    weights : libpysal.weights
        libpysal.weights of order 1
    contiguity : str (default 'queen')
        type of contiguity weights. Can be 'queen' or 'rook'.
    silent : bool (default True)
        silence libpysal islands warnings

    Returns
    -------
    libpysal.weights
        libpysal.weights object

    Examples
    --------
    >>> first_order = libpysal.weights.Queen.from_dataframe(geodataframe)
    >>> first_order.mean_neighbors
    5.848032564450475
    >>> fourth_order = sw_high(k=4, gdf=geodataframe)
    >>> fourth.mean_neighbors
    85.73188602442333

    """
    if weights is not None:
        first_order = weights
    elif gdf is not None:
        if contiguity == "queen":
            first_order = libpysal.weights.Queen.from_dataframe(
                gdf, ids=ids, silence_warnings=silent
            )
        elif contiguity == "rook":
            first_order = libpysal.weights.Rook.from_dataframe(
                gdf, ids=ids, silence_warnings=silent
            )
        else:
            raise ValueError(
                "{} is not supported. Use 'queen' or 'rook'.".format(contiguity)
            )
    else:
        raise AttributeError("GeoDataFrame or spatial weights must be given.")

    joined = first_order
    for i in list(range(2, k + 1)):
        i_order = libpysal.weights.higher_order(
            first_order, k=i, silence_warnings=silent
        )
        joined = libpysal.weights.w_union(joined, i_order, silence_warnings=silent)
    return joined


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
        Decide wheter genereate 'primal' or 'dual' graph.
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


def _points_to_gdf(net, spatial_weights):
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
        gdf_nodes = _points_to_gdf(net, spatial_weights)

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
    Convert networkx.Graph to LineString GeoDataFrame and Point GeoDataFrame

    Parameters
    ----------
    net : networkx.Graph
        networkx.Graph
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
    if "approach" in net.graph.keys():
        if net.graph["approach"] == "primal":
            nid = 1
            for n in net:
                net.nodes[n][nodeID] = nid
                nid += 1
            return _primal_to_gdf(
                net,
                points=points,
                lines=lines,
                spatial_weights=spatial_weights,
                nodeID=nodeID,
            )
        if net.graph["approach"] == "dual":
            return _dual_to_gdf(net)
        raise ValueError(
            "Approach {} is not supported. Use 'primal' or 'dual'.".format(
                net.graph["approach"]
            )
        )

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
    If feature's circular compactness (:py:func:`momepy.circular_compactness`)
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
        print("Loop", l + 1, "out of 2.")
        blg.reset_index(inplace=True, drop=True)
        blg["mm_uid"] = range(len(blg))
        sw = libpysal.weights.contiguity.Rook.from_dataframe(blg, silence_warnings=True)
        blg["neighbors"] = sw.neighbors
        blg["neighbors"] = blg["neighbors"].map(sw.neighbors)
        blg["n_count"] = blg.apply(lambda row: len(row.neighbors), axis=1)
        blg["circu"] = CircularCompactness(blg).series

        # idetify those smaller than x with only one neighbor and attaches it to it.
        join = {}
        delete = []

        for row in tqdm(
            blg.itertuples(), total=blg.shape[0], desc="Identifying changes"
        ):
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
                            shares[n] = row.geometry.intersection(
                                blg.at[n, "geometry"]
                            ).length
                        maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                        uid = blg.loc[maximal].mm_uid
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]
                    else:
                        delete.append(row.Index)
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
                            shares[n] = row.geometry.intersection(
                                blg.at[n, "geometry"]
                            ).length
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
                    shared = row.geometry.intersection(
                        blg.at[row.neighbors[0], "geometry"]
                    ).length
                    if shared == row.geometry.exterior.length:
                        uid = blg.iloc[row.neighbors[0]].mm_uid
                        if uid in join:
                            existing = join[uid]
                            existing.append(row.mm_uid)
                            join[uid] = existing
                        else:
                            join[uid] = [row.mm_uid]

        for key in tqdm(join, total=len(join), desc="Changing geometry"):
            selection = blg[blg["mm_uid"] == key]
            if not selection.empty:
                geoms = [selection.iloc[0].geometry]

                for j in join[key]:
                    subset = blg[blg["mm_uid"] == j]
                    if not subset.empty:
                        geoms.append(blg[blg["mm_uid"] == j].iloc[0].geometry)
                        blg.drop(blg[blg["mm_uid"] == j].index[0], inplace=True)
                new_geom = shapely.ops.unary_union(geoms)
                blg.loc[blg.loc[blg["mm_uid"] == key].index[0], "geometry"] = new_geom

        blg.drop(delete, inplace=True)
    return blg[buildings.columns]


def network_false_nodes(gdf, tolerance=0.1, precision=3):
    """
    Check topology of street network and eliminate nodes of degree 2 by joining
    affected edges. Attributes are not preserved.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        GeoDataFrame  or GeoSeries containg edge representation of street network.
    tolerance : float
        nodes wihtin tolearance are seen as identical (floating point precision fix)
    precision : int
        rounding parameter in estimating uniqueness of two points based on their
        coordinates

    Returns
    -------
    gdf : GeoDataFrame, GeoSeries
    """
    if not isinstance(gdf, (gpd.GeoSeries, gpd.GeoDataFrame)):
        raise TypeError(
            "'gdf' should be GeoDataFrame or GeoSeries, got {}".format(type(gdf))
        )
    streets = gdf.reset_index(drop=True).explode()
    if isinstance(streets, gpd.GeoDataFrame):
        series = False
        streets = streets.reset_index(drop=True).geometry
    elif isinstance(streets, gpd.GeoSeries):
        streets = streets.reset_index(drop=True)
        series = True

    sindex = streets.sindex

    false_xy = []
    print("Identifying false points...")
    for idx, line in tqdm(streets.iteritems(), total=streets.shape[0]):
        l_coords = list(line.coords)
        start = Point(l_coords[0]).buffer(tolerance)
        end = Point(l_coords[-1]).buffer(tolerance)

        # find out whether ends of the line are connected or not
        possible_first_index = list(sindex.intersection(start.bounds))
        possible_first_matches = streets.iloc[possible_first_index]
        possible_first_matches_clean = possible_first_matches.drop(idx, axis=0)
        real_first_matches = possible_first_matches_clean[
            possible_first_matches_clean.intersects(start)
        ]

        possible_second_index = list(sindex.intersection(end.bounds))
        possible_second_matches = streets.iloc[possible_second_index]
        possible_second_matches_clean = possible_second_matches.drop(idx, axis=0)
        real_second_matches = possible_second_matches_clean[
            possible_second_matches_clean.intersects(end)
        ]

        if len(real_first_matches) == 1:
            false_xy.append(
                (round(l_coords[0][0], precision), round(l_coords[0][1], precision))
            )
        if len(real_second_matches) == 1:
            false_xy.append(
                (round(l_coords[-1][0], precision), round(l_coords[-1][1], precision))
            )

    false_unique = [Point(x) for x in set(false_xy)]

    geoms = streets

    print("Merging segments...")
    for point in tqdm(false_unique):
        matches = list(geoms[geoms.intersects(point.buffer(tolerance))].index)
        idx = max(geoms.index) + 1
        try:
            snap = shapely.ops.snap(geoms[matches[0]], geoms[matches[1]], tolerance)
            multiline = snap.union(geoms[matches[1]])
            linestring = shapely.ops.linemerge(multiline)
            geoms = geoms.append(gpd.GeoSeries(linestring, index=[idx]))
            geoms = geoms.drop(matches)
        except (IndexError, ValueError):
            import warnings

            warnings.warn(
                "An exception during merging occured. "
                "Lines at point [{x}, {y}] were not merged.".format(
                    x=point.x, y=point.y
                )
            )

    geoms.crs = streets.crs
    streets = geoms.explode().reset_index(drop=True)
    if series:
        return streets
    geoms_gdf = gpd.GeoDataFrame(geometry=streets, crs=streets.crs)
    return geoms_gdf


def snap_street_network_edge(
    edges,
    buildings,
    tolerance_street,
    tessellation=None,
    tolerance_edge=None,
    edge=None,
):
    """
    Fix street network before performing blocks()

    Extends unjoined ends of street segments to join with other segmets or
    tessellation boundary.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing street network
    buildings : GeoDataFrame
        GeoDataFrame containing building footprints
    tolerance_street : float
        tolerance in snapping to street network (by how much could be street segment
        extended).
    tessellation : GeoDataFrame (default None)
        GeoDataFrame containing morphological tessellation. If edge is not passed it
        will be used as edge.
    tolerance_edge : float (default None)
        tolerance in snapping to edge of tessellated area (by how much could be street
        segment extended).
    edge : Polygon
        edge of area covered by morphological tessellation (same as `limit` in
        :py:func:`momepy.tessellation`)

    Returns
    -------
    GeoDataFrame
        GeoDataFrame of extended street network.

    """
    # makes line as a extrapolation of existing with set length (tolerance)
    def getExtrapoledLine(p1, p2, tolerance):
        """
        Creates a line extrapoled in p1->p2 direction.
        """
        EXTRAPOL_RATIO = tolerance  # length of a line
        a = p2

        # defining new point based on the vector between existing points
        if p1[0] >= p2[0] and p1[1] >= p2[1]:
            b = (
                p2[0]
                - EXTRAPOL_RATIO
                * math.cos(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
                p2[1]
                - EXTRAPOL_RATIO
                * math.sin(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
            )
        elif p1[0] <= p2[0] and p1[1] >= p2[1]:
            b = (
                p2[0]
                + EXTRAPOL_RATIO
                * math.cos(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
                p2[1]
                - EXTRAPOL_RATIO
                * math.sin(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
            )
        elif p1[0] <= p2[0] and p1[1] <= p2[1]:
            b = (
                p2[0]
                + EXTRAPOL_RATIO
                * math.cos(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
                p2[1]
                + EXTRAPOL_RATIO
                * math.sin(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
            )
        else:
            b = (
                p2[0]
                - EXTRAPOL_RATIO
                * math.cos(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
                p2[1]
                + EXTRAPOL_RATIO
                * math.sin(
                    math.atan(
                        math.fabs(p1[1] - p2[1] + 0.000001)
                        / math.fabs(p1[0] - p2[0] + 0.000001)
                    )
                ),
            )
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
        extrapolation = getExtrapoledLine(
            *extra, tolerance=tolerance
        )  # we use the last two points

        possible_intersections_index = list(sindex.intersection(extrapolation.bounds))
        possible_intersections_lines = network.iloc[possible_intersections_index]
        possible_intersections_clean = possible_intersections_lines.drop(idx, axis=0)
        possible_intersections = possible_intersections_clean.intersection(
            extrapolation
        )

        if not possible_intersections.is_empty.all():

            true_int = []
            for one in list(possible_intersections.index):
                if possible_intersections[one].type == "Point":
                    true_int.append(possible_intersections[one])
                elif possible_intersections[one].type == "MultiPoint":
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
                possible_buildings_index = list(
                    bindex.intersection(new_extended_line.bounds)
                )
                possible_buildings = buildings.iloc[possible_buildings_index]
                possible_intersections = possible_buildings.intersection(
                    new_extended_line
                )

                if possible_intersections.any():
                    pass
                else:
                    network.loc[idx, "geometry"] = new_extended_line
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
        extrapolation = getExtrapoledLine(
            *extra, tolerance
        )  # we use the last two points

        # possible_intersections_index = list(qindex.intersection(extrapolation.bounds))
        # possible_intersections_lines = geometry_cut.iloc[possible_intersections_index]
        possible_intersections = geometry.intersection(extrapolation)

        if possible_intersections.type != "GeometryCollection":

            true_int = []

            if possible_intersections.type == "Point":
                true_int.append(possible_intersections)
            elif possible_intersections.type == "MultiPoint":
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
                possible_buildings_index = list(
                    bindex.intersection(new_extended_line.bounds)
                )
                possible_buildings = buildings.iloc[possible_buildings_index]
                possible_intersections = possible_buildings.intersection(
                    new_extended_line
                )

                if not possible_intersections.is_empty.all():
                    pass
                else:
                    network.loc[idx, "geometry"] = new_extended_line

    network = edges.copy()
    # generating spatial index (rtree)
    print("Building R-tree for network...")
    sindex = network.sindex
    print("Building R-tree for buildings...")
    bindex = buildings.sindex

    def _get_geometry():
        if edge is not None:
            return edge.boundary
        if tessellation is not None:
            print("Dissolving tesselation...")
            return tessellation.geometry.unary_union.boundary
        return None

    geometry = _get_geometry()

    print("Snapping...")
    # iterating over each street segment
    for idx, line in tqdm(network.geometry.iteritems(), total=network.shape[0]):

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
                if geometry is not None:
                    extend_line_edge(tolerance_edge, idx)
        # end connected, extend start
        elif not first and second:
            l_coords.reverse()
            if extend_line(tolerance_street, idx) is False:
                if geometry is not None:
                    extend_line_edge(tolerance_edge, idx)
        # unconnected, extend both ends
        elif not first and not second:
            if extend_line(tolerance_street, idx) is False:
                if geometry is not None:
                    extend_line_edge(tolerance_edge, idx)
            l_coords.reverse()
            if extend_line(tolerance_street, idx) is False:
                if geometry is not None:
                    extend_line_edge(tolerance_edge, idx)
        else:
            print("Something went wrong.")

    return network


def _azimuth(point1, point2):
    """azimuth between 2 shapely points (interval 0 - 180)"""
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180
