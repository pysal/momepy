#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math
import operator

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pygeos
import shapely
from shapely.geometry import LineString, Point
from tqdm import tqdm

from .shape import CircularCompactness

__all__ = [
    "preprocess",
    "network_false_nodes",
    "remove_false_nodes",
    "snap_street_network_edge",
    "CheckTessellationInput",
    "close_gaps",
]


def preprocess(
    buildings, size=30, compactness=0.2, islands=True, loops=2, verbose=True
):
    """
    Preprocesses building geometry to eliminate additional structures being single
    features.

    Certain data providers (e.g. Ordnance Survey in GB) do not provide building geometry
    as one feature, but divided into different features depending their level (if they
    are on ground floor or not - passages, overhangs). Ideally, these features should
    share one building ID on which they could be dissolved. If this is not the case,
    series of steps needs to be done to minimize errors in morphological analysis.

    This script attempts to preprocess such geometry based on several condidions:
    If feature area is smaller than set size it will be a) deleted if it does not
    touch any other feature; b) will be joined to feature with which it shares the
    longest boundary. If feature is fully within other feature, these will be joined.
    If feature's circular compactness (:py:class:`momepy.CircularCompactness`)
    is < 0.2, it will be joined to feature with which it shares the longest boundary.
    Function does multiple loops through.


    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        geopandas.GeoDataFrame containing building layer
    size : float (default 30)
        maximum area of feature to be considered as additional structure. Set to
        None if not wanted.
    compactness : float (default .2)
        if set, function will resolve additional structures identified based on
        their circular compactness.
    islands : bool (default True)
        if True, function will resolve additional structures which are fully within
        other structures (share 100% of exterior boundary).
    loops : int (default 2)
        number of loops
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing preprocessed geometry
    """
    blg = buildings.copy()
    blg = blg.explode()
    blg.reset_index(drop=True, inplace=True)
    for loop in range(0, loops):
        print("Loop", loop + 1, f"out of {loops}.") if verbose else None
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
            blg.itertuples(),
            total=blg.shape[0],
            desc="Identifying changes",
            disable=not verbose,
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
                if row.circu < compactness:
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

        for key in tqdm(
            join, total=len(join), desc="Changing geometry", disable=not verbose
        ):
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


def remove_false_nodes(gdf):
    """
    Clean topology of existing LineString geometry by removal of nodes of degree 2.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries, array of pygeos geometries
        (Multi)LineString data of street network

    Returns
    -------
    gdf : GeoDataFrame, GeoSeries
    """
    if isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # explode to avoid MultiLineStrings
        # double reset index due to the bug in GeoPandas explode
        df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

        # get underlying pygeos geometry
        geom = df.geometry.values.data
    else:
        geom = gdf

    # extract array of coordinates and number per geometry
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = pygeos.STRtree(geom)
    inp, res = tree.query_bulk(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    merge = res[np.isin(inp, unique[counts == 2])]

    if len(merge) > 0:
        # filter duplications and create a dictionary with indication of components to
        # be merged together
        dups = [item for item, count in collections.Counter(merge).items() if count > 1]
        split = np.split(merge, len(merge) / 2)
        components = {}
        for i, a in enumerate(split):
            if a[0] in dups or a[1] in dups:
                if a[0] in components.keys():
                    i = components[a[0]]
                elif a[1] in components.keys():
                    i = components[a[1]]
            components[a[0]] = i
            components[a[1]] = i

        # iterate through components and create new geometries
        new = []
        for c in set(components.values()):
            keys = []
            for item in components.items():
                if item[1] == c:
                    keys.append(item[0])
            new.append(pygeos.line_merge(pygeos.union_all(geom[keys])))

        # remove incorrect geometries and append fixed versions
        df = df.drop(merge)
        final = gpd.GeoSeries(new).explode().reset_index(drop=True)
        if isinstance(gdf, gpd.GeoDataFrame):
            return df.append(
                gpd.GeoDataFrame({df.geometry.name: final}, geometry=df.geometry.name),
                ignore_index=True,
            )
        return df.append(final, ignore_index=True)


def snap_street_network_edge(
    edges,
    buildings,
    tolerance_street,
    tessellation=None,
    tolerance_edge=None,
    edge=None,
    verbose=True,
):
    """
    Fix street network before performing :class:`momepy.Blocks`.

    Extends unjoined ends of street segments to join with other segments or
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
        edge of area covered by morphological tessellation (same as ``limit`` in
        :py:class:`momepy.Tessellation`)
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

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
        Extends a line geometry withing GeoDataFrame to snap on itself withing 
        tolerance.
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

    # function extending line to closest object within set distance to edge defined by
    # tessellation
    def extend_line_edge(tolerance, idx):
        """
        Extends a line geometry withing GeoDataFrame to snap on the boundary of 
        tessellation withing tolerance.
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
    print("Building spatial index for network...") if verbose else None
    sindex = network.sindex
    print("Building spatial index for buildings...") if verbose else None
    bindex = buildings.sindex

    def _get_geometry():
        if edge is not None:
            return edge.boundary
        if tessellation is not None:
            print("Dissolving tesselation...") if verbose else None
            return tessellation.geometry.unary_union.boundary
        return None

    geometry = _get_geometry()

    # iterating over each street segment
    for idx, line in tqdm(
        network.geometry.iteritems(),
        total=network.shape[0],
        desc="Snapping",
        disable=not verbose,
    ):

        l_coords = list(line.coords)
        # network_w = network.drop(idx, axis=0)['geometry']  # ensure that it wont
        # intersect itself
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
            print("Something went wrong.") if verbose else None

    return network


class CheckTessellationInput:
    """
    Check input data for :class:`Tessellation` for potential errors.

    :class:`Tessellation` requires data of relatively high level of precision and there
    are three particular patterns causing issues.\n
    1. Features will collapse into empty polygon - these do not have tessellation
    cell in the end.\n
    2. Features will split into MultiPolygon - at some cases, features with narrow links
    between parts split into two during 'shrinking'. In most cases that is not an issue
    and resulting tessellation is correct anyway, but sometimes this result in a cell
    being MultiPolygon, which is not correct.\n
    3. Overlapping features - features which overlap even after 'shrinking' cause
    invalid tessellation geometry.\n

    :class:`CheckTessellationInput` will check for all of these. Overlapping features
    have to be fixed prior Tessellation. Features which will split will cause issues
    only sometimes, so
    should be checked and fixed if necessary. Features which will collapse could
    be ignored, but they will have to excluded from next steps of 
    tessellation-based analysis.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        GeoDataFrame containing objects to be used as ``gdf`` in :class:`Tessellation`
    shrink : float (default 0.4)
        distance for negative buffer
    collapse : bool (default True)
        check for features which would collapse to empty polygon
    split : bool (default True)
        check for features which would split into Multi-type
    overlap : bool (default True)
        check for overlapping features (after negative buffer)

    Attributes
    ----------
    collapse : GeoDataFrame or GeoSeries
        features which would collapse to empty polygon
    split : GeoDataFrame or GeoSeries
        features which would split into Multi-type
    overlap : GeoDataFrame or GeoSeries
        overlapping features (after negative buffer)


    Examples
    --------
    >>> check = CheckTessellationData(df)
    Collapsed features  : 3157
    Split features      : 519
    Overlapping features: 22
    """

    import warnings

    warnings.filterwarnings("ignore", "GeoSeries.isna", UserWarning)

    def __init__(self, gdf, shrink=0.4, collapse=True, split=True, overlap=True):
        data = gdf[~gdf.is_empty]

        if split:
            types = data.type

        if shrink != 0:
            shrink = data.buffer(-shrink)
        else:
            shrink = data

        if collapse:
            emptycheck = shrink.is_empty
            self.collapse = gdf[emptycheck]
            collapsed = len(self.collapse)
        else:
            collapsed = "NA"

        if split:
            type_check = shrink.type != types
            self.split = gdf[type_check]
            split_count = len(self.split)
        else:
            split_count = "NA"

        if overlap:
            shrink = shrink.reset_index(drop=True)
            shrink = shrink[~(shrink.is_empty | shrink.geometry.isna())]
            sindex = shrink.sindex
            hits = shrink.bounds.apply(
                lambda row: list(sindex.intersection(row)), axis=1
            )
            od_matrix = pd.DataFrame(
                {
                    "origin": np.repeat(hits.index, hits.apply(len)),
                    "dest": np.concatenate(hits.values),
                }
            )
            od_matrix = od_matrix[od_matrix.origin != od_matrix.dest]
            duplicated = pd.DataFrame(np.sort(od_matrix, axis=1)).duplicated()
            od_matrix = od_matrix.reset_index(drop=True)[~duplicated]
            od_matrix = od_matrix.join(
                shrink.geometry.rename("o_geom"), on="origin"
            ).join(shrink.geometry.rename("d_geom"), on="dest")
            intersection = od_matrix.o_geom.values.intersection(od_matrix.d_geom.values)
            type_filter = gpd.GeoSeries(intersection).type == "Polygon"
            empty_filter = intersection.is_empty
            overlapping = od_matrix.reset_index(drop=True)[empty_filter ^ type_filter]
            over_rows = sorted(overlapping.origin.append(overlapping.dest).unique())

            self.overlap = gdf.iloc[over_rows]
            overlapping_c = len(self.overlap)
        else:
            overlapping_c = "NA"

        print(
            "Collapsed features  : {0}\n"
            "Split features      : {1}\n"
            "Overlapping features: {2}".format(collapsed, split_count, overlapping_c)
        )


def network_false_nodes(gdf, tolerance=0.1, precision=3, verbose=True):
    """
    Check topology of street network and eliminate nodes of degree 2 by joining
    affected edges.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        GeoDataFrame  or GeoSeries containing edge representation of street network.
    tolerance : float
        nodes within a tolerance are seen as identical (floating point precision fix)
    precision : int
        rounding parameter in estimating uniqueness of two points based on their
        coordinates
    verbose : bool
        if True, shows progress bars in loops

    Returns
    -------
    gdf : GeoDataFrame, GeoSeries
    """
    import warnings

    warnings.warn(
        "network_false_nodes() is deprecated and will be removed in momepy 0.5.0. "
        "Use remove_false_nodes() instead.",
        FutureWarning,
    )

    if not isinstance(gdf, (gpd.GeoSeries, gpd.GeoDataFrame)):
        raise TypeError(
            "'gdf' should be GeoDataFrame or GeoSeries, got {}".format(type(gdf))
        )
    # double reset_index due to geopandas issue in explode
    streets = gdf.reset_index(drop=True).explode().reset_index(drop=True)
    if isinstance(streets, gpd.GeoDataFrame):
        series = False
    elif isinstance(streets, gpd.GeoSeries):
        series = True

    sindex = streets.sindex

    false_xy = []
    for line in tqdm(
        streets.geometry,
        total=streets.shape[0],
        desc="Identifying false points",
        disable=not verbose,
    ):
        l_coords = list(line.coords)
        start = Point(l_coords[0]).buffer(tolerance)
        end = Point(l_coords[-1]).buffer(tolerance)

        real_first_matches = sindex.query(start, predicate="intersects")
        real_second_matches = sindex.query(end, predicate="intersects")

        if len(real_first_matches) == 2:
            false_xy.append(
                (round(l_coords[0][0], precision), round(l_coords[0][1], precision))
            )
        if len(real_second_matches) == 2:
            false_xy.append(
                (round(l_coords[-1][0], precision), round(l_coords[-1][1], precision))
            )

    false_unique = list(set(false_xy))
    x, y = zip(*false_unique)
    points = gpd.points_from_xy(x, y).buffer(tolerance)

    geoms = streets
    idx = max(geoms.index) + 1

    for x, y, point in tqdm(
        zip(x, y, points), desc="Merging segments", total=len(x), disable=not verbose
    ):

        predic = geoms.sindex.query(point, predicate="intersects")
        matches = geoms.iloc[predic].geometry

        try:
            snap = shapely.ops.snap(matches.iloc[0], matches.iloc[1], tolerance,)
            multiline = snap.union(matches.iloc[1])
            linestring = shapely.ops.linemerge(multiline)
            if linestring.type == "LineString":
                if series:
                    geoms.loc[idx] = linestring
                else:
                    geoms.loc[idx, "geometry"] = linestring
                idx += 1
            elif linestring.type == "MultiLineString":
                for g in linestring.geoms:
                    if series:
                        geoms.loc[idx] = g
                    else:
                        geoms.loc[idx, "geometry"] = g
                    idx += 1

            geoms = geoms.drop(matches.index)
        except (IndexError, ValueError):

            warnings.warn(
                "An exception during merging occured. "
                f"Lines at point [{x}, {y}] were not merged."
            )

    streets = geoms.explode().reset_index(drop=True)
    if series:
        streets.crs = gdf.crs
        return streets
    return streets


def close_gaps(gdf, tolerance):
    """Close gaps in LineString geometry where it should be contiguous.

    Snaps both lines to a centroid of a gap in between.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        GeoDataFrame  or GeoSeries containing LineString representation of a network.
    tolerance : float
        nodes within a tolerance will be snapped together

    Returns
    -------
    GeoSeries

    """
    geom = gdf.geometry.values.data
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    buffered = pygeos.buffer(points, tolerance / 2)

    dissolved = pygeos.union_all(buffered)

    exploded = [
        pygeos.get_geometry(dissolved, i)
        for i in range(pygeos.get_num_geometries(dissolved))
    ]

    centroids = pygeos.centroid(exploded)

    snapped = pygeos.snap(geom, pygeos.union_all(centroids), tolerance)

    return gpd.GeoSeries(snapped, crs=gdf.crs)
