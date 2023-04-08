#!/usr/bin/env python

import collections
import math
import operator
import warnings

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import shapely
from packaging.version import Version
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, polygonize
from tqdm.auto import tqdm

from .coins import COINS
from .shape import CircularCompactness

__all__ = [
    "preprocess",
    "remove_false_nodes",
    "CheckTessellationInput",
    "close_gaps",
    "extend_lines",
    "roundabout_simplification",
]

GPD_10 = Version(gpd.__version__) >= Version("0.10")
GPD_09 = Version(gpd.__version__) >= Version("0.9")


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
    if GPD_10:
        blg = blg.explode(ignore_index=True)
    else:
        blg = blg.explode()
        blg.reset_index(drop=True, inplace=True)
    for loop in range(0, loops):
        print("Loop", loop + 1, f"out of {loops}.") if verbose else None
        blg.reset_index(inplace=True, drop=True)
        blg["mm_uid"] = range(len(blg))
        sw = libpysal.weights.contiguity.Rook.from_dataframe(blg, silence_warnings=True)
        blg["neighbors"] = sw.neighbors.values()
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
            if size and row.geometry.area < size:
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
                            blg.at[n, blg.geometry.name]
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
            if compactness and row.circu < compactness:
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
                            blg.at[n, blg.geometry.name]
                        ).length
                    maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                    uid = blg.loc[maximal].mm_uid
                    if uid in join:
                        existing = join[uid]
                        existing.append(row.mm_uid)
                        join[uid] = existing
                    else:
                        join[uid] = [row.mm_uid]

            if islands and row.n_count == 1:
                shared = row.geometry.intersection(
                    blg.at[row.neighbors[0], blg.geometry.name]
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
                blg.loc[
                    blg.loc[blg["mm_uid"] == key].index[0], blg.geometry.name
                ] = new_geom

        blg.drop(delete, inplace=True)
    return blg[buildings.columns]


def remove_false_nodes(gdf):
    """
    Clean topology of existing LineString geometry by removal of nodes of degree 2.

    Returns the original gdf if there's no node of degree 2.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries, array of shapely geometries
        (Multi)LineString data of street network

    Returns
    -------
    gdf : GeoDataFrame, GeoSeries

    See also
    --------
    momepy.extend_lines
    momepy.close_gaps
    """
    if isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # explode to avoid MultiLineStrings
        # reset index due to the bug in GeoPandas explode
        if GPD_10:
            df = gdf.reset_index(drop=True).explode(ignore_index=True)
        else:
            df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

        # get underlying shapely geometry
        geom = df.geometry.array
    else:
        geom = gdf
        df = gpd.GeoSeries(gdf)

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = shapely.STRtree(geom)
    inp, res = tree.query(points, predicate="intersects")
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
                if a[0] in components:
                    i = components[a[0]]
                elif a[1] in components:
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
            new.append(shapely.line_merge(shapely.union_all(geom[keys])))

        # remove incorrect geometries and append fixed versions
        df = df.drop(merge)
        if GPD_10:
            final = gpd.GeoSeries(new).explode(ignore_index=True)
        else:
            final = gpd.GeoSeries(new).explode().reset_index(drop=True)
        if isinstance(gdf, gpd.GeoDataFrame):
            return pd.concat(
                [
                    df,
                    gpd.GeoDataFrame(
                        {df.geometry.name: final}, geometry=df.geometry.name
                    ),
                ],
                ignore_index=True,
            )
        return pd.concat([df, final], ignore_index=True)

    # if there's nothing to fix, return the original dataframe
    return gdf


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

    warnings.filterwarnings("ignore", "GeoSeries.isna", UserWarning)

    def __init__(self, gdf, shrink=0.4, collapse=True, split=True, overlap=True):
        data = gdf[~gdf.is_empty]

        if split:
            types = data.geom_type

        shrink = data.buffer(-shrink) if shrink != 0 else data

        if collapse:
            emptycheck = shrink.is_empty
            self.collapse = gdf[emptycheck]
            collapsed = len(self.collapse)
        else:
            collapsed = "NA"

        if split:
            type_check = shrink.geom_type != types
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
            type_filter = gpd.GeoSeries(intersection).geom_type == "Polygon"
            empty_filter = intersection.is_empty
            overlapping = od_matrix.reset_index(drop=True)[empty_filter ^ type_filter]
            over_rows = sorted(
                pd.concat([overlapping.origin, overlapping.dest]).unique()
            )

            self.overlap = gdf.iloc[over_rows]
            overlapping_c = len(self.overlap)
        else:
            overlapping_c = "NA"

        print(
            f"Collapsed features  : {collapsed}\n"
            f"Split features      : {split_count}\n"
            f"Overlapping features: {overlapping_c}"
        )


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

    See also
    --------
    momepy.extend_lines
    momepy.remove_false_nodes

    """
    geom = gdf.geometry.array
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    buffered = shapely.buffer(points, tolerance / 2)

    dissolved = shapely.union_all(buffered)

    exploded = [
        shapely.get_geometry(dissolved, i)
        for i in range(shapely.get_num_geometries(dissolved))
    ]

    centroids = shapely.centroid(exploded)

    snapped = shapely.snap(geom, shapely.union_all(centroids), tolerance)

    return gpd.GeoSeries(snapped, crs=gdf.crs)


def extend_lines(gdf, tolerance, target=None, barrier=None, extension=0):
    """Extends lines from gdf to itself or target within a set tolerance

    Extends unjoined ends of LineString segments to join with other segments or
    target. If ``target`` is passed, extend lines to target. Otherwise extend
    lines to itself.

    If ``barrier`` is passed, each extended line is checked for intersection
    with ``barrier``. If they intersect, extended line is not returned. This
    can be useful if you don't want to extend street network segments through
    buildings.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing LineString geometry
    tolerance : float
        tolerance in snapping (by how much could be each segment
        extended).
    target : GeoDataFrame, GeoSeries
        target geometry to which ``gdf`` gets extended. Has to be
        (Multi)LineString geometry.
    barrier : GeoDataFrame, GeoSeries
        extended line is not used if it intersects barrier
    extension : float
        by how much to extend line beyond the snapped geometry. Useful
        when creating enclosures to avoid floating point imprecision.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame of with extended geometry

    See also
    --------
    momepy.close_gaps
    momepy.remove_false_nodes

    """
    # explode to avoid MultiLineStrings
    # reset index due to the bug in GeoPandas explode
    if GPD_10:
        df = gdf.reset_index(drop=True).explode(ignore_index=True)
    else:
        df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

    if target is None:
        target = df
        itself = True
    else:
        itself = False

    # get underlying shapely geometry
    geom = df.geometry.array

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = shapely.STRtree(geom)
    inp, res = tree.query(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:
        l_coords = shapely.get_coordinates(geom[line])

        start = shapely.points(l_coords[0])
        end = shapely.points(l_coords[-1])

        first = list(tree.query(start, predicate="intersects"))
        second = list(tree.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        t = target if not itself else target.drop(line)

        if first and not second:
            snapped = _extend_line(l_coords, t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    shapely.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and second:
            snapped = _extend_line(np.flip(l_coords, axis=0), t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    shapely.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and not second:
            one_side = _extend_line(l_coords, t, tolerance)
            one_side_e = _extend_line(one_side, t, extension, snap=False)
            snapped = _extend_line(np.flip(one_side_e, axis=0), t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    shapely.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )

    df.iloc[ends, df.columns.get_loc(df.geometry.name)] = new_geoms
    return df


def _extend_line(coords, target, tolerance, snap=True):
    """
    Extends a line geometry to snap on the target within a tolerance.
    """
    if snap:
        extrapolation = _get_extrapolated_line(
            coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
            tolerance,
        )
        int_idx = target.sindex.query(extrapolation, predicate="intersects")
        intersection = shapely.intersection(
            target.iloc[int_idx].geometry.array, extrapolation
        )
        if intersection.size > 0:
            if len(intersection) > 1:
                distances = {}
                ix = 0
                for p in intersection:
                    distance = shapely.distance(p, shapely.points(coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = shapely.get_coordinates(intersection[minimal])

            else:
                new_point_coords = shapely.get_coordinates(intersection[0])
            coo = np.append(coords, new_point_coords)
            new = np.reshape(coo, (int(len(coo) / 2), 2))

            return new
        return coords

    extrapolation = _get_extrapolated_line(
        coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
        tolerance,
        point=True,
    )
    return np.vstack([coords, extrapolation])


def _get_extrapolated_line(coords, tolerance, point=False):
    """
    Creates a shapely line extrapoled in p1->p2 direction.
    """
    p1 = coords[:2]
    p2 = coords[2:]
    a = p2

    # defining new point based on the vector between existing points
    if p1[0] >= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
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
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
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
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
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
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    if point:
        return b
    return shapely.linestrings([a, b])


def _polygonize_ifnone(edges, polys):
    if polys is None:
        pre_polys = polygonize(edges.geometry)
        polys = gpd.GeoDataFrame(geometry=list(pre_polys), crs=edges.crs)
    return polys


def _selecting_rabs_from_poly(
    gdf,
    area_col="area",
    circom_threshold=0.7,
    area_threshold=0.85,
    include_adjacent=True,
    diameter_factor=1.5,
):
    """
    From a GeoDataFrame of polygons, returns a GDF of polygons that are
    above the Circular Compactness threshold.

    Return
    ________
    GeoDataFrames : roundabouts and adjacent polygons
    """
    # calculate parameters
    if area_col == "area":
        gdf.loc[:, area_col] = gdf.geometry.area
    circom_serie = CircularCompactness(gdf, area_col).series
    # selecting roundabout polygons based on compactness
    mask = circom_serie > circom_threshold
    rab = gdf[mask]
    # exclude those above the area threshold
    area_threshold_val = gdf.area.quantile(area_threshold)
    rab = rab[rab[area_col] < area_threshold_val]

    if include_adjacent is True:
        bounds = rab.geometry.bounds
        rab = pd.concat([rab, bounds], axis=1)
        rab["deltax"] = rab.maxx - rab.minx
        rab["deltay"] = rab.maxy - rab.miny
        rab["rab_diameter"] = rab[["deltax", "deltay"]].max(axis=1)

        # selecting the adjacent areas that are of smaller than itself
        if GPD_10:
            rab_adj = gpd.sjoin(gdf, rab, predicate="intersects")
        else:
            rab_adj = gpd.sjoin(gdf, rab, op="intersects")

        area_right = area_col + "_right"
        area_left = area_col + "_left"
        area_mask = rab_adj[area_right] >= rab_adj[area_left]
        rab_adj = rab_adj[area_mask]
        rab_adj.index.name = "index"

        # adding a hausdorff_distance threshold
        rab_adj["hdist"] = 0
        # TODO: (should be a way to vectorize)
        for i, group in rab_adj.groupby("index_right"):
            for g in group.itertuples():
                hdist = g.geometry.hausdorff_distance(rab.loc[i].geometry)
                rab_adj.loc[g.Index, "hdist"] = hdist

        rab_plus = rab_adj[rab_adj.hdist < (rab_adj.rab_diameter * diameter_factor)]

    else:
        rab["index_right"] = rab.index
        rab_plus = rab

    # only keeping relevant fields
    geom_col = rab_plus.geometry.name
    rab_plus = rab_plus[[geom_col, "index_right"]]

    return rab_plus


def _rabs_center_points(gdf, center_type="centroid"):
    """
    From a selection of roundabouts, returns an aggregated GeoDataFrame
    per roundabout with extra column with center_type.
    """
    # temporary DataFrame where geometry is the array of shapely geometries
    # Hack until shapely 2.0 is out.
    # TODO: replace shapely with shapely 2.0
    tmp = pd.DataFrame(gdf.copy())  # creating a copy avoids warnings
    tmp["geometry"] = tmp.geometry.array

    shapely_geoms = (
        tmp.groupby("index_right")
        .geometry.apply(shapely.multipolygons)
        .rename("geometry")
    )
    shapely_geoms = shapely.make_valid(shapely_geoms)

    rab_multipolygons = gpd.GeoDataFrame(shapely_geoms, crs=gdf.crs)
    # make_valid is transforming the multipolygons into geometry collections because of
    # shared edges

    if center_type == "centroid":
        # geometry centroid of the actual circle
        rab_multipolygons["center_pt"] = gdf[
            gdf.index == gdf.index_right
        ].geometry.centroid

    elif center_type == "mean":
        coords, idxs = shapely.get_coordinates(shapely_geoms, return_index=True)
        means = {}
        for i in np.unique(idxs):
            tmps = coords[idxs == i]
            target_idx = rab_multipolygons.index[i]
            means[target_idx] = Point(tmps.mean(axis=0))

        rab_multipolygons["center_pt"] = gpd.GeoSeries(means, crs=gdf.crs)

    # centerpoint of minimum_bounding_circle
    # TODO
    # minimun_bounding_circle() should be available in Shapely 2.0.

    return rab_multipolygons


def _coins_filtering_many_incoming(incoming_many, angle_threshold=0):
    """
    Used only for the cases when more than one incoming line touches the
    roundabout.
    """
    idx_out_many_incoming = []
    # For each new connection, evaluate COINS and select the group from which the new
    # line belongs
    # TODO ideally use the groupby object on line_wkt used earlier
    for _g, x in incoming_many.groupby("line_wkt"):
        gs = gpd.GeoSeries(pd.concat([x.geometry, x.line]), crs=incoming_many.crs)
        gdf = gpd.GeoDataFrame(geometry=gs)
        gdf = gdf.drop_duplicates()

        coins = COINS(gdf, angle_threshold=angle_threshold)
        group_series = coins.stroke_attribute()
        gdf["coins_group"] = group_series
        # selecting the incoming and its extension
        coins_group_filter = gdf.groupby("coins_group").count() == 1
        f = gdf.coins_group.map(coins_group_filter.geometry)
        idxs_remove = gdf[f].index
        idx_out_many_incoming.extend(idxs_remove)

    incoming_many_reduced = incoming_many.drop(idx_out_many_incoming, axis=0)

    return incoming_many_reduced


def _selecting_incoming_lines(rab_multipolygons, edges, angle_threshold=0):
    """Selecting only the lines that are touching but not covered by
    the ``rab_plus``.
    If more than one LineString is incoming to ``rab_plus``, COINS algorithm
    is used to select the line to be extended further.
    """
    # selecting the lines that are touching but not covered by
    if GPD_10:
        touching = gpd.sjoin(edges, rab_multipolygons, predicate="touches")
        edges_idx, rabs_idx = rab_multipolygons.sindex.query_bulk(
            edges.geometry, predicate="covered_by"
        )
    else:
        touching = gpd.sjoin(edges, rab_multipolygons, op="touches")
        edges_idx, rabs_idx = rab_multipolygons.sindex.query_bulk(
            edges.geometry, op="covered_by"
        )
    idx_drop = edges.index.take(edges_idx)
    touching_idx = touching.index
    ls = list(set(touching_idx) - set(idx_drop))

    incoming = touching.loc[ls]

    # figuring out which ends of incoming edges need to be connected to the center_pt
    incoming["first_pt"] = incoming.geometry.apply(lambda x: Point(x.coords[0]))
    incoming["dist_first_pt"] = incoming.center_pt.distance(incoming.first_pt)
    incoming["last_pt"] = incoming.geometry.apply(lambda x: Point(x.coords[-1]))
    incoming["dist_last_pt"] = incoming.center_pt.distance(incoming.last_pt)
    lines = []
    for _i, row in incoming.iterrows():
        if row.dist_first_pt < row.dist_last_pt:
            lines.append(LineString([row.first_pt, row.center_pt]))
        else:
            lines.append(LineString([row.last_pt, row.center_pt]))
    incoming["line"] = gpd.GeoSeries(lines, index=incoming.index, crs=edges.crs)

    # checking if there are more than one incoming lines arriving to the same point
    # which would create several new lines
    incoming["line_wkt"] = incoming.line.to_wkt()
    grouped_lines = incoming.groupby(["line_wkt"])["line_wkt"]
    count_s = grouped_lines.count()

    # separating the incoming roads that come on their own to those that come in groups
    filter_count_one = pd.DataFrame(count_s[count_s == 1])
    filter_count_many = pd.DataFrame(count_s[count_s > 1])
    incoming_ones = pd.merge(
        incoming, filter_count_one, left_on="line_wkt", right_index=True, how="inner"
    )
    incoming_many = pd.merge(
        incoming, filter_count_many, left_on="line_wkt", right_index=True, how="inner"
    )
    incoming_many_reduced = _coins_filtering_many_incoming(
        incoming_many, angle_threshold=angle_threshold
    )

    incoming_all = gpd.GeoDataFrame(
        pd.concat([incoming_ones, incoming_many_reduced]), crs=edges.crs
    )

    return incoming_all, idx_drop


def _ext_lines_to_center(edges, incoming_all, idx_out):
    """
    Extends the LineStrings geometries to the centerpoint defined by
    _rabs_center_points. Also deletes the lines that originally defined the roundabout.
    Creates a new column labled with the 'rab' number.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with updated geometry
    """

    incoming_all["geometry"] = incoming_all.apply(
        lambda row: linemerge([row.geometry, row.line]), axis=1
    )
    new_edges = edges.drop(idx_out, axis=0)

    # creating a unique group label for returned gdf
    _, inv = np.unique(incoming_all.index_right, return_inverse=True)
    incoming_label = pd.Series(inv, index=incoming_all.index)
    incoming_label = incoming_label[~incoming_label.index.duplicated(keep="first")]

    # maintaining the same gdf shape as the original
    incoming_all = incoming_all[edges.columns]
    new_edges = pd.concat([new_edges, incoming_all])

    # adding a new column to match
    new_edges["simplification_group"] = incoming_label.astype("Int64")

    return new_edges


def roundabout_simplification(
    edges,
    polys=None,
    area_col="area",
    circom_threshold=0.7,
    area_threshold=0.85,
    include_adjacent=True,
    diameter_factor=1.5,
    center_type="centroid",
    angle_threshold=0,
):
    """
    Selects the roundabouts from ``polys`` to create a center point to merge all
    incoming edges. If None is passed, the function will perform shapely polygonization.

    All ``edges`` attributes are preserved and roundabouts are deleted.
    Note that some attributes, like length, may no longer reflect the reality of newly
    constructed geometry.

    If ``include_adjacent`` is True, adjacent polygons to the actual roundabout are
    also selected for simplification if two conditions are met:
        - the area of adjacent polygons is less than the actual roundabout
        - adjacent polygons do not extend beyond a factor of the diameter of the actual
        roundabout.
        This uses hausdorff_distance algorithm.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing LineString geometry of urban network
    polys : GeoDataFrame
        GeoDataFrame containing Polygon geometry derived from polygonyzing
        ``edges`` GeoDataFrame.
    area_col : string
        Column name containing area values if ``polys`` GeoDataFrame contains such
        information. Otherwise, it will
    circom_threshold : float (default 0.7)
        Circular compactness threshold to select roundabouts from ``polys``
        GeoDataFrame.
        Polygons with a higher or equal threshold value will be considered for
        simplification.
    area_threshold : float (default 0.85)
        Percentile threshold value from the area of ``polys`` to leave as input
        geometry.
        Polygons with a higher or equal threshold will be considered as urban blocks
        not considered
        for simplification.
    include_adjacent : boolean (default True)
        Adjacent polygons to be considered also as part of the simplification.
    diameter_factor : float (default 1.5)
        The factor to be applied to the diameter of each roundabout that determines
        how far an adjacent polygon can stretch until it is no longer considered part
        of the overall roundabout group. Only applyies when include_adjacent = True.
    center_type : string (default 'centroid')
        Method to use for converging the incoming LineStrings.
        Current list of options available : 'centroid', 'mean'.
        - 'centroid': selects the centroid of the actual roundabout (ignoring adjacent
        geometries)
        - 'mean': calculates the mean coordinates from the points of polygons (including
         adjacent geometries)
    angle_threshold : int, float (default 0)
        The angle threshold for the COINS algorithm. Only used when multiple incoming
        LineStrings
        arrive at the same Point to the roundabout or to the adjacent polygons if set
        as True.
        eg. when two 'edges' touch the roundabout at the same point, COINS algorithm
        will evaluate which of those
        incoming lines should be extended according to their deflection angle.
        Segments will only be considered a part of the same street if the deflection
        angle
        is above the threshold.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an updated geometry and an additional
        column labeling modified edges.
    """
    if not GPD_09:
        raise ImportError(
            "`roundabout_simplification` requires geopandas 0.9.0 or newer. "
            f"Your current version is {gpd.__version__}."
        )

    if len(edges[edges.geom_type != "LineString"]) > 0:
        raise TypeError(
            "Only LineString geometries are allowed. "
            "Try using the `explode()` method to explode MultiLineStrings."
        )

    polys = _polygonize_ifnone(edges, polys)
    rab = _selecting_rabs_from_poly(
        polys,
        area_col=area_col,
        circom_threshold=circom_threshold,
        area_threshold=area_threshold,
        include_adjacent=include_adjacent,
        diameter_factor=diameter_factor,
    )
    rab_multipolygons = _rabs_center_points(rab, center_type=center_type)
    incoming_all, idx_drop = _selecting_incoming_lines(
        rab_multipolygons, edges, angle_threshold=angle_threshold
    )
    output = _ext_lines_to_center(edges, incoming_all, idx_drop)

    return output
