#!/usr/bin/env python

import math
import operator
import warnings
from copy import deepcopy

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from packaging.version import Version
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, polygonize, split
from tqdm.auto import tqdm

from .coins import COINS
from .graph import node_degree
from .shape import CircularCompactness
from .utils import gdf_to_nx, nx_to_gdf

__all__ = [
    "preprocess",
    "remove_false_nodes",
    "CheckTessellationInput",
    "close_gaps",
    "extend_lines",
    "roundabout_simplification",
    "consolidate_intersections",
    "FaceArtifacts",
]

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")


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
    blg = blg.explode(ignore_index=True)
    for loop in range(0, loops):
        print("Loop", loop + 1, f"out of {loops}.") if verbose else None
        blg.reset_index(inplace=True, drop=True)
        blg["mm_uid"] = range(len(blg))
        sw = libpysal.weights.contiguity.Rook.from_dataframe(
            blg, silence_warnings=True, use_index=False
        )
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
                    join.setdefault(uid, []).append(row.mm_uid)
                elif row.n_count > 1:
                    shares = {}
                    for n in row.neighbors:
                        shares[n] = row.geometry.intersection(
                            blg.at[n, blg.geometry.name]
                        ).length
                    maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                    uid = blg.loc[maximal].mm_uid
                    join.setdefault(uid, []).append(row.mm_uid)
                else:
                    delete.append(row.Index)
            if compactness and row.circu < compactness:
                if row.n_count == 1:
                    uid = blg.iloc[row.neighbors[0]].mm_uid
                    join.setdefault(uid, []).append(row.mm_uid)
                elif row.n_count > 1:
                    shares = {}
                    for n in row.neighbors:
                        shares[n] = row.geometry.intersection(
                            blg.at[n, blg.geometry.name]
                        ).length
                    maximal = max(shares.items(), key=operator.itemgetter(1))[0]
                    uid = blg.loc[maximal].mm_uid
                    join.setdefault(uid, []).append(row.mm_uid)

            if islands and row.n_count == 1:
                shared = row.geometry.intersection(
                    blg.at[row.neighbors[0], blg.geometry.name]
                ).length
                if shared == row.geometry.exterior.length:
                    uid = blg.iloc[row.neighbors[0]].mm_uid
                    join.setdefault(uid, []).append(row.mm_uid)

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
                new_geom = shapely.union_all(geoms)
                blg.loc[blg.loc[blg["mm_uid"] == key].index[0], blg.geometry.name] = (
                    new_geom
                )

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
    if isinstance(gdf, gpd.GeoDataFrame | gpd.GeoSeries):
        # explode to avoid MultiLineStrings
        # reset index due to the bug in GeoPandas explode
        df = gdf.reset_index(drop=True).explode(ignore_index=True)

        # get underlying shapely geometry
        geom = df.geometry.array
    else:
        geom = gdf
        df = gpd.GeoSeries(gdf)

    # extract array of coordinates and number per geometry
    start_points = shapely.get_point(geom, 0)
    end_points = shapely.get_point(geom, -1)

    points = shapely.points(
        np.unique(
            shapely.get_coordinates(np.concatenate([start_points, end_points])), axis=0
        )
    )

    # query LineString geometry to identify points intersecting 2 geometries
    tree = shapely.STRtree(geom)
    inp, res = tree.query(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    mask = np.isin(inp, unique[counts == 2])
    merge_res = res[mask]
    merge_inp = inp[mask]

    if len(merge_res):
        g = nx.Graph(list(zip(merge_inp * -1, merge_res, strict=True)))
        new_geoms = []
        for c in nx.connected_components(g):
            valid = [ix for ix in c if ix > -1]
            new_geoms.append(shapely.line_merge(shapely.union_all(geom[valid])))

        df = df.drop(merge_res)
        final = gpd.GeoSeries(new_geoms, crs=df.crs).explode(ignore_index=True)
        if isinstance(gdf, gpd.GeoDataFrame):
            combined = pd.concat(
                [
                    df,
                    gpd.GeoDataFrame(
                        {df.geometry.name: final}, geometry=df.geometry.name, crs=df.crs
                    ),
                ],
                ignore_index=True,
            )
        else:
            combined = pd.concat([df, final], ignore_index=True)

        # re-order closed loops
        fixed_loops = []
        fixed_index = []
        nodes = nx_to_gdf(
            node_degree(
                gdf_to_nx(
                    combined
                    if isinstance(combined, gpd.GeoDataFrame)
                    else combined.to_frame("geometry")
                )
            ),
            lines=False,
        )
        loops = combined[combined.is_ring]
        node_ix, loop_ix = loops.sindex.query(nodes.geometry, predicate="intersects")
        for ix in np.unique(loop_ix):
            loop_geom = loops.geometry.iloc[ix]
            target_nodes = nodes.geometry.iloc[node_ix[loop_ix == ix]]
            if len(target_nodes) == 2:
                node_coords = shapely.get_coordinates(target_nodes)
                coords = np.array(loop_geom.coords)
                new_start = (
                    node_coords[0]
                    if (node_coords[0] != coords[0]).all()
                    else node_coords[1]
                )
                new_start_idx = np.where(coords == new_start)[0][0]
                rolled_coords = np.roll(coords[:-1], -new_start_idx, axis=0)
                new_sequence = np.append(rolled_coords, rolled_coords[[0]], axis=0)
                fixed_loops.append(shapely.LineString(new_sequence))
                fixed_index.append(ix)
        fixed_loops = gpd.GeoSeries(fixed_loops, crs=df.crs).explode(ignore_index=True)

        if isinstance(gdf, gpd.GeoDataFrame):
            return pd.concat(
                [
                    combined.drop(loops.iloc[fixed_index].index),
                    gpd.GeoDataFrame(
                        {df.geometry.name: fixed_loops},
                        geometry=df.geometry.name,
                        crs=df.crs,
                    ),
                ],
                ignore_index=True,
            )
        else:
            return pd.concat(
                [combined.drop(loops.iloc[fixed_index].index), fixed_loops],
                ignore_index=True,
            )

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
    df = gdf.reset_index(drop=True).explode(ignore_index=True)

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
            new = np.reshape(coo, (len(coo) // 2, 2))

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
        rab_adj = gpd.sjoin(gdf, rab, predicate="intersects")

        area_right = area_col + "_right"
        area_left = area_col + "_left"
        area_mask = rab_adj[area_right] >= rab_adj[area_left]
        rab_adj = rab_adj[area_mask]
        rab_adj.index.name = "index"

        # adding a hausdorff_distance threshold
        rab_adj["hdist"] = 0.0
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
    touching = gpd.sjoin(edges, rab_multipolygons, predicate="touches")
    if GPD_GE_013:
        edges_idx, _ = rab_multipolygons.sindex.query(
            edges.geometry, predicate="covered_by"
        )
    else:
        edges_idx, _ = rab_multipolygons.sindex.query_bulk(
            edges.geometry, predicate="covered_by"
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
      roundabout. This uses hausdorff_distance algorithm.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame containing LineString geometry of urban network
    polys : GeoDataFrame
        GeoDataFrame containing Polygon geometry derived from polygonyzing ``edges``
        GeoDataFrame.
    area_col : string
        Column name containing area values if ``polys`` GeoDataFrame contains such
        information. Otherwise, it will
    circom_threshold : float (default 0.7)
        Circular compactness threshold to select roundabouts from ``polys``
        GeoDataFrame. Polygons with a higher or equal threshold value will be considered
        for simplification.
    area_threshold : float (default 0.85)
        Percentile threshold value from the area of ``polys`` to leave as input
        geometry. Polygons with a higher or equal threshold will be considered as urban
        blocks not considered for simplification.
    include_adjacent : boolean (default True)
        Adjacent polygons to be considered also as part of the simplification.
    diameter_factor : float (default 1.5)
        The factor to be applied to the diameter of each roundabout that determines how
        far an adjacent polygon can stretch until it is no longer considered part of the
        overall roundabout group. Only applyies when include_adjacent = True.
    center_type : string (default 'centroid')
        Method to use for converging the incoming LineStrings. Current list of options
        available : 'centroid', 'mean'. - 'centroid': selects the centroid of the actual
        roundabout (ignoring adjacent geometries) - 'mean': calculates the mean
        coordinates from the points of polygons (including adjacent geometries)
    angle_threshold : int, float (default 0)
        The angle threshold for the COINS algorithm. Only used when multiple incoming
        LineStrings arrive at the same Point to the roundabout or to the adjacent
        polygons if set as True. eg. when two 'edges' touch the roundabout at the same
        point, COINS algorithm will evaluate which of those incoming lines should be
        extended according to their deflection angle. Segments will only be considered a
        part of the same street if the deflection angle is above the threshold.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an updated geometry and an additional column labeling modified
        edges.
    """
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


def consolidate_intersections(
    graph,
    tolerance=30,
    rebuild_graph=True,
    rebuild_edges_method="spider",
    x_att="x",
    y_att="y",
    edge_from_att="from",
    edge_to_att="to",
):
    """
    Consolidate close street intersections into a single node, collapsing short edges.

    If rebuild_graph is True, new edges are drawn according to ``rebuild_edges_method``
    which is one of:

    1. Extension reconstruction:
        Edges are linearly extended from original endpoints until the new nodes. This
        method preserves most faithfully the network geometry but can result in
        overlapping geometry.
    2. Spider-web reconstruction:
        Edges are cropped within a buffer of the new endpoints and linearly extended
        from there. This method improves upon linear reconstruction by mantaining, when
        possible, network planarity.
    3. Euclidean reconstruction:
        Edges are ignored and new edges are built as straight lines between new origin
        and new destination. This method ignores geometry, but efficiently preserves
        adjacency.

    If ``rebuild_graph`` is False, graph is returned with consolidated nodes but without
    reconstructed edges i.e. graph is intentionally disconnected.

    Graph must be configured so that

    1. All nodes have attributes determining their x and y coordinates;
    2. All edges have attributes determining their origin, destination, and geometry.

    Parameters
    ----------
    graph : Networkx.Graph, Networkx.DiGraph, Networkx.MultiGraph, or
        Networkx.MultiDiGraph
    tolerance : float, default 30
        distance in network units below which nodes will be consolidated
    rebuild_graph : bool
    rebuild_edges_method : str
        'extend' or 'spider' or 'euclidean', ignored if rebuild_graph is False
    x_att : str
        node attribute with the valid x-coordinate
    y_att : str
        node attribute with the valid y-coordinate
    edge_from_att : str
        edge attribute with the valid origin node id
    edge_to_att : str
        edge attribute with the valid destination node id

    Returns
    -------
    Networkx.MultiGraph or Networkx.MultiDiGraph
        directionality inferred from input type

    """
    # Collect nodes and their data:
    nodes, nodes_dict = zip(*graph.nodes(data=True), strict=False)
    nodes_df = pd.DataFrame(nodes_dict, index=nodes)
    graph_crs = graph.graph.get("crs")

    # Create a graph without the edges above a certain length and clean it
    #  from isolated nodes (the unsimplifiable nodes):
    components_graph = deepcopy(graph)
    components_graph.remove_edges_from(
        [
            edge
            for edge in graph.edges(keys=True, data=True)
            if edge[-1]["length"] > tolerance
        ]
    )
    isolated_nodes_list = list(nx.isolates(components_graph))
    components_graph.remove_nodes_from(isolated_nodes_list)

    # The connected components of this graph are node clusters we must individually
    #  simplify. We collect them in a dataframe and retrieve node properties (x, y
    #  coords mainly) from the original graph.
    components = nx.connected_components(components_graph)
    components_dict = dict(enumerate(components, start=max(nodes) + 1))
    nodes_to_merge_dict = {
        node: cpt for cpt, nodes in components_dict.items() for node in nodes
    }
    new_nodes_df = pd.DataFrame.from_dict(
        nodes_to_merge_dict, orient="index", columns=["cluster"]
    )
    nodes_to_merge_df = pd.concat(
        [new_nodes_df, nodes_df[[x_att, y_att]]], axis=1, join="inner"
    )

    # The two node attributes we need for the clusters are the position of the cluster
    #  centroids. Those are obtained by averaging the x and y columns. We also add
    # . attribtues referring to the original node ids in every cluster:
    cluster_centroids_df = nodes_to_merge_df.groupby("cluster").mean()
    cluster_centroids_df["simplified"] = True
    cluster_centroids_df["original_node_ids"] = cluster_centroids_df.index.map(
        components_dict
    )
    cluster_geometries = gpd.points_from_xy(
        cluster_centroids_df[x_att], cluster_centroids_df[y_att]
    )
    cluster_gdf = gpd.GeoDataFrame(
        cluster_centroids_df, crs=graph_crs, geometry=cluster_geometries
    )
    cluster_nodes_list = list(cluster_gdf.to_dict("index").items())

    # Create a simplified graph object:
    simplified_graph = graph.copy()

    # Rebuild edges if necessary:
    if rebuild_graph:
        rebuild_edges_method = rebuild_edges_method.lower()
        simplified_graph.graph["approach"] = "primal"
        edges_gdf = nx_to_gdf(simplified_graph, points=False, lines=True)
        simplified_edges = _get_rebuilt_edges(
            edges_gdf,
            nodes_to_merge_dict,
            cluster_gdf,
            method=rebuild_edges_method,
            buffer=1.5 * tolerance,
            edge_from_att=edge_from_att,
            edge_to_att=edge_to_att,
        )

    # Replacing the collapsed nodes with centroids and adding edges:
    simplified_graph.remove_nodes_from(nodes_to_merge_df.index)
    simplified_graph.add_nodes_from(cluster_nodes_list)
    if rebuild_graph:
        simplified_graph.add_edges_from(simplified_edges)

    return simplified_graph


def _get_rebuilt_edges(
    edges_gdf,
    nodes_dict,
    cluster_gdf,
    method="spider",
    buffer=45,
    edge_from_att="from",
    edge_to_att="to",
):
    """
    Update origin and destination on network edges when original endpoints were replaced
    by a
      consolidated node cluster. New edges are drawn according to method which is one
      of:

    1. Extension reconstruction:
        Edges are linearly extended from original endpoints until the new nodes. This
        method preserves most faithfully the network geometry.
    2. Spider-web reconstruction:
        Edges are cropped within a buffer of the new endpoints and linearly extended
        from there. This method improves upon linear reconstruction by mantaining, when
        possible, network planarity.
    3. Euclidean reconstruction:
        Edges are ignored and new edges are built as straightlines between new origin
        and new destination. This method ignores geometry, but efficiently preserves
        adjacency.

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        GeoDataFrame containing LineString geometry and columns determining origin
        and destination node ids
    nodes_dict : dict
        Dictionary whose keys are node ids and values are the corresponding consolidated
        node cluster ids. Only consolidated nodes are in the dictionary.
    cluster_gdf : GeoDataFrame
        GeoDataFrame containing consolidated node ids.
    method: string
        'extension' or 'spider' or 'euclidean'
    buffer : float
        distance to buffer consolidated nodes in the Spider-web reconstruction
    edge_from_att : str
        edge attribute with the valid origin node id
    edge_to_att : str
        edge attribute with the valid destination node id

    Returns
    ----------
    List
        list of edges that should be added to the network. Edges are in the format
        (origin_id, destination_id, data), where data is inferred from edges_gdf

    """
    # Determine what endpoints were made into clusters:
    edges_gdf["origin_cluster"] = edges_gdf[edge_from_att].apply(
        lambda u: nodes_dict.get(u, -1)
    )
    edges_gdf["destination_cluster"] = edges_gdf[edge_to_att].apply(
        lambda v: nodes_dict.get(v, -1)
    )

    # Determine what edges need to be simplified (either between diff.
    #  clusters or self-loops in a cluster):
    edges_tosimplify_gdf = edges_gdf.query(
        "origin_cluster != destination_cluster or "
        f"(('{edge_to_att}' == '{edge_from_att}') and origin_cluster >= 0)"
    )

    # Determine the new point geometries (when exists):
    edges_tosimplify_gdf = edges_tosimplify_gdf.assign(
        new_origin_pt=edges_tosimplify_gdf.origin_cluster.map(
            cluster_gdf.geometry, None
        )
    )
    edges_tosimplify_gdf = edges_tosimplify_gdf.assign(
        new_destination_pt=edges_tosimplify_gdf.destination_cluster.map(
            cluster_gdf.geometry, None
        )
    )

    # Determine the new geometry according to the simplification method:
    if method == "extend":
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _extension_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )
    elif method == "euclidean":
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _euclidean_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )
    elif method == "spider":
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _spider_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt, buffer
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )
    else:
        msg = (
            f"Simplification '{method}' not recognized. See documentation for options."
        )
        raise ValueError(msg)

    # Rename and update the columns:
    cols_rename = {
        edge_from_att: "original_from",
        edge_to_att: "original_to",
        "origin_cluster": edge_from_att,
        "destination_cluster": edge_to_att,
        "geometry": "original_geometry",
    }
    new_edges_gdf = edges_simplified_gdf.rename(cols_rename, axis=1)

    cols_drop = ["new_origin_pt", "new_destination_pt"]
    new_edges_gdf = new_edges_gdf.drop(columns=cols_drop)

    new_edges_gdf = new_edges_gdf.set_geometry("new_geometry")
    new_edges_gdf.loc[:, "length"] = new_edges_gdf.length

    # Update the indices:
    new_edges_gdf.loc[:, edge_from_att] = new_edges_gdf[edge_from_att].where(
        new_edges_gdf[edge_from_att] >= 0, new_edges_gdf["original_from"]
    )
    new_edges_gdf.loc[:, edge_to_att] = new_edges_gdf[edge_to_att].where(
        new_edges_gdf[edge_to_att] >= 0, new_edges_gdf["original_to"]
    )

    # Get the edge list with (from, to, data):
    new_edges_list = list(
        zip(
            new_edges_gdf[edge_from_att],
            new_edges_gdf[edge_to_att],
            new_edges_gdf.iloc[:, 2:].to_dict("index").values(),
            strict=False,
        )
    )

    return new_edges_list


def _extension_simplification(geometry, new_origin, new_destination):
    """
    Extends edge geometry to new endpoints.

    If either new_origin or new_destination is None, maintains the
      respective current endpoint.

    Parameters
    ----------
    geometry : shapely.LineString
    new_origin : shapely.Point or None
    new_destination: shapely.Point or None

    Returns
    ----------
    shapely.LineString

    """
    # If we are dealing with a self-loop the line has no endpoints:
    if new_origin == new_destination:
        current_node = Point(geometry.coords[0])
        geometry = linemerge([LineString([new_origin, current_node]), geometry])
    # Assuming the line is not closed, we can find its endpoints:
    else:
        current_origin, current_destination = geometry.boundary.geoms
        if new_origin is not None:
            geometry = linemerge([LineString([new_origin, current_origin]), geometry])
        if new_destination is not None:
            geometry = linemerge(
                [geometry, LineString([current_destination, new_destination])]
            )
    return geometry


def _spider_simplification(geometry, new_origin, new_destination, buff=15):
    """
    Extends edge geometry to new endpoints via a "spider-web" method. Breaks
      current geometry within a buffer of the new endpoint and then extends
      it linearly. Useful to maintain planarity.

    If either new_origin or new_destination is None, maintains the
      respective current endpoint.

    Parameters
    ----------
    geometry : shapely.LineString
    new_origin : shapely.Point or None
    new_destination: shapely.Point or None
    buff : float
        distance from new endpoint to break current geometry

    Returns
    ----------
    shapely.LineString

    """
    # If we are dealing with a self-loop the line has no boundary
    # . and we just use the first coordinate:
    if new_origin == new_destination:
        current_node = Point(geometry.coords[0])
        geometry = linemerge([LineString([new_origin, current_node]), geometry])
    # Assuming the line is not closed, we can find its endpoints
    #  via the boundary attribute:
    else:
        current_origin, current_destination = geometry.boundary.geoms
        if new_origin is not None:
            # Create a buffer around the new origin:
            new_origin_buffer = new_origin.buffer(buff)
            # Use shapely.ops.split to break the edge where it
            #  intersects the buffer:
            geometry_split_by_buffer_list = list(
                split(geometry, new_origin_buffer).geoms
            )
            # If only one geometry results, edge does not intersect
            #  buffer and line should connect new origin to old origin
            if len(geometry_split_by_buffer_list) == 1:
                geometry_split_by_buffer = geometry_split_by_buffer_list[0]
                splitting_point = current_origin
            # If more than one geometry, merge all linestrings
            #  but the first and get their origin
            else:
                geometry_split_by_buffer = linemerge(geometry_split_by_buffer_list[1:])
                splitting_point = geometry_split_by_buffer.boundary.geoms[0]
            # Merge this into new geometry:
            additional_line = [LineString([new_origin, splitting_point])]
            # Consider MultiLineStrings separately:
            if geometry_split_by_buffer.geom_type == "MultiLineString":
                geometry = linemerge(
                    additional_line + list(geometry_split_by_buffer.geoms)
                )
            else:
                geometry = linemerge(additional_line + [geometry_split_by_buffer])

        if new_destination is not None:
            # Create a buffer around the new destination:
            new_destination_buffer = new_destination.buffer(buff)
            # Use shapely.ops.split to break the edge where it
            #  intersects the buffer:
            geometry_split_by_buffer_list = list(
                split(geometry, new_destination_buffer).geoms
            )
            # If only one geometry results, edge does not intersect
            # . buffer and line should connect new destination to old destination
            if len(geometry_split_by_buffer_list) == 1:
                geometry_split_by_buffer = geometry_split_by_buffer_list[0]
                splitting_point = current_destination
            # If more than one geometry, merge all linestrings
            #  but the last and get their destination
            else:
                geometry_split_by_buffer = linemerge(geometry_split_by_buffer_list[:-1])
                splitting_point = geometry_split_by_buffer.boundary.geoms[1]
            # Merge this into new geometry:
            additional_line = [LineString([splitting_point, new_destination])]
            # Consider MultiLineStrings separately:
            if geometry_split_by_buffer.geom_type == "MultiLineString":
                geometry = linemerge(
                    list(geometry_split_by_buffer.geoms) + additional_line
                )
            else:
                geometry = linemerge([geometry_split_by_buffer] + additional_line)

    return geometry


def _euclidean_simplification(geometry, new_origin, new_destination):
    """
    Rebuilds edge geometry to new endpoints. Ignores current geometry
      and traces a straight line between new endpoints.

    If either new_origin or new_destination is None, maintains the
      respective current endpoint.

    Parameters
    ----------
    geometry : shapely.LineString
    new_origin : shapely.Point or None
    new_destination : shapely.Point or None

    Returns
    ----------
    shapely.LineString

    """
    # If we are dealing with a self-loop, geometry will be null!
    if new_origin == new_destination:
        geometry = None
    # Assuming the line is not closed, we can find its endpoints:
    else:
        current_origin, current_destination = geometry.boundary.geoms
        if new_origin is not None:
            if new_destination is not None:
                geometry = LineString([new_origin, new_destination])
            else:
                geometry = LineString([new_origin, current_destination])
        else:
            if new_destination is not None:
                geometry = LineString([current_origin, new_destination])
    return geometry


class FaceArtifacts:
    """Identify face artifacts in street networks

    For a given street network composed of transportation-oriented geometry containing
    features representing things like roundabouts, dual carriegaways and complex
    intersections, identify areas enclosed by geometry that is considered a `face
    artifact` as per :cite:`fleischmann2023`. Face artifacts highlight areas with a high
    likelihood of being of non-morphological (e.g. transporation) origin and may require
    simplification prior morphological analysis. See :cite:`fleischmann2023` for more
    details.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing street network represented as (Multi)LineString geometry
    index : str, optional
        A type of the shape compacntess index to be used. Available are
        ['circlular_compactness', 'isoperimetric_quotient', 'diameter_ratio'], by
        default "circular_compactness"
    height_mins : float, optional
        Required depth of valleys, by default -np.inf
    height_maxs : float, optional
        Required height of peaks, by default 0.008
    prominence : float, optional
        Required prominence of peaks, by default 0.00075

    Attributes
    ----------
    threshold : float
        Identified threshold between face polygons and face artifacts
    face_artifacts : GeoDataFrame
        A GeoDataFrame of geometries identified as face artifacts
    polygons : GeoDataFrame
        All polygons resulting from polygonization of the input gdf with the
        face_artifact_index
    kde : scipy.stats._kde.gaussian_kde
        Representation of a kernel-density estimate using Gaussian kernels.
    pdf : numpy.ndarray
        Probability density function
    peaks : numpy.ndarray
        locations of peaks in pdf
    valleys : numpy.ndarray
        locations of valleys in pdf

    Examples
    --------
    >>> fa = momepy.FaceArtifacts(street_network_prague)
    >>> fa.threshold
    6.9634555986177045
    >>> fa.face_artifacts.head()
                                                 geometry  face_artifact_index
    6   POLYGON ((-744164.625 -1043922.362, -744167.39...             5.112844
    9   POLYGON ((-744154.119 -1043804.734, -744152.07...             6.295660
    10  POLYGON ((-744101.275 -1043738.053, -744103.80...             2.862871
    12  POLYGON ((-744095.511 -1043623.478, -744095.35...             3.712403
    17  POLYGON ((-744488.466 -1044533.317, -744489.33...             5.158554
    """

    def __init__(
        self,
        gdf,
        index="circular_compactness",
        height_mins=-np.inf,
        height_maxs=0.008,
        prominence=0.00075,
    ):
        try:
            from esda import shape
        except (ImportError, ModuleNotFoundError) as err:
            raise ImportError(
                "The `esda` package is required. You can install it using "
                "`pip install esda` or `conda install esda -c conda-forge`."
            ) from err

        # Polygonize street network
        polygons = gpd.GeoSeries(
            shapely.polygonize(  # polygonize
                [gdf.dissolve().geometry.item()]
            )
        ).explode(ignore_index=True)

        # Store geometries as a GeoDataFrame
        self.polygons = gpd.GeoDataFrame(geometry=polygons)
        if index == "circular_compactness":
            self.polygons["face_artifact_index"] = np.log(
                shape.minimum_bounding_circle_ratio(polygons) * polygons.area
            )
        elif index == "isoperimetric_quotient":
            self.polygons["face_artifact_index"] = np.log(
                shape.isoperimetric_quotient(polygons) * polygons.area
            )
        elif index == "diameter_ratio":
            self.polygons["face_artifact_index"] = np.log(
                shape.diameter_ratio(polygons) * polygons.area
            )
        else:
            raise ValueError(
                f"'{index}' is not supported. Use one of ['circlular_compactness', "
                "'isoperimetric_quotient', 'diameter_ratio']"
            )

        # parameters for peak/valley finding
        peak_parameters = {
            "height_mins": height_mins,
            "height_maxs": height_maxs,
            "prominence": prominence,
        }
        mylinspace = np.linspace(
            self.polygons["face_artifact_index"].min(),
            self.polygons["face_artifact_index"].max(),
            1000,
        )

        self.kde = gaussian_kde(
            self.polygons["face_artifact_index"], bw_method="silverman"
        )
        self.pdf = self.kde.pdf(mylinspace)

        # find peaks
        self.peaks, self.d_peaks = find_peaks(
            x=self.pdf,
            height=peak_parameters["height_maxs"],
            threshold=None,
            distance=None,
            prominence=peak_parameters["prominence"],
            width=1,
            plateau_size=None,
        )

        # find valleys
        self.valleys, self.d_valleys = find_peaks(
            x=-self.pdf + 1,
            height=peak_parameters["height_mins"],
            threshold=None,
            distance=None,
            prominence=peak_parameters["prominence"],
            width=1,
            plateau_size=None,
        )

        # check if we have at least 2 peaks
        condition_2peaks = len(self.peaks) > 1

        # check if we have at least 1 valley
        condition_1valley = len(self.valleys) > 0

        conditions = [condition_2peaks, condition_1valley]

        # if both these conditions are true, we find the artifact index
        if all(conditions):
            # find list order of highest peak
            highest_peak_listindex = np.argmax(self.d_peaks["peak_heights"])
            # find index (in linspace) of highest peak
            highest_peak_index = self.peaks[highest_peak_listindex]
            # define all possible peak ranges fitting our definition
            peak_bounds = list(zip(self.peaks[:-1], self.peaks[1:], strict=True))
            peak_bounds_accepted = [b for b in peak_bounds if highest_peak_index in b]
            # find all valleys that lie between two peaks
            valleys_accepted = [
                v_index
                for v_index in self.valleys
                if any(v_index in range(b[0], b[1]) for b in peak_bounds_accepted)
            ]
            # the value of the first of those valleys is our artifact index
            # get the order of the valley
            valley_index = valleys_accepted[0]

            # derive threshold value for given option from index/linspace
            self.threshold = float(mylinspace[valley_index])
            self.face_artifacts = self.polygons[
                self.polygons.face_artifact_index < self.threshold
            ]
        else:
            warnings.warn(
                "No threshold found. Either your dataset it too small or the "
                "distribution of the face artifact index does not follow the "
                "expected shape.",
                UserWarning,
                stacklevel=2,
            )
            self.threshold = None
            self.face_artifacts = None
