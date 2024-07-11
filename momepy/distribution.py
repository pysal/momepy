#!/usr/bin/env python

# distribution.py
# definitions of spatial distribution characters
import math
import os
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from packaging.version import Version
from tqdm.auto import tqdm  # progress bar

from .utils import _azimuth, deprecated, removed

__all__ = [
    "Orientation",
    "SharedWalls",
    "SharedWallsRatio",
    "StreetAlignment",
    "CellAlignment",
    "Alignment",
    "NeighborDistance",
    "MeanInterbuildingDistance",
    "NeighboringStreetOrientationDeviation",
    "BuildingAdjacency",
    "Neighbors",
]

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")


@deprecated("orientation")
class Orientation:
    """
    Calculate the orientation of object. The deviation of orientation from cardinal
    directions are captured. Here 'orientation' is defined as an orientation of the
    longest axis of bounding rectangle in range 0 - 45. The orientation of LineStrings
    is represented by the orientation of the line connecting the first and the last
    point of the segment.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings_df['orientation'] = momepy.Orientation(buildings_df).series
    100%|██████████| 144/144 [00:00<00:00, 630.54it/s]
    >>> buildings_df['orientation'][0]
    41.05146788287027
    """

    def __init__(self, gdf, verbose=True):
        self.gdf = gdf
        # define empty list for results
        results_list = []

        def _dist(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])

        bboxes = shapely.minimum_rotated_rectangle(gdf.geometry)
        for geom, bbox in tqdm(
            zip(gdf.geometry, bboxes, strict=True),
            total=gdf.shape[0],
            disable=not verbose,
        ):
            if geom.geom_type in ["Polygon", "MultiPolygon", "LinearRing"]:
                bbox = list(bbox.exterior.coords)
                axis1 = _dist(bbox[0], bbox[3])
                axis2 = _dist(bbox[0], bbox[1])

                if axis1 <= axis2:
                    az = _azimuth(bbox[0], bbox[1])
                else:
                    az = _azimuth(bbox[0], bbox[3])
            elif geom.geom_type in ["LineString", "MultiLineString"]:
                coords = geom.coords
                az = _azimuth(coords[0], coords[-1])
            else:
                results_list.append(np.nan)
                continue

            results_list.append(az)

        # get a deviation from cardinal directions
        results = np.abs((np.array(results_list, dtype=float) + 45) % 90 - 45)

        self.series = pd.Series(results, index=gdf.index)


class SharedWalls:
    """
    Calculate the length of shared walls of adjacent elements (typically buildings).

    .. math::
        \\textit{length of shared walls}

    Note that data needs to be topologically correct.
    Overlapping polygons will lead to incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings_df['swr'] = momepy.SharedWalls(buildings_df).series

    See also
    --------
    SharedWallsRatio
    """

    def __init__(self, gdf):
        if os.getenv("ALLOW_LEGACY_MOMEPY", "False").lower() not in (
            "true",
            "1",
            "yes",
        ):
            warnings.warn(
                "Class based API like `momepy.SharedWalls` or `momepy.SharedWallsRatio`"
                " is deprecated. Replace it with `momepy.shared_walls` or explicitly "
                "computing `momepy.shared_walls / gdf.length` respectively to use "
                "functional API instead or pin momepy version <1.0. Class-based API "
                "will be removed in 1.0. ",
                # "See details at https://docs.momepy.org/en/stable/migration.html",
                FutureWarning,
                stacklevel=2,
            )
        self.gdf = gdf

        if GPD_GE_013:
            inp, res = gdf.sindex.query(gdf.geometry, predicate="intersects")
        else:
            inp, res = gdf.sindex.query_bulk(gdf.geometry, predicate="intersects")
        left = gdf.geometry.take(inp).reset_index(drop=True)
        right = gdf.geometry.take(res).reset_index(drop=True)
        intersections = left.intersection(right).length
        results = intersections.groupby(inp).sum().reset_index(
            drop=True
        ) - gdf.geometry.length.reset_index(drop=True)
        results.index = gdf.index

        self.series = results


class SharedWallsRatio(SharedWalls):
    """
    Calculate shared walls ratio of adjacent elements (typically buildings).

    .. math::
        \\textit{length of shared walls} \\over perimeter

    Note that data needs to be topologically correct.
    Overlapping polygons will lead to incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    perimeters : str, list, np.array, pd.Series (default None, optional)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where perimeter values are stored.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    perimeters : GeoDataFrame
        A Series containing used perimeters values.

    Examples
    --------
    >>> buildings_df['swr'] = momepy.SharedWallsRatio(buildings_df).series
    >>> buildings_df['swr'][10]
    0.3424804411228673

    See also
    --------
    SharedWalls
    """

    def __init__(self, gdf, perimeters=None):
        super().__init__(gdf)

        if perimeters is None:
            self.perimeters = gdf.geometry.length
        elif isinstance(perimeters, str):
            self.perimeters = gdf[perimeters]
        else:
            self.perimeters = perimeters

        self.series = self.series / self.perimeters


@deprecated("street_alignment")
class StreetAlignment:
    """
    Calculate the difference between street orientation and orientation of
    another object in degrees. The orientation of a street segment is represented
    by the orientation of line connecting the first and the last point of the
    segment. A network ID linking each object to specific street segment is needed,
    and can be generated by :func:`momepy.get_network_id`. Either ``network_id`` or
    both ``left_network_id`` and ``right_network_id`` are required.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{street orientation}}\\right|

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    right : GeoDataFrame
        A GeoDataFrame containing a street network.
    orientations : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        object orientation values are stored. The object can be calculated
        using :class:`momepy.Orientation`.
    network_id : str (default None)
        The name of the column storing network ID in both left and right.
    left_network_id : str, list, np.array, pd.Series (default None)
        The name of the left dataframe column, ``np.array``, or ``pd.Series`` where
        object network IDs are stored.
    right_network_id : str, list, np.array, pd.Series (default None)
        The name of the right dataframe column, ``np.array``, or ``pd.Series`` of
        streets with unique network IDs. These IDs have to be defined beforehand and
        can be defined using :func:`momepy.unique_id`.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    left : GeoDataFrame
        The original ``left`` GeoDataFrame.
    right : GeoDataFrame
        The original ``right`` GeoDataFrame.
    network_id : str
        The name of the column storing network ID in both ``left`` and ``right``.
    left_network_id : Series
        A Series containing used ``left`` ID.
    right_network_id : Series
        A Series containing used ``right`` ID.

    Examples
    --------
    >>> buildings_df['street_alignment'] = momepy.StreetAlignment(buildings_df,
    ...                                                           streets_df,
    ...                                                           'orientation',
    ...                                                           'nID',
    ...                                                           'nID').series
    >>> buildings_df['street_alignment'][0]
    0.29073888476702336
    """

    def __init__(
        self,
        left,
        right,
        orientations,
        network_id=None,
        left_network_id=None,
        right_network_id=None,
    ):
        self.left = left
        self.right = right
        self.network_id = network_id

        left = left.copy()
        right = right.copy()

        if network_id:
            left_network_id = network_id
            right_network_id = network_id
        else:
            if left_network_id is None and right_network_id is not None:
                raise ValueError("left_network_id not set.")
            if left_network_id is not None and right_network_id is None:
                raise ValueError("right_network_id not set.")
            if left_network_id is None and right_network_id is None:
                raise ValueError(
                    "Network ID not set. Use either network_id or left_network_id "
                    "and right_network_id."
                )

        if not isinstance(orientations, str):
            left["mm_o"] = orientations
            orientations = "mm_o"
        self.orientations = left[orientations]

        if not isinstance(left_network_id, str):
            left["mm_nid"] = left_network_id
            left_network_id = "mm_nid"
        self.left_network_id = left[left_network_id]
        if not isinstance(right_network_id, str):
            right["mm_nis"] = right_network_id
            right_network_id = "mm_nis"
        self.right_network_id = right[right_network_id]

        right["_orientation"] = Orientation(right, verbose=False).series

        merged = left[[left_network_id, orientations]].merge(
            right[[right_network_id, "_orientation"]],
            left_on=left_network_id,
            right_on=right_network_id,
            how="left",
        )

        self.series = np.abs(merged[orientations] - merged["_orientation"])
        self.series.index = left.index


@deprecated("cell_alignment")
class CellAlignment:
    """
    Calculate the difference between cell orientation and the orientation of object.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{cell orientation}}\\right|

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    right : GeoDataFrame
        A GeoDataFrame containing tessellation cells (or relevant spatial units).
    left_orientations : str, list, np.array, pd.Series
        The name of the ``left`` dataframe column, ``np.array``, or
        `pd.Series`` where object orientation values are stored. This
        can be calculated using :class:`momepy.Orientation`.
    right_orientations : str, list, np.array, pd.Series
        The name of the ``right`` dataframe column, ``np.array``, or ``pd.Series``
        where object orientation values are stored. This
        can be calculated using :class:`momepy.Orientation`.
    left_unique_id : str
        The name of the ``left`` dataframe column with a unique
        ID shared between the ``left`` and ``right`` GeoDataFrame objects.
    right_unique_id : str
        The name of the ``right`` dataframe column with a unique
        ID shared between the ``left`` and ``right`` GeoDataFrame objects.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    left : GeoDataFrame
        The original ``left`` GeoDataFrame.
    right : GeoDataFrame
        The original ``right`` GeoDataFrame.
    left_orientations : Series
        A Series containing used ``left`` orientations.
    right_orientations : Series
        A Series containing used ``right`` orientations.
    left_unique_id : Series
        A Series containing used ``left`` ID.
    right_unique_id : Series
        A Series containing used ``right`` ID.

    Examples
    --------
    >>> buildings_df['cell_alignment'] = momepy.CellAlignment(buildings_df,
    ...                                                       tessellation_df,
    ...                                                       'bl_orient',
    ...                                                       'tes_orient',
    ...                                                       'uID',
    ...                                                       'uID').series
    >>> buildings_df['cell_alignment'][0]
    0.8795123936951939

    """

    def __init__(
        self,
        left,
        right,
        left_orientations,
        right_orientations,
        left_unique_id,
        right_unique_id,
    ):
        self.left = left
        self.right = right

        left = left.copy()
        right = right.copy()
        if not isinstance(left_orientations, str):
            left["mm_o"] = left_orientations
            left_orientations = "mm_o"
        self.left_orientations = left[left_orientations]
        if not isinstance(right_orientations, str):
            right["mm_o"] = right_orientations
            right_orientations = "mm_o"
        self.right_orientations = right[right_orientations]
        self.left_unique_id = left[left_unique_id]
        self.right_unique_id = right[right_unique_id]

        comp = left[[left_unique_id, left_orientations]].merge(
            right[[right_unique_id, right_orientations]],
            left_on=left_unique_id,
            right_on=right_unique_id,
            how="left",
        )
        if left_orientations == right_orientations:
            left_orientations = left_orientations + "_x"
            right_orientations = right_orientations + "_y"
        self.series = np.absolute(comp[left_orientations] - comp[right_orientations])
        self.series.index = left.index


@deprecated("alignment")
class Alignment:
    """
    Calculate the mean deviation of solar orientation of objects on adjacent cells
    from an object.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    spatial_weights : libpysal.weights, optional
        A spatial weights matrix.
    orientations : str, list, np.array, pd.Series
        The name of the left dataframe column, ``np.array``, or ``pd.Series``
        where object orientation values are stored. This
        can be calculated using :class:`momepy.Orientation`.
    unique_id : str
        The name of the unique ID column used as the ``spatial_weights`` index.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    orientations : Series
        A Series containing used orientation values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.

    Examples
    --------
    >>> buildings_df['alignment'] = momepy.Alignment(buildings_df,
    ...                                              sw,
    ...                                              'uID',
    ...                                              bl_orient).series
    100%|██████████| 144/144 [00:01<00:00, 140.84it/s]
    >>> buildings_df['alignment'][0]
    18.299481296455237
    """

    def __init__(self, gdf, spatial_weights, unique_id, orientations, verbose=True):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        # define empty list for results
        results_list = []
        gdf = gdf.copy()
        if not isinstance(orientations, str):
            gdf["mm_o"] = orientations
            orientations = "mm_o"
        self.orientations = gdf[orientations]

        data = gdf.set_index(unique_id)[orientations]

        # iterating over rows one by one
        for index, orient in tqdm(
            data.items(), total=data.shape[0], disable=not verbose
        ):
            if index in spatial_weights.neighbors:
                neighbours = spatial_weights.neighbors[index]
                if neighbours:
                    orientation = data.loc[neighbours]
                    results_list.append(abs(orientation - orient).mean())
                else:
                    results_list.append(np.nan)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("neighbor_distance")
class NeighborDistance:
    """
    Calculate the mean distance to adjacent buildings (based on ``spatial_weights``).
    If no neighbours are found, return ``np.nan``.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dist_i=\\frac{dist_1+dist_2+\\cdots+dist_n}{n}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    spatial_weights : libpysal.weights
        A spatial weights matrix based on ``unique_id``.
    unique_id : str
        The name of the unique ID column used as the ``spatial_weights`` index.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.

    Examples
    --------
    >>> buildings_df['neighbour_distance'] = momepy.NeighborDistance(buildings_df,
    ...                                                              sw,
    ...                                                              'uID').series
    100%|██████████| 144/144 [00:00<00:00, 345.78it/s]
    >>> buildings_df['neighbour_distance'][0]
    29.18589019096464
    """

    def __init__(self, gdf, spatial_weights, unique_id, verbose=True):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        # define empty list for results
        results_list = []

        data = gdf.set_index(unique_id).geometry

        # iterating over rows one by one
        for index, geom in tqdm(data.items(), total=data.shape[0], disable=not verbose):
            if geom is not None and index in spatial_weights.neighbors:
                neighbours = spatial_weights.neighbors[index]
                building_neighbours = data.loc[neighbours]
                if len(building_neighbours):
                    results_list.append(
                        building_neighbours.geometry.distance(geom).mean()
                    )
                else:
                    results_list.append(np.nan)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("mean_interbuilding_distance")
class MeanInterbuildingDistance:
    """
    Calculate the mean interbuilding distance. Interbuilding distances are
    calculated between buildings on adjacent cells based on
    ``spatial_weights``, while the extent is defined as order of contiguity.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    unique_id : str
        The name of the unique ID column used as the ``spatial_weights`` index.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    order : int
        The order of contiguity defining the extent.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    sw_higher : libpysal.weights
        The spatial weights matrix of higher order.
    order : int
        Order of contiguity.

    Notes
    -----
    Fix UserWarning.

    Examples
    --------
    >>> buildings_df['mean_interbuilding_distance'] = momepy.MeanInterbuildingDistance(
    ...     buildings_df,
    ...     sw,
    ...     'uID'
    ... ).series
    Computing mean interbuilding distances...
    100%|██████████| 144/144 [00:00<00:00, 317.42it/s]
    >>> buildings_df['mean_interbuilding_distance'][0]
    29.305457092042744
    """

    def __init__(
        self,
        gdf,
        spatial_weights,
        unique_id,
        order=3,
        verbose=True,
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.set_index(unique_id).geometry

        # define empty list for results
        results_list = []

        # define adjacency list from lipysal
        adj_list = spatial_weights.to_adjlist(drop_islands=True)
        adj_list["weight"] = (
            data.loc[adj_list.focal]
            .reset_index(drop=True)
            .distance(data.loc[adj_list.neighbor].reset_index(drop=True))
            .values
        )

        # generate graph
        graph = nx.from_pandas_edgelist(
            adj_list, source="focal", target="neighbor", edge_attr="weight"
        )

        print("Computing mean interbuilding distances...") if verbose else None
        # iterate over subgraphs to get the final values
        for uid in tqdm(data.index, total=data.shape[0], disable=not verbose):
            try:
                sub = nx.ego_graph(graph, uid, radius=order)
                results_list.append(
                    np.nanmean([x[-1] for x in list(sub.edges.data("weight"))])
                )
            except Exception:
                results_list.append(np.nan)
        self.series = pd.Series(results_list, index=gdf.index)


@removed("mean_deviation")
class NeighboringStreetOrientationDeviation:
    """
    Calculate the mean deviation of solar orientation of adjacent streets. The
    orientation of a street segment is represented by the orientation of the line
    connecting the first and last point of the segment.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    orientation : Series
        A Series containing used street orientation values.

    Examples
    --------
    >>> streets_df['orient_dev'] = momepy.NeighboringStreetOrientationDeviation(
    ...     streets_df
    ... ).series
    >>> streets_df['orient_dev'][6]
    7.043096518688273
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.orientation = gdf.geometry.apply(self._orient)

        if GPD_GE_013:
            inp, res = gdf.sindex.query(gdf.geometry, predicate="intersects")
        else:
            inp, res = gdf.sindex.query_bulk(gdf.geometry, predicate="intersects")
        itself = inp == res
        inp = inp[~itself]
        res = res[~itself]

        left = self.orientation.take(inp).reset_index(drop=True)
        right = self.orientation.take(res).reset_index(drop=True)
        deviations = (left - right).abs()

        results = deviations.groupby(inp).mean()

        match = gdf.iloc[list(results.index)]
        match["result"] = results.to_list()

        self.series = match.result

    def _orient(self, geom):
        start = geom.coords[0]
        end = geom.coords[-1]
        az = _azimuth(start, end)
        if 90 > az >= 45:
            diff = az - 45
            az = az - 2 * diff
        elif 135 > az >= 90:
            diff = az - 90
            az = az - 2 * diff
            diff = az - 45
            az = az - 2 * diff
        elif 181 > az >= 135:
            diff = az - 135
            az = az - 2 * diff
            diff = az - 90
            az = az - 2 * diff
            diff = az - 45
            az = az - 2 * diff
        return az


@deprecated("building_adjacency")
class BuildingAdjacency:
    """
    Calculate the level of building adjacency. Building adjacency reflects how much
    buildings tend to join together into larger structures. It is calculated as a
    ratio of joined built-up structures and buildings within the extent defined
    in ``spatial_weights_higher``.

    Adapted from :cite:`vanderhaegen2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    spatial_weights_higher : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the unique ID column used as the ``spatial_weights`` index.
    spatial_weights : libpysal.weights, optional
        A spatial weights matrix. If ``None``, a Queen contiguity matrix will
        be calculated based on ``gdf``. It is to denote adjacent buildings
        and is based on ``unique_id``.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    sw_higher : libpysal.weights
        A higher order spatial weights matrix.
    id : Series
        A Series containing used unique IDs.
    sw : libpysal.weights
        The spatial weights matrix.

    Examples
    --------
    >>> buildings_df['adjacency'] = momepy.BuildingAdjacency(buildings_df,
    ...                                                      swh,
    ...                                                      unique_id='uID').series
    Calculating spatial weights...
    Spatial weights ready...
    Calculating adjacency: 100%|██████████| 144/144 [00:00<00:00, 335.55it/s]
    >>> buildings_df['adjacency'][10]
    0.23809523809523808
    """

    def __init__(
        self, gdf, spatial_weights_higher, unique_id, spatial_weights=None, verbose=True
    ):
        self.gdf = gdf
        self.sw_higher = spatial_weights_higher
        self.id = gdf[unique_id]
        results_list = []

        # if weights matrix is not passed, generate it from gdf
        if spatial_weights is None:
            print("Calculating spatial weights...") if verbose else None
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(
                gdf, silence_warnings=True, ids=unique_id
            )
            print("Spatial weights ready...") if verbose else None

        self.sw = spatial_weights
        patches = dict(
            zip(gdf[unique_id], spatial_weights.component_labels, strict=True)
        )

        for uid in tqdm(
            self.id,
            total=gdf.shape[0],
            disable=not verbose,
            desc="Calculating adjacency",
        ):
            if uid in spatial_weights_higher.neighbors:
                neighbours = spatial_weights_higher.neighbors[uid].copy()
                if neighbours:
                    neighbours.append(uid)

                    patches_sub = [patches[x] for x in neighbours]
                    patches_nr = len(set(patches_sub))

                    results_list.append(patches_nr / len(neighbours))
                else:
                    results_list.append(np.nan)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


@deprecated("neighbors")
class Neighbors:
    """
    Calculate the number of neighbours captured by ``spatial_weights``. If
    ``weighted=True``, the number of neighbours will be divided by the perimeter of
    the object to return relative value.

    Adapted from :cite:`hermosilla2012`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    spatial_weights : libpysal.weights
        A spatial weights matrix.
    unique_id : str
        The name of the unique ID column used as the ``spatial_weights`` index.
    weighted : bool (default False)
        If ``True``, the number of neighbours will be divided
        by the perimeter of object, to return the relative value.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    values : Series
        A Series containing used values.
    sw : libpysal.weights
        The spatial weights matrix.
    id : Series
        A Series containing used unique ID.
    weighted : bool
        Whether object is weighted or not.

    Examples
    --------
    >>> sw = libpysal.weights.contiguity.Queen.from_dataframe(tessellation_df,
    ...                                                       ids='uID')
    >>> tessellation_df['neighbours'] = momepy.Neighbors(tessellation_df,
    ...                                                  sw,
    ...                                                  'uID').series
    100%|██████████| 144/144 [00:00<00:00, 6909.50it/s]
    >>> tessellation_df['neighbours'][0]
    4
    """

    def __init__(self, gdf, spatial_weights, unique_id, weighted=False, verbose=True):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.weighted = weighted

        neighbours = []
        for index, geom in tqdm(
            gdf.set_index(unique_id).geometry.items(),
            total=gdf.shape[0],
            disable=not verbose,
        ):
            if index in spatial_weights.neighbors:
                if weighted is True:
                    neighbours.append(
                        spatial_weights.cardinalities[index] / geom.length
                    )
                else:
                    neighbours.append(spatial_weights.cardinalities[index])
            else:
                neighbours.append(np.nan)

        self.series = pd.Series(neighbours, index=gdf.index)
