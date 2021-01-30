#!/usr/bin/env python
# -*- coding: utf-8 -*-

# distribution.py
# definitions of spatial distribution characters

import math

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar

from .utils import _azimuth

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


class Orientation:
    """
    Calculate the orientation of object

    Captures the deviation of orientation from cardinal directions.
    Defined as an orientation of the longext axis of bounding rectangle in range 0 - 45.
    Orientation of LineStrings is represented by the orientation of line
    connecting first and last point of the segment.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame

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

        for geom in tqdm(gdf.geometry, total=gdf.shape[0], disable=not verbose):
            if geom.type in ["Polygon", "MultiPolygon", "LinearRing"]:
                # TODO: vectorize once minimum_rotated_rectangle is in geopandas
                bbox = list(geom.minimum_rotated_rectangle.exterior.coords)
                axis1 = _dist(bbox[0], bbox[3])
                axis2 = _dist(bbox[0], bbox[1])

                if axis1 <= axis2:
                    az = _azimuth(bbox[0], bbox[1])
                else:
                    az = _azimuth(bbox[0], bbox[3])
            elif geom.type in ["LineString", "MultiLineString"]:
                coords = geom.coords
                az = _azimuth(coords[0], coords[-1])
            else:
                results_list.append(np.nan)
                continue

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
            results_list.append(az)

        self.series = pd.Series(results_list, index=gdf.index)


class SharedWalls:
    """
    Calculate the length of shared walls of adjacent elements (typically buildings)

    .. math::
        \\textit{length of shared walls}

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing gdf to analyse

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings_df['swr'] = momepy.SharedWalls(buildings_df).series

    See also
    --------
    SharedWallsRatio
    """

    def __init__(self, gdf):
        self.gdf = gdf

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
    Calculate shared walls ratio of adjacent elements (typically buildings)

    .. math::
        \\textit{length of shared walls} \\over perimeter

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing gdf to analyse
    perimeters : str, list, np.array, pd.Series (default None, optional)
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is
        stored perimeter value

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    perimeters : GeoDataFrame
        Series containing used perimeters values

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
        super(SharedWallsRatio, self).__init__(gdf)

        if perimeters is None:
            self.perimeters = gdf.geometry.length
        elif isinstance(perimeters, str):
            self.perimeters = gdf[perimeters]
        else:
            self.perimeters = perimeters

        self.series = self.series / self.perimeters


class StreetAlignment:
    """
    Calculate the difference between street orientation and orientation of object in
    degrees

    Orientation of street segment is represented by the orientation of line
    connecting first and last point of the segment. Network ID linking each object
    to specific street segment is needed. Can be generated by
    :func:`momepy.get_network_id`.
    Either ``network_id`` or both ``left_network_id`` and ``right_network_id``
    are required.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{street orientation}}\\right|

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing objects to analyse
    right : GeoDataFrame
        GeoDataFrame containing street network
    orientations : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is
        stored object orientation value
        (can be calculated using :class:`momepy.Orientation`)
    network_id : str (default None)
        the name of the column storing network ID in both left and right
    left_network_id : str, list, np.array, pd.Series (default None)
        the name of the left dataframe column, ``np.array``, or ``pd.Series`` where is
        stored object network ID
    right_network_id : str, list, np.array, pd.Series (default None)
        the name of the right dataframe column, ``np.array``, or ``pd.Series`` of
        streets with unique network id (has to be defined beforehand)
        (can be defined using :func:`momepy.unique_id`)

    Attributes
    ----------
    series : Series
        Series containing resulting values
    left : GeoDataFrame
        original left GeoDataFrame
    right : GeoDataFrame
        original right GeoDataFrame
    network_id : str
        the name of the column storing network ID in both left and right
    left_network_id : Series
        Series containing used left ID
    right_network_id : Series
        Series containing used right ID

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


class CellAlignment:
    """
    Calculate the difference between cell orientation and orientation of object

    .. math::
        \\left|{\\textit{building orientation} - \\textit{cell orientation}}\\right|

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing objects to analyse
    right : GeoDataFrame
        GeoDataFrame containing tessellation cells (or relevant spatial units)
    left_orientations : str, list, np.array, pd.Series
        the name of the left dataframe column, ``np.array``, or ``pd.Series`` where is
        stored object orientation value
        (can be calculated using :class:`momepy.Orientation`)
    right_orientations : str, list, np.array, pd.Series
        the name of the right dataframe column, ``np.array``, or ``pd.Series`` where is
        stored object orientation value
        (can be calculated using :class:`momepy.Orientation`)
    left_unique_id : str
        the name of the ``left`` dataframe column with unique id shared between ``left``
         and ``right`` gdf
    right_unique_id : str
        the name of the ``right`` dataframe column with unique id shared between
        ``left`` and ``right`` gdf

    Attributes
    ----------
    series : Series
        Series containing resulting values
    left : GeoDataFrame
        original left GeoDataFrame
    right : GeoDataFrame
        original right GeoDataFrame
    left_orientations : Series
        Series containing used left orientations
    right_orientations : Series
        Series containing used right orientations
    left_unique_id : Series
        Series containing used left ID
    right_unique_id : Series
        Series containing used right ID

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


class Alignment:

    """
    Calculate the mean deviation of solar orientation of objects on adjacent cells
    from an object

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix
    orientations : str, list, np.array, pd.Series
        the name of the left dataframe column, ``np.array``, or ``pd.Series`` where is
        stored object orientation value
        (can be calculated using :class:`momepy.Orientation`)
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    orientations : Series
        Series containing used orientation values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID

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
            data.iteritems(), total=data.shape[0], disable=not verbose
        ):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index]
                if neighbours:
                    orientation = data.loc[neighbours]
                    results_list.append(abs(orientation - orient).mean())
                else:
                    results_list.append(np.nan)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class NeighborDistance:
    """
    Calculate the mean distance to adjacent buildings (based on ``spatial_weights``)

    If no neighbours are found, return ``np.nan``.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dist_i=\\frac{dist_1+dist_2+\\cdots+dist_n}{n}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights
        spatial weights matrix based on unique_id
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID

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
        for index, geom in tqdm(
            data.iteritems(), total=data.shape[0], disable=not verbose
        ):
            if geom is not None and index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index]
                building_neighbours = data.loc[neighbours]
                if len(building_neighbours) > 0:
                    results_list.append(
                        building_neighbours.geometry.distance(geom).mean()
                    )
                else:
                    results_list.append(np.nan)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class MeanInterbuildingDistance:
    """
    Calculate the mean interbuilding distance

    Interbuilding distances are calculated between buildings on adjacent cells based on
    ``spatial_weights``, while the extent is defined as order of contiguity.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    spatial_weights : libpysal.weights
        spatial weights matrix
    order : int
        Order of contiguity defining the extent
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    sw_higher : libpysal.weights
        Spatial weights matrix of higher order
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
        spatial_weights_higher=None,
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
        adj_list = spatial_weights.to_adjlist()
        adj_list["weight"] = (
            data.loc[adj_list.focal]
            .reset_index(drop=True)
            .distance(data.loc[adj_list.neighbor].reset_index(drop=True))
            .values
        )

        # generate graph
        G = nx.from_pandas_edgelist(
            adj_list, source="focal", target="neighbor", edge_attr="weight"
        )

        print("Computing mean interbuilding distances...") if verbose else None
        # iterate over subgraphs to get the final values
        for uid in tqdm(data.index, total=data.shape[0], disable=not verbose):
            try:
                sub = nx.ego_graph(G, uid, radius=order)
                results_list.append(
                    np.nanmean([x[-1] for x in list(sub.edges.data("weight"))])
                )
            except Exception:
                results_list.append(np.nan)
        self.series = pd.Series(results_list, index=gdf.index)


class NeighboringStreetOrientationDeviation:
    """
    Calculate the mean deviation of solar orientation of adjacent streets

    Orientation of street segment is represented by the orientation of line
    connecting first and last point of the segment.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing street network to analyse

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    orientation : Series
        Series containing used street orientation values

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


class BuildingAdjacency:
    """
    Calculate the level of building adjacency

    Building adjacency reflects how much buildings tend to join together into larger
    structures.
    It is calculated as a ratio of joined built-up structures and buildings within
    the extent defined in ``spatial_weights_higher``.

    Adapted from :cite:`vanderhaegen2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights_higher : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on gdf. It is to denote adjacent buildings (note: based on unique ID).
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw_higher : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    sw : libpysal.weights
        spatial weights matrix

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
        patches = dict(zip(gdf[unique_id], spatial_weights.component_labels))

        for uid in tqdm(
            self.id,
            total=gdf.shape[0],
            disable=not verbose,
            desc="Calculating adjacency",
        ):
            if uid in spatial_weights_higher.neighbors.keys():
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


class Neighbors:
    """
    Calculate the number of neighbours captured by ``spatial_weights``

    If ``weighted=True``, number of neighbours will be divided by the perimeter of
    object to return relative value.

    Adapted from :cite:`hermosilla2012`.


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    weighted : bool (default False)
        if ``True``, number of neighbours will be divided by the perimeter of object,
        to return relative value
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : Series
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    weighted : bool
        used weighted value

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
            gdf.set_index(unique_id).geometry.iteritems(),
            total=gdf.shape[0],
            disable=not verbose,
        ):
            if index in spatial_weights.neighbors.keys():
                if weighted is True:
                    neighbours.append(
                        spatial_weights.cardinalities[index] / geom.length
                    )
                else:
                    neighbours.append(spatial_weights.cardinalities[index])
            else:
                neighbours.append(np.nan)

        self.series = pd.Series(neighbours, index=gdf.index)
