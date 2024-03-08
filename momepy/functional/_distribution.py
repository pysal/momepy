import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from packaging.version import Version
from pandas import Series

__all__ = [
    "orientation",
    "shared_walls",
    "alignment",
    "neighbor_distance",
    "mean_interbuilding_distance",
    "building_adjacency",
]

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")


def orientation(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculate the orientation of objects.

    The 'orientation' is defined as the deviation of orientation of the bounding
    rectangle from cardinal directions. As such it is within a range 0 - 45. The
    orientation of LineStrings is represented by the orientation of the line connecting
    the first and the last point of the segment.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    geom_types = geometry.geom_type
    poly_mask = geom_types.str.contains("Polygon")
    line_mask = geom_types.str.contains("Line")

    result = np.full(len(geometry), np.nan, dtype=float)

    if poly_mask.any():
        bboxes = shapely.minimum_rotated_rectangle(geometry.geometry.loc[poly_mask])
        coords = shapely.get_coordinates(bboxes)
        pt0 = coords[::5]
        pt1 = coords[1::5]
        angle = np.arctan2(pt1[:, 0] - pt0[:, 0], pt1[:, 1] - pt0[:, 1])
        result[poly_mask] = np.degrees(angle)

    if line_mask.any():
        first = shapely.get_point(geometry.geometry, 0)
        last = shapely.get_point(geometry.geometry, -1)
        pt0 = shapely.get_coordinates(first)
        pt1 = shapely.get_coordinates(last)
        angle = np.arctan2(pt1[:, 0] - pt0[:, 0], pt1[:, 1] - pt0[:, 1])
        result[line_mask] = np.degrees(angle)

    return Series(
        np.abs((result + 45) % 90 - 45),
        index=geometry.index,
        dtype=float,
        name="orientation",
    )


def shared_walls(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculate the length of shared walls of adjacent elements (typically buildings).

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    if GPD_GE_013:
        inp, res = geometry.sindex.query(geometry.geometry, predicate="touches")
    else:
        inp, res = geometry.sindex.query_bulk(geometry.geometry, predicate="touches")
    left = geometry.geometry.take(inp).reset_index(drop=True)
    right = geometry.geometry.take(res).reset_index(drop=True)
    intersections = left.intersection(right).length
    walls = intersections.groupby(inp).sum()
    walls.index = geometry.index.take(walls.index)

    results = Series(0.0, index=geometry.index, name="shared_walls")
    results.loc[walls.index] = walls
    return results


def alignment(orientation: Series, graph: Graph) -> Series:
    """Calculate the mean deviation of orientation adjacent elements

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Takes orientation of adjacent elements defined in ``graph`` and calculates the
    mean deviation.

    Notes
    -----
    The index of ``orientation`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    orientation : pd.Series
        A series containing orientation (e.g. measured by the :func:`orientation`
        function) indexed using the same index that has been used to build the graph.
    graph : libpysal.graph.Graph
        Graph representing spatial relationships between elements.

    Returns
    -------
    Series
    """
    orientation_expanded = orientation.loc[graph._adjacency.index.get_level_values(1)]
    orientation_expanded.index = graph._adjacency.index.get_level_values(0)
    return (orientation_expanded - orientation).abs().groupby(level=0).mean()


def neighbor_distance(geometry: GeoDataFrame | GeoSeries, graph: Graph) -> Series:
    """Calculate the mean distance to adjacent elements.

    Takes geometry of adjacent elements defined in ``graph`` and calculates the
    mean distance.

    Notes
    -----
    The index of ``geometry`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing geometries to analyse.
    graph : libpysal.graph.Graph
        Graph representing spatial relationships between elements.

    Returns
    -------
    Series
    """
    geoms = geometry.geometry.loc[graph._adjacency.index.get_level_values(1)]
    geoms.index = graph._adjacency.index.get_level_values(0)
    mean_distance = (
        (geoms.distance(geometry.geometry, align=True)).groupby(level=0).mean()
    )
    mean_distance.loc[graph.isolates] = np.nan
    return mean_distance


def mean_interbuilding_distance(
    geometry: GeoDataFrame | GeoSeries, graph: Graph, order: int = 3
) -> Series:
    """Calculate the mean distance between adjacent geometries within a set neighborhood

    For each building, this function defines a neighborhood (ego graph) based on the
    neighbors within a defined ``order`` of contigity along the graph. It then
    calculates the mean distance between adjacent buildings within this neighborhood.
    Typically, ``graph`` represents contiguity derived from tessellation cells or plots
    linked to buildings.

    Notes
    -----
    The index of ``geometry`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing geometries to analyse.
    graph : libpysal.graph.Graph
        Graph representing spatial relationships between elements.
    order : int
        The order of contiguity defining the extent of the neighborhood.

    Returns
    -------
    Series
    """
    distance = pd.Series(
        shapely.distance(
            geometry.geometry.loc[graph._adjacency.index.get_level_values(0)].values,
            geometry.geometry.loc[graph._adjacency.index.get_level_values(1)].values,
        ),
        index=graph._adjacency.index,
        name="distance",
    )

    nx_graph = nx.from_pandas_edgelist(
        distance.reset_index(), source="focal", target="neighbor", edge_attr="distance"
    )

    results_list = []
    for uid in geometry.index:
        try:
            sub = nx.ego_graph(nx_graph, uid, radius=order)
            results_list.append(
                np.mean(
                    np.array([data["distance"] for _, _, data in sub.edges(data=True)])
                )
            )
        # this may happen if the graph comes from tessellation thad does not fully match
        except nx.NodeNotFound:
            warnings.warn(
                f"Geometry with the index {uid} not found in the graph.",
                UserWarning,
                stacklevel=2,
            )
            results_list.append(np.nan)

    return Series(
        results_list, index=geometry.index, name="mean_interbuilding_distance"
    )
    # 57.4 s ± 1.57 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # 1min 2s ± 3.78 s per loop (mean ± std. dev. of 7 runs, 1 loop each)


def building_adjacency(
    neighborhood_graph: Graph,
    contiguity_graph: Graph,
) -> Series:
    """Calculate the level of building adjacency.

    Building adjacency reflects how much buildings tend to join together into larger
    structures. It is calculated as a ratio of joined built-up structures captured by
    ``contiguity_graph`` and buildings within the neighborhood defined in
    ``neighborhood_graph``.

    Adapted from :cite:`vanderhaegen2017`.

    Notes
    -----
    Both graphs must be built on the same index.

    If you want to consider the geometry
    part of its own neighborhood and include it in calculation, ensure you assign
    self-weights to the ``contiguity_graph`` using
    ``contiguity_graph.assign_self_weight()``.

    Parameters
    ----------
    neighborhood_graph : Graph
        Graph representing the extent around each geometry within which to calculate
        the level of building adjacency. This can be a distance based graph, KNN graph,
        higher order contiguity, etc.
    contiguity_graph : Graph
        Graph representing contiguity between geometries.

    Returns
    -------
    Series
    """
    components = contiguity_graph.component_labels

    grouper = components.loc[
        neighborhood_graph._adjacency.index.get_level_values(1)
    ].groupby(neighborhood_graph._adjacency.index.get_level_values(0))
    result = grouper.agg("nunique") / grouper.agg("count")
    result.name = "building_adjacency"
    result.index.name = None
    return result

    # old: 251 ms ± 14.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # new: 57.3 ms ± 1.26 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
