import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from numpy.typing import NDArray
from packaging.version import Version
from pandas import Series
from scipy import sparse

__all__ = [
    "orientation",
    "shared_walls",
    "alignment",
    "neighbor_distance",
    "mean_interbuilding_distance",
    "building_adjacency",
    "neighbors",
    "street_alignment",
    "cell_alignment",
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
    orientation : Series
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
    geometry: GeoDataFrame | GeoSeries,
    adjacency_graph: Graph,
    neighborhood_graph: Graph,
) -> Series:
    """Calculate the mean distance between adjacent geometries within a set neighborhood

    For each building, this function takes a neighborhood based on the neighbors within
    a ``neighborhood_graph`` and calculates the mean distance between adjacent buildings
    within this neighborhood where adjacency is captured by ``adjacency_graph``.

    Notes
    -----
    The index of ``geometry`` must match the index along which both of the graphs are
    built.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing geometries to analyse.
    adjacency_graph : libpysal.graph.Graph
        Graph representing the adjacency of geometries. Typically, this is a contiguity
        graph derived from tessellation cells linked to buildings.
    neighborhood_graph : libpysal.graph.Graph
        Graph representing the extent around each geometry within which to calculate
        the mean interbuilding distance. This can be a distance based graph, KNN graph,
        higher order contiguity, etc.

    Returns
    -------
    Series
    """
    distance = Series(
        shapely.distance(
            geometry.geometry.loc[
                adjacency_graph._adjacency.index.get_level_values(0)
            ].values,
            geometry.geometry.loc[
                adjacency_graph._adjacency.index.get_level_values(1)
            ].values,
        ),
        index=adjacency_graph._adjacency.index,
        name="distance",
    )

    distance_matrix = (
        distance.astype("Sparse[float]").sparse.to_coo(sort_labels=True)[0].tocsr()
    )
    neighborhood_matrix = sparse.coo_matrix(neighborhood_graph.sparse).tocsr()

    mean_distances = np.zeros(distance_matrix.shape[0], dtype=float)

    for i in range(distance_matrix.shape[0]):
        neighborhood_indices = np.append(neighborhood_matrix[i].indices, i)
        sub_matrix = distance_matrix[neighborhood_indices][:, neighborhood_indices]
        mean_distances[i] = sub_matrix.sum() / sub_matrix.nnz

    return Series(
        mean_distances, index=geometry.index, name="mean_interbuilding_distance"
    )


def building_adjacency(
    contiguity_graph: Graph,
    neighborhood_graph: Graph,
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

    Parameters
    ----------
    contiguity_graph : libpysal.graph.Graph
        Graph representing contiguity between geometries, typically a rook contiguity
        graph derived from buildings.
    neighborhood_graph : libpysal.graph.Graph
        Graph representing the extent around each geometry within which to calculate
        the level of building adjacency. This can be a distance based graph, KNN graph,
        higher order contiguity, etc.

    Returns
    -------
    Series
    """
    components = contiguity_graph.component_labels

    # check if self-weights are present, otherwise assign them to treat self as part of
    # the neighborhood
    has_self_weights = (
        neighborhood_graph._adjacency.index.get_level_values("focal")
        == neighborhood_graph._adjacency.index.get_level_values("neighbor")
    ).sum() == neighborhood_graph.n
    if not has_self_weights:
        neighborhood_graph = neighborhood_graph.assign_self_weight()

    grouper = components.loc[
        neighborhood_graph._adjacency.index.get_level_values(1)
    ].groupby(neighborhood_graph._adjacency.index.get_level_values(0))
    result = grouper.agg("nunique") / grouper.agg("count")
    result.name = "building_adjacency"
    result.index.name = None
    return result


def neighbors(
    geometry: GeoDataFrame | GeoSeries, graph: Graph, weighted=False
) -> Series:
    """Calculate the number of neighbours captured by ``graph``.

    If ``weighted=True``, the number of neighbours will be divided by the perimeter of
    the object to return a relative value (neighbors per meter).

    Adapted from :cite:`hermosilla2012`.

    Notes
    -----
    The index of ``geometry`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    gdf : GeoDataFrame | GeoSeries
        GeoDataFrame containing geometries to analyse.
    graph : libpysal.graph.Graph
        Graph representing spatial relationships between elements.
    weighted : bool
        If True, the number of neighbours will be divided by the perimeter of the object
        to return a relative value (neighbors per meter).

    Returns
    -------
    Series
    """
    if weighted:
        r = graph.cardinalities / geometry.length
    else:
        r = graph.cardinalities

    r.name = "neighbors"
    return r


def street_alignment(
    building_orientation: Series,
    street_orientation: Series,
    street_index: Series,
) -> Series:
    """Calulate the deviation of the building orientation from the street orientation.

    Parameters
    ----------
    building_orientation : Series
        Series with the orientation of buildings. Can be measured using
        :func:`orientation`.
    street_orientation : Series
        Series with the orientation of streets. Can be measured using
        :func:`orientation`.
    street_index : Series
        Series with the index of the street to which the building belongs. Can be
        retrieved using :func:`momepy.get_nearest_street`.

    Returns
    -------
    Series
    """
    return (building_orientation - street_orientation.loc[street_index].values).abs()


def cell_alignment(
    left_orientations: NDArray[np.float64] | Series,
    right_orientations: NDArray[np.float64] | Series,
) -> Series:
    """
    Calculate the difference between cell orientation and the orientation of object.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{cell orientation}}\\right|

    Note
    ----
    ``left_orientations`` and ``right_orientations`` must be aligned or have an index.

    Parameters
    ----------
    left_orientations : np.array, pd.Series
        The ``np.array``, or `pd.Series`` with orientation of cells.
        This can be calculated using :func:`orientation`.
    right_orientations : np.array, pd.Series
        The ``np.array`` or ``pd.Series`` with orientation of objects.
        This can be calculated using :func:`orientation`.

    Returns
    -------
    Series
    """

    if not isinstance(left_orientations, Series):
        left_orientations = Series(left_orientations)
    if not isinstance(right_orientations, Series):
        right_orientations = Series(right_orientations)
    return (left_orientations - right_orientations).abs()
