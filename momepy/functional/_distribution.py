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

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.orientation(buildings)
    0      41.051468
    1      20.829200
    2      20.969610
    3      20.811525
    4      20.412895
             ...
    139    21.107057
    140    13.669062
    141    13.671217
    142    21.463914
    143    13.754569
    Name: orientation, Length: 144, dtype: float64
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


def shared_walls(
    geometry: GeoDataFrame | GeoSeries, strict: bool = True, tolerance: float = 0.01
) -> Series:
    """Calculate the length of shared walls of adjacent elements (typically buildings).

    Note that data needs to be topologically correct. Overlapping polygons will lead to
    incorrect results.

    Adapted from :cite:`hamaina2012a`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    strict : bool
        Perform calculations based on strict contiguity. If set to `False`,
        consider overlapping or nearly overlapping polygons as touching.
    tolerance: float
        Tolerance for non-strict calculations, if strict is True, tolerance
        has no effect on the results.

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.shared_walls(buildings)
    0       0.000000
    1      33.805154
    2      39.766297
    3      40.604643
    4      40.455735
            ...
    139     9.886397
    140     0.000000
    141     0.000000
    142    19.332735
    143    10.876113
    Name: shared_walls, Length: 144, dtype: float64
    """

    predicate = "touches"
    if not strict:
        orig_lengths = geometry.length
        geometry = geometry.buffer(tolerance)
        predicate = "intersects"

    if GPD_GE_013:
        inp, res = geometry.sindex.query(geometry.geometry, predicate=predicate)
    else:
        inp, res = geometry.sindex.query_bulk(geometry.geometry, predicate=predicate)

    mask = inp != res
    inp, res = inp[mask], res[mask]
    left = geometry.geometry.take(inp).reset_index(drop=True)
    right = geometry.geometry.take(res).reset_index(drop=True)
    intersections = left.intersection(right).length
    walls = intersections.groupby(inp).sum()
    walls.index = geometry.index.take(walls.index)

    results = Series(0.0, index=geometry.index, name="shared_walls")
    results.loc[walls.index] = walls

    if not strict:
        results = (results / 2) - (2 * tolerance)
        results = results.clip(0, orig_lengths)

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

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> orientation = momepy.orientation(buildings)

    Define a spatial graph that includes observations within its own neighborhood:

    >>> delaunay = graph.Graph.build_triangulation(
    ...     buildings.centroid
    ... ).assign_self_weight()
    >>> delaunay
    <Graph of 144 nodes and 970 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Alignment of orienation within triangulated neighbors:

    >>> momepy.alignment(orientation, delaunay)
    0      14.639585
    1       0.217417
    2       0.205626
    3       0.151730
    4       0.352692
             ...
    139     1.970642
    140     0.127322
    141     0.161906
    142     4.350890
    143     0.084884
    Name: alignment, Length: 144, dtype: float64
    """
    orientation_expanded = orientation.loc[graph._adjacency.index.get_level_values(1)]
    orientation_expanded.index = graph._adjacency.index.get_level_values(0)
    r = (orientation_expanded - orientation).abs().groupby(level=0).mean()
    r.name = "alignment"
    return r


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

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a spatial graph:

    >>> delaunay = graph.Graph.build_triangulation(buildings.centroid)
    >>> delaunay
    <Graph of 144 nodes and 826 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Mean distance to adjacent buildings within triangulated neighbors:

    >>> momepy.neighbor_distance(buildings, delaunay)
    0      29.185890
    1      30.244905
    2      47.052305
    3      22.831824
    4      16.183615
            ...
    139    39.698734
    140    20.634252
    141    38.208668
    142     6.304569
    143    14.551355
    Length: 144, dtype: float64
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

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a spatial graph denoting building adjacency:

    >>> delaunay = graph.Graph.build_triangulation(buildings.centroid)
    >>> delaunay
    <Graph of 144 nodes and 826 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Define a spatial graph denoting the neighborhood:

    >>> knn15 = graph.Graph.build_knn(buildings.centroid, k=15)
    >>> knn15
    <Graph of 144 nodes and 2160 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

     Measure mean interbuilding distance:

    >>> momepy.mean_interbuilding_distance(buildings, delaunay, knn15)
    0      29.516506
    1      18.673132
    2      23.277728
    3      25.409034
    4      18.454463
            ...
    139    21.642580
    140    16.427126
    141    17.792155
    142    10.844367
    143    14.896066
    Name: mean_interbuilding_distance, Length: 144, dtype: float64
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

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a spatial graph denoting building contiguity:

    >>> contig = graph.Graph.build_contiguity(buildings)
    >>> contig
    <Graph of 144 nodes and 248 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Define a spatial graph denoting the neighborhood:

    >>> knn15 = graph.Graph.build_knn(buildings.centroid, k=15)
    >>> knn15
    <Graph of 144 nodes and 2160 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

     Measure mean interbuilding distance:

    >>> momepy.building_adjacency(contig, knn15)
    0      0.6875
    1      0.1875
    2      0.1875
    3      0.2500
    4      0.1875
            ...
    139    0.4375
    140    0.1875
    141    0.1875
    142    0.1875
    143    0.2500
    Name: building_adjacency, Length: 144, dtype: float64
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

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> tessellation = momepy.morphological_tessellation(buildings)

    Define a spatial graph denoting adjacency:

    >>> contig = graph.Graph.build_contiguity(tessellation)
    >>> contig
    <Graph of 144 nodes and 768 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Number of neighbors of each tessellation cell:

    >>> momepy.neighbors(tessellation, contig)
    focal
    0       4
    1       9
    2       3
    3       3
    4       7
        ..
    139     3
    140     6
    141    12
    142     2
    143     5
    Name: neighbors, Length: 144, dtype: int64

    Weighted by the tessellation area:

    >>> momepy.neighbors(tessellation, contig, weighted=True)
    focal
    0      0.012732
    1      0.010116
    2      0.013350
    3      0.010172
    4      0.038916
            ...
    139    0.020037
    140    0.036766
    141    0.045287
    142    0.044147
    143    0.051799
    Name: neighbors, Length: 144, dtype: float64
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

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Get street index.

    >>> buildings["street_index"] = momepy.get_nearest_street(buildings, streets)
    >>> buildings.head()
       uID                                           geometry  street_index
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...           0.0
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...          33.0
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...          10.0
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...           8.0
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...           8.0

    Compute orientations.

    >>> blg_orient = momepy.orientation(buildings)
    >>> str_orient = momepy.orientation(streets)

    Compute street alignment.

    >>> momepy.street_alignment(blg_orient, str_orient, buildings["street_index"])
    0      0.290739
    1      4.542071
    2      0.105745
    3      0.424903
    4      0.823533
            ...
    139    1.779876
    140    0.109254
    141    0.466453
    142    1.223387
    143    0.455081
    Name: street_alignment, Length: 144, dtype: float64
    """
    r = (building_orientation - street_orientation.loc[street_index].values).abs()
    r.name = "street_alignment"
    return r


def cell_alignment(
    left_orientation: NDArray[np.float64] | Series,
    right_orientation: NDArray[np.float64] | Series,
) -> Series:
    """
    Calculate the difference between cell orientation and the orientation of object.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{cell orientation}}\\right|

    Notes
    -----
    ``left_orientation`` and ``right_orientation`` must be aligned or have an index.

    Parameters
    ----------
    left_orientation : np.array, pd.Series
        The ``np.array``, or `pd.Series`` with orientation of cells.
        This can be calculated using :func:`orientation`.
    right_orientation : np.array, pd.Series
        The ``np.array`` or ``pd.Series`` with orientation of objects.
        This can be calculated using :func:`orientation`.

    Returns
    -------
    Series

    Examples
    --------
    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> tessellation = momepy.morphological_tessellation(buildings)

    Measure orientations:

    >>> blg_orient = momepy.orientation(buildings)
    >>> tess_orient = momepy.orientation(tessellation)

    Compute alignment:

    >>> momepy.cell_alignment(blg_orient, tess_orient)
    0       0.854788
    1      20.829200
    2       5.552120
    3       4.052674
    4       0.159289
             ...
    139     0.189750
    140    17.920139
    141     0.393708
    142     0.024618
    143     0.252122
    Name: orientation, Length: 144, dtype: float64
    """

    if not isinstance(left_orientation, Series):
        left_orientation = Series(left_orientation)
    if not isinstance(right_orientation, Series):
        right_orientation = Series(right_orientation)
    return (left_orientation - right_orientation).abs()
