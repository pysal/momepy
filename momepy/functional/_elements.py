import warnings

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from joblib import Parallel, delayed
from libpysal.cg import voronoi_frames
from libpysal.graph import Graph
from packaging.version import Version
from pandas import MultiIndex, Series

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")
GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")
LPS_GE_411 = Version(libpysal.__version__) >= Version("4.11.dev")

__all__ = [
    "morphological_tessellation",
    "enclosed_tessellation",
    "verify_tessellation",
    "get_nearest_street",
    "get_nearest_node",
    "generate_blocks",
    "buffered_limit",
]


def morphological_tessellation(
    geometry: GeoSeries | GeoDataFrame,
    clip: str | shapely.Geometry | GeoSeries | GeoDataFrame | None = "bounding_box",
    shrink: float = 0.4,
    segment: float = 0.5,
) -> GeoDataFrame:
    """Generate morphological tessellation.

    Morpohological tessellation is a method to divide space into cells based on
    building footprints and Voronoi tessellation. The function wraps
    :func:`libpysal.cg.voronoi_frames` and provides customized default parameters
    following :cite:`fleischmann2020`.

    Tessellation requires data of relatively high level of precision
    and there are three particular patterns causing issues:

    1. Features will collapse into empty polygon - these
       do not have tessellation cell in the end.
    2. Features will split into MultiPolygons - in some cases,
       features with narrow links between parts split into two
       during 'shrinking'. In most cases that is not an issue
       and the resulting tessellation is correct anyway, but
       sometimes this results in a cell being a MultiPolygon,
       which is not correct.
    3. Overlapping features - features which overlap even
       after 'shrinking' cause invalid tessellation geometry.

    All three types can be tested using :class:`momepy.CheckTessellationInput`.

    See :cite:`fleischmann2020` for details of implementation.

    Parameters
    ----------
    geometry : GeoSeries | GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate the space around.
    clip : str | shapely.Geometry | GeoSeries | GeoDataFrame | None
        Polygon used to clip the Voronoi polygons, by default "bounding_box". You can
        pass any option accepted by :func:`libpysal.cg.voronoi_frames` or geopandas
        object that will be automatically unioned.
    shrink : float, optional
        The distance for negative buffer to generate space between adjacent polygons).
        By default 0.4
    segment : float, optional
        The maximum distance between points after discretization. By default 0.5

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an index matching the index of input geometry

    See also
    --------
    momepy.enclosed_tessellation
    momepy.CheckTessellationInput
    momepy.verify_tessellation

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a limit used to clip the extent:

    >>> limit = momepy.buffered_limit(buildings, buffer="adaptive")

    Generate tessellation:

    >>> momepy.morphological_tessellation(buildings).head()
                                                geometry
    0  POLYGON ((1603577.153 6464348.291, 1603576.946...
    1  POLYGON ((1603166.356 6464326.62, 1603166.425 ...
    2  POLYGON ((1603006.941 6464167.63, 1603009.97 6...
    3  POLYGON ((1602995.269 6464132.007, 1603001.768...
    4  POLYGON ((1603084.231 6464104.386, 1603083.773...

    """

    if isinstance(geometry.index, MultiIndex):
        raise ValueError(
            "MultiIndex is not supported in `momepy.morphological_tessellation`."
        )

    if isinstance(clip, GeoSeries | GeoDataFrame):
        clip = clip.union_all() if GPD_GE_10 else clip.unary_union

    return voronoi_frames(
        geometry,
        clip=clip,
        shrink=shrink,
        segment=segment,
        return_input=False,
        as_gdf=True,
    )


def enclosed_tessellation(
    geometry: GeoSeries | GeoDataFrame,
    enclosures: GeoSeries | GeoDataFrame,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
) -> GeoDataFrame:
    """Generate enclosed tessellation

    Enclosed tessellation is an enhanced :func:`morphological_tessellation`, based on
    predefined enclosures and building footprints. We can see enclosed tessellation as
    two-step partitioning of space based on building footprints and boundaries (e.g.
    street network, railway). Original morphological tessellation is used under the hood
    to partition each enclosure.

    Tessellation requires data of relatively high level of precision and there are three
    particular patterns causing issues:

    1. Features will collapse into empty polygon - these
       do not have tessellation cell in the end.
    2. Features will split into MultiPolygons - in some cases,
       features with narrow links between parts split into two during 'shrinking'.
       In most cases that is not an issue and the resulting tessellation is correct
       anyway, but sometimes this results in a cell being a MultiPolygon, which is
       not correct.
    3. Overlapping features - features which overlap even
       after 'shrinking' cause invalid tessellation geometry.

    All three types can be tested using :class:`momepy.CheckTessellationInput`.

    The index of the resulting GeoDataFrame links the input buildings with the output
    geometry. Enclosures with no buildings are also included in the output with negative
    index. Ensure that the input geometry has unique non-negative index for this to work
    correctly.

    Parameters
    ----------
    geometry : GeoSeries | GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate the space around.
    enclosures : GeoSeries | GeoDataFrame
        The enclosures geometry, which can be generated using :func:`momepy.enclosures`.
    shrink : float, optional
        The distance for negative buffer to generate space between adjacent polygons).
        By default 0.4
    segment : float, optional
        The maximum distance between points after discretization. By default 0.5
    threshold : float, optional
        The minimum threshold for a building to be considered within an enclosure.
        Threshold is a ratio of building area which needs to be within an enclosure to
        inlude it in the tessellation of that enclosure. Resolves sliver geometry
        issues. If None, the check is skipped and all intersecting buildings are
        considered. By default 0.05
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using all available cores.
        By default -1

    Warnings
    --------
    Due to the floating point precision issues in clipping the tessellation cells to the
    extent of their parental enclosures, the result does not form a precise polygonal
    coverage. To build a contiguity graph, use fuzzy contiguity builder with a small
    buffer, e.g.::

        from libpysal import graph

        graph.Graph.build_fuzzy_contiguity(tessellation, buffer=1e-6)

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an index matching the index of input geometry and a column
        matching the index of input enclosures.

    See also
    --------
    momepy.enclosures
    momepy.morphological_tessellation
    momepy.CheckTessellationInput
    momepy.verify_tessellation

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Generate enclosures:

    >>> enclosures = momepy.enclosures(streets)

    Generate tessellation:

    >>> momepy.enclosed_tessellation(buildings, enclosures).head()
                                                  geometry  enclosure_index
    0    POLYGON ((1603572.779 6464354.58, 1603572.505 ...                0
    113  POLYGON ((1603543.601 6464322.376, 1603543.463...                0
    114  POLYGON ((1603525.157 6464283.592, 1603524.725...                0
    125  POLYGON ((1603601.446 6464256.455, 1603600.982...                0
    126  POLYGON ((1603528.593 6464221.033, 1603527.796...                0
    """

    if isinstance(geometry.index, MultiIndex):
        raise ValueError(
            "MultiIndex is not supported in `momepy.enclosed_tessellation`."
        )

    # convert to GeoDataFrame and add position (we will need it later)
    enclosures = enclosures.geometry.to_frame()
    enclosures["position"] = range(len(enclosures))

    # preserve index name if exists
    index_name = enclosures.index.name
    if not index_name:
        index_name = "enclosure_index"
    enclosures[index_name] = enclosures.index

    # figure out which enlosures contain which buildings
    if GPD_GE_013:
        inp, res = geometry.sindex.query(enclosures.geometry, predicate="intersects")
    else:
        inp, res = geometry.sindex.query_bulk(
            enclosures.geometry, predicate="intersects"
        )

    # find out which enclosures contain one and multiple buildings
    unique, counts = np.unique(inp, return_counts=True)
    splits = unique[counts > 1]
    single = unique[counts == 1]
    altered = unique[counts > 0]

    # prepare input for parallel processing
    tuples = [
        (
            enclosures.index[i],  # enclosure index
            enclosures.geometry.iloc[i],  # enclosure geometry
            geometry.iloc[res[inp == i]],  # buildings within the enclosure
        )
        for i in splits
    ]

    # generate tessellation in parallel
    new = Parallel(n_jobs=n_jobs)(
        delayed(_tess)(*t, threshold, shrink, segment, index_name) for t in tuples
    )

    new_df = pd.concat(new, axis=0)

    # some enclosures had building intersections that did not meet the threshold
    if -1 in new_df.index:
        unchanged_in_new = new_df.loc[[-1]]
        new_df = new_df.drop(-1)
        clean_blocks = pd.concat(
            [
                enclosures.drop(enclosures.index[altered]).drop(columns="position"),
                unchanged_in_new,
            ]
        )
    else:
        clean_blocks = enclosures.drop(enclosures.index[altered]).drop(
            columns="position"
        )

    # assign negative index to enclosures with no buildings
    clean_blocks.index = range(-len(clean_blocks), 0, 1)

    # get building index for enclosures with single building
    singles = enclosures.iloc[single]
    singles.index = singles.position.loc[singles.index].apply(
        lambda ix: geometry.iloc[res[inp == ix]].index[0]
    )
    # combine results
    return pd.concat([new_df, singles.drop(columns="position"), clean_blocks])


def _tess(ix, poly, blg, threshold, shrink, segment, enclosure_id):
    """Generate tessellation for a single enclosure. Helper for enclosed_tessellation"""
    # check if threshold is set and filter buildings based on the threshold
    if threshold:
        blg = blg[
            shapely.area(shapely.intersection(blg.geometry.array, poly))
            > (shapely.area(blg.geometry.array) * threshold)
        ]

    if len(blg) > 1:
        tess = voronoi_frames(
            blg,
            clip=poly,
            shrink=shrink,
            segment=segment,
            return_input=False,
            as_gdf=True,
        )
        tess[enclosure_id] = ix
        return tess

    ## in case a single building is left in blg
    if len(blg) == 1:
        assigned_ix = blg.index[0]
    else:
        assigned_ix = -1

    return GeoDataFrame(
        {enclosure_id: ix},
        geometry=[poly],
        index=[assigned_ix],
        crs=blg.crs,
    )


def verify_tessellation(tessellation, geometry):
    """Check whether result matches buildings and contains only Polygons.

    Checks if the generated tessellation fully matches the input buildings, i.e. if
    there are all building indices present in the tessellation. Also checks if there are
    any MultiPolygons present in the tessellation. The former is often caused by
    buildings collapsing during the generation process, the latter is usually caused by
    errors in the input geometry, overlapping buildings, or narrow links between parts
    of the building.

    Parameters
    ----------
    tessellation : GeoSeries | GeoDataFrame
        tessellation geometry
    geometry : GeoSeries | GeoDataFrame
        building geometry used to generate tessellation

    Returns
    -------
    tuple(excluded, multipolygons)
        Tuple of indices of building IDs not present in tessellations and MultiPolygons.

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a limit used to clip the extent:

    >>> limit = momepy.buffered_limit(buildings, buffer="adaptive")

    Generate tessellation:

    >>> tessellation = momepy.morphological_tessellation(buildings)

    Verify the result.

    >>> excluded, multipolygons = momepy.verify_tessellation(tessellation, buildings)
    """

    if isinstance(geometry.index, MultiIndex) or isinstance(
        tessellation.index, MultiIndex
    ):
        raise ValueError("MultiIndex is not supported in `momepy.verify_tessellation`.")

    # check against input layer
    ids_original = geometry.index
    ids_generated = tessellation.index
    collapsed = pd.Index([])
    if len(ids_original) != len(ids_generated):
        collapsed = ids_original.difference(ids_generated)
        warnings.warn(
            message=(
                "Tessellation does not fully match buildings. "
                f"{len(collapsed)} element(s) disappeared "
                f"during generation. Index of the affected elements: {collapsed}."
            ),
            category=UserWarning,
            stacklevel=2,
        )

    # check MultiPolygons - usually caused by error in input geometry
    multipolygons = tessellation[
        tessellation.geometry.geom_type == "MultiPolygon"
    ].index
    if len(multipolygons) > 0:
        warnings.warn(
            message=(
                "Tessellation contains MultiPolygon elements. Initial "
                "objects should  be edited. Index of affected "
                f"elements: {list(multipolygons)}."
            ),
            category=UserWarning,
            stacklevel=2,
        )
    return collapsed, multipolygons


def get_nearest_street(
    buildings: GeoSeries | GeoDataFrame,
    streets: GeoSeries | GeoDataFrame,
    max_distance: float | None = None,
) -> Series:
    """Identify the nearest street for each building.

    Parameters
    ----------
    buildings : GeoSeries | GeoDataFrame
        GeoSeries or GeoDataFrame of buildings
    streets : GeoSeries | GeoDataFrame
        GeoSeries or GeoDataFrame of streets
    max_distance : float | None, optional
        Maximum distance within which to query for nearest street. Must be
        greater than 0. By default None, indicating no distance limit. Note that it is
        advised to set a limit to avoid long processing times.

    Notes
    -----
    In case of multiple streets within the same distance, only one is returned.

    Returns
    -------
    np.ndarray
        array containing the index of the nearest street for each building

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Get street index.

    >>> momepy.get_nearest_street(buildings, streets)
    0       0.0
    1      33.0
    2      10.0
    3       8.0
    4       8.0
        ...
    139    34.0
    140    32.0
    141    21.0
    142    16.0
    143    19.0
    Length: 144, dtype: float64
    """
    blg_idx, str_idx = streets.sindex.nearest(
        buildings.geometry, return_all=False, max_distance=max_distance
    )

    ids = pd.Series(None, index=buildings.index, dtype=streets.index.dtype)

    ids.iloc[blg_idx] = streets.index[str_idx]
    return ids


def get_nearest_node(
    buildings: GeoSeries | GeoDataFrame,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    nearest_edge: Series,
) -> Series:
    """Identify the nearest node for each building.

    Snap each building to the closest street network node on the closest network edge.
    This assumes that the nearest street network edge has already been identified using
    :func:`get_nearest_street`.

    The ``edges`` and ``nodes`` GeoDataFrames are expected to be an outcome of
    :func:`momepy.nx_to_gdf` or match its structure with ``["node_start", "node_end"]``
    columns and their meaning.


    Parameters
    ----------
    buildings : GeoSeries | GeoDataFrame
        GeoSeries or GeoDataFrame of buildings.
    nodes : GeoDataFrame
        A GeoDataFrame containing street nodes.
    edges : GeoDataFrame
        A GeoDataFrame containing street edges with ``["node_start", "node_end"]``
        columns marking start and end nodes of each edge. These are the default
        outcome of :func:`momepy.nx_to_gdf`.
    nearest_edge : Series
        A Series aligned with ``buildings`` containing the information on the nearest
        street edge. Matches the outcome of :func:`get_nearest_street`.

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Pass an object via ``networkx`` to get the nodes and necessary information.

    >>> G = momepy.gdf_to_nx(streets)
    >>> nodes, edges = momepy.nx_to_gdf(G)

    Get nearest edge:

    >>> buildings["edge_index"] = momepy.get_nearest_street(buildings, edges)

    Get nearest node:

    >>> momepy.get_nearest_node(buildings, nodes, edges, buildings["edge_index"])
    0       0.0
    1       9.0
    2      11.0
    3      11.0
    4      11.0
        ...
    139     1.0
    140    20.0
    141    15.0
    142     2.0
    143    22.0
    Length: 144, dtype: float64
    """

    if (
        isinstance(buildings.index, MultiIndex)
        or isinstance(nearest_edge.index, MultiIndex)
        or isinstance(nodes.index, MultiIndex)
        or isinstance(edges.index, MultiIndex)
    ):
        raise ValueError("MultiIndex is not supported in `momepy.get_nearest_node`.")

    # treat possibly missing edge index
    a = np.empty(len(buildings))
    na_mask = np.isnan(nearest_edge)
    a[na_mask] = np.nan

    streets = edges.loc[nearest_edge[~na_mask]]
    starts = nodes.loc[streets["node_start"]].distance(buildings[~na_mask], align=False)
    ends = nodes.loc[streets["node_end"]].distance(buildings[~na_mask], align=False)
    mask = starts.values > ends.values
    r = starts.index.to_numpy(copy=True)
    r[mask] = ends.index[mask]

    a[~na_mask] = r
    return pd.Series(a, index=buildings.index)


def generate_blocks(
    tessellation: GeoDataFrame, edges: GeoDataFrame, buildings: GeoDataFrame
) -> tuple[GeoDataFrame, Series]:
    """
    Generate blocks based on buildings, tessellation, and street network.
    Dissolves tessellation cells based on street-network based polygons.
    Links resulting ID to ``tessellation`` and returns
    ``blocks`` and ``tessellation`` ids.

    Parameters
    ----------
    tessellation : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    edges : GeoDataFrame
        A GeoDataFrame containing a street network.
    buildings : GeoDataFrame
        A GeoDataFrame containing buildings.

    Notes
    -----
    This function assumes morphological tessellation and 1:1 relationship
    between buildings and cells. Tesselation cells that do not have buildings
    can break the functionality.

    Returns
    -------
    blocks : GeoDataFrame
        A GeoDataFrame containing generated blocks.
    tessellation_ids : Series
        A Series derived from morphological tessellation with block ID.

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Generate tessellation:

    >>> tessellation = momepy.morphological_tessellation(buildings)
    >>> tessellation
                                                geometry
    0  POLYGON ((1603577.153 6464348.291, 1603576.946...
    1  POLYGON ((1603166.356 6464326.62, 1603166.425 ...
    2  POLYGON ((1603006.941 6464167.63, 1603009.97 6...
    3  POLYGON ((1602995.269 6464132.007, 1603001.768...
    4  POLYGON ((1603084.231 6464104.386, 1603083.773...

    >>> blocks, tessellation_id = momepy.generate_blocks(
    ...     tessellation, streets, buildings
    ... )
    >>> blocks.head()
                                                geometry
    0  POLYGON ((1603500.079 6464214.019, 1603499.565...
    1  POLYGON ((1603431.893 6464278.302, 1603431.553...
    2  POLYGON ((1603321.257 6464125.859, 1603320.938...
    3  POLYGON ((1603137.411 6464124.658, 1603137.116...
    4  POLYGON ((1603179.384 6463961.584, 1603179.357...

    ``tessellation_id`` can be directly assigned to its
    respective parental DataFrame directly.

    >>> tessellation["block_id"] = tessellation_id
    """

    if (
        isinstance(buildings.index, MultiIndex)
        or isinstance(tessellation.index, MultiIndex)
        or isinstance(edges.index, MultiIndex)
    ):
        raise ValueError("MultiIndex is not supported in `momepy.generate_blocks`.")
    id_name: str = "bID"

    # slice the tessellations by the street network
    cut = gpd.overlay(
        tessellation,
        gpd.GeoDataFrame(geometry=edges.buffer(0.001)),
        how="difference",
    )
    cut = cut.explode(ignore_index=True)
    # touching tessellations form a block
    weights = Graph.build_contiguity(cut, rook=False)
    cut["component"] = weights.component_labels

    # generate block geometries
    buildings_c = buildings.copy()
    buildings_c.geometry = buildings_c.representative_point()  # make points
    centroids_temp_id = gpd.sjoin(
        buildings_c,
        cut[[cut.geometry.name, "component"]],
        how="left",
        predicate="within",
    )
    cells_copy = tessellation[[tessellation.geometry.name]].merge(
        centroids_temp_id[["component"]], right_index=True, left_index=True, how="left"
    )
    blocks = cells_copy.dissolve(by="component").explode(ignore_index=True)

    # assign block ids to buildings and tessellations
    centroids_w_bl_id2 = gpd.sjoin(buildings_c, blocks, how="left", predicate="within")
    buildings_id = centroids_w_bl_id2["index_right"]
    buildings_id.name = id_name
    cells_m = tessellation.merge(
        buildings_id, left_index=True, right_index=True, how="left"
    )
    tessellation_id = cells_m[id_name]

    return blocks, tessellation_id


def buffered_limit(
    gdf: GeoDataFrame | GeoSeries,
    buffer: float | str = 100,
    min_buffer: float = 0,
    max_buffer: float = 100,
    **kwargs,
) -> shapely.Geometry:
    """
    Define limit for tessellation as a buffer around buildings.

    The function calculates a buffer around buildings and returns a MultiPolygon or
    Polygon defining the study area. The buffer can be either a fixed number or
    "adaptive" which calculates the buffer based on Gabriel graph.

    See :cite:`fleischmann2020` for details.

    Parameters
    ----------
    gdf : GeoDataFrame | GeoSeries
        A GeoDataFrame containing building footprints.
    buffer : float | str, optional
        A buffer around buildings limiting the extend of tessellation. If "adaptive",
        the buffer is calculated based on Gabriel graph as the half of the maximum
        distance between neighbors (represented as centroids) of each node + 10% of
        such the maximum distance. The lower and upper bounds can be furhter specified
        by ``min_buffer`` and ``max_buffer``. By default 100.
    min_buffer : float, optional
        The minimum adaptive buffer distance. By default 0.
    max_buffer : float, optional
        The maximum adaptive buffer distance. By default 100.
    **kwargs
        Keyword arguments passed to :meth:`geopandas.GeoSeries.buffer`.

    Returns
    -------
    MultiPolygon
        A MultiPolygon or Polygon defining the study area.

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    >>> limit = momepy.buffered_limit(buildings)
    >>> type(limit)
    <class 'shapely.geometry.polygon.Polygon'>
    """
    if buffer == "adaptive":
        if not LPS_GE_411:
            raise ImportError(
                "Adaptive buffer requires libpysal 4.11 or higher."
            )  # because https://github.com/pysal/libpysal/pull/709
        gabriel = Graph.build_triangulation(gdf.centroid, "gabriel", kernel="identity")
        max_dist = gabriel.aggregate("max")
        buffer = np.clip(max_dist / 2 + max_dist * 0.1, min_buffer, max_buffer).values

    elif not isinstance(buffer, int | float):
        raise ValueError("`buffer` must be either 'adaptive' or a number.")

    return (
        gdf.buffer(buffer, **kwargs).union_all()
        if GPD_GE_10
        else gdf.buffer(buffer, **kwargs).unary_union
    )
