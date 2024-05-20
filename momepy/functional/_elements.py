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

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")
GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")
LPS_GE_411 = Version(libpysal.__version__) >= Version("4.11.dev")

__all__ = [
    "morphological_tessellation",
    "enclosed_tessellation",
    "verify_tessellation",
    "get_nearest_street",
    "buffered_limit",
]


def morphological_tessellation(
    geometry: GeoSeries | GeoDataFrame,
    clip: str | shapely.Geometry | GeoSeries | GeoDataFrame | None = "bounding_box",
    shrink: float = 0.4,
    segment: float = 0.5,
) -> GeoSeries:
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
    GeoSeries
        GeoSeries with an index matching the index of input geometry

    See also
    --------
    momepy.enclosed_tessellation
    momepy.CheckTessellationInput
    momepy.verify_tessellation
    """
    if isinstance(clip, GeoSeries | GeoDataFrame):
        clip = clip.union_all() if GPD_GE_10 else clip.unary_union

    return voronoi_frames(
        geometry,
        clip=clip,
        shrink=shrink,
        segment=segment,
        return_input=False,
        as_gdf=False,
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
    """

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

    return GeoDataFrame(
        {enclosure_id: ix},
        geometry=[poly],
        index=[-1],
        crs=blg.crs,
    )


def verify_tessellation(tesselation, geometry):
    """Check whether result matches buildings and contains only Polygons.

    Checks if the generated tessellation fully matches the input buildings, i.e. if
    there are all building indices present in the tessellation. Also checks if there are
    any MultiPolygons present in the tessellation. The former is often caused by
    buildings collapsing during the generation process, the latter is usually caused by
    errors in the input geometry, overlapping buildings, or narrow links between parts
    of the building.

    Parameters
    ----------
    tesselation : GeoSeries | GeoDataFrame
        tessellation geometry
    geometry : GeoSeries | GeoDataFrame
        building geometry used to generate tessellation

    Returns
    -------
    tuple(excluded, multipolygons)
        Tuple of indices of building IDs not present in tessellations and MultiPolygons.
    """
    # check against input layer
    ids_original = geometry.index
    ids_generated = tesselation.index
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
    multipolygons = tesselation[tesselation.geometry.geom_type == "MultiPolygon"].index
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
) -> np.ndarray:
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
    """
    blg_idx, str_idx = streets.sindex.nearest(
        buildings.geometry, return_all=False, max_distance=max_distance
    )

    if streets.index.dtype == "object":
        ids = np.empty(len(buildings), dtype=object)
    else:
        ids = np.empty(len(buildings), dtype=np.float32)
        ids[:] = np.nan

    ids[blg_idx] = streets.index[str_idx]
    return ids


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
    >>> limit = mm.buffered_limit(buildings_df)
    >>> type(limit)
    shapely.geometry.polygon.Polygon
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
