import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed
from libpysal.cg import voronoi_frames
from packaging.version import Version

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")
GPD_GE_10 = Version(gpd.__version__) >= Version("1.0.0")

__all__ = [
    "morphological_tessellation",
    "enclosed_tessellation",
    "verify_tessellation",
]


def morphological_tessellation(
    geometry: gpd.GeoSeries | gpd.GeoDataFrame,
    clip: str
    | shapely.Geometry
    | gpd.GeoSeries
    | gpd.GeoDataFrame
    | None = "bounding_box",
    shrink: float = 0.4,
    segment: float = 0.5,
) -> gpd.GeoSeries:
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
    geometry : gpd.GeoSeries | gpd.GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate the space around.
    clip : str | shapely.Geometry | gpd.GeoSeries | gpd.GeoDataFrame | None
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
    gpd.GeoSeries
        GeoSeries with an index matching the index of input geometry

    See also
    --------
    momepy.enclosed_tessellation
    momepy.CheckTessellationInput
    momepy.verify_tessellation
    """
    if isinstance(clip, gpd.GeoSeries | gpd.GeoDataFrame):
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
    geometry: gpd.GeoSeries | gpd.GeoDataFrame,
    enclosures: gpd.GeoSeries,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
) -> gpd.GeoDataFrame:
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

    Parameters
    ----------
    geometry : gpd.GeoSeries | gpd.GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate the space around.
    enclosures : gpd.GeoSeries
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
    gpd.GeoDataFrame
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
    enclosures = enclosures.to_frame()
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

    # get building index for enclosures with single building
    singles = enclosures.iloc[single]
    singles.index = singles.position.loc[single].apply(
        lambda ix: geometry.iloc[res[inp == ix]].index[0]
    )
    # combine results
    clean_blocks = enclosures.drop(altered)
    clean_blocks.index = range(
        -len(clean_blocks), 0, 1
    )  # assign negative index to enclosures with no buildings
    return pd.concat(
        new + [singles.drop(columns="position"), clean_blocks.drop(columns="position")]
    )


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

    return gpd.GeoDataFrame(
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
    tesselation : gpd.GeoSeries | gpd.GeoDataFrame
        tessellation geometry
    geometry : gpd.GeoSeries | gpd.GeoDataFrame
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
                f"{len(collapsed)} element(s) collapsed "
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
