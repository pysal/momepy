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


    See :cite:`fleischmann2020` for details of implementation.


    Parameters
    ----------
    geometry : gpd.GeoSeries | gpd.GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate.
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
    geometry,
    enclosures: gpd.GeoSeries,
    shrink=0.4,
    segment=0.5,
    threshold=0.05,
    n_jobs=-1,
):
    enclosures = enclosures.to_frame()
    enclosures["position"] = range(len(enclosures))
    index_name = enclosures.index.name
    if not index_name:
        index_name = "enclosure_index"
    enclosures[index_name] = enclosures.index

    if GPD_GE_013:
        inp, res = geometry.sindex.query(enclosures.geometry, predicate="intersects")
    else:
        inp, res = geometry.sindex.query_bulk(
            enclosures.geometry, predicate="intersects"
        )

    unique, counts = np.unique(inp, return_counts=True)
    splits = unique[counts > 1]
    single = unique[counts == 1]
    altered = unique[counts > 0]

    tuples = []
    for i in splits:
        tuples.append(
            (
                enclosures.index[i],
                enclosures.geometry.iloc[i],
                geometry.iloc[res[inp == i]],
            )
        )
    new = Parallel(n_jobs=n_jobs)(
        delayed(_tess)(*t, threshold, shrink, segment, index_name) for t in tuples
    )

    # finalise the result
    singles = enclosures.iloc[single]
    singles.index = singles.position.loc[single].apply(
        lambda ix: geometry.iloc[res[inp == ix]].index[0]
    )
    clean_blocks = enclosures.drop(altered)
    clean_blocks.index = range(-len(clean_blocks), 0, 1)
    return pd.concat(
        new + [singles.drop(columns="position"), clean_blocks.drop(columns="position")]
    )


def _tess(ix, enclosure, blg, threshold, shrink, segment, enclosure_id):
    poly = enclosure
    within = blg[
        shapely.area(shapely.intersection(blg.geometry.array, poly))
        > (shapely.area(blg.geometry.array) * threshold)
    ].copy()
    if len(within) > 1:
        tess = voronoi_frames(
            within,
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
    """Check whether result matches buildings and contains only Polygons."""
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
            stacklevel=4,
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
            stacklevel=4,
        )
    return collapsed, multipolygons
