import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from joblib import Parallel, delayed
from libpysal.cg import voronoi_frames
from libpysal.graph import Graph
from packaging.version import Version
from pandas import Series

GPD_GE_013 = Version(gpd.__version__) >= Version("0.13.0")
GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")

__all__ = [
    "morphological_tessellation",
    "enclosed_tessellation",
    "verify_tessellation",
    "get_nearest_street",
    "generate_blocks",
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
    enclosures: GeoSeries,
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
    enclosures : GeoSeries
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

    new_df = pd.concat(new, axis=0)

    # some enclosures had building intersections that did not meet the threshold
    if -1 in new_df.index:
        unchanged_in_new = new_df.loc[[-1]]
        new_df = new_df.drop(-1)
        clean_blocks = pd.concat(
            [enclosures.drop(altered).drop(columns="position"), unchanged_in_new]
        )
    else:
        clean_blocks = enclosures.drop(altered).drop(columns="position")

    # assign negative index to enclosures with no buildings
    clean_blocks.index = range(-len(clean_blocks), 0, 1)

    # get building index for enclosures with single building
    singles = enclosures.iloc[single]
    singles.index = singles.position.loc[single].apply(
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


def generate_blocks(
    tessellation: GeoDataFrame, edges: GeoDataFrame, buildings: GeoDataFrame
) -> tuple[Series, Series, Series]:
    """
    Generate blocks based on buildings, tessellation, and street network.
    Dissolves tessellation cells based on street-network based polygons.
    Links resulting ID to ``buildings`` and ``tessellation`` and returns
    ``blocks``, ``buildings_ds`` and ``tessellation`` ids.

    Parameters
    ----------
    tessellation : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    edges : GeoDataFrame
        A GeoDataFrame containing a street network.
    buildings : GeoDataFrame
        A GeoDataFrame containing buildings.

    Returns
    -------
    blocks : GeoDataFrame
        A GeoDataFrame containing generated blocks.
    buildings_ids : Series
        A Series derived from buildings with block ID.
    tessellation_ids : Series
        A Series derived from morphological tessellation with block ID.

    Examples
    --------
    >>> blocks, buildings_id, tessellation_id = mm.generate_blocks(tessellation_df,
    ... streets_df, buildings_df)
    >>> blocks.head()
            geometry
    0	    POLYGON ((1603560.078648818 6464202.366899694,...
    1	    POLYGON ((1603457.225976106 6464299.454696888,...
    2	    POLYGON ((1603056.595487018 6464093.903488506,...
    3	    POLYGON ((1603260.943782872 6464141.327631323,...
    4	    POLYGON ((1603183.399594798 6463966.109982309,...
    """

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
    # return blocks, centroids_w_bl_id2
    buildings_id = centroids_w_bl_id2["index_right"]
    buildings_id.name = id_name
    cells_m = tessellation.merge(
        buildings_id, left_index=True, right_index=True, how="left"
    )
    tessellation_id = cells_m[id_name]

    return blocks, buildings_id, tessellation_id
