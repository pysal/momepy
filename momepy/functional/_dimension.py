import math

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from numpy.typing import NDArray
from pandas import DataFrame, Series

__all__ = [
    "volume",
    "floor_area",
    "courtyard_area",
    "longest_axis_length",
    "perimeter_wall",
    "street_profile",
    "weighted_character",
]

try:
    from numba import njit

    HAS_NUMBA = True
except (ModuleNotFoundError, ImportError):
    HAS_NUMBA = False
    from libpysal.common import jit as njit


def volume(
    area: NDArray[np.float_] | Series,
    height: NDArray[np.float_] | Series,
) -> NDArray[np.float_] | Series:
    """
    Calculates volume of each object in given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    area : NDArray[np.float_] | Series
        array of areas
    height : NDArray[np.float_] | Series
        array of heights

    Returns
    -------
    NDArray[np.float_] | Series
        array of a type depending on the input
    """
    return area * height


def floor_area(
    area: NDArray[np.float_] | Series,
    height: NDArray[np.float_] | Series,
    floor_height: float | NDArray[np.float_] | Series = 3,
) -> NDArray[np.float_] | Series:
    """Calculates floor area of each object based on height and area.

    The number of
    floors is simplified into the formula: ``height // floor_height``. B
    y default one floor is approximated to 3 metres.

    .. math::
        area * \\frac{height}{floor_height}


    Parameters
    ----------
    area : NDArray[np.float_] | Series
        array of areas
    height : NDArray[np.float_] | Series
        array of heights
    floor_height : float | NDArray[np.float_] | Series, optional
        float denoting the uniform floor height or an aarray reflecting the building
        height by geometry, by default 3

    Returns
    -------
    NDArray[np.float_] | Series
        array of a type depending on the input
    """
    return area * (height // floor_height)


def courtyard_area(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates area of holes within geometry - area of courtyards.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    return Series(
        shapely.area(
            shapely.polygons(shapely.get_exterior_ring(geometry.geometry.array))
        )
        - geometry.area,
        index=geometry.index,
        name="courtyard_area",
    )


def longest_axis_length(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the length of the longest axis of object.

    Axis is defined as a
    diameter of minimal bounding circle around the geometry. It does
    not have to be fully inside an object.

    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.

    Returns
    -------
    Series
    """
    return shapely.minimum_bounding_radius(geometry.geometry) * 2


def perimeter_wall(
    geometry: GeoDataFrame | GeoSeries, graph: Graph | None = None
) -> Series:
    """Calculate the perimeter wall length the joined structure.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    graph : Graph | None, optional
        Graph encoding Queen contiguity of ``geometry``. If ``None`` Queen contiguity is
        built on the fly.

    Returns
    -------
    Series
    """

    if graph is None:
        graph = Graph.build_contiguity(geometry)

    isolates = graph.isolates

    # measure perimeter walls of connected components while ignoring isolates
    blocks = geometry.drop(isolates)
    component_perimeter = (
        blocks[[blocks.geometry.name]]
        .set_geometry(blocks.buffer(0.01))  # type: ignore
        .dissolve(by=graph.component_labels.drop(isolates))
        .exterior.length
    )

    # combine components with isolates
    results = Series(np.nan, index=geometry.index, name="perimeter_wall")
    results.loc[isolates] = geometry.geometry[isolates].exterior.length
    results.loc[results.index.drop(isolates)] = component_perimeter.loc[
        graph.component_labels.loc[results.index.drop(isolates)]
    ].values

    return results


def weighted_character(
    y: NDArray[np.float_] | Series, area: NDArray[np.float_] | Series, graph: Graph
) -> Series:
    """Calculates the weighted character.

    Character weighted by the area of the objects within neighbors defined in ``graph``.
    Results are index based on ``graph``.

    .. math::
        \\frac{\\sum_{i=1}^{n} {character_{i} * area_{i}}}{\\sum_{i=1}^{n} area_{i}}

    Adapted from :cite:`dibble2017`.

    Notes
    -----
    The index of ``y`` and ``area`` must match the index along which the ``graph`` is
    built.

    Parameters
    ----------
    y : NDArray[np.float_] | Series
        The character values to be weighted.
    area : NDArray[np.float_] | Series
        The area values to be used as weightss
    graph : libpysal.graph.Graph
        A spatial weights matrix for values and areas.

    Returns
    -------
    Series
        A Series containing the resulting values.

    Examples
    --------
    >>> res = mm.weighted_character(buildings_df['height'],
    ...                     buildings_df.geometry.area, graph)
    """

    stats = graph.describe(y * area, statistics=["sum"])["sum"]
    agg_area = graph.describe(area, statistics=["sum"])["sum"]

    return stats / agg_area


def street_profile(
    streets: GeoDataFrame,
    buildings: GeoDataFrame,
    distance: float = 10,
    tick_length: float = 50,
    height: None | Series = None,
) -> DataFrame:
    """Calculates the street profile characters.

    This functions returns a DataFrame with widths, standard deviation of width,
    openness, heights, standard deviation of height and
    ratio height/width. The algorithm generates perpendicular lines to the ``streets``
    dataframe features every ``distance`` and measures values on intersections with
    features of ``buildings``. If no feature is reached within ``tick_length`` its value
    is set as width (being a theoretical maximum).

    Derived from :cite:`araldi2019`.

    Parameters
    ----------
    streets : GeoDataFrame
        A GeoDataFrame containing streets to analyse.
    buildings : GeoDataFrame
        A GeoDataFrame containing buildings along the streets.
        Only Polygon geometries are currently supported.
    distance : int (default 10)
        The distance between perpendicular ticks.
    tick_length : int (default 50)
        The length of ticks.
    height: pd.Series (default None)
        The ``pd.Series`` where building height are stored. If set to ``None``,
        height and ratio height/width will not be calculated.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> street_prof = momepy.street_profile(streets_df,
    ...                 buildings_df, height=buildings_df['height'])
    >>> streets_df['width'] = street_prof['width']
    >>> streets_df['deviations'] = street_prof['width_deviation']
    """

    # filter relevant buildings and streets
    inp, res = shapely.STRtree(streets.geometry).query(
        buildings.geometry, predicate="dwithin", distance=tick_length // 2
    )
    buildings_near_streets = np.unique(inp)
    streets_near_buildings = np.unique(res)
    relevant_buildings = buildings.iloc[buildings_near_streets].reset_index(drop=True)
    relevant_streets = streets.iloc[streets_near_buildings].reset_index(drop=True)
    if height is not None:
        height = height.iloc[buildings_near_streets].reset_index(drop=True)

    # calculate street profile on a subset of the data
    partial_res = _street_profile(
        relevant_streets,
        relevant_buildings,
        distance=distance,
        tick_length=tick_length,
        height=height,
    )

    # return full result with defaults
    final_res = pd.DataFrame(np.nan, index=streets.index, columns=partial_res.columns)
    final_res.iloc[streets_near_buildings[partial_res.index.values]] = (
        partial_res.values
    )
    ## streets with no buildings get the theoretical width and max openness
    final_res.loc[final_res["width"].isna(), "width"] = tick_length
    final_res.loc[final_res["openness"].isna(), "openness"] = 1

    return final_res


def _street_profile(
    streets: GeoDataFrame,
    buildings: GeoDataFrame,
    distance: float = 10,
    tick_length: float = 50,
    height: None | Series = None,
) -> DataFrame:
    """Helper function that actually calculates the street profile characters."""

    ## generate points for every street at `distance` intervals
    segments = streets.segmentize(distance)
    coords, coord_indxs = shapely.get_coordinates(segments, return_index=True)
    starts = ~pd.Series(coord_indxs).duplicated(keep="first")
    ends = ~pd.Series(coord_indxs).duplicated(keep="last")
    end_markers = starts | ends

    ## generate tick streings
    njit_ticks = generate_ticks(coords, end_markers.values, tick_length)
    ticks = shapely.linestrings(njit_ticks.reshape(-1, 2, 2))

    ## find the length of intersection of the nearest building for every tick
    inp, res = buildings.geometry.sindex.query(ticks, predicate="intersects")
    intersections = shapely.intersection(ticks[inp], buildings.geometry.array[res])
    distances = shapely.distance(intersections, shapely.points(coords[inp // 2]))

    # streets which intersect buildings have 0 distance to them
    distances[np.isnan(distances)] = 0
    min_distances = pd.Series(distances).groupby(inp).min()
    dists = np.full((len(ticks),), np.nan)
    dists[min_distances.index.values] = min_distances.values

    ## generate tick values and groupby street
    tick_coords = np.repeat(coord_indxs, 2)
    ## multiple agg to avoid custom apply
    left_ticks = (
        pd.Series(dists[::2])
        .groupby(tick_coords[::2])
        .mean()
        .replace(np.nan, tick_length // 2)
    )
    right_ticks = (
        pd.Series(dists[1::2])
        .groupby(tick_coords[1::2])
        .mean()
        .replace(np.nan, tick_length // 2)
    )
    w = left_ticks + right_ticks

    grouper = pd.Series(dists).groupby(tick_coords)
    openness_agg = grouper.agg(["size", "count"])
    # proportion of NAs
    o = (openness_agg["size"] - openness_agg["count"]) / openness_agg["size"]
    # needs to be seperate to pass ddof
    wd = grouper.std(ddof=0)

    final_result = pd.DataFrame(
        np.nan, columns=["width", "openness", "width_deviation"], index=streets.index
    )
    final_result["width"] = w
    final_result["openness"] = o
    final_result["width_deviation"] = wd

    ## if heights are available add heights stats to the result
    if height is not None:
        min_heights = height.loc[res].groupby(inp).min()
        tick_heights = np.full((len(ticks),), np.nan)
        tick_heights[min_heights.index.values] = min_heights.values
        heights_res = pd.Series(tick_heights).groupby(tick_coords).agg(["mean", "std"])
        final_result["height"] = heights_res["mean"]
        final_result["height_deviation"] = heights_res["std"]
        final_result["hw_ratio"] = final_result["height"] / final_result[
            "width"
        ].replace(0, np.nan)

    return final_result


# angle between two points
@njit
def _get_angle_njit(x1, y1, x2, y2):
    """
    pt1, pt2 : tuple
    """
    x_diff = x2 - x1
    y_diff = y2 - y1
    return math.degrees(math.atan2(y_diff, x_diff))


# get the second end point of a tick
# p1 = bearing + 90
@njit
def _get_point_njit(x1, y1, bearing, dist):
    bearing = math.radians(bearing)
    x = x1 + dist * math.cos(bearing)
    y = y1 + dist * math.sin(bearing)
    return np.array((x, y))


@njit
def generate_ticks(list_points, end_markers, tick_length):
    ticks = np.empty((len(list_points) * 2, 4), dtype=np.float64)

    for i in range(len(list_points)):
        tick_pos = i * 2
        end = end_markers[i]
        pt = list_points[i]

        if end:
            ticks[tick_pos, :] = np.array([pt[0], pt[1], pt[0], pt[1]])
            ticks[tick_pos + 1, :] = np.array([pt[0], pt[1], pt[0], pt[1]])
        else:
            next_pt = list_points[i + 1]
            njit_angle1 = _get_angle_njit(pt[0], pt[1], next_pt[0], next_pt[1])
            njit_end_1 = _get_point_njit(
                pt[0], pt[1], njit_angle1 + 90, tick_length / 2
            )
            njit_angle2 = _get_angle_njit(njit_end_1[0], njit_end_1[1], pt[0], pt[1])
            njit_end_2 = _get_point_njit(
                njit_end_1[0], njit_end_1[1], njit_angle2, tick_length
            )
            ticks[tick_pos, :] = np.array([njit_end_1[0], njit_end_1[1], pt[0], pt[1]])
            ticks[tick_pos + 1, :] = np.array(
                [njit_end_2[0], njit_end_2[1], pt[0], pt[1]]
            )

    return ticks
