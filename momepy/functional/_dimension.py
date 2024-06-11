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
    """
    Calculate the perimeter wall length the joined structure.

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


def street_profile(
    streets: GeoDataFrame,
    buildings: GeoDataFrame,
    distance: float = 10,
    tick_length: float = 50,
    heights: None | Series = None,
) -> DataFrame:
    """Calculates the street profile characters.

    This functions returns a DataFrame with widths, standard deviation of width,
    openness, heights, standard deviation of height and
    ratio height/width. The algorithm generates perpendicular lines to the ``streets``
    dataframe features  every ``distance`` and measures values on intersections with
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
    heights: pd.Series (default None)
        The ``pd.Series`` where building height are stored. If set to ``None``,
        height and ratio height/width will not be calculated.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> street_prof = momepy.street_profile(streets_df,
    ...                 buildings_df, heights=buildings_df['height'])
    100%|██████████| 33/33 [00:02<00:00, 15.66it/s]
    >>> streets_df['width'] = street_prof.w
    >>> streets_df['deviations'] = street_prof.wd
    """

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
    intersections = shapely.intersection(ticks[inp], buildings.boundary.array[res])
    distances = shapely.distance(intersections, shapely.points(coords[inp // 2]))
    min_distances = pd.Series(distances).groupby(inp).min()
    dists = np.full((len(ticks),), np.nan)
    dists[min_distances.index.values] = min_distances.values

    ## generate tick values and groupby street
    tick_coords = np.repeat(coord_indxs, 2)
    street_res = (
        pd.Series(dists)
        .groupby(tick_coords)
        .apply(generate_street_tick_values, tick_length)
    )
    njit_result = np.concatenate(street_res).reshape((-1, 3))
    njit_result[np.isnan(njit_result[:, 0]), 0] = tick_length

    final_result = pd.DataFrame(
        np.nan, columns=["width", "openness", "width_deviation"], index=streets.index
    )
    final_result.loc[street_res.index] = njit_result

    ## if heights are available add heights stats to the result
    if heights is not None:
        min_heights = heights[res].groupby(inp).min()
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


def generate_street_tick_values(group, tick_length):
    s = group.values

    lefts = s[::2]
    rights = s[1::2]
    left_mean = np.nanmean(lefts) if ~np.isnan(lefts).all() else tick_length / 2
    right_mean = np.nanmean(rights) if ~np.isnan(rights).all() else tick_length / 2
    width = np.mean([left_mean, right_mean]) * 2
    f_sum = s.shape[0]
    s_nan = np.isnan(s)

    openness_score = np.nan if not f_sum else s_nan.sum() / f_sum  # how few buildings

    deviation_score = np.nan if s_nan.all() else np.nanstd(s)  # deviation of buildings
    return [width, openness_score, deviation_score]
