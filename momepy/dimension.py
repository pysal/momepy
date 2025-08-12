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
    area: NDArray[np.float64] | Series,
    height: NDArray[np.float64] | Series,
) -> NDArray[np.float64] | Series:
    """
    Calculates volume of each object in given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    area : NDArray[np.float64] | Series
        array of areas
    height : NDArray[np.float64] | Series
        array of heights

    Returns
    -------
    NDArray[np.float64] | Series
        array of a type depending on the input

    Examples
    --------
    >>> import pandas as pd
    >>> area = pd.Series([100, 30, 40, 75, 230])
    >>> height = pd.Series([22, 6.5, 12, 9, 4.5])
    >>> momepy.volume(area, height)
    0    2200.0
    1     195.0
    2     480.0
    3     675.0
    4    1035.0
    dtype: float64
    """
    return area * height


def floor_area(
    area: NDArray[np.float64] | Series,
    height: NDArray[np.float64] | Series,
    floor_height: float | NDArray[np.float64] | Series = 3,
) -> NDArray[np.float64] | Series:
    """Calculates floor area of each object based on height and area.

    The number of
    floors is simplified into the formula: ``height // floor_height``. B
    y default one floor is approximated to 3 metres.

    .. math::
        area * \\frac{height}{floor_height}


    Parameters
    ----------
    area : NDArray[np.float64] | Series
        array of areas
    height : NDArray[np.float64] | Series
        array of heights
    floor_height : float | NDArray[np.float64] | Series, optional
        float denoting the uniform floor height or an aarray reflecting the building
        height by geometry, by default 3

    Returns
    -------
    NDArray[np.float64] | Series
        array of a type depending on the input

    Examples
    --------
    >>> import pandas as pd
    >>> area = pd.Series([100, 30, 40, 75, 230])
    >>> height = pd.Series([22, 6.5, 12, 9, 4.5])
    >>> momepy.floor_area(area, height)
    0    700.0
    1     60.0
    2    160.0
    3    225.0
    4    230.0
    dtype: float64

    If you know average height of floors per each building, you can pass it directly:

    >>> floor_height = pd.Series([3.2, 3, 4, 3, 4.5])
    >>> momepy.floor_area(area, height, floor_height=floor_height)
    0    600.0
    1     60.0
    2    120.0
    3    225.0
    4    230.0
    dtype: float64
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

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> ca = momepy.courtyard_area(buildings)
    >>> ca
    0      0.0
    1      0.0
    2      0.0
    3      0.0
    4      0.0
        ...
    139    0.0
    140    0.0
    141    0.0
    142    0.0
    143    0.0
    Name: courtyard_area, Length: 144, dtype: float64

    Verify that at least some buildings have courtyards:

    >>> ca.sum()
    np.float64(353.33274206543274)
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

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.longest_axis_length(buildings)
    0       40.265562
    1      191.254382
    2       37.247151
    3       47.022428
    4       37.170142
            ...
    139     11.587272
    140     27.747002
    141     52.566435
    142     11.091309
    143     15.472821
    Name: geometry, Length: 144, dtype: float64
    """
    return shapely.minimum_bounding_radius(geometry.geometry) * 2


def perimeter_wall(
    geometry: GeoDataFrame | GeoSeries, graph: Graph | None = None, buffer: float = 0.01
) -> Series:
    """Calculate the perimeter wall length of the joined structure.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    graph : Graph | None, optional
        Graph encoding Queen contiguity of ``geometry``. If ``None`` Queen contiguity is
        built on the fly.
    buffer: float
        Buffer value for the geometry. It can be used
        to account for topological problems.

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.perimeter_wall(buildings)
    0      137.186310
    1      663.342296
    2      663.342296
    3      663.342296
    4      663.342296
            ...
    139     42.839590
    140     78.562927
    141    147.342182
    142    118.354123
    143    342.909172
    Name: perimeter_wall, Length: 144, dtype: float64

    By default, ``momepy`` calculates a Queen contiguity graph to determine connected
    components. Alternatively, you can pass that yourself. This can be useful when
    the graph is already computed or when you need to use a different method due to
    topological issues.

    >>> from libpysal import graph
    >>> strict_contig = graph.Graph.build_contiguity(
    ...     buildings, rook=False, strict=True,
    ... )
    >>> momepy.perimeter_wall(buildings, graph=strict_contig)
    0      137.186310
    1      663.342296
    2      663.342296
    3      663.342296
    4      663.342296
            ...
    139     42.839590
    140     78.562927
    141    147.342182
    142    118.354123
    143    342.909172
    Name: perimeter_wall, Length: 144, dtype: float64
    """

    if graph is None:
        graph = Graph.build_contiguity(geometry, rook=False)

    isolates = graph.isolates

    # measure perimeter walls of connected components while ignoring isolates
    blocks = geometry.drop(isolates)
    component_perimeter = (
        blocks[[blocks.geometry.name]]
        .set_geometry(blocks.buffer(buffer))  # type: ignore
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
    y: NDArray[np.float64] | Series, area: NDArray[np.float64] | Series, graph: Graph
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
    y : NDArray[np.float64] | Series
        The character values to be weighted.
    area : NDArray[np.float64] | Series
        The area values to be used as weightss
    graph : libpysal.graph.Graph
        A spatial weights matrix for values and areas.

    Returns
    -------
    Series
        A Series containing the resulting values.

    Examples
    --------
    Area-weighted elongation within 5 nearest neighbors:

    >>> from libpysal import graph
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    Measure elongation (or anything else):

    >>> elongation = momepy.elongation(buildings)
    >>> elongation.head()
    0    0.908244
    1    0.581318
    2    0.726527
    3    0.838840
    4    0.727294
    Name: elongation, dtype: float64

    Define spatial graph:

    >>> knn5 = graph.Graph.build_knn(buildings.centroid, k=5)
    >>> knn5
    <Graph of 144 nodes and 720 nonzero edges indexed by
     [0, 1, 2, 3, 4, ...]>

    Measure the area-weighted character:

    >>> momepy.weighted_character(elongation, buildings.area, knn5)
    focal
    0      0.808190
    1      0.817309
    2      0.627589
    3      0.794769
    4      0.806403
             ...
    139    0.780744
    140    0.875046
    141    0.753670
    142    0.440000
    143    0.901127
    Name: sum, Length: 144, dtype: float64
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
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")
    >>> streets.head()
                                                geometry
    0  LINESTRING (1603585.64 6464428.774, 1603413.20...
    1  LINESTRING (1603268.502 6464060.781, 1603296.8...
    2  LINESTRING (1603607.303 6464181.853, 1603592.8...
    3  LINESTRING (1603678.97 6464477.215, 1603675.68...
    4  LINESTRING (1603537.194 6464558.112, 1603557.6...

    >>> result = momepy.street_profile(streets, buildings)
    >>> result.head()
           width  openness  width_deviation
    0  47.905964  0.946429         0.020420
    1  42.418068  0.615385         2.644521
    2  32.131831  0.608696         2.864438
    3  50.000000  1.000000              NaN
    4  50.000000  1.000000              NaN

    If you know height of each building, you can pass that along to get back
    more information:

    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(seed=42)
    >>> height = pd.Series(rng.integers(low=9, high=30, size=len(buildings)))
    >>> result = momepy.street_profile(streets, buildings, height=height)
    >>> result.head()
           width  openness  width_deviation     height  height_deviation  hw_ratio
    0  47.905964  0.946429         0.020420  12.666667          4.618802  0.264407
    1  42.418068  0.615385         2.644521  21.500000          6.467869  0.506859
    2  32.131831  0.608696         2.864438  17.555556          4.901647  0.546360
    3  50.000000  1.000000              NaN        NaN               NaN       NaN
    4  50.000000  1.000000              NaN        NaN               NaN       NaN
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
    ticks = np.empty((len(list_points) * 2, 4), dtype=float)

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
