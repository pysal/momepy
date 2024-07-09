import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from packaging.version import Version
from pandas import DataFrame, MultiIndex, Series

from momepy.functional import _dimension

__all__ = [
    "form_factor",
    "fractal_dimension",
    "facade_ratio",
    "circular_compactness",
    "square_compactness",
    "convexity",
    "courtyard_index",
    "rectangularity",
    "shape_index",
    "corners",
    "squareness",
    "equivalent_rectangular_index",
    "elongation",
    "centroid_corner_distance",
    "linearity",
    "compactness_weighted_axis",
]

GPD_013 = Version(gpd.__version__) >= Version("0.13")


def form_factor(
    geometry: GeoDataFrame | GeoSeries,
    height: NDArray[np.float64] | Series,
) -> Series:
    """Calculates the form factor of each object given its geometry and height.

    .. math::
        surface \\over {volume^{2 \\over 3}}

    where

    .. math::
        surface = (perimeter * height) + area

    Adapted from :cite:`bourdic2012`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    height : NDArray[np.float64] | Series
        array of heights

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Synthesize some  building height information.

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> height = rng.integers(low=9, high=30, size=len(buildings))

    >>> momepy.form_factor(buildings, height)
    0      5.588952
    1      8.403204
    2      5.149302
    3      5.381587
    4      5.030861
            ...
    139    6.039771
    140    5.904980
    141    5.508910
    142    5.869335
    143    5.378663
    Name: form_factor, Length: 144, dtype: float64
    """
    area = geometry.area
    volume = area * height
    surface = (geometry.length * height) + geometry.area
    zeros = volume == 0
    res = np.empty(len(geometry))
    res[zeros] = np.nan
    res[~zeros] = surface[~zeros] / (volume[~zeros] ** (2 / 3))
    return Series(res, index=geometry.index, name="form_factor")


def fractal_dimension(
    geometry: GeoDataFrame | GeoSeries,
) -> Series:
    """Calculates fractal dimension based on area and perimeter.

    .. math::
        {2log({{perimeter} \\over {4}})} \\over log(area)

    Based on :cite:`mcgarigal1995fragstats`.

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
    >>> momepy.fractal_dimension(buildings)
    0      1.072678
    1      1.182350
    2      1.018422
    3      1.048314
    4      1.017328
            ...
    139    1.014975
    140    1.033581
    141    1.064103
    142    1.022617
    143    1.000008
    Length: 144, dtype: float64
    """
    return (2 * np.log(geometry.length / 4)) / np.log(geometry.area)


def facade_ratio(
    geometry: GeoDataFrame | GeoSeries,
) -> Series:
    """
    Calculates the facade ratio of each object given its geometry.

    .. math::
        area \\over perimeter

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
    >>> momepy.facade_ratio(buildings)
    0       5.310716
    1      11.314008
    2       5.963959
    3       6.376086
    4       5.987687
            ...
    139     1.868981
    140     4.046407
    141     5.963454
    142     1.711740
    143     2.734681
    Length: 144, dtype: float64
    """
    return geometry.area / geometry.length


def circular_compactness(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the circular compactness of each object given its geometry.

    .. math::
        area \\over \\textit{area of enclosing circle}

    Adapted from :cite:`dibble2017`.

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
    >>> momepy.circular_compactness(buildings)
    0      0.572145
    1      0.390417
    2      0.588332
    3      0.520411
    4      0.591297
            ...
    139    0.563433
    140    0.525733
    141    0.404872
    142    0.530405
    143    0.636387
    Length: 144, dtype: float64
    """
    return geometry.area / (
        np.pi * shapely.minimum_bounding_radius(geometry.geometry.array) ** 2
    )


def square_compactness(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the square compactness of each object given its geometry.

    .. math::
        \\begin{equation*}
        \\left(\\frac{4 \\sqrt{area}}{perimeter}\\right) ^ 2
        \\end{equation*}

    Adapted from :cite:`feliciotti2018`.

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
    >>> momepy.square_compactness(buildings)
    0      0.619387
    1      0.182604
    2      0.887750
    3      0.719750
    4      0.894034
            ...
    139    0.940666
    140    0.824085
    141    0.647576
    142    0.914813
    143    0.999961
    Length: 144, dtype: float64
    """
    return ((np.sqrt(geometry.area) * 4) / geometry.length) ** 2


def convexity(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the convexity of each object given its geometry.

    .. math::
        \\frac{\\textit{area}}{\\textit{area of convex hull}}

    Adapted from :cite:`dibble2017`.

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
    >>> momepy.convexity(buildings)
    0      0.815196
    1      0.703008
    2      0.953398
    3      0.890489
    4      0.957323
            ...
    139    1.000000
    140    0.904176
    141    0.795774
    142    1.000000
    143    1.000000
    Length: 144, dtype: float64
    """
    return geometry.area / geometry.geometry.convex_hull.area


def courtyard_index(
    geometry: GeoDataFrame | GeoSeries,
    courtyard_area: NDArray[np.float64] | Series | None = None,
) -> Series:
    """Calculates the courtyard index of each object given its geometry.

    .. math::
        \\textit{area of courtyards} \\over \\textit{total area}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    courtyard_area : NDArray[np.float64] | Series | None, optional
        array of courtyard areas. If None, it will be calculated, by default None

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> ci = momepy.courtyard_index(buildings)
    >>> ci
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
    Length: 144, dtype: float64

    >>> ci.max()
    np.float64(0.16605915738643523)

    If you know the courtyard area, you can pass it to skip the computation step.

    >>> courtyard_area = momepy.courtyard_area(buildings)
    >>> momepy.courtyard_index(buildings, courtyard_area=courtyard_area)
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
    Length: 144, dtype: float64
    """
    if courtyard_area is None:
        courtyard_area = _dimension.courtyard_area(geometry)
    return courtyard_area / geometry.area


def rectangularity(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the rectangularity of each object given its geometry.

    .. math::
        \\frac{\\textit{area}}{\\textit{area of minimum bounding rectangle}}

    Adapted from :cite:`dibble2017`.

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
    >>> momepy.rectangularity(buildings)
    0      0.694268
    1      0.702242
    2      0.901582
    3      0.821797
    4      0.912858
             ...
    139    0.996876
    140    0.820865
    141    0.659281
    142    0.971600
    143    0.999400
    Length: 144, dtype: float64
    """
    return geometry.area / shapely.area(
        shapely.minimum_rotated_rectangle(geometry.geometry.array)
    )


def shape_index(
    geometry: GeoDataFrame | GeoSeries,
    longest_axis_length: NDArray[np.float64] | Series | None = None,
) -> Series:
    """Calculates the shape index of each object given its geometry.

    .. math::
        {\\sqrt{{area} \\over {\\pi}}} \\over {0.5 * \\textit{longest axis}}

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
         A GeoDataFrame or GeoSeries containing polygons to analyse.
    longest_axis_length : NDArray[np.float64] | Series | None, optional
        array of longest axis lengths. If None, it will be calculated, by default None

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.shape_index(buildings)
    0      0.756403
    1      0.624834
    2      0.767028
    3      0.721395
    4      0.768958
            ...
    139    0.750622
    140    0.725074
    141    0.636296
    142    0.728289
    143    0.797739
    Length: 144, dtype: float64

    If you know the longest axis length, you can pass it to skip the computation step.

    >>> lal = momepy.longest_axis_length(buildings)
    >>> momepy.shape_index(buildings, longest_axis_length=lal)
    0      0.756403
    1      0.624834
    2      0.767028
    3      0.721395
    4      0.768958
            ...
    139    0.750622
    140    0.725074
    141    0.636296
    142    0.728289
    143    0.797739
    Length: 144, dtype: float64

    """
    if longest_axis_length is None:
        longest_axis_length = _dimension.longest_axis_length(geometry)
    return np.sqrt(geometry.area / np.pi) / (0.5 * longest_axis_length)


def corners(
    geometry: GeoDataFrame | GeoSeries,
    eps: float = 10,
    include_interiors: bool = False,
) -> Series:
    """Calculates the number of corners of each object given its geometry.

    As a corner is considered a point where the angle between two consecutive segments
    deviates from 180 degrees by more than ``eps``.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    eps : float, optional
        Deviation from 180 degrees to consider a corner, by default 10
    include_interiors : bool, optional
        If True, polygon interiors are included in the calculation. If False, only
        exterior is considered, by default False

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.corners(buildings)
    0      24
    1      43
    2       8
    3      16
    4       8
        ..
    139     4
    140     6
    141     6
    142     4
    143     4
    Length: 144, dtype: int64
    """
    if not GPD_013:
        raise ImportError("momepy.corners requires geopandas 0.13 or later. ")

    def _count_corners(points: DataFrame, eps: float) -> int:
        pts = points.values[:-1]
        true_angles = _true_angles_mask(pts, eps=eps)
        corners = np.count_nonzero(true_angles)

        return corners

    if include_interiors:
        coords = geometry.reset_index(drop=True).get_coordinates(index_parts=False)
    else:
        coords = geometry.reset_index(drop=True).exterior.get_coordinates(
            index_parts=False
        )

    cc = coords.groupby(level=0).apply(_count_corners, eps=eps)
    cc.index = geometry.index
    return cc


def squareness(
    geometry: GeoDataFrame | GeoSeries,
    eps: float = 10,
    include_interiors: bool = False,
) -> Series:
    """Calculates the squareness of each object given its geometry.

    Squareness is a mean deviation of angles at corners from 90 degrees.

    As a corner is considered a point where the angle between two consecutive segments
    deviates from 180 degrees by more than ``eps``.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    eps : float, optional
        Deviation from 180 degrees to consider a corner, by default 10
    include_interiors : bool, optional
        If True, polygon interiors are included in the calculation. If False, only
        exterior is considered, by default False

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.squareness(buildings)
    0      3.707582
    1      2.990318
    2      0.437987
    3      4.573564
    4      0.382146
            ...
    139    0.214733
    140    0.307916
    141    0.343259
    142    0.899731
    143    0.028140
    Length: 144, dtype: float64
    """
    if not GPD_013:
        raise ImportError("momepy.squareness requires geopandas 0.13 or later. ")

    def _squareness(points: DataFrame, eps: float):
        pts = points.values[:-1]
        true_angles, degrees = _true_angles_mask(pts, eps=eps, return_degrees=True)

        return np.nanmean(np.abs(90 - degrees[true_angles]))

    if include_interiors:
        coords = geometry.reset_index(drop=True).get_coordinates(index_parts=False)
    else:
        coords = geometry.reset_index(drop=True).exterior.get_coordinates(
            index_parts=False
        )

    sq = coords.groupby(level=0).apply(_squareness, eps=eps)
    sq.index = geometry.index
    return sq


def equivalent_rectangular_index(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the equivalent rectangular index of each object given its geometry.

    .. math::
        \\sqrt{{area} \\over \\textit{area of bounding rectangle}} *
        {\\textit{perimeter of bounding rectangle} \\over {perimeter}}

    Based on :cite:`basaraner2017`.

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
    >>> momepy.equivalent_rectangular_index(buildings)
    0      0.787923
    1      0.443137
    2      0.954252
    3      0.851658
    4      0.957543
             ...
    139    1.000050
    140    0.907837
    141    0.813269
    142    0.995926
    143    0.999999
    Length: 144, dtype: float64
    """
    bbox = shapely.minimum_rotated_rectangle(geometry.geometry.array)
    return np.sqrt(geometry.area / shapely.area(bbox)) * (
        shapely.length(bbox) / geometry.length
    )


def elongation(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the elongation of each object given its geometry.

    The elongation is defined as the elongation of the minimum bounding rectangle.

    .. math::
        {{p - \\sqrt{p^2 - 16a}} \\over {4}} \\over
        {{{p} \\over {2}} - {{p - \\sqrt{p^2 - 16a}} \\over {4}}}

    where `a` is the area of the object and `p` its perimeter.

    Based on :cite:`gil2012`.

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
    >>> momepy.elongation(buildings)
    0      0.908244
    1      0.581318
    2      0.726527
    3      0.838840
    4      0.727294
             ...
    139    0.608004
    140    0.979998
    141    0.747326
    142    0.564060
    143    0.987953
    Name: elongation, Length: 144, dtype: float64
    """
    bbox = shapely.minimum_rotated_rectangle(geometry.geometry.array)
    a = shapely.area(bbox)
    p = shapely.length(bbox)
    sqrt = np.maximum(p**2 - 16 * a, 0)

    elo1 = ((p - np.sqrt(sqrt)) / 4) / ((p / 2) - ((p - np.sqrt(sqrt)) / 4))
    elo2 = ((p + np.sqrt(sqrt)) / 4) / ((p / 2) - ((p + np.sqrt(sqrt)) / 4))

    res = np.where(elo1 <= elo2, elo1, elo2)

    return Series(res, index=geometry.index, name="elongation")


def centroid_corner_distance(
    geometry: GeoDataFrame | GeoSeries,
    eps: float = 10,
    include_interiors: bool = False,
) -> DataFrame:
    """Calculates the centroid-corner distance of each object given its geometry.

    As a corner is considered a point where the angle between two consecutive segments
    deviates from 180 degrees by more than ``eps``.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    eps : float, optional
        Deviation from 180 degrees to consider a corner, by default 10
    include_interiors : bool, optional
        If True, polygon interiors are included in the calculation. If False, only
        exterior is considered, by default False

    Returns
    -------
    DataFrame
        DataFrame with columns 'mean' and 'std'

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.centroid_corner_distance(buildings).head()
           mean        std
    0  15.961532   3.081063
    1  58.763388  22.922368
    2  14.988106   3.648731
    3  15.000439   4.999226
    4  14.965557   3.660826
    """
    if not GPD_013:
        raise ImportError(
            "momepy.centroid_corner_distance requires geopandas 0.13 or later. "
        )

    result_index = geometry.index
    if isinstance(geometry.index, MultiIndex):
        geometry = geometry.reset_index(drop=True)

    def _ccd(points: DataFrame, eps: float) -> Series:
        centroid = points.values[0, 2:]
        pts = points.values[:-1, :2]

        true_angles = _true_angles_mask(pts, eps=eps)
        dists = np.linalg.norm(pts[true_angles] - centroid, axis=1)
        return Series({"mean": np.nanmean(dists), "std": np.nanstd(dists)})

    if include_interiors:
        coords = geometry.get_coordinates(index_parts=False)
    else:
        coords = geometry.exterior.get_coordinates(index_parts=False)
    coords[["cent_x", "cent_y"]] = geometry.centroid.get_coordinates(index_parts=False)
    ccd = coords.groupby(level=0).apply(_ccd, eps=eps)
    ccd.index = result_index
    return ccd


def linearity(geometry: GeoDataFrame | GeoSeries) -> Series:
    """Calculates the linearity of each LineString

    The linearity is defined as the ratio of the length of the segment between the first
    and last point to the length of the LineString. While other geometry types are
    accepted, the result is not well defined.

    .. math::
        \\frac{l_{euclidean}}{l_{segment}}

    where `l` is the length of the LineString.

    Adapted from :cite:`araldi2019`.


    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing lines to analyse.

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> streets = geopandas.read_file(path, layer="streets")
    >>> momepy.linearity(streets).head()
    0    1.000000
    1    0.995987
    2    0.999653
    3    0.999997
    4    1.000000
    dtype: float64
    """
    return (
        shapely.distance(
            shapely.get_point(geometry.geometry.array, 0),
            shapely.get_point(geometry.geometry.array, -1),
        )
        / geometry.length
    )


def compactness_weighted_axis(
    geometry: GeoDataFrame | GeoSeries,
    longest_axis_length: NDArray[np.float64] | Series | None = None,
) -> Series:
    """Calculates the compactness-weighted axis of each object in a given GeoDataFrame.

    .. math::
        d_{i} \\times\\left(\\frac{4}{\\pi}-\\frac{16 (area_{i})}
        {perimeter_{i}^{2}}\\right)

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
         A GeoDataFrame or GeoSeries containing polygons to analyse.
    longest_axis_length : NDArray[np.float64] | Series | None, optional
        array of longest axis lengths. If None, it will be calculated, by default None

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> momepy.compactness_weighted_axis(buildings)
    0       26.327730
    1      208.588747
    2       14.358370
    3       26.026429
    4       14.095118
            ...
    139      3.853623
    140     12.462700
    141     32.888900
    142      3.975422
    143      4.228395
    Length: 144, dtype: float64

    If you know the longest axis length, you can pass it to skip the computation step.

    >>> lal = momepy.longest_axis_length(buildings)
    >>> momepy.compactness_weighted_axis(buildings, longest_axis_length=lal)
    0       26.327730
    1      208.588747
    2       14.358370
    3       26.026429
    4       14.095118
            ...
    139      3.853623
    140     12.462700
    141     32.888900
    142      3.975422
    143      4.228395
    Length: 144, dtype: float64
    """
    if longest_axis_length is None:
        longest_axis_length = _dimension.longest_axis_length(geometry)

    return longest_axis_length * (
        (4 / np.pi) - (16 * geometry.area) / (geometry.length**2)
    )


# helper functions


def _true_angles_mask(
    points: NDArray[np.float64], eps: float, return_degrees: bool = False
) -> NDArray[np.bool_] | tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Calculates the mask of true angles.

    Parameters
    ----------
    points : NDArray[np.float64]
        array of points
    eps : float
        Deviation from 180 degrees to consider a corner
    return_degrees : bool, optional
        If True, returns also degrees, by default False

    Returns
    -------
    NDArray[np.bool_] | tuple[NDArray[np.bool_], NDArray[np.float64]]
        boolean array or a tuple of boolean array and float array of degrees
    """
    a = np.roll(points, 1, axis=0)
    b = points
    c = np.roll(points, -1, axis=0)

    ba = a - b
    bc = c - b

    cosine_angle = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    angles = np.arccos(cosine_angle)
    degrees = np.degrees(angles)

    if return_degrees:
        return np.logical_or(degrees <= 180 - eps, degrees >= 180 + eps), degrees
    return np.logical_or(degrees <= 180 - eps, degrees >= 180 + eps)
