import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from packaging.version import Version
from pandas import DataFrame, Series

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
    """
    if not GPD_013:
        raise ImportError(
            "momepy.centroid_corner_distance requires geopandas 0.13 or later. "
        )

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
    ccd.index = geometry.index
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
