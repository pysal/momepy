import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from pandas import DataFrame, Series

from momepy.functional import _dimension

__all__ = [
    "form_factor",
    "fractal_dimension",
    "volume_facade_ratio",
    "circular_compactness",
    "square_compactness",
    "convexity",
    "courtyard_index",
    "rectangularity",
    "shape_index",
    "corners",
    "squareness",
]


def form_factor(
    geometry: GeoDataFrame | GeoSeries,
    height: NDArray[np.float_] | Series,
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
    height : NDArray[np.float_] | Series
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
    area: NDArray[np.float_] | Series,
    perimeter: NDArray[np.float_] | Series,
) -> NDArray[np.float_] | Series:
    """Calculates fractal dimensionbased on area and perimeter.

    .. math::
        {2log({{perimeter} \\over {4}})} \\over log(area)

    Based on :cite:`mcgarigal1995fragstats`.

    Parameters
    ----------
    area : NDArray[np.float_] | Series
        array of areas
    perimeter : NDArray[np.float_] | Series
        array of perimeters

    Returns
    -------
    NDArray[np.float_] | Series
        array of a type depending on the input
    """
    return (2 * np.log(perimeter / 4)) / np.log(area)


def volume_facade_ratio(
    geometry: GeoDataFrame | GeoSeries,
    height: NDArray[np.float_] | Series,
) -> Series:
    """
    Calculates the volume-to-facade ratio of each object given its geometry and height.

    .. math::
        volume \\over perimeter * height

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    height : NDArray[np.float_] | Series
        array of heights

    Returns
    -------
    Series
    """
    return (geometry.area * height) / (geometry.length * height)


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
        np.pi * shapely.minimum_bounding_radius(geometry.geometry) ** 2
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
    courtyard_area: NDArray[np.float_] | Series | None = None,
) -> Series:
    """Calculates the courtyard index of each object given its geometry.

    .. math::
        \\textit{area of courtyards} \\over \\textit{total area}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
        A GeoDataFrame or GeoSeries containing polygons to analyse.
    courtyard_area : NDArray[np.float_] | Series | None, optional
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
    return geometry.area / shapely.minimum_rotated_rectangle(geometry.geometry).area


def shape_index(
    geometry: GeoDataFrame | GeoSeries,
    longest_axis_length: NDArray[np.float_] | Series | None = None,
) -> Series:
    """Calculates the shape index of each object given its geometry.

    .. math::
        {\\sqrt{{area} \\over {\\pi}}} \\over {0.5 * \\textit{longest axis}}

    Parameters
    ----------
    geometry : GeoDataFrame | GeoSeries
         A GeoDataFrame or GeoSeries containing polygons to analyse.
    longest_axis_length : NDArray[np.float_] | Series | None, optional
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

    def _count_corners(points: DataFrame, eps: float) -> int:
        points = points.values[:-1]
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

        true_angles = np.logical_or(degrees <= 180 - eps, degrees >= 180 + eps)
        corners = np.count_nonzero(true_angles)

        return corners

    if include_interiors:
        coords = geometry.get_coordinates(index_parts=False)
    else:
        coords = geometry.exterior.get_coordinates(index_parts=False)

    return coords.groupby(level=0).apply(_count_corners, eps=eps)


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

    def _squareness(points: DataFrame, eps: float) -> int:
        points = points.values[:-1]
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

        true_angles = np.logical_or(degrees <= 180 - eps, degrees >= 180 + eps)

        return np.nanmean(np.abs(90 - degrees[true_angles]))

    if include_interiors:
        coords = geometry.get_coordinates(index_parts=False)
    else:
        coords = geometry.exterior.get_coordinates(index_parts=False)

    return coords.groupby(level=0).apply(_squareness, eps=eps)


def eri(geometry: GeoDataFrame | GeoSeries) -> Series:
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
    bbox = shapely.minimum_rotated_rectangle(geometry.geometry)
    return np.sqrt(geometry.area / bbox.area) * (bbox.length / geometry.length)
