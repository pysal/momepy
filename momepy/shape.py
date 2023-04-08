#!/usr/bin/env python

# shape.py
# definitions of shape characters

import math
import random

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from tqdm.auto import tqdm  # progress bar

__all__ = [
    "FormFactor",
    "FractalDimension",
    "VolumeFacadeRatio",
    "CircularCompactness",
    "SquareCompactness",
    "Convexity",
    "CourtyardIndex",
    "Rectangularity",
    "ShapeIndex",
    "Corners",
    "Squareness",
    "EquivalentRectangularIndex",
    "Elongation",
    "CentroidCorners",
    "Linearity",
    "CompactnessWeightedAxis",
]


def _form_factor(height, geometry, area=None, perimeter=None, volume=None):
    """Helper for FormFactor."""
    if area is None:
        area = geometry.area
    if perimeter is None:
        perimeter = geometry.length
    if volume is None:
        volume = area * height

    surface = (perimeter * height) + area
    zeros = volume == 0
    res = np.empty(len(geometry))
    res[zeros] = np.nan
    res[~zeros] = surface[~zeros] / (volume[~zeros] ** (2 / 3))
    return res


class FormFactor:
    """
    Calculates the form factor of each object in a given GeoDataFrame.

    .. math::
        surface \\over {volume^{2 \\over 3}}

    where

    .. math::
        surface = (perimeter * height) + area

    Adapted from :cite:`bourdic2012`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    volumes : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where volume
        values are stored. To calculate volume you can use :py:func:`momepy.volume`.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.
    heights : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where height
        values are stored. Note that it cannot be ``None``.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    volumes : Series
        A Series containing used volume values.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['formfactor'] = momepy.FormFactor(buildings_df, 'volume').series
    >>> buildings_df.formfactor[0]
    1.9385988170288635

    >>> volume = momepy.Volume(buildings_df, 'height').series
    >>> buildings_df['formfactor'] = momepy.FormFactor(buildings_df, volume).series
    >>> buildings_df.formfactor[0]
    1.9385988170288635
    """

    def __init__(self, gdf, volumes, areas=None, heights=None):
        if heights is None:
            raise ValueError("`heights` cannot be None.")
            # TODO: this shouldn't be needed but it would be a breaking change now.
            # remove during the functional refactor
        self.gdf = gdf

        gdf = gdf.copy()
        if not isinstance(volumes, str):
            gdf["mm_v"] = volumes
            volumes = "mm_v"
        self.volumes = gdf[volumes]
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]

        if isinstance(heights, str):
            heights = gdf[heights]

        self.series = pd.Series(
            _form_factor(
                height=np.asarray(heights),
                geometry=gdf.geometry,
                area=self.areas,
                volume=self.volumes,
            ),
            index=gdf.index,
        )


class FractalDimension:
    """
    Calculates fractal dimension of each object in given GeoDataFrame.

    .. math::
        {2log({{perimeter} \\over {4}})} \\over log(area)

    Based on :cite:`mcgarigal1995fragstats`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        perimeter values stored. If set to ``None``, this function will calculate
        perimeters during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    perimeters : Series
        A Series containing used perimeter values.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['fractal'] = momepy.FractalDimension(buildings_df,
    ...                                                   'area',
    ...                                                   'peri').series
    >>> buildings_df.fractal[0]
    1.0726778567038908
    """

    def __init__(self, gdf, areas=None, perimeters=None):
        self.gdf = gdf

        gdf = gdf.copy()

        if perimeters is None:
            gdf["mm_p"] = gdf.geometry.length
            perimeters = "mm_p"
        else:
            if not isinstance(perimeters, str):
                gdf["mm_p"] = perimeters
                perimeters = "mm_p"
        self.perimeters = gdf[perimeters]
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"

        self.series = pd.Series(
            (2 * np.log(gdf[perimeters] / 4)) / np.log(gdf[areas]), index=gdf.index
        )


class VolumeFacadeRatio:
    """
    Calculates the volume/facade ratio of each object in a given GeoDataFrame.

    .. math::
        volume \\over perimeter * height

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    heights : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where height values are stored.
    volumes : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where volume values are stored.
    perimeters : , list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where perimeter values are stored.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    perimeters : Series
        A Series containing used perimeter values.
    volumes : Series
        A Series containing used volume values.

    Examples
    --------
    >>> buildings_df['vfr'] = momepy.VolumeFacadeRatio(buildings_df, 'height').series
    >>> buildings_df.vfr[0]
    5.310715735236504
    """

    def __init__(self, gdf, heights, volumes=None, perimeters=None):
        self.gdf = gdf

        gdf = gdf.copy()
        if perimeters is None:
            gdf["mm_p"] = gdf.geometry.length
            perimeters = "mm_p"
        else:
            if not isinstance(perimeters, str):
                gdf["mm_p"] = perimeters
                perimeters = "mm_p"
        self.perimeters = gdf[perimeters]

        if volumes is None:
            gdf["mm_v"] = gdf.geometry.area * gdf[heights]
            volumes = "mm_v"
        else:
            if not isinstance(volumes, str):
                gdf["mm_v"] = volumes
                volumes = "mm_v"
        self.volumes = gdf[volumes]

        self.series = gdf[volumes] / (gdf[perimeters] * gdf[heights])


#######################################################################################
# Smallest enclosing circle - Library (Python)

# Copyright (c) 2017 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program (see COPYING.txt and COPYING.LESSER.txt).
# If not, see <http://www.gnu.org/licenses/>.

# Data conventions: A point is a pair of floats (x, y).
# A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points.
# Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].

# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given,
# a circle of radius 0 is returned.
#
# Initially: No boundary points known


def _make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for i, p in enumerate(shuffled):
        if c is None or not _is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


def _make_circle_one_point(points, p):
    """One boundary point known."""
    c = (p[0], p[1], 0.0)
    for i, q in enumerate(points):
        if not _is_in_circle(c, q):
            if c[2] == 0.0:
                c = _make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


def _make_circle_two_points(points, p, q):
    """Two boundary points known."""
    circ = _make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if _is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = _make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
            left is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            > _cross_product(px, py, qx, qy, left[0], left[1])
        ):
            left = c
        elif cross < 0.0 and (
            right is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            < _cross_product(px, py, qx, qy, right[0], right[1])
        ):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    if left is None:
        return right
    if right is None:
        return left
    if left[2] <= right[2]:
        return left
    return right


def _make_circumcircle(p0, p1, p2):
    """Mathematical algorithm from Wikipedia: Circumscribed circle."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
    ax -= ox
    ay -= oy
    bx -= ox
    by -= oy
    cx -= ox
    cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = (
        ox
        + (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        )
        / d
    )
    y = (
        oy
        + (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        )
        / d
    )
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))


def _make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def _is_in_circle(c, p):
    return (
        c is not None
        and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON
    )


def _cross_product(x0, y0, x1, y1, x2, y2):
    """
    Returns twice the signed area of the
    triangle defined by (x0, y0), (x1, y1), (x2, y2).
    """
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


# end of Nayuiki script to define the smallest enclosing circle
#######################################################################################


def _circle_area(points):
    """calculate the area of circumcircle."""
    if len(points[0]) == 3:
        points = [x[:2] for x in points]
    circ = _make_circle(points)
    return math.pi * circ[2] ** 2


def _circle_radius(points):
    if len(points[0]) == 3:
        points = [x[:2] for x in points]
    circ = _make_circle(points)
    return circ[2]


class CircularCompactness:
    """
    Calculates the compactness index of each object in a given GeoDataFrame.

    .. math::
        area \\over \\textit{area of enclosing circle}

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['comp'] = momepy.CircularCompactness(buildings_df, 'area').series
    >>> buildings_df['comp'][0]
    0.572145421828038
    """

    def __init__(self, gdf, areas=None):
        self.gdf = gdf

        if areas is None:
            areas = gdf.geometry.area
        elif isinstance(areas, str):
            areas = gdf[areas]
        self.areas = areas
        hull = gdf.convex_hull.exterior
        radius = hull.apply(
            lambda g: _circle_radius(list(g.coords)) if g is not None else None
        )
        self.series = areas / (np.pi * radius**2)


class SquareCompactness:
    """
    Calculates the compactness index of each object in a given GeoDataFrame.

    .. math::
        \\begin{equation*}
        \\left(\\frac{4 \\sqrt{area}}{perimeter}\\right) ^ 2
        \\end{equation*}

    Adapted from :cite:`feliciotti2018`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        perimeter values stored. If set to ``None``, this function will calculate
        perimeters during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : Series
        A Series containing used area values.
    perimeters : Series
        A Series containing used perimeter values.

    Examples
    --------
    >>> buildings_df['squ_comp'] = momepy.SquareCompactness(buildings_df).series
    >>> buildings_df['squ_comp'][0]
    0.6193872538650996
    """

    def __init__(self, gdf, areas=None, perimeters=None):
        self.gdf = gdf

        gdf = gdf.copy()

        if perimeters is None:
            gdf["mm_p"] = gdf.geometry.length
            perimeters = "mm_p"
        else:
            if not isinstance(perimeters, str):
                gdf["mm_p"] = perimeters
                perimeters = "mm_p"
        self.perimeters = gdf[perimeters]
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        self.series = ((np.sqrt(gdf[areas]) * 4) / gdf[perimeters]) ** 2


class Convexity:
    """
    Calculates the Convexity index of each object in a given GeoDataFrame.

    .. math::
        area \\over \\textit{convex hull area}

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['convexity'] = momepy.Convexity(buildings_df).series
    >>> buildings_df['convexity'][0]
    0.8151964258521672
    """

    def __init__(self, gdf, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()

        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        self.series = gdf[areas] / gdf.geometry.convex_hull.area


class CourtyardIndex:
    """
    Calculates the courtyard index of each object in a given GeoDataFrame.

    .. math::
        \\textit{area of courtyards} \\over \\textit{total area}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    courtyard_areas : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where
        courtyard area values are stored. To calculate volume you can use
        :py:class:`momepy.CourtyardArea`.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    courtyard_areas : Series
        A Series containing used courtyard areas values.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['courtyard_index'] = momepy.CourtyardIndex(buildings,
    ...                                                         'courtyard_area',
    ...                                                         'area').series
    >>> buildings_df.courtyard_index[80]
    0.16605915738643523
    """

    def __init__(self, gdf, courtyard_areas, areas=None):
        self.gdf = gdf
        gdf = gdf.copy()

        if not isinstance(courtyard_areas, str):
            gdf["mm_ca"] = courtyard_areas
            courtyard_areas = "mm_ca"
        self.courtyard_areas = gdf[courtyard_areas]
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        self.series = gdf[courtyard_areas] / gdf[areas]


class Rectangularity:
    """
    Calculates the rectangularity of each object in a given GeoDataFrame.

    .. math::
        {area \\over \\textit{minimum bounding rotated rectangle area}}

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['rect'] = momepy.Rectangularity(buildings_df, 'area').series
    100%|██████████| 144/144 [00:00<00:00, 866.62it/s]
    >>> buildings_df.rect[0]
    0.6942676157646379
    """

    def __init__(self, gdf, areas=None):
        self.gdf = gdf
        gdf = gdf.copy()
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        mrr = shapely.minimum_rotated_rectangle(gdf.geometry.array)
        mrr_area = shapely.area(mrr)
        self.series = gdf[areas] / mrr_area


class ShapeIndex:
    """
    Calculates the shape index of each object in a given GeoDataFrame.

    .. math::
        {\\sqrt{{area} \\over {\\pi}}} \\over {0.5 * \\textit{longest axis}}

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    longest_axis : str, list, np.array, pd.Series
        The name of the dataframe column, ``np.array``, or ``pd.Series``
        where is longest axis values are stored.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    longest_axis : Series
        A Series containing used longest axis values.
    areas : Series
        A Series containing used area values.

    Examples
    --------
    >>> buildings_df['shape_index'] = momepy.ShapeIndex(buildings_df,
    ...                                                 longest_axis='long_ax',
    ...                                                 areas='area').series
    100%|██████████| 144/144 [00:00<00:00, 5558.33it/s]
    >>> buildings_df['shape_index'][0]
    0.7564029493781987
    """

    def __init__(self, gdf, longest_axis, areas=None):
        self.gdf = gdf
        gdf = gdf.copy()

        if not isinstance(longest_axis, str):
            gdf["mm_la"] = longest_axis
            longest_axis = "mm_la"
        self.longest_axis = gdf[longest_axis]
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        self.series = pd.Series(
            np.sqrt(gdf[areas] / np.pi) / (0.5 * gdf[longest_axis]), index=gdf.index
        )


class Corners:
    """
    Calculates the number of corners of each object in a given GeoDataFrame. Uses only
    external shape (``shapely.geometry.exterior``), courtyards are not included.

    .. math::
        \\sum corner

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings_df['corners'] = momepy.Corners(buildings_df).series
    100%|██████████| 144/144 [00:00<00:00, 1042.15it/s]
    >>> buildings_df.corners[0]
    24
    """

    def __init__(self, gdf, verbose=True):
        self.gdf = gdf

        # define empty list for results
        results_list = []

        # calculate angle between points, return true or false if real corner
        def _true_angle(a, b, c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            # TODO: add arg to specify these values
            if np.degrees(angle) <= 170:
                return True
            if np.degrees(angle) >= 190:
                return True
            return False

        # fill new column with the value of area, iterating over rows one by one
        for geom in tqdm(gdf.geometry, total=gdf.shape[0], disable=not verbose):
            if geom.geom_type == "Polygon":
                corners = 0  # define empty variables
                points = list(geom.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue
            elif geom.geom_type == "MultiPolygon":
                corners = 0  # define empty variables
                for g in geom.geoms:
                    points = list(g.exterior.coords)  # get points of a shape
                    stop = len(points) - 1  # define where to stop
                    for i in np.arange(
                        len(points)
                    ):  # for every point, calculate angle and add 1 if True angle
                        if i == 0:
                            continue
                        elif i == stop:
                            a = np.asarray(points[i - 1])
                            b = np.asarray(points[i])
                            c = np.asarray(points[1])

                            if _true_angle(a, b, c) is True:
                                corners = corners + 1
                            else:
                                continue

                        else:
                            a = np.asarray(points[i - 1])
                            b = np.asarray(points[i])
                            c = np.asarray(points[i + 1])

                            if _true_angle(a, b, c) is True:
                                corners = corners + 1
                            else:
                                continue
            else:
                corners = np.nan

            results_list.append(corners)

        self.series = pd.Series(results_list, index=gdf.index)


class Squareness:
    """
    Calculates the squareness of each object in a given GeoDataFrame. Uses only
    external shape (``shapely.geometry.exterior``), courtyards are not included.
     Returns ``np.nan`` for MultiPolygons.

    .. math::
        \\mu=\\frac{\\sum_{i=1}^{N} d_{i}}{N}

    where :math:`d` is the deviation of angle of corner :math:`i` from 90 degrees.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings_df['squareness'] = momepy.Squareness(buildings_df).series
    100%|██████████| 144/144 [00:01<00:00, 129.49it/s]
    >>> buildings_df.squareness[0]
    3.7075816043359864
    """

    def __init__(self, gdf, verbose=True):
        self.gdf = gdf
        # define empty list for results
        results_list = []

        def _angle(a, b, c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(cosine_angle))

            return angle

        # fill new column with the value of area, iterating over rows one by one
        for geom in tqdm(gdf.geometry, total=gdf.shape[0], disable=not verbose):
            if geom.geom_type == "Polygon":
                angles = []
                points = list(geom.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])
                        ang = _angle(a, b, c)

                        if ang <= 175 or _angle(a, b, c) >= 185:
                            angles.append(ang)
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])
                        ang = _angle(a, b, c)

                        if _angle(a, b, c) <= 175 or _angle(a, b, c) >= 185:
                            angles.append(ang)
                        else:
                            continue
                deviations = [abs(90 - i) for i in angles]
                results_list.append(np.mean(deviations))

            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class EquivalentRectangularIndex:
    """
    Calculates the equivalent rectangular index of each object in a given GeoDataFrame.

    .. math::
        \\sqrt{{area} \\over \\textit{area of bounding rectangle}} *
        {\\textit{perimeter of bounding rectangle} \\over {perimeter}}

    Based on :cite:`basaraner2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area values are stored. If set to ``None``, this function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        perimeter values are stored. If set to ``None``, the function will calculate
        perimeters during the process without saving them separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : Series
        A Series containing used area values.
    perimeters : Series
        A Series containing used perimeter values.

    Examples
    --------
    >>> buildings_df['eri'] = momepy.EquivalentRectangularIndex(buildings_df,
    ...                                                         'area',
    ...                                                         'peri').series
    >>> buildings_df['eri'][0]
    0.7879229963118455
    """

    def __init__(self, gdf, areas=None, perimeters=None):
        self.gdf = gdf
        # define empty list for results

        if perimeters is None:
            perimeters = gdf.geometry.length
        else:
            if isinstance(perimeters, str):
                perimeters = gdf[perimeters]

        self.perimeters = perimeters

        if areas is None:
            areas = gdf.geometry.area
        else:
            if isinstance(areas, str):
                areas = gdf[areas]

        self.areas = areas
        bbox = shapely.minimum_rotated_rectangle(gdf.geometry)
        res = np.sqrt(areas / bbox.area) * (bbox.length / perimeters)

        self.series = pd.Series(res, index=gdf.index)


class Elongation:
    """
    Calculates the elongation of each object seen as
    elongation of its minimum bounding rectangle.

    .. math::
        {{p - \\sqrt{p^2 - 16a}} \\over {4}} \\over
        {{{p} \\over {2}} - {{p - \\sqrt{p^2 - 16a}} \\over {4}}}

    where `a` is the area of the object and `p` its perimeter.

    Based on :cite:`gil2012`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.

    Attributes
    ----------
    e : Series
        A Series containing resulting values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> buildings_df['elongation'] = momepy.Elongation(buildings_df).series
    >>> buildings_df['elongation'][0]
    0.9082437463675544
    """

    def __init__(self, gdf):
        self.gdf = gdf

        bbox = shapely.minimum_rotated_rectangle(gdf.geometry)
        a = bbox.area
        p = bbox.length
        cond1 = p**2
        cond2 = 16 * a
        bigger = cond1 >= cond2
        sqrt = np.empty(len(a))
        sqrt[bigger] = cond1[bigger] - cond2[bigger]
        sqrt[~bigger] = 0

        # calculate both width/length and length/width
        elo1 = ((p - np.sqrt(sqrt)) / 4) / ((p / 2) - ((p - np.sqrt(sqrt)) / 4))
        elo2 = ((p + np.sqrt(sqrt)) / 4) / ((p / 2) - ((p + np.sqrt(sqrt)) / 4))

        # use the smaller one (e.g. shorter/longer)
        res = np.empty(len(a))
        res[elo1 <= elo2] = elo1[elo1 <= elo2]
        res[~(elo1 <= elo2)] = elo2[~(elo1 <= elo2)]

        self.series = pd.Series(res, index=gdf.index)


class CentroidCorners:
    """
    Calculates the mean distance centroid - corners and standard deviation.
    Returns ``np.nan`` for MultiPolygons.

    .. math::
        \\overline{x}=\\frac{1}{n}\\left(\\sum_{i=1}^{n} dist_{i}\\right);
        \\space \\mathrm{SD}=\\sqrt{\\frac{\\sum|x-\\overline{x}|^{2}}{n}}

    Adapted from :cite:`schirmer2015` and :cite:`cimburova2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    mean : Series
        A Series containing mean distance values.
    std : Series
        A Series containing standard deviation values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> ccd = momepy.CentroidCorners(buildings_df)
    100%|██████████| 144/144 [00:00<00:00, 846.58it/s]
    >>> buildings_df['ccd_means'] = ccd.means
    >>> buildings_df['ccd_stdev'] = ccd.std
    >>> buildings_df['ccd_means'][0]
    15.961531913184833
    >>> buildings_df['ccd_stdev'][0]
    3.0810634305400177
    """

    def __init__(self, gdf, verbose=True):
        self.gdf = gdf
        # define empty list for results
        results_list = []
        results_list_sd = []

        # calculate angle between points, return true or false if real corner
        def true_angle(a, b, c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            if np.degrees(angle) <= 170:
                return True
            if np.degrees(angle) >= 190:
                return True
            return False

        # iterating over rows one by one
        for geom in tqdm(gdf.geometry, total=gdf.shape[0], disable=not verbose):
            if geom.geom_type == "Polygon":
                distances = []  # set empty list of distances
                centroid = geom.centroid  # define centroid
                points = list(geom.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])
                        p = Point(points[i])

                        if true_angle(a, b, c) is True:
                            distance = centroid.distance(
                                p
                            )  # calculate distance point - centroid
                            distances.append(distance)  # add distance to the list
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])
                        p = Point(points[i])

                        if true_angle(a, b, c) is True:
                            distance = centroid.distance(p)
                            distances.append(distance)
                        else:
                            continue
                if not distances:  # circular buildings
                    if geom.has_z:
                        coords = [
                            (coo[0], coo[1]) for coo in geom.convex_hull.exterior.coords
                        ]
                    else:
                        coords = geom.convex_hull.exterior.coords
                    results_list.append(_circle_radius(coords))
                    results_list_sd.append(0)
                else:
                    results_list.append(np.mean(distances))  # calculate mean
                    results_list_sd.append(np.std(distances))  # calculate st.dev
            else:
                results_list.append(np.nan)
                results_list_sd.append(np.nan)

        self.mean = pd.Series(results_list, index=gdf.index)
        self.std = pd.Series(results_list_sd, index=gdf.index)


class Linearity:
    """
    Calculates the linearity of each LineString object in a given GeoDataFrame.
    MultiLineString returns ``np.nan``.

    .. math::
        \\frac{l_{euclidean}}{l_{segment}}

    where `l` is the length of the LineString.

    Adapted from :cite:`araldi2019`.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Attributes
    ----------
    series : Series
        A Series containing mean distance values.
    gdf : GeoDataFrame
        The original GeoDataFrame.

    Examples
    --------
    >>> streets_df['linearity'] = momepy.Linearity(streets_df).series
    >>> streets_df['linearity'][0]
    1.0
    """

    def __init__(self, gdf):
        self.gdf = gdf

        euclidean = gdf.geometry.apply(
            lambda geom: self._dist(geom.coords[0], geom.coords[-1])
            if geom.geom_type == "LineString"
            else np.nan
        )
        self.series = euclidean / gdf.geometry.length

    def _dist(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])


class CompactnessWeightedAxis:
    """
    Calculates the compactness-weighted axis of each object in a given GeoDataFrame.
    Initially designed for blocks.

    .. math::
        d_{i} \\times\\left(\\frac{4}{\\pi}-\\frac{16 (area_{i})}
        {perimeter_{i}^{2}}\\right)

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing objects.
    areas : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        area value are stored . If set to ``None``, this function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        perimeter values are stored. If set to ``None``, this function will calculate
        perimeters during the process without saving them separately.
    longest_axis : str, list, np.array, pd.Series (default None)
        The name of the dataframe column, ``np.array``, or ``pd.Series`` where
        longest axis length values are stored. If set to ``None``, this function will
        calculate longest axis lengths during the process without saving them
        separately.

    Attributes
    ----------
    series : Series
        A Series containing resulting values
    gdf : GeoDataFrame
        The original GeoDataFrame.
    areas : Series
        A Series containing used area values.
    longest_axis : Series
        A Series containing used area values.
    perimeters : Series
        A Series containing used area values.

    Examples
    --------
    >>> blocks_df['cwa'] = mm.CompactnessWeightedAxis(blocks_df).series
    """

    def __init__(self, gdf, areas=None, perimeters=None, longest_axis=None):
        self.gdf = gdf
        gdf = gdf.copy()

        if perimeters is None:
            gdf["mm_p"] = gdf.geometry.length
            perimeters = "mm_p"
        else:
            if not isinstance(perimeters, str):
                gdf["mm_p"] = perimeters
                perimeters = "mm_p"
        self.perimeters = gdf[perimeters]
        if longest_axis is None:
            from .dimension import LongestAxisLength

            gdf["mm_la"] = LongestAxisLength(gdf).series
            longest_axis = "mm_la"
        else:
            if not isinstance(longest_axis, str):
                gdf["mm_la"] = longest_axis
                longest_axis = "mm_la"
        self.longest_axis = gdf[longest_axis]
        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        self.series = pd.Series(
            gdf[longest_axis]
            * ((4 / np.pi) - (16 * gdf[areas]) / ((gdf[perimeters]) ** 2)),
            index=gdf.index,
        )
