#!/usr/bin/env python
# -*- coding: utf-8 -*-

# shape.py
# definitons of shape characters

import pandas as pd
from tqdm import tqdm  # progress bar
import math
import random
import numpy as np
from shapely.geometry import Point


def form_factor(gdf, volumes, areas=None):
    """
    Calculates form factor of each object in given geoDataFrame.

    .. math::
        area \\over {volume^{2 \\over 3}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    volumes : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored volume value.
        (To calculate volume you can use :py:func:`momepy.volume`)
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Bourdic, L., Salat, S. and Nowacki, C. (2012) ‘Assessing cities: a new system
    of cross-scale spatial indicators’, Building Research & Information, 40(5),
    pp. 592–605. doi: 10.1080/09613218.2012.703488self.

    Examples
    --------
    >>> buildings_df['formfactor'] = momepy.form_factor(buildings_df, 'volume')
    Calculating form factor...
    Form factor calculated.
    >>> buildings_df.formfactor[0]
    1.9385988170288635

    >>> buildings_df['formfactor'] = momepy.form_factor(buildings_df, momepy.volume(buildings_df, 'height'))
    Calculating volumes...
    Volumes calculated.
    Calculating form factor...
    Form factor calculated.
    >>> buildings_df.formfactor[0]
    1.9385988170288635

    """

    print('Calculating form factor...')
    gdf = gdf.copy()
    if not isinstance(volumes, str):
        gdf['mm_v'] = volumes
        volumes = 'mm_v'
    if areas is None:
        series = gdf.apply(lambda row: row.geometry.area / (row[volumes] ** (2 / 3)) if row[volumes] != 0 else 0, axis=1)
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: row[areas] / (row[volumes] ** (2 / 3)) if row[volumes] != 0 else 0, axis=1)

    print('Form factor calculated.')
    return series


def fractal_dimension(gdf, areas=None, perimeters=None):
    """
    Calculates fractal dimension of each object in given geoDataFrame.

    .. math::
        {log({{perimeter} \\over {4}})} \\over log(area)

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored perimeter value. If set to None, function will calculate perimeters
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Hermosilla, T. et al. (2014) ‘Using street based metrics to characterize urban
    typologies’, Computers, Environment and Urban Systems, 44, pp. 68–79.
    doi: 10.1016/j.compenvurbsys.2013.12.002.

    Examples
    --------
    >>> buildings_df['fractal'] = momepy.fractal_dimension(buildings_df, 'area', 'peri')
    Calculating fractal dimension...
    100%|██████████| 144/144 [00:00<00:00, 3928.09it/s]
    Fractal dimension calculated.
    >>> buildings_df.fractal[0]
    0.5363389283519454
    """

    print('Calculating fractal dimension...')
    gdf = gdf.copy()

    if perimeters is None:
        gdf['mm_p'] = gdf.geometry.length
        perimeters = 'mm_p'
    else:
        if not isinstance(perimeters, str):
            gdf['mm_p'] = perimeters
            perimeters = 'mm_p'
    if areas is None:
        series = gdf.apply(lambda row: math.log(row[perimeters] / 4) / math.log(row.geometry.area), axis=1)
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: math.log(row[perimeters] / 4) / math.log(row[areas]), axis=1)

    print('Fractal dimension calculated.')
    return series


def volume_facade_ratio(gdf, heights, volumes=None, perimeters=None):
    """
    Calculates volume/facade ratio of each object in given geoDataFrame.

    .. math::
        volume \\over perimeter * heigth

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    heights : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored height value
    volumes : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored volume value
    perimeters : , list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored perimeter value


    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Schirmer, P. M. and Axhausen, K. W. (2015) ‘A multiscale classification of
    urban morphology’, Journal of Transport and Land Use, 9(1), pp. 101–130.
    doi: 10.5198/jtlu.2015.667.

    Examples
    -----
    >>> buildings_df['vfr'] = momepy.volume_facade_ratio(buildings_df, 'height')
    Calculating volume/facade ratio...
    Volume/facade ratio calculated.
    >>> buildings_df.vfr[0]
    5.310715735236504
    """

    print('Calculating volume/facade ratio...')
    gdf = gdf.copy()
    if perimeters is None:
        gdf['mm_p'] = gdf.geometry.length
        perimeters = 'mm_p'
    else:
        if not isinstance(perimeters, str):
            gdf['mm_p'] = perimeters
            perimeters = 'mm_p'

    if volumes is None:
        gdf['mm_v'] = gdf.geometry.area * gdf[heights]
        volumes = 'mm_v'
    else:
        if not isinstance(volumes, str):
            gdf['mm_v'] = volumes
            volumes = 'mm_v'

    series = gdf[volumes] / (gdf[perimeters] * gdf[heights])

    print('Volume/facade ratio calculated.')
    return series


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

# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
#
# Initially: No boundary points known


def _make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not _is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not _is_in_circle(c, q):
            if c[2] == 0.0:
                c = _make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
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
        elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
            left = c
        elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    if (left[2] <= right[2]):
        return left
    return right


def _make_circumcircle(p0, p1, p2):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
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
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
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
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
# end of Nayuiki script to define the smallest enclosing circle


# calculate the area of circumcircle
def _circle_area(points):
    circ = _make_circle(points)
    return(math.pi * circ[2] ** 2)


def circular_compactness(gdf, areas=None):
    """
    Calculates compactness index of each object in given geoDataFrame.

    .. math::
        area \\over \\textit{area of enclosing circle}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Dibble, J. (2016) Urban Morphometrics: Towards a Quantitative Science of Urban
    Form. University of Strathclyde.

    Examples
    --------
    >>> buildings_df['circ_comp'] = momepy.circular_compactness(buildings_df, 'area')
    Calculating compactness index...
    100%|██████████| 144/144 [00:00<00:00, 2498.75it/s]
    Compactness index calculated.
    >>> buildings_df['circ_comp'][0]
    0.572145421828038
    """

    print('Calculating compactness index...')
    gdf = gdf.copy()

    if areas is None:
        series = gdf.apply(lambda row: (row.geometry.area) / (_circle_area(list(row['geometry'].convex_hull.exterior.coords))), axis=1)
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: (row[areas]) / (_circle_area(list(row['geometry'].convex_hull.exterior.coords))), axis=1)

    print('Compactness index calculated.')
    return series


def square_compactness(gdf, areas=None, perimeters=None):
    """
    Calculates compactness index of each object in given geoDataFrame.

    .. math::
        \\begin{equation*}
        \\left(\\frac{4 \\sqrt{area}}{perimeter}\\right) ^ 2
        \\end{equation*}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored perimeter value. If set to None, function will calculate perimeters
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Feliciotti A (2018) RESILIENCE AND URBAN DESIGN:A SYSTEMS APPROACH TO THE
    STUDY OF RESILIENCE IN URBAN FORM. LEARNING FROM THE CASE OF GORBALS. Glasgow.

    Examples
    --------
    >>> buildings_df['squ_comp'] = momepy.square_compactness(buildings_df)
    Calculating compactness index...
    Compactness index calculated.
    >>> buildings_df['squ_comp'][0]
    0.6193872538650996

    """

    print('Calculating compactness index...')
    gdf = gdf.copy()

    if perimeters is None:
        gdf['mm_p'] = gdf.geometry.length
        perimeters = 'mm_p'
    else:
        if not isinstance(perimeters, str):
            gdf['mm_p'] = perimeters
            perimeters = 'mm_p'
    if areas is None:
        series = gdf.apply(lambda row: ((4 * math.sqrt(row.geometry.area)) / (row[perimeters])) ** 2, axis=1)
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: ((4 * math.sqrt(row[areas])) / (row[perimeters])) ** 2, axis=1)

    print('Compactness index calculated.')
    return series


def convexeity(gdf, areas=None):
    """
    Calculates convexeity index of each object in given geoDataFrame.

    .. math::
        area \\over \\textit{convex hull area}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Dibble, J. (2016) Urban Morphometrics: Towards a Quantitative Science of Urban
    Form. University of Strathclyde.

    Examples
    --------
    >>> buildings_df['convexeity'] = momepy.convexeity(buildings_df)
    Calculating convexeity...
    Convexeity calculated.
    >>> buildings_df.convexeity[0]
    0.8151964258521672
    """

    print('Calculating convexeity...')
    gdf = gdf.copy()

    if areas is None:
        series = gdf.geometry.area / gdf.geometry.convex_hull.area
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf[areas] / gdf.geometry.convex_hull.area

    print('Convexeity calculated.')
    return series


def courtyard_index(gdf, courtyard_areas, areas=None):
    """
    Calculates courtyard index of each object in given geoDataFrame.

    .. math::
        \\textit{area of courtyards} \\over \\textit{total area}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    courtyard_areas : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored area value
        (To calculate volume you can use :py:func:`momepy.dimension.courtyard_area`)
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Schirmer, P. M. and Axhausen, K. W. (2015) ‘A multiscale classification of
    urban morphology’, Journal of Transport and Land Use, 9(1), pp. 101–130.
    doi: 10.5198/jtlu.2015.667.

    Examples
    --------
    >>> buildings_df['courtyard_index'] = momepy.courtyard_index(buildings, 'courtyard_area', 'area')
    Calculating courtyard index...
    Courtyard index calculated.
    >>> buildings_df.courtyard_index[80]
    0.16605915738643523

    >>> buildings_df['courtyard_index2'] = momepy.courtyard_index(buildings_df, momepy.courtyard_area(buildings_df))
    Calculating courtyard areas...
    Courtyard areas calculated.
    Calculating courtyard index...
    Courtyard index calculated.
    >>> buildings_df.courtyard_index2[80]
    0.16605915738643523
    """

    print('Calculating courtyard index...')
    gdf = gdf.copy()

    if not isinstance(courtyard_areas, str):
        gdf['mm_ca'] = courtyard_areas
        courtyard_areas = 'mm_ca'
    if areas is None:
        series = gdf[courtyard_areas] / gdf.geometry.area
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf[courtyard_areas] / gdf[areas]

    print('Courtyard index calculated.')
    return series


def rectangularity(gdf, areas=None):
    """
    Calculates rectangularity of each object in given geoDataFrame.

    .. math::
        {area \\over \\textit{minimum bounding rotated rectangle area}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Dibble, J. (2016) Urban Morphometrics: Towards a Quantitative Science of Urban
    Form. University of Strathclyde.

    Examples
    --------
    >>> buildings_df['rectangularity'] = momepy.rectangularity(buildings_df, 'area')
    Calculating rectangularity...
    100%|██████████| 144/144 [00:00<00:00, 866.62it/s]
    Rectangularity calculated.
    >>> buildings_df.rectangularity[0]
    0.6942676157646379
    """

    print('Calculating rectangularity...')
    gdf = gdf.copy()
    if areas is None:
        series = gdf.apply(lambda row: row.geometry.area / (row.geometry.minimum_rotated_rectangle.area), axis=1)

    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: row[areas] / (row.geometry.minimum_rotated_rectangle.area), axis=1)

    print('Rectangularity calculated.')
    return series


def shape_index(gdf, longest_axis, areas=None):
    """
    Calculates shape index of each object in given geoDataFrame.

    .. math::
        {\\sqrt{{area} \\over {\\pi}}} \\over {0.5 * \\textit{longest axis}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    longest_axis : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored longest axis value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Ale?

    Examples
    --------
    >>> buildings_df['shape_index'] = momepy.shape_index(buildings_df, longest_axis='long_ax', areas='area')
    Calculating shape index...
    100%|██████████| 144/144 [00:00<00:00, 5558.33it/s]
    Shape index calculated.
    >>> buildings_df['shape_index'][0]
    0.7564029493781987
    """
    print('Calculating shape index...')
    gdf = gdf.copy()

    if not isinstance(longest_axis, str):
        gdf['mm_la'] = longest_axis
        longest_axis = 'mm_la'

    if areas is None:
        series = gdf.apply(lambda row: math.sqrt(row.geometry.area / math.pi) / (0.5 * row[longest_axis]), axis=1)
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: math.sqrt(row[areas] / math.pi) / (0.5 * row[longest_axis]), axis=1)

    print('Shape index calculated.')
    return series


def corners(gdf):
    """
    Calculates number of corners of each object in given geoDataFrame.

    Uses only external shape (shapely.geometry.exterior), courtyards are not included.

    .. math::
        \\sum corner

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    ?

    Examples
    --------
    >>> buildings_df['corners'] = momepy.corners(buildings_df)
    Calculating corners...
    100%|██████████| 144/144 [00:00<00:00, 1042.15it/s]
    Corners calculated.
    >>> buildings_df.corners[0]
    24


    """
    # define empty list for results
    results_list = []
    print('Calculating corners...')

    # calculate angle between points, return true or false if real corner
    def true_angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        if np.degrees(angle) <= 170:
            return True
        elif np.degrees(angle) >= 190:
            return True
        return False

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        corners = 0  # define empty variables
        points = list(row['geometry'].exterior.coords)  # get points of a shape
        stop = len(points) - 1  # define where to stop
        for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
            if i == 0:
                continue
            elif i == stop:
                a = np.asarray(points[i - 1])
                b = np.asarray(points[i])
                c = np.asarray(points[1])

                if true_angle(a, b, c) is True:
                    corners = corners + 1
                else:
                    continue

            else:
                a = np.asarray(points[i - 1])
                b = np.asarray(points[i])
                c = np.asarray(points[i + 1])

                if true_angle(a, b, c) is True:
                    corners = corners + 1
                else:
                    continue

        results_list.append(corners)

    series = pd.Series(results_list, index=gdf.index)

    print('Corners calculated.')
    return series


def squareness(gdf):
    """
    Calculates squareness of each object in given geoDataFrame.

    Uses only external shape (shapely.geometry.exterior), courtyards are not included.

    .. math::
        \\textit{mean deviation of all corners from 90 degrees}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Dibble, J. (2016) Urban Morphometrics: Towards a Quantitative Science of Urban
    Form. University of Strathclyde.

    Examples
    --------
    >>> buildings_df['squareness'] = momepy.squareness(buildings_df)
    Calculating squareness...
    100%|██████████| 144/144 [00:01<00:00, 129.49it/s]
    Squareness calculated.
    >>> buildings_df.squareness[0]
    3.7075816043359864
    """
    # define empty list for results
    results_list = []
    print('Calculating squareness...')

    def _angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        angles = []
        points = list(row['geometry'].exterior.coords)  # get points of a shape
        stop = len(points) - 1  # define where to stop
        for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
            if i == 0:
                continue
            elif i == stop:
                a = np.asarray(points[i - 1])
                b = np.asarray(points[i])
                c = np.asarray(points[1])
                ang = _angle(a, b, c)

                if ang <= 175:
                    angles.append(ang)
                elif _angle(a, b, c) >= 185:
                    angles.append(ang)
                else:
                    continue

            else:
                a = np.asarray(points[i - 1])
                b = np.asarray(points[i])
                c = np.asarray(points[i + 1])
                ang = _angle(a, b, c)

                if _angle(a, b, c) <= 175:
                    angles.append(ang)
                elif _angle(a, b, c) >= 185:
                    angles.append(ang)
                else:
                    continue
        deviations = []
        for i in angles:
            dev = abs(90 - i)
            deviations.append(dev)
        results_list.append(np.mean(deviations))

    series = pd.Series(results_list, index=gdf.index)

    print('Squareness calculated.')
    return series


def equivalent_rectangular_index(gdf, areas=None, perimeters=None):
    """
    Calculates equivalent rectangular index of each object in given geoDataFrame.

    .. math::
        \\sqrt{{area} \\over \\textit{area of bounding rectangle}} * {\\textit{perimeter of bounding rectangle} \\over {perimeter}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored perimeter value. If set to None, function will calculate perimeters
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Basaraner M and Cetinkaya S (2017) Performance of shape indices and classification
    schemes for characterising perceptual shape complexity of building footprints in GIS.
    2nd ed. International Journal of Geographical Information Science, Taylor & Francis
    31(10): 1952–1977. Available from:
    https://www.tandfonline.com/doi/full/10.1080/13658816.2017.1346257.

    Examples
    --------
    >>> buildings_df['eri'] = momepy.equivalent_rectangular_index(buildings_df, 'area', 'peri')
    Calculating equivalent rectangular index...
    100%|██████████| 144/144 [00:00<00:00, 895.57it/s]
    Equivalent rectangular index calculated.
    >>> buildings_df['eri'][0]
    0.7879229963118455
    """
    # define empty list for results
    results_list = []
    print('Calculating equivalent rectangular index...')
    gdf = gdf.copy()

    if perimeters is None:
        gdf['mm_p'] = gdf.geometry.length
        perimeters = 'mm_p'
    else:
        if not isinstance(perimeters, str):
            gdf['mm_p'] = perimeters
            perimeters = 'mm_p'
    if areas is None:
        gdf['mm_a'] = gdf.geometry.area
        areas = 'mm_a'
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        bbox = row['geometry'].minimum_rotated_rectangle
        results_list.append(math.sqrt(row[areas] / bbox.area) * (bbox.length / row[perimeters]))

    series = pd.Series(results_list, index=gdf.index)

    print('Equivalent rectangular index calculated.')
    return series


def elongation(gdf):
    """
    Calculates elongation of object seen as elongation of its minimum bounding rectangle.

    .. math::
        {{p - \\sqrt{p^2 - 16a}} \\over {4}} \\over {{{p} \\over {2}} - {{p - \\sqrt{p^2 - 16a}} \\over {4}}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Gil J, Montenegro N, Beirão JN, et al. (2012) On the Discovery of
    Urban Typologies: Data Mining the Multi-dimensional Character of
    Neighbourhoods. Urban Morphology 16(1): 27–40.

    Examples
    --------
    >>> buildings_df['elongation'] = momepy.elongation(buildings_df)
    Calculating elongation...
    100%|██████████| 144/144 [00:00<00:00, 1032.62it/s]
    Elongation calculated.
    >>> buildings_df['elongation'][0]
    0.9082437463675544
    """
    # define empty list for results
    results_list = []
    print('Calculating elongation...')

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        bbox = row['geometry'].minimum_rotated_rectangle
        a = bbox.area
        p = bbox.length
        cond1 = (p ** 2)
        cond2 = (16 * a)
        if cond1 >= cond2:
            sqrt = cond1 - cond2
        else:
            sqrt = 0

        # calculate both width/length and length/width
        elo1 = ((p - math.sqrt(sqrt)) / 4) / ((p / 2) - ((p - math.sqrt(sqrt)) / 4))
        elo2 = ((p + math.sqrt(sqrt)) / 4) / ((p / 2) - ((p + math.sqrt(sqrt)) / 4))
        # use the smaller one (e.g. shorter/longer)
        if elo1 <= elo2:
            elo = elo1
        else:
            elo = elo2

        results_list.append(elo)

    series = pd.Series(results_list, index=gdf.index)

    print('Elongation calculated.')
    return series


def centroid_corners(gdf):
    """
    Calculates mean distance centroid - corners and st. deviation.

    .. math::
        \\overline{x}=\\frac{1}{n}\\left(\\sum_{i=1}^{n} dist_{i}\\right);\\space \\mathrm{SD}=\\sqrt{\\frac{\\sum|x-\\overline{x}|^{2}}{n}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects

    Returns
    -------
    means, st_devs : tuple

    means : Series
        Series containing mean distance values.
    st_devs : Series
        Series containing standard deviation values.

    References
    ----------
    Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
    Journal of Transport and Land Use 9(1): 101–130.
    + Cimburova (ADD)

    Examples
    --------
    >>> means, deviations = momepy.centroid_corners(buildings_df)
    Calculating distance centroid - corner...
    100%|██████████| 144/144 [00:00<00:00, 846.58it/s]
    Distances centroid - corner calculated.
    >>> buildings_df['ccd_means'] = means
    >>> buildings_df['ccd_stdev'] = deviations
    >>> buildings_df['ccd_means'][0]
    15.961531913184833
    >>> buildings_df['ccd_stdev'][0]
    3.0810634305400177
    """
    # define empty list for results
    results_list = []
    results_list_sd = []
    print('Calculating distance centroid - corner...')

    # calculate angle between points, return true or false if real corner
    def true_angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        if np.degrees(angle) <= 170:
            return True
        elif np.degrees(angle) >= 190:
            return True
        return False

    # iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        distances = []  # set empty list of distances
        centroid = row['geometry'].centroid  # define centroid
        points = list(row['geometry'].exterior.coords)  # get points of a shape
        stop = len(points) - 1  # define where to stop
        for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
            if i == 0:
                continue
            elif i == stop:
                a = np.asarray(points[i - 1])
                b = np.asarray(points[i])
                c = np.asarray(points[1])
                p = Point(points[i])

                if true_angle(a, b, c) is True:
                    distance = centroid.distance(p)  # calculate distance point - centroid
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
        if not distances:
            from momepy.dimension import _longest_axis
            results_list.append(_longest_axis(row['geometry'].convex_hull.exterior.coords) / 2)
            results_list_sd.append(0)
        else:
            results_list.append(np.mean(distances))  # calculate mean
            results_list_sd.append(np.std(distances))  # calculate st.dev
    means = pd.Series(results_list, index=gdf.index)
    st_devs = pd.Series(results_list_sd, index=gdf.index)

    print('Distances centroid - corner calculated.')
    return means, st_devs


def linearity(gdf):
    """
    Calculates linearity of each LineString object in given geoDataFrame.

    .. math::
        \\frac{l_{euclidean}}{l_{segment}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ---------
    Araldi A and Fusco G (2017) Decomposing and Recomposing Urban Fabric:
    The City from the Pedestrian Point of View. In:, pp. 365–376. Available
    from: http://link.springer.com/10.1007/978-3-319-62407-5.

    Examples
    --------
    >>> streets_df['linearity'] = momepy.linearity(streets_df)
    Calculating linearity...
    100%|██████████| 33/33 [00:00<00:00, 1737.64it/s]
    Linearity calculated.
    >>> streets_df['linearity'][0]
    1.0
    """
    # define empty list for results
    results_list = []
    print('Calculating linearity...')

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        euclidean = Point(row['geometry'].coords[0]).distance(Point(row['geometry'].coords[-1]))
        results_list.append(euclidean / row['geometry'].length)

    series = pd.Series(results_list, index=gdf.index)

    print('Linearity calculated.')
    return series


def compactness_weighted_axis(gdf, areas=None, perimeters=None, longest_axis=None):
    """
    Calculates compactness-weighted axis of each object in given geoDataFrame.

    Initially designed for blocks.

    .. math::
        d_{i} \\times\\left(\\frac{4}{\\pi}-\\frac{16 (area_{i})}{perimeter_{i}^{2}}\\right)

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.
    perimeters : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored perimeter value. If set to None, function will calculate perimeters
        during the process without saving them separately.
    longest_axis : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored longest axis length value. If set to None, function will calculate it
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> blocks_df['cwa'] = mm.compactness_weighted_axis(blocks_df)
    Calculating compactness-weighted axis...
    Compactness-weighted axis calculated.
    """

    print('Calculating compactness-weighted axis...')
    gdf = gdf.copy()

    if perimeters is None:
        gdf['mm_p'] = gdf.geometry.length
        perimeters = 'mm_p'
    else:
        if not isinstance(perimeters, str):
            gdf['mm_p'] = perimeters
            perimeters = 'mm_p'

    if longest_axis is None:
        from .dimension import longest_axis_length
        gdf['mm_la'] = longest_axis_length(gdf)
        longest_axis = 'mm_la'
    else:
        if not isinstance(longest_axis, str):
            gdf['mm_la'] = longest_axis
            longest_axis = 'mm_la'

    if areas is None:
        series = gdf.apply(lambda row: row[longest_axis] * ((4 / math.pi) - (16 * row.geometry.area) / ((row[perimeters]) ** 2)), axis=1)
    else:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: row[longest_axis] * ((4 / math.pi) - (16 * row[areas]) / ((row[perimeters]) ** 2)), axis=1)

    print('Compactness-weighted axis calculated.')
    return series
