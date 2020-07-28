#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dimension.py
# definitions of dimension characters

import math
from distutils.version import LooseVersion

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy as sp
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm

from .shape import _make_circle

GPD_08 = str(gpd.__version__) >= LooseVersion("0.8.0")

__all__ = [
    "Area",
    "Perimeter",
    "Volume",
    "FloorArea",
    "CourtyardArea",
    "LongestAxisLength",
    "AverageCharacter",
    "StreetProfile",
    "WeightedCharacter",
    "CoveredArea",
    "PerimeterWall",
    "SegmentsLength",
]


class Area:
    """
    Calculates area of each object in given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for GeoPandas ``.area`` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['area'] = momepy.Area(buildings).series
    >>> buildings.area[0]
    728.5574947044363

    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.series = self.gdf.geometry.area


class Perimeter:
    """
    Calculates perimeter of each object in given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for GeoPandas ``.length`` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Attributes
    ----------
    series : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['perimeter'] = momepy.Perimeter(buildings).series
    >>> buildings.perimeter[0]
    137.18630991119903
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.series = self.gdf.geometry.length


class Volume:
    """
    Calculates volume of each object in given GeoDataFrame based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    heights : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    heights : Series
        Series containing used heights values

    areas : GeoDataFrame
        Series containing used areas values

    Examples
    --------
    >>> buildings['volume'] = momepy.Volume(buildings, heights='height_col').series
    >>> buildings.volume[0]
    7285.5749470443625

    >>> buildings['volume'] = momepy.Volume(buildings, heights='height_col', areas='area_col').series
    >>> buildings.volume[0]
    7285.5749470443625
    """

    def __init__(self, gdf, heights, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()
        if not isinstance(heights, str):
            gdf["mm_h"] = heights
            heights = "mm_h"
        self.heights = gdf[heights]

        if areas is not None:
            if not isinstance(areas, str):
                gdf["mm_a"] = areas
                areas = "mm_a"
            self.areas = gdf[areas]
        else:
            self.areas = gdf.geometry.area
        try:
            self.series = self.areas * self.heights

        except KeyError:
            raise KeyError(
                "ERROR: Column not found. Define heights and areas or set areas to None."
            )


class FloorArea:
    """
    Calculates floor area of each object based on height and area.

    Number of floors is simplified into formula height / 3
    (it is assumed that on average one floor is approximately 3 metres)

    .. math::
        area * \\frac{height}{3}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    heights : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    heights : Series
        Series containing used heights values
    areas : GeoDataFrame
        Series containing used areas values

    Examples
    --------
    >>> buildings['floor_area'] = momepy.FloorArea(buildings, heights='height_col').series
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309

    >>> buildings['floor_area'] = momepy.FloorArea(buildings, heights='height_col', areas='area_col').series
    >>> buildings.floor_area[0]
    2185.672484113309
    """

    def __init__(self, gdf, heights, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()
        if not isinstance(heights, str):
            gdf["mm_h"] = heights
            heights = "mm_h"
        self.heights = gdf[heights]

        if areas is not None:
            if not isinstance(areas, str):
                gdf["mm_a"] = areas
                areas = "mm_a"
            self.areas = gdf[areas]
        else:
            self.areas = gdf.geometry.area
        try:
            self.series = self.areas * (self.heights // 3)

        except KeyError:
            raise KeyError(
                "ERROR: Column not found. Define heights and areas or set areas to None."
            )


class CourtyardArea:
    """
    Calculates area of holes within geometry - area of courtyards.

    Ensure that your geometry is ``shapely.geometry.Polygon``.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    series : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    areas : GeoDataFrame
        Series containing used areas values

    Examples
    --------
    >>> buildings['courtyard_area'] = momepy.CourtyardArea(buildings).series
    >>> buildings.courtyard_area[80]
    353.33274206543274
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

        exts = gdf.geometry.apply(lambda g: Polygon(g.exterior))

        self.series = pd.Series(exts.area - gdf[areas], index=gdf.index)


# calculate the radius of circumcircle
def _longest_axis(points):
    circ = _make_circle(points)
    return circ[2] * 2


class LongestAxisLength:
    """
    Calculates the length of the longest axis of object.

    Axis is defined as a diameter of minimal circumscribed circle around the convex hull.
    It does not have to be fully inside an object.

    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Attributes
    ----------
    series : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings['lal'] = momepy.LongestAxisLength(buildings).series
    >>> buildings.lal[0]
    40.2655616057102
    """

    def __init__(self, gdf):
        self.gdf = gdf
        hulls = gdf.geometry.convex_hull
        self.series = hulls.apply(lambda hull: _longest_axis(hull.exterior.coords))


class AverageCharacter:
    """
    Calculates the average of a character within a set neighbourhood defined in ``spatial_weights``

    Average value of the character within a set neighbourhood defined in ``spatial_weights``.
    Can be set to ``mean``, ``median`` or ``mode``. ``mean`` is defined as:

    .. math::
        \\frac{1}{n}\\left(\\sum_{i=1}^{n} value_{i}\\right)

    Adapted from :cite:`hausleitner2017`.


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, ``np.array``, or ``pd.Series`` where is stored character value.
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index.
    spatial_weights : libpysal.weights
        spatial weights matrix
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.
    mode : str (default 'all')
        mode of average calculation. Can be set to `all`, `mean`, `median` or `mode` or
        list of any of the options.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting mean values
    mean : Series
        Series containing resulting mean values
    median : Series
        Series containing resulting median values
    mode : Series
        Series containing resulting mode values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : GeoDataFrame
        Series containing used values
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    rng : tuple
        range
    modes : str
        mode


    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation, threshold=100, silence_warnings=True, ids='uID')
    >>> tessellation['mean_area'] = momepy.AverageCharacter(tessellation, values='area', spatial_weights=sw, unique_id='uID').mean
    100%|██████████| 144/144 [00:00<00:00, 1433.32it/s]
    >>> tessellation.mean_area[0]
    4823.1334436678835
    """

    def __init__(
        self,
        gdf,
        values,
        spatial_weights,
        unique_id,
        rng=None,
        mode="all",
        verbose=True,
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng
        self.modes = mode

        if rng:
            from momepy import limit_range

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)[values]

        means = []
        medians = []
        modes = []

        allowed = ["mean", "median", "mode"]

        if mode == "all":
            mode = allowed
        elif isinstance(mode, list):
            for m in mode:
                if m not in allowed:
                    raise ValueError("{} is not supported as mode.".format(mode))
        elif isinstance(mode, str):
            if mode not in allowed:
                raise ValueError("{} is not supported as mode.".format(mode))
            mode = [mode]

        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]

                values_list = data.loc[neighbours]

                if rng:
                    values_list = limit_range(values_list, rng=rng)
                if "mean" in mode:
                    means.append(np.mean(values_list))
                if "median" in mode:
                    medians.append(np.median(values_list))
                if "mode" in mode:
                    modes.append(sp.stats.mode(values_list)[0][0])

            else:
                if "mean" in mode:
                    means.append(np.nan)
                if "median" in mode:
                    medians.append(np.nan)
                if "mode" in mode:
                    modes.append(np.nan)

        if "mean" in mode:
            self.series = self.mean = pd.Series(means, index=gdf.index)
        if "median" in mode:
            self.median = pd.Series(medians, index=gdf.index)
        if "mode" in mode:
            self.mode = pd.Series(modes, index=gdf.index)


class StreetProfile:
    """
    Calculates the street profile characters.

    Returns a dictionary with widths, standard deviation of width, openness, heights,
    standard deviation of height and ratio height/width. Algorithm generates perpendicular
    lines to ``right`` dataframe features every ``distance`` and measures values on intersection
    with features of ``left``. If no feature is reached within
    ``tick_length`` its value is set as width (being a theoretical maximum).

    Derived from :cite:`araldi2019`.

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing streets to analyse
    right : GeoDataFrame
        GeoDataFrame containing buildings along the streets (only Polygon geometry type is supported)
    heights: str, list, np.array, pd.Series (default None)
        the name of the buildings dataframe column, ``np.array``, or ``pd.Series`` where is stored building height. If set to None,
        height and ratio height/width will not be calculated.
    distance : int (default 10)
        distance between perpendicular ticks
    tick_length : int (default 50)
        length of ticks
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    w : Series
        Series containing street profile width values
    wd : Series
        Series containing street profile standard deviation values
    o : Series
        Series containing street profile openness values
    h : Series
        Series containing street profile heights values. Returned only when heights is set.
    hd : Series
        Series containing street profile heights standard deviation values. Returned only when heights is set.
    p : Series
        Series containing street profile height/width ratio values. Returned only when heights is set.
    left : GeoDataFrame
        original left GeoDataFrame
    right : GeoDataFrame
        original right GeoDataFrame
    distance : int
        distance between perpendicular ticks
    tick_length : int
        length of ticks
    heights : GeoDataFrame
        Series containing used height values

    Examples
    --------
    >>> street_profile = momepy.StreetProfile(streets_df, buildings_df, heights='height')
    100%|██████████| 33/33 [00:02<00:00, 15.66it/s]
    >>> streets_df['width'] = street_profile.w
    >>> streets_df['deviations'] = street_profile.wd
    """

    def __init__(
        self, left, right, heights=None, distance=10, tick_length=50, verbose=True
    ):
        self.left = left
        self.right = right
        self.distance = distance
        self.tick_length = tick_length

        if heights is not None:
            if not isinstance(heights, str):
                right = right.copy()
                right["mm_h"] = heights
                heights = "mm_h"

            self.heights = right[heights]

        sindex = right.sindex

        results_list = []
        deviations_list = []
        heights_list = []
        heights_deviations_list = []
        openness_list = []

        for shapely_line in tqdm(
            left.geometry, total=left.shape[0], disable=not verbose
        ):
            # list to hold all the point coords
            list_points = []
            # set the current distance to place the point
            current_dist = distance
            # get the total length of the line
            line_length = shapely_line.length
            # append the starting coordinate to the list
            list_points.append(list(shapely_line.coords)[0])
            # https://nathanw.net/2012/08/05/generating-chainage-distance-nodes-in-qgis/
            # while the current cumulative distance is less than the total length of the line
            while current_dist < line_length:
                # use interpolate and increase the current distance
                list_points.append(
                    list(shapely_line.interpolate(current_dist).coords)[0]
                )
                current_dist += distance
            # append end coordinate to the list
            list_points.append(list(shapely_line.coords)[-1])

            ticks = []
            for num, pt in enumerate(list_points, 1):
                # start chainage 0
                if num == 1:
                    angle = self._getAngle(pt, list_points[num])
                    line_end_1 = self._getPoint1(pt, angle, tick_length / 2)
                    angle = self._getAngle(line_end_1, pt)
                    line_end_2 = self._getPoint2(line_end_1, angle, tick_length)
                    tick1 = LineString([line_end_1, pt])
                    tick2 = LineString([line_end_2, pt])
                    ticks.append([tick1, tick2])

                # everything in between
                if num < len(list_points) - 1:
                    angle = self._getAngle(pt, list_points[num])
                    line_end_1 = self._getPoint1(
                        list_points[num], angle, tick_length / 2
                    )
                    angle = self._getAngle(line_end_1, list_points[num])
                    line_end_2 = self._getPoint2(line_end_1, angle, tick_length)
                    tick1 = LineString([line_end_1, list_points[num]])
                    tick2 = LineString([line_end_2, list_points[num]])
                    ticks.append([tick1, tick2])

                # end chainage
                if num == len(list_points):
                    angle = self._getAngle(list_points[num - 2], pt)
                    line_end_1 = self._getPoint1(pt, angle, tick_length / 2)
                    angle = self._getAngle(line_end_1, pt)
                    line_end_2 = self._getPoint2(line_end_1, angle, tick_length)
                    tick1 = LineString([line_end_1, pt])
                    tick2 = LineString([line_end_2, pt])
                    ticks.append([tick1, tick2])
            # widths = []
            m_heights = []
            lefts = []
            rights = []
            for duo in ticks:
                for ix, tick in enumerate(duo):
                    if GPD_08:
                        int_blg = right.iloc[sindex.query(tick, predicate="intersects")]
                    else:
                        possible_intersections_index = list(
                            sindex.intersection(tick.bounds)
                        )
                        possible_intersections = right.iloc[
                            possible_intersections_index
                        ]
                        real_intersections = possible_intersections.intersects(tick)
                        int_blg = possible_intersections[real_intersections]
                    if not int_blg.empty:
                        true_int = int_blg.intersection(tick)
                        dist = true_int.distance(Point(tick.coords[-1]))
                        if ix == 0:
                            lefts.append(dist.min())
                        else:
                            rights.append(dist.min())
                        if heights is not None:
                            m_heights.append(int_blg.loc[dist.idxmin()][heights])

            openness = (len(lefts) + len(rights)) / len(ticks * 2)
            openness_list.append(1 - openness)
            if rights and lefts:
                results_list.append(2 * np.mean(lefts + rights))
                deviations_list.append(np.std(lefts + rights))
            elif not lefts and rights:
                results_list.append(2 * np.mean([np.mean(rights), tick_length / 2]))
                deviations_list.append(np.std(rights))
            elif not rights and lefts:
                results_list.append(2 * np.mean([np.mean(lefts), tick_length / 2]))
                deviations_list.append(np.std(lefts))
            else:
                results_list.append(tick_length)
                deviations_list.append(0)

            if heights is not None:
                if m_heights:
                    heights_list.append(np.mean(m_heights))
                    heights_deviations_list.append(np.std(m_heights))
                else:
                    heights_list.append(0)
                    heights_deviations_list.append(0)

        self.w = pd.Series(results_list, index=left.index)
        self.wd = pd.Series(deviations_list, index=left.index)
        self.o = pd.Series(openness_list, index=left.index)

        if heights is not None:
            self.h = pd.Series(heights_list, index=left.index)
            self.hd = pd.Series(heights_deviations_list, index=left.index)
            self.p = self.h / self.w

    # http://wikicode.wikidot.com/get-angle-of-line-between-two-points
    # https://glenbambrick.com/tag/perpendicular/
    # angle between two points
    def _getAngle(self, pt1, pt2):
        """
        pt1, pt2 : tuple
        """
        x_diff = pt2[0] - pt1[0]
        y_diff = pt2[1] - pt1[1]
        return math.degrees(math.atan2(y_diff, x_diff))

    # start and end points of chainage tick
    # get the first end point of a tick
    def _getPoint1(self, pt, bearing, dist):
        """
        pt : tuple
        """
        angle = bearing + 90
        bearing = math.radians(angle)
        x = pt[0] + dist * math.cos(bearing)
        y = pt[1] + dist * math.sin(bearing)
        return (x, y)

    # get the second end point of a tick
    def _getPoint2(self, pt, bearing, dist):
        """
        pt : tuple
        """
        bearing = math.radians(bearing)
        x = pt[0] + dist * math.cos(bearing)
        y = pt[1] + dist * math.sin(bearing)
        return (x, y)


class WeightedCharacter:
    """
    Calculates the weighted character

    Character weighted by the area of the objects within ``k`` topological steps defined in ``spatial_weights``.

    .. math::
        \\frac{\\sum_{i=1}^{n} {character_{i} * area_{i}}}{\\sum_{i=1}^{n} area_{i}}

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    values : str, list, np.array, pd.Series
        the name of the gdf dataframe column, ``np.array``, or ``pd.Series`` where is stored character to be weighted
    spatial_weights : libpysal.weights
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on left.
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index.
    areas : str, list, np.array, pd.Series (default None)
        the name of the left dataframe column, ``np.array``, or ``pd.Series`` where is stored area value
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps


    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    values : GeoDataFrame
        Series containing used values
    areas : GeoDataFrame
        Series containing used areas
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID

    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation_df, threshold=100, silence_warnings=True)
    >>> buildings_df['w_height_100'] = momepy.WeightedCharacter(buildings_df, values='height', spatial_weights=sw,
                                                                 unique_id='uID').series
    100%|██████████| 144/144 [00:00<00:00, 361.60it/s]
    """

    def __init__(
        self, gdf, values, spatial_weights, unique_id, areas=None, verbose=True
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.copy()
        if areas is None:
            areas = gdf.geometry.area

        if not isinstance(areas, str):
            data["mm_a"] = areas
            areas = "mm_a"
        if not isinstance(values, str):
            data["mm_vals"] = values
            values = "mm_vals"

        self.areas = data[areas]
        self.values = data[values]

        data = data.set_index(unique_id)[[values, areas]]

        results_list = []
        for index in tqdm(data.index, total=data.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]

                subset = data.loc[neighbours]
                results_list.append(
                    (sum(subset[values] * subset[areas])) / (sum(subset[areas]))
                )
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class CoveredArea:
    """
    Calculates the area covered by neighbours

    Total area covered by neighbours defined in ``spatial_weights`` and element itself.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing Polygon geometry
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID

    Examples
    --------
    >>> sw = momepy.sw_high(k=3, gdf=tessellation_df, ids='uID')
    >>> tessellation_df['covered3steps'] = mm.CoveredArea(tessellation_df, sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 549.15it/s]

    """

    def __init__(self, gdf, spatial_weights, unique_id, verbose=True):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf
        area = data.set_index(unique_id).geometry.area

        results_list = []
        for index in tqdm(area.index, total=area.shape[0], disable=not verbose):
            if index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index].copy()
                if neighbours:
                    neighbours.append(index)
                else:
                    neighbours = [index]

                areas = area.loc[neighbours]
                results_list.append(sum(areas))
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)


class PerimeterWall:
    """
    Calculate the perimeter wall length the joined structure.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on gdf. It is to denote adjacent buildings (note: based on index, not ID).
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix

    Examples
    --------
    >>> buildings_df['wall_length'] = mm.PerimeterWall(buildings_df).series
    Calculating spatial weights...
    Spatial weights ready...
    100%|██████████| 144/144 [00:00<00:00, 4171.39it/s]

    Notes
    -----
    It might take a while to compute this character.
    """

    def __init__(self, gdf, spatial_weights=None, verbose=True):
        self.gdf = gdf

        if spatial_weights is None:

            print("Calculating spatial weights...") if verbose else None
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True)
            print("Spatial weights ready...") if verbose else None
        self.sw = spatial_weights

        # dict to store walls for each uID
        walls = {}
        components = pd.Series(spatial_weights.component_labels, index=range(len(gdf)))
        geom = gdf.geometry

        for i in tqdm(range(gdf.shape[0]), total=gdf.shape[0], disable=not verbose):
            # if the id is already present in walls, continue (avoid repetition)
            if i in walls:
                continue
            else:
                comp = spatial_weights.component_labels[i]
                to_join = components[components == comp].index
                joined = geom.iloc[to_join]
                dissolved = joined.buffer(
                    0.01
                ).unary_union  # buffer to avoid multipolygons where buildings touch by corners only
                for b in to_join:
                    walls[b] = dissolved.exterior.length

        results_list = []
        for i in tqdm(range(gdf.shape[0]), total=gdf.shape[0], disable=not verbose):
            results_list.append(walls[i])
        self.series = pd.Series(results_list, index=gdf.index)


class SegmentsLength:
    """
    Calculate the cummulative and/or mean length of segments.

    Length of segments within set topological distance from each of them.
    Reached topological distance should be captured by ``spatial_weights``. If ``mean=False`` it
    will compute sum of length, if ``mean=True`` it will compute sum and mean.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing streets (edges) to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on streets (note: spatial_weights should be based on index, not unique ID).
    mean : boolean, optional
        If mean=False it will compute sum of length, if mean=True it will compute
        sum and mean
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting total lengths
    mean : Series
        Series containing resulting total lengths
    sum : Series
        Series containing resulting total lengths
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix

    Examples
    --------
    >>> streets_df['length_neighbours'] = mm.SegmentsLength(streets_df, mean=True).mean
    Calculating spatial weights...
    Spatial weights ready...
    """

    def __init__(self, gdf, spatial_weights=None, mean=False, verbose=True):
        self.gdf = gdf

        if spatial_weights is None:
            print("Calculating spatial weights...") if verbose else None
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True)
            print("Spatial weights ready...") if verbose else None
        self.sw = spatial_weights

        lenghts = gdf.geometry.length

        sums = []
        means = []
        for index in tqdm(gdf.index, total=gdf.shape[0], disable=not verbose):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            dims = lenghts.iloc[neighbours]
            if mean:
                means.append(np.mean(dims))
            sums.append(sum(dims))

        self.series = self.sum = pd.Series(sums, index=gdf.index)
        if mean:
            self.mean = pd.Series(means, index=gdf.index)
