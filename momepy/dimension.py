#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dimension.py
# definitons of dimension characters

import math

import numpy as np
import pandas as pd
import scipy as sp
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm

from .shape import _make_circle

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

    It is a simple wrapper for geopandas `gdf.geometry.area` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Attributes
    ----------
    area : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['area'] = momepy.Area(buildings).area
    >>> buildings.area[0]
    728.5574947044363

    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.area = self.gdf.geometry.area


class Perimeter:
    """
    Calculates perimeter of each object in given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for geopandas `gdf.geometry.length` for the consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Attributes
    ----------
    perimeter : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['perimeter'] = momepy.Perimeter(buildings).perimeter
    >>> buildings.perimeter[0]
    137.18630991119903
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.perimeter = self.gdf.geometry.length


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
        the name of the dataframe column, np.array, or pd.Series where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    volume : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    heights : Series
        Series containing used heights values

    areas : GeoDataFrame
        Series containing used areas values

    Examples
    --------
    >>> buildings['volume'] = momepy.Volume(buildings, heights='height_col')
    >>> buildings.volume[0]
    7285.5749470443625

    >>> buildings['volume'] = momepy.Volume(buildings, heights='height_col', areas='area_col').volume
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
            self.volume = self.areas * self.heights

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
        the name of the dataframe column, np.array, or pd.Series where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    fa : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    heights : Series
        Series containing used heights values
    areas : GeoDataFrame
        Series containing used areas values

    Examples
    --------
    >>> buildings['floor_area'] = momepy.FloorArea(buildings, heights='height_col')
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309

    >>> buildings['floor_area'] = momepy.FloorArea(buildings, heights='height_col', areas='area_col').fa
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
            self.fa = self.areas * (self.heights // 3)

        except KeyError:
            raise KeyError(
                "ERROR: Column not found. Define heights and areas or set areas to None."
            )


class CourtyardArea:
    """
    Calculates area of holes within geometry - area of courtyards.

    Ensure that your geometry is shapely Polygon.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Attributes
    ----------
    ca : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    areas : GeoDataFrame
        Series containing used areas values

    Examples
    --------
    >>> buildings['courtyard_area'] = momepy.CourtyardArea(buildings).ca
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

        self.ca = gdf.apply(
            lambda row: Polygon(row.geometry.exterior).area - row[areas], axis=1
        )


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
    lal : Series
        Series containing resulting values

    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> buildings['lal'] = momepy.LongestAxisLength(buildings).lal
    >>> buildings.lal[0]
    40.2655616057102
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.lal = gdf.apply(
            lambda row: self._longest_axis(row.geometry.convex_hull.exterior.coords),
            axis=1,
        )

    # calculate the radius of circumcircle
    def _longest_axis(self, points):
        circ = _make_circle(points)
        return circ[2] * 2


class AverageCharacter:
    """
    Calculates the average of a character within k steps of morphological tessellation

    Average value of the character within k topological steps defined in spatial_weights.
    Can be set to `mean`, `median` or `mode`.

    .. math::
        \\frac{1}{n}\\left(\\sum_{i=1}^{n} value_{i}\\right)

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    values : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored character value.
    unique_id : str
        name of the column with unique id used as spatial_weights index.
    spatial_weights : libpysal.weights
        spatial weights matrix
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.
    mode : str (default 'mean')
        mode of average calculation. Can be set to `mean`, `median` or `mode`.

    Attributes
    ----------
    ac : Series
        Series containing resulting values
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
    mode : str
        mode

    References
    ----------
    Hausleitner B and Berghauser Pont M (2017) Development of a configurational
    typology for micro-businesses integrating geometric and configurational variables. [adapted]

    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation, threshold=100, silence_warnings=True, ids='uID')
    >>> tessellation['mesh_100'] = momepy.AverageCharacter(tessellation, values='area', spatial_weights=sw, unique_id='uID').ac
    100%|██████████| 144/144 [00:00<00:00, 1433.32it/s]
    >>> tessellation.mesh_100[0]
    4823.1334436678835
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, rng=None, mode="mean"):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        self.rng = rng
        self.mode = mode

        data = gdf.copy()
        if values is not None:
            if not isinstance(values, str):
                data["mm_v"] = values
                values = "mm_v"
        self.values = data[values]

        data = data.set_index(unique_id)

        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            values_list = data.loc[neighbours][values]

            if rng:
                from momepy import limit_range

                values_list = limit_range(values_list.tolist(), rng=rng)
            if mode == "mean":
                results_list.append(np.mean(values_list))
            elif mode == "median":
                results_list.append(np.median(values_list))
            elif mode == "mode":
                results_list.append(sp.stats.mode(values_list)[0][0])
            else:
                raise ValueError("{} is not supported as mode.".format(mode))

        self.ac = pd.Series(results_list, index=gdf.index)


class StreetProfile:
    """
    Calculates the street profile characters.

    Returns a dictionary with widths, standard deviation of width, openness, heights,
    standard deviation of height and ratio height/width. Algorithm generates perpendicular
    lines to `right` dataframe features every `distance` and measures values on intersection
    with features of `left.` If no feature is reached within
    `tick_length` its value is set as width (being theoretical maximum).

    .. math::
        \\

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing streets to analyse
    right : GeoDataFrame
        GeoDataFrame containing buildings along the streets (only Polygon geometry type is supported)
    heights: str, list, np.array, pd.Series (default None)
        the name of the buildings dataframe column, np.array, or pd.Series where is stored building height. If set to None,
        height and ratio height/width will not be calculated.
    distance : int (default 10)
        distance between perpendicular ticks
    tick_length : int (default 50)
        lenght of ticks

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
        lenght of ticks
    heights : GeoDataFrame
        Series containing used height values

    References
    ----------
    Oliveira V (2013) Morpho: a methodology for assessing urban form. Urban Morphology 17(1): 21–33.

    Araldi A and Fusco G (2017) Decomposing and Recomposing Urban Fabric: The City from the Pedestrian
    Point of View. In: Gervasi O, Murgante B, Misra S, et al. (eds), Computational Science and Its
    Applications – ICCSA 2017, Lecture Notes in Computer Science, Cham: Springer International
    Publishing, pp. 365–376. Available from: http://link.springer.com/10.1007/978-3-319-62407-5.

    Examples
    --------
    >>> street_profile = momepy.StreetProfile(streets_df, buildings_df, heights='height')
    100%|██████████| 33/33 [00:02<00:00, 15.66it/s]
    >>> streets_df['width'] = street_profile.w
    >>> streets_df['deviations'] = street_profile.wd
    """

    def __init__(self, left, right, heights=None, distance=10, tick_length=50):
        self.left = left
        self.right = right
        self.distance = distance
        self.tick_length = tick_length

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

        for idx, row in tqdm(left.iterrows(), total=left.shape[0]):
            # if idx == 2:
            #     ee
            # list to hold all the point coords
            list_points = []
            # set the current distance to place the point
            current_dist = distance
            # make shapely MultiLineString object
            shapely_line = row.geometry
            # get the total length of the line
            line_length = shapely_line.length
            # append the starting coordinate to the list
            list_points.append(Point(list(shapely_line.coords)[0]))
            # https://nathanw.net/2012/08/05/generating-chainage-distance-nodes-in-qgis/
            # while the current cumulative distance is less than the total length of the line
            while current_dist < line_length:
                # use interpolate and increase the current distance
                list_points.append(shapely_line.interpolate(current_dist))
                current_dist += distance
            # append end coordinate to the list
            list_points.append(Point(list(shapely_line.coords)[-1]))

            ticks = []
            for num, pt in enumerate(list_points, 1):
                # start chainage 0
                if num == 1:
                    angle = self._getAngle(pt, list_points[num])
                    line_end_1 = self._getPoint1(pt, angle, tick_length / 2)
                    angle = self._getAngle(line_end_1, pt)
                    line_end_2 = self._getPoint2(line_end_1, angle, tick_length)
                    tick1 = LineString([(line_end_1.x, line_end_1.y), (pt.x, pt.y)])
                    tick2 = LineString([(line_end_2.x, line_end_2.y), (pt.x, pt.y)])
                    ticks.append([tick1, tick2])

                # everything in between
                if num < len(list_points) - 1:
                    angle = self._getAngle(pt, list_points[num])
                    line_end_1 = self._getPoint1(
                        list_points[num], angle, tick_length / 2
                    )
                    angle = self._getAngle(line_end_1, list_points[num])
                    line_end_2 = self._getPoint2(line_end_1, angle, tick_length)
                    tick1 = LineString(
                        [
                            (line_end_1.x, line_end_1.y),
                            (list_points[num].x, list_points[num].y),
                        ]
                    )
                    tick2 = LineString(
                        [
                            (line_end_2.x, line_end_2.y),
                            (list_points[num].x, list_points[num].y),
                        ]
                    )
                    ticks.append([tick1, tick2])

                # end chainage
                if num == len(list_points):
                    angle = self._getAngle(list_points[num - 2], pt)
                    line_end_1 = self._getPoint1(pt, angle, tick_length / 2)
                    angle = self._getAngle(line_end_1, pt)
                    line_end_2 = self._getPoint2(line_end_1, angle, tick_length)
                    tick1 = LineString([(line_end_1.x, line_end_1.y), (pt.x, pt.y)])
                    tick2 = LineString([(line_end_2.x, line_end_2.y), (pt.x, pt.y)])
                    ticks.append([tick1, tick2])
            # widths = []
            m_heights = []
            lefts = []
            rights = []
            for duo in ticks:

                for ix, tick in enumerate(duo):
                    possible_intersections_index = list(
                        sindex.intersection(tick.bounds)
                    )
                    possible_intersections = right.iloc[possible_intersections_index]
                    real_intersections = possible_intersections.intersects(tick)
                    get_height = right.loc[list(real_intersections.index)]
                    possible_int = get_height.exterior.intersection(tick)

                    if not possible_int.is_empty.all():
                        true_int = []
                        for one in list(possible_int.index):
                            if possible_int[one].type == "Point":
                                true_int.append(possible_int[one])
                            elif possible_int[one].type == "MultiPoint":
                                for p in possible_int[one]:
                                    true_int.append(p)

                        if len(true_int) > 1:
                            distances = []
                            ix = 0
                            for p in true_int:
                                distance = p.distance(Point(tick.coords[-1]))
                                distances.append(distance)
                                ix = ix + 1
                            minimal = min(distances)
                            if ix == 0:
                                lefts.append(minimal)
                            else:
                                rights.append(minimal)
                        else:
                            if ix == 0:
                                lefts.append(
                                    true_int[0].distance(Point(tick.coords[-1]))
                                )
                            else:
                                rights.append(
                                    true_int[0].distance(Point(tick.coords[-1]))
                                )
                        if heights is not None:
                            indices = {}
                            for idx, row in get_height.iterrows():
                                dist = row.geometry.distance(Point(tick.coords[-1]))
                                indices[idx] = dist
                            minim = min(indices, key=indices.get)
                            m_heights.append(right.loc[minim][heights])

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
        x_diff = pt2.x - pt1.x
        y_diff = pt2.y - pt1.y
        return math.degrees(math.atan2(y_diff, x_diff))

    # start and end points of chainage tick
    # get the first end point of a tick
    def _getPoint1(self, pt, bearing, dist):
        angle = bearing + 90
        bearing = math.radians(angle)
        x = pt.x + dist * math.cos(bearing)
        y = pt.y + dist * math.sin(bearing)
        return Point(x, y)

    # get the second end point of a tick
    def _getPoint2(self, pt, bearing, dist):
        bearing = math.radians(bearing)
        x = pt.x + dist * math.cos(bearing)
        y = pt.y + dist * math.sin(bearing)
        return Point(x, y)


class WeightedCharacter:
    """
    Calculates the weighted character

    Character weighted by the area of the objects within `k` topological steps defined in spatial_weights.

    .. math::
        \\frac{\\sum_{i=1}^{n} {character_{i} * area_{i}}}{\\sum_{i=1}^{n} area_{i}}

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    values : str, list, np.array, pd.Series
        the name of the gdf dataframe column, np.array, or pd.Series where is stored character to be weighted
    spatial_weights : libpysal.weights
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on left.
    unique_id : str
        name of the column with unique id used as spatial_weights index.
    areas : str, list, np.array, pd.Series (default None)
        the name of the left dataframe column, np.array, or pd.Series where is stored area value


    Attributes
    ----------
    wc : Series
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

    References
    ----------
    Dibble J, Prelorendjos A, Romice O, et al. (2017) On the origin of spaces: Morphometric foundations of urban form evolution.
    Environment and Planning B: Urban Analytics and City Science 46(4): 707–730.

    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation_df, threshold=100, silence_warnings=True)
    >>> buildings_df['w_height_100'] = momepy.WeightedCharacter(buildings_df, values='height', spatial_weights=sw,
                                                                 unique_id='uID').wc
    100%|██████████| 144/144 [00:00<00:00, 361.60it/s]
    """

    def __init__(self, gdf, values, spatial_weights, unique_id, areas=None):
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

        data = data.set_index(unique_id)

        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            subset = data.loc[neighbours]
            results_list.append(
                (sum(subset[values] * subset[areas])) / (sum(subset[areas]))
            )

        self.wc = pd.Series(results_list, index=gdf.index)


class CoveredArea:
    """
    Calculates the area covered by k steps of morphological tessellation

    Total area covered within k topological steps defined in spatial_weights.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    spatial_weights : libpysal.weights
        spatial weights matrix
    unique_id : str
        name of the column with unique id used as spatial_weights index.

    Attributes
    ----------
    ca : Series
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
    >>> tessellation_df['covered3steps'] = mm.CoveredArea(tessellation_df, sw, 'uID').ca
    100%|██████████| 144/144 [00:00<00:00, 549.15it/s]

    """

    def __init__(self, gdf, spatial_weights, unique_id):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.set_index(unique_id)

        results_list = []
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            areas = data.loc[neighbours].geometry.area
            results_list.append(sum(areas))

        self.ca = pd.Series(results_list, index=gdf.index)


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

    Attributes
    ----------
    wall : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix

    Examples
    --------
    >>> buildings_df['wall_length'] = mm.PerimeterWall(buildings_df).wall
    Calculating spatial weights...
    Spatial weights ready...
    100%|██████████| 144/144 [00:00<00:00, 4171.39it/s]

    Notes
    -----
    It might take a while to compute this character.
    """

    def __init__(self, gdf, spatial_weights=None):
        self.gdf = gdf

        if spatial_weights is None:
            print("Calculating spatial weights...")
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True)
            print("Spatial weights ready...")
        self.sw = spatial_weights

        # dict to store walls for each uID
        walls = {}
        components = pd.Series(spatial_weights.component_labels, index=gdf.index)

        for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            # if the id is already present in walls, continue (avoid repetition)
            if index in walls:
                continue
            else:
                comp = spatial_weights.component_labels[index]
                to_join = components[components == comp].index
                joined = gdf.iloc[to_join]
                dissolved = joined.geometry.buffer(
                    0.01
                ).unary_union  # buffer to avoid multipolygons where buildings touch by corners only
                for b in to_join:
                    walls[b] = dissolved.exterior.length

        results_list = []
        for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            results_list.append(walls[index])
        self.wall = pd.Series(results_list, index=gdf.index)


class SegmentsLength:
    """
    Calculate the cummulative or mean length of segments.

    Length of segments within set topological distance from each of them.
    Reached topological distance should be captured by spatial_weights. If mean=False
    it will return total length, if mean=True it will return mean value.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing streets (edges) to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on streets (note: spatial_weights shoud be based on index, not unique ID).
    mean : boolean, optional
        If mean=False it will return total length, if mean=True it will return mean value

    Attributes
    ----------
    sl : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix
    mean : boolean
        used mean boolean value

    Examples
    --------
    >>> streets_df['length_neighbours'] = mm.SegmentsLength(streets_df, mean=True).sl
    Calculating spatial weights...
    Spatial weights ready...
    """

    def __init__(self, gdf, spatial_weights=None, mean=False):
        self.gdf = gdf
        self.mean = mean

        if spatial_weights is None:
            print("Calculating spatial weights...")
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True)
            print("Spatial weights ready...")
        self.sw = spatial_weights

        results_list = []
        for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
            neighbours = spatial_weights.neighbors[index].copy()
            if neighbours:
                neighbours.append(index)
            else:
                neighbours = [index]

            dims = gdf.iloc[neighbours].geometry.length
            if mean:
                results_list.append(np.mean(dims))
            else:
                results_list.append(sum(dims))

        self.sl = pd.Series(results_list, index=gdf.index)
