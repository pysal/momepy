#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dimension.py
# definitons of dimension characters

from tqdm import tqdm  # progress bar
from shapely.geometry import Polygon, LineString, Point
from .shape import make_circle
import pandas as pd
import math
import numpy as np


def area(objects):
    """
    Calculate area of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating areas...')

    # fill results_list with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        results_list.append(row['geometry'].area)

    series = pd.Series(results_list)

    print('Areas calculated.')
    return series


def perimeter(objects):
    """
    Calculate perimeter of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating perimeters...')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        results_list.append(row['geometry'].length)

    series = pd.Series(results_list)

    print('Perimeters calculated.')
    return series


def height_os(objects, original_column='relh2'):
    """
    Copy values from GB Ordance Survey data.

    Function tailored to GB Ordance Survey data of OS Building Heights (Alpha).
    It will copy RelH2 (relative height from ground level to base of the roof) values to new column.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    original_column : str
        name of column where is stored original height value


    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating heights...')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        results_list.append(row[original_column])

    series = pd.Series(results_list)

    print('Heights defined.')
    return series


def height_prg(objects, floors_column='od_POCET_P', floor_type='od_TYP'):
    """
    Define building heights based on Geoportal Prague Data.

    Function tailored to Geoportal Prague.
    It will calculate estimated building heights based on floor number and type.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    floor_type : str
        name of the column defining buildings type (to differentiate height of the floor)

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating heights...')

    # fill new column with the value of perimeter, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        if row[floor_type] == 7:
            height = row[floors_column] * 3.5  # old buildings with high ceiling
        elif row[floor_type] == 3:
            height = row[floors_column] * 5  # warehouses
        else:
            height = row[floors_column] * 3  # standard buildings

        results_list.append(height)

    series = pd.Series(results_list)

    print('Heights defined.')
    return series


def volume(objects, area_column, height_column, area_calculated):
    """
    Calculate volume of each object in given shapefile based on its height and area.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    area_column : str
        name of column where is stored area value
    height_column : str
        name of column where is stored height value
    area_calculated : bool
        boolean value checking whether area has been previously calculated and
        stored in separate column. If set to False, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating volumes...')

    if area_calculated:
        try:
            # fill new column with the value of perimeter, iterating over rows one by one
            for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
                results_list.append(row[area_column] * row[height_column])
            series = pd.Series(results_list)

            print('Volumes calculated.')
            return series

        except KeyError:
            print('ERROR: Building area column named', area_column, 'not found. Define area_column or set area_calculated to False.')
    else:
        # fill new column with the value of perimeter, iterating over rows one by one
        for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
            results_list.append(row['geometry'].area * row[height_column])

        series = pd.Series(results_list)

        print('Volumes calculated.')
        return series


def floor_area(objects, area_column, height_column, area_calculated):
    """
    Calculate floor area of each object based on height and area.

    Number of floors is simplified into formula height / 3
    (it is assumed that on average one floor is approximately 3 metres)

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    area_column : str
        name of column where is stored area value
    height_column : str
        name of column where is stored height value
    area_calculated : bool
        boolean value checking whether area has been previously calculated and
        stored in separate column. If set to False, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating floor areas...')

    if area_calculated:
        try:
            # fill new column with the value of perimeter, iterating over rows one by one
            for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
                results_list.append(row[area_column] * (row[height_column] // 3))

            series = pd.Series(results_list)

            print('Floor areas calculated.')

        except KeyError:
            print('ERROR: Building area column named', area_column, 'not found. Define area_column or set area_calculated to False.')
    else:
        # fill new column with the value of perimeter, iterating over rows one by one
        for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
            results_list.append(row['geometry'].area * (row[height_column] // 3))

        series = pd.Series(results_list)

        print('Floor areas calculated.')
    return series


def courtyard_area(objects, area_column, area_calculated):
    """
    Calculate area of holes within geometry - area of courtyards.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    area_column : str
        name of column where is stored area value
    area_calculated : bool
        boolean value checking whether area has been previously calculated and
        stored in separate column. If set to False, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating courtyard areas...')

    if area_calculated:
        try:
            for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
                results_list.append(Polygon(row['geometry'].exterior).area - row[area_column])

            series = pd.Series(results_list)

            print('Core area indices calculated.')
        except KeyError:
            print('ERROR: Building area column named', area_column, 'not found. Define area_column or set area_calculated to False.')
    else:
        for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
            results_list.append(Polygon(row['geometry'].exterior).area - row['geometry'].area)

        series = pd.Series(results_list)

        print('Core area indices calculated.')
    return series


# calculate the area of circumcircle
def _longest_axis(points):
    circ = make_circle(points)
    return(circ[2] * 2)


def longest_axis_length(objects):
    """
    Calculate the length of the longest axis of object.

    Axis is defined as a diameter of minimal circumscribed circle around the convex hull.
    It does not have to be fully inside an object.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []
    print('Calculating the longest axis...')

    def sort_NoneType(geom):
        if geom is not None:
            if geom.type is not 'Polygon':
                return(0)
            else:
                return(_longest_axis(list(geom.boundary.coords)))
        else:
            return(0)

    # fill new column with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        results_list.append(_longest_axis(row['geometry'].convex_hull.exterior.coords))

    series = pd.Series(results_list)

    print('The longest axis calculated.')
    return series


def effective_mesh(objects, spatial_weights, area_column):
    """
    Calculate the effective mesh size

    Effective mesh size of the area within k topological steps defined in spatial_weights.

    .. math::
        \\

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights
        spatial weights matrix
    area_column : str
        name of the column of objects gdf where is stored area value

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Hausleitner B and Berghauser Pont M (2017) Development of a configurational
    typology for micro-businesses integrating geometric and configurational variables.

    Notes
    -----
    Resolve the issues if there is no spatial weights matrix. Corellation with block_density()

    """
    # define empty list for results
    results_list = []

    print('Calculating effective mesh size...')

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbours = spatial_weights.neighbors[index]
        total_area = sum(objects.iloc[neighbours][area_column]) + row[area_column]
        results_list.append(total_area / (len(neighbours) + 1))

    series = pd.Series(results_list)
    print('Effective mesh size calculated.')
    return series


def street_profile(streets, buildings, height_column=None, distance=10, tick_length=50):
    """
    Calculate the street profile widths, heights, and ratio height/width

    .. math::
        \\

    Parameters
    ----------
    streets : GeoDataFrame
        GeoDataFrame containing streets to analyse
    buildings : GeoDataFrame
        GeoDataFrame containing buildings along the streets
    height_column: str, optional
        name of the column of buildings gdf where is stored building height. If set to None,
        height and ratio height/width will not be calculated.
    distance : int, optional
        distance between perpendicular ticks
    tick_length : int, optional
        lenght of ticks

    Returns
    -------
    widths, heights, profile_ratio : tuple

    widths : Series
        Series containing street profile width values.
    heights : Series
        Series containing street profile heights values.
    profile_ratio : Series
        Series containing street profile height/width ratio values.

    References
    ----------
    Oliveira V (2013) Morpho: a methodology for assessing urban form. Urban Morphology 17(1): 21–33.

    Notes
    -----
    Add condition for segments without buildings around
    Add explanation of algorithm

    """

    print('Calculating street profile...')

    # http://wikicode.wikidot.com/get-angle-of-line-between-two-points
    # https://glenbambrick.com/tag/perpendicular/
    # angle between two points
    def _getAngle(pt1, pt2):
        x_diff = pt2.x - pt1.x
        y_diff = pt2.y - pt1.y
        return math.degrees(math.atan2(y_diff, x_diff))

    # start and end points of chainage tick
    # get the first end point of a tick
    def _getPoint1(pt, bearing, dist):
        angle = bearing + 90
        bearing = math.radians(angle)
        x = pt.x + dist * math.cos(bearing)
        y = pt.y + dist * math.sin(bearing)
        return Point(x, y)

    # get the second end point of a tick
    def _getPoint2(pt, bearing, dist):
        bearing = math.radians(bearing)
        x = pt.x + dist * math.cos(bearing)
        y = pt.y + dist * math.sin(bearing)
        return Point(x, y)

    # distance between each points
    distance = 10
    # the length of each tick
    tick_length = 50
    sindex = buildings.sindex

    results_list = []
    heights_list = []

    for idx, row in tqdm(streets.iterrows(), total=streets.shape[0]):
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
                angle = _getAngle(pt, list_points[num])
                line_end_1 = _getPoint1(pt, angle, tick_length / 2)
                angle = _getAngle(line_end_1, pt)
                line_end_2 = _getPoint2(line_end_1, angle, tick_length)
                tick1 = LineString([(line_end_1.x, line_end_1.y), (pt.x, pt.y)])
                tick2 = LineString([(line_end_2.x, line_end_2.y), (pt.x, pt.y)])
                ticks.append([tick1, tick2])

            # everything in between
            if num < len(list_points) - 1:
                angle = _getAngle(pt, list_points[num])
                line_end_1 = _getPoint1(list_points[num], angle, tick_length / 2)
                angle = _getAngle(line_end_1, list_points[num])
                line_end_2 = _getPoint2(line_end_1, angle, tick_length)
                tick1 = LineString([(line_end_1.x, line_end_1.y), (list_points[num].x, list_points[num].y)])
                tick2 = LineString([(line_end_2.x, line_end_2.y), (list_points[num].x, list_points[num].y)])
                ticks.append([tick1, tick2])

            # end chainage
            if num == len(list_points):
                angle = _getAngle(list_points[num - 2], pt)
                line_end_1 = _getPoint1(pt, angle, tick_length / 2)
                angle = _getAngle(line_end_1, pt)
                line_end_2 = _getPoint2(line_end_1, angle, tick_length)
                tick1 = LineString([(line_end_1.x, line_end_1.y), (pt.x, pt.y)])
                tick2 = LineString([(line_end_2.x, line_end_2.y), (pt.x, pt.y)])
                ticks.append([tick1, tick2])
        widths = []
        heights = []
        for duo in ticks:
            width = []
            for tick in duo:
                possible_intersections_index = list(sindex.intersection(tick.bounds))
                possible_intersections = buildings.iloc[possible_intersections_index]
                real_intersections = possible_intersections.intersects(tick)
                get_height = buildings.iloc[list(real_intersections.index)]
                possible_int = get_height.exterior.intersection(tick)

                if possible_int.any():
                    true_int = []
                    for one in list(possible_int.index):
                        if possible_int[one].type == 'Point':
                            true_int.append(possible_int[one])
                        elif possible_int[one].type == 'MultiPoint':
                            true_int.append(possible_int[one][0])
                            true_int.append(possible_int[one][1])

                    if len(true_int) > 1:
                        distances = []
                        ix = 0
                        for p in true_int:
                            distance = p.distance(Point(tick.coords[-1]))
                            distances.append(distance)
                            ix = ix + 1
                        minimal = min(distances)
                        width.append(minimal)
                    else:
                        width.append(true_int[0].distance(Point(tick.coords[-1])))
                    if height_column is not None:
                        indices = {}
                        for idx, row in get_height.iterrows():
                            dist = row.geometry.distance(Point(tick.coords[-1]))
                            indices[idx] = dist
                        minim = min(indices, key=indices.get)
                        heights.append(buildings.iloc[minim][height_column])
                else:
                    width.append(np.nan)
            widths.append(width[0] + width[1])

        results_list.append(np.nanmean(widths))
        if height_column is not None:
            heights_list.append(np.mean(heights))

    widths_series = pd.Series(results_list)
    if height_column is not None:
        heights_series = pd.Series(heights_list)
        profile_ratio = heights_series / widths_series
        return widths_series, heights_series, profile_ratio
    else:
        return widths_series
    print('Street profile calculated.')


def weighted_character(objects, tessellation, spatial_weights, character_column, area_column):
    """
    Calculate the weighted character

    Character weighted by the area of the objects of the area within `k` topological steps defined in spatial_weights.

    .. math::
        \\

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights
        spatial weights matrix
    character_column : str
        name of the column of objects gdf where is stored character
    area_column : str
        name of the column of objects gdf where is stored area value

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Jacob

    Notes
    -----
    Resolve the issues if there is no spatial weights matrix.

    """
    # define empty list for results
    results_list = []
    weights_matrix = spatial_weights
    print('Calculating weighted {}...'.format(character_column))

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        id = tessellation.loc[tessellation['uID'] == row['uID']].index[0]
        neighbours = weights_matrix.neighbors[id]

        if len(neighbours) > 0:
            neighbours_ids = tessellation.iloc[neighbours]['uID']
            building_neighbours = objects.loc[objects['uID'].isin(neighbours_ids)]

            results_list.append((sum(building_neighbours[character_column] *
                                     building_neighbours[area_column]) +
                                (row[character_column] * row[area_column])) /
                                (sum(building_neighbours[area_column]) + row[area_column]))
        else:
            results_list.append(row[character_column])
    series = pd.Series(results_list)

    print('Weighted {} calculated.'.format(character_column))
    return series
# to be deleted, keep at the end

# path = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/test/voronoi_10.shp"
# objects = gpd.read_file(path)

# longest_axis_length2(objects='longest_axis')

# objects.head
# objects.to_file(path)
