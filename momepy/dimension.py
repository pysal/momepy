#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dimension.py
# definitons of dimension characters

from tqdm import tqdm  # progress bar
from shapely.geometry import Polygon, LineString, Point
from .shape import _make_circle
import pandas as pd
import math
import numpy as np


def area(objects):
    """
    Calculates area of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for geopandas gdf.geometry.area for consistency of momepy.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['area'] = momepy.area(buildings)
    Calculating areas...
    Areas calculated.
    >>> buildings.area[0]
    728.5574947044363

    """

    print('Calculating areas...')

    series = objects.geometry.area

    print('Areas calculated.')
    return series


def perimeter(objects):
    """
    Calculates perimeter of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for geopandas gdf.geometry.length for consistency of momepy.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['perimeter'] = momepy.perimeter(buildings)
    Calculating perimeters...
    Perimeters calculated.
    >>> buildings.perimeter[0]
    137.18630991119903
    """

    print('Calculating perimeters...')

    series = objects.geometry.length

    print('Perimeters calculated.')
    return series


def volume(objects, heights, areas=None):
    """
    Calculates volume of each object in given shapefile based on its height and area.

    .. math::
        area * height

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    heights : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings['volume'] = momepy.volume(buildings, heights='height_col')
    Calculating volumes...
    Volumes calculated.
    >>> buildings.volume[0]
    7285.5749470443625

    >>> buildings['volume'] = momepy.volume(buildings, heights='height_col', areas='area_col')
    Calculating volumes...
    Volumes calculated.
    >>> buildings.volume[0]
    7285.5749470443625
    """
    print('Calculating volumes...')
    if not isinstance(heights, str):
        objects['mm_h'] = heights
        heights = 'mm_h'

    if areas is not None:
        if not isinstance(areas, str):
            objects['mm_a'] = areas
            areas = 'mm_a'
        try:
            series = objects[areas] * objects[heights]

        except KeyError:
            raise KeyError('ERROR: Column not found. Define heights and areas or set areas to None.')
    else:
        series = objects.geometry.area * objects[heights]

    if 'mm_h' in objects.columns:
        objects.drop(columns=['mm_h'], inplace=True)
    if 'mm_a' in objects.columns:
        objects.drop(columns=['mm_a'], inplace=True)

    print('Volumes calculated.')
    return series


def floor_area(objects, heights, areas=None):
    """
    Calculates floor area of each object based on height and area.

    Number of floors is simplified into formula height / 3
    (it is assumed that on average one floor is approximately 3 metres)

    .. math::
        area * \\frac{height}{3}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    heights : str, list, np.array, pd.Series
        the name of the dataframe column, np.array, or pd.Series where is stored height value
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings['floor_area'] = momepy.volume(buildings, heights='height_col')
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309

    >>> buildings['floor_area'] = momepy.volume(buildings, heights='height_col', areas='area_col')
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309
    """
    print('Calculating floor areas...')
    if not isinstance(heights, str):
        objects['mm_h'] = heights
        heights = 'mm_h'

    if areas is not None:
        if not isinstance(areas, str):
            objects['mm_a'] = areas
            areas = 'mm_a'
        try:
            series = objects[areas] * (objects[heights] // 3)

        except KeyError:
            raise KeyError('ERROR: Column not found. Define heights and areas or set areas to None.')
    else:
        series = objects.geometry.area * (objects[heights] // 3)

    if 'mm_h' in objects.columns:
        objects.drop(columns=['mm_h'], inplace=True)
    if 'mm_a' in objects.columns:
        objects.drop(columns=['mm_a'], inplace=True)

    print('Floor areas calculated.')
    return series


def courtyard_area(objects, areas=None):
    """
    Calculates area of holes within geometry - area of courtyards.

    Ensure that your geometry is shapely Polygon.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings['courtyard_area'] = momepy.courtyard_area(buildings)
    Calculating courtyard areas...
    Courtyard areas calculated.
    >>> buildings.courtyard_area[80]
    353.33274206543274
    """

    print('Calculating courtyard areas...')

    if areas is not None:
        if not isinstance(areas, str):
            objects['mm_a'] = areas
            areas = 'mm_a'
        series = objects.apply(lambda row: Polygon(row.geometry.exterior).area - row[areas], axis=1)

    else:
        series = objects.apply(lambda row: Polygon(row.geometry.exterior).area - row.geometry.area, axis=1)

    if 'mm_a' in objects.columns:
        objects.drop(columns=['mm_a'], inplace=True)

    print('Courtyard areas calculated.')
    return series


# calculate the radius of circumcircle
def _longest_axis(points):
    circ = _make_circle(points)
    return(circ[2] * 2)


def longest_axis_length(objects):
    """
    Calculates the length of the longest axis of object.

    Axis is defined as a diameter of minimal circumscribed circle around the convex hull.
    It does not have to be fully inside an object.

    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings['lal'] = momepy.longest_axis_length(buildings)
    Calculating the longest axis...
    The longest axis calculated.
    >>> buildings.lal[0]
    40.2655616057102
    """

    print('Calculating the longest axis...')

    series = objects.apply(lambda row: _longest_axis(row['geometry'].convex_hull.exterior.coords), axis=1)

    print('The longest axis calculated.')
    return series


def effective_mesh(objects, spatial_weights=None, areas=None, order=3, mode=None):
    """
    Calculates the effective mesh size of morphological tessellation

    Effective mesh size of the area within k topological steps defined in spatial_weights.

    .. math::
        \\frac{1}{n}\\left(\\sum_{i=1}^{n} area_{i}\\right)

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.
    order : int
        order of Queen contiguity
    mode : str
        mode of calculation. None will use all values, `iq` will use values within interquartile range, `id` will use values within interdecile range

    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Hausleitner B and Berghauser Pont M (2017) Development of a configurational
    typology for micro-businesses integrating geometric and configurational variables. [adapted]

    Examples
    --------
    >>> tessellation['mesh'] = momepy.effective_mesh(tessellation)
    Calculating effective mesh size...
    Generating weights matrix (Queen) of 3 topological steps...
    100%|██████████| 144/144 [00:00<00:00, 900.83it/s]
    Effective mesh size calculated.
    >>> tessellation.mesh[0]
    2922.957260196682

    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation, threshold=100, silence_warnings=True)
    >>> tessellation['mesh_100'] = momepy.effective_mesh(tessellation, spatial_weights=sw, areas='area')
    Calculating effective mesh size...
    100%|██████████| 144/144 [00:00<00:00, 1433.32it/s]
    Effective mesh size calculated.
    >>> tessellation.mesh_100[0]
    4823.1334436678835

    Notes
    -----
    Check corellation with block_density()

    """
    # define empty list for results
    results_list = []

    print('Calculating effective mesh size...')

    if spatial_weights is None:
        print('Generating weights matrix (Queen) of {} topological steps...'.format(order))
        from momepy import Queen_higher
        # matrix to define area of analysis (more steps)
        spatial_weights = Queen_higher(k=order, geodataframe=objects)
    else:
        if not all(objects.index == range(len(objects))):
            raise ValueError('Index is not consecutive range 0:x, spatial weights will not match objects.')

    if areas is not None:
        if not isinstance(areas, str):
            objects['mm_a'] = areas
            areas = 'mm_a'

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbours = spatial_weights.neighbors[index]
        if areas is not None:
            area_list = objects.iloc[neighbours][areas].tolist()
            area_list.append(row[areas])
        else:
            area_list = objects.iloc[neighbours].geometry.area.tolist()
            area_list.append(row.geometry.area)
        if mode:
            from momepy import limit_range
            area_list = limit_range(area_list, mode=mode)
        results_list.append(sum(area_list) / len(area_list))

    series = pd.Series(results_list)

    if 'mm_a' in objects.columns:
        objects.drop(columns=['mm_a'], inplace=True)

    print('Effective mesh size calculated.')
    return series


def street_profile(streets, buildings, heights=None, distance=10, tick_length=50):
    """
    Calculates the street profile widths, heights, and ratio height/width

    .. math::
        \\

    Parameters
    ----------
    streets : GeoDataFrame
        GeoDataFrame containing streets to analyse
    buildings : GeoDataFrame
        GeoDataFrame containing buildings along the streets
    heights: str, list, np.array, pd.Series (default None)
        the name of the buildings dataframe column, np.array, or pd.Series where is stored building height. If set to None,
        height and ratio height/width will not be calculated.
    distance : int, optional
        distance between perpendicular ticks
    tick_length : int, optional
        lenght of ticks

    Returns
    -------
    widths, (heights, profile_ratio) : tuple

    widths : Series
        Series containing street profile width values.
    heights : Series, optional
        Series containing street profile heights values. Returned only when heights is set.
    profile_ratio : Series, optional
        Series containing street profile height/width ratio values. Returned only when heights is set.

    References
    ----------
    Oliveira V (2013) Morpho: a methodology for assessing urban form. Urban Morphology 17(1): 21–33.

    Examples
    --------
    >>> widths, heights, profile = momepy.street_profile(streets_df, buildings_df, heights='height')
    Calculating street profile...
    100%|██████████| 33/33 [00:02<00:00, 15.66it/s]
    Street profile calculated.
    >>> streets_df['width'] = widths
    >>> streets_df['height'] = heights
    >>> streets_df['profile'] = profile

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

    sindex = buildings.sindex

    results_list = []
    heights_list = []

    if heights is not None:
        if not isinstance(heights, str):
            buildings['mm_h'] = heights
            heights = 'mm_h'

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
        m_heights = []
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
                    if heights is not None:
                        indices = {}
                        for idx, row in get_height.iterrows():
                            dist = row.geometry.distance(Point(tick.coords[-1]))
                            indices[idx] = dist
                        minim = min(indices, key=indices.get)
                        m_heights.append(buildings.iloc[minim][heights])
                else:
                    width.append(np.nan)
            widths.append(width[0] + width[1])

        results_list.append(np.nanmean(widths))
        if heights is not None:
            heights_list.append(np.mean(m_heights))

    widths_series = pd.Series(results_list)
    if heights is not None:
        heights_series = pd.Series(heights_list)
        profile_ratio = heights_series / widths_series
        if 'mm_h' in buildings.columns:
            buildings.drop(columns=['mm_h'], inplace=True)
        print('Street profile calculated.')
        return widths_series, heights_series, profile_ratio

    print('Street profile calculated.')
    return widths_series


def weighted_character(objects, tessellation, characters, unique_id, spatial_weights=None, areas=None, order=3):
    """
    Calculates the weighted character

    Character weighted by the area of the objects within `k` topological steps defined in spatial_weights.

    .. math::
        \\frac{\\sum_{i=1}^{n} {character_{i} * area_{i}}}{\\sum_{i=1}^{n} area_{i}}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    tessellation : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    characters : str, list, np.array, pd.Series (default None)
        the name of the objects dataframe column, np.array, or pd.Series where is stored character to be weighted
    spatial_weights : libpysal.weights (default None)
        spatial weights matrix - If None, Queen contiguity matrix of set order will be calculated
        based on objects.
    areas : str, list, np.array, pd.Series (default None)
        the name of the objects dataframe column, np.array, or pd.Series where is stored area value
    order : int (default 3)
        order of Queen contiguity. Used only when spatial_weights=None.


    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Jacob

    Examples
    --------
    >>> buildings_df['w_height'] = momepy.weighted_character(buildings_df, tessellation_df, characters='height', unique_id='uID')
    Generating weights matrix (Queen) of 3 topological steps...
    Calculating weighted height...
    100%|██████████| 144/144 [00:00<00:00, 385.52it/s]
    Weighted height calculated.

    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation_df, threshold=100, silence_warnings=True)
    >>> buildings_df['w_height_100'] = momepy.weighted_character(buildings_df, tessellation_df, characters='height',
                                                                 unique_id='uID', spatial_weights=sw)
    Calculating weighted height...
    100%|██████████| 144/144 [00:00<00:00, 361.60it/s]
    Weighted height calculated.
    """
    # define empty list for results
    results_list = []

    if spatial_weights is None:
        print('Generating weights matrix (Queen) of {} topological steps...'.format(order))
        from momepy import Queen_higher
        # matrix to define area of analysis (more steps)
        spatial_weights = Queen_higher(k=order, geodataframe=tessellation)

    print('Calculating weighted {}...'.format(characters))

    if areas is not None:
        if not isinstance(areas, str):
            objects['mm_a'] = areas
            areas = 'mm_a'

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        uid = tessellation.loc[tessellation[unique_id] == row[unique_id]].index[0]
        neighbours = spatial_weights.neighbors[uid]

        if neighbours:
            neighbours_ids = tessellation.iloc[neighbours][unique_id]
            building_neighbours = objects.loc[objects[unique_id].isin(neighbours_ids)]

            if areas is not None:
                results_list.append((sum(building_neighbours[characters]
                                         * building_neighbours[areas])
                                    + (row[characters] * row[areas]))
                                    / (sum(building_neighbours[areas]) + row[areas]))
            else:
                results_list.append((sum(building_neighbours[characters]
                                         * building_neighbours.geometry.area)
                                    + (row[characters] * row.geometry.area))
                                    / (sum(building_neighbours.geometry.area) + row.geometry.area))
        else:
            results_list.append(row[characters])
    series = pd.Series(results_list)

    if 'mm_a' in objects.columns:
        objects.drop(columns=['mm_a'], inplace=True)

    print('Weighted {} calculated.'.format(characters))
    return series
# to be deleted, keep at the end

# path = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/test/voronoi_10.shp"
# objects = gpd.read_file(path)

# longest_axis_length2(objects='longest_axis')

# objects.head
# objects.to_file(path)
