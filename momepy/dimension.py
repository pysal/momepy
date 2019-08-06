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


def area(gdf):
    """
    Calculates area of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for geopandas `gdf.geometry.area` for consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
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

    series = gdf.geometry.area

    print('Areas calculated.')
    return series


def perimeter(gdf):
    """
    Calculates perimeter of each object in given shapefile. It can be used for any
    suitable element (building footprint, plot, tessellation, block).

    It is a simple wrapper for geopandas `gdf.geometry.length` for consistency of momepy.

    Parameters
    ----------
    gdf : GeoDataFrame
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

    series = gdf.geometry.length

    print('Perimeters calculated.')
    return series


def volume(gdf, heights, areas=None):
    """
    Calculates volume of each object in given shapefile based on its height and area.

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
    gdf = gdf.copy()
    if not isinstance(heights, str):
        gdf['mm_h'] = heights
        heights = 'mm_h'

    if areas is not None:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        try:
            series = gdf[areas] * gdf[heights]

        except KeyError:
            raise KeyError('ERROR: Column not found. Define heights and areas or set areas to None.')
    else:
        series = gdf.geometry.area * gdf[heights]

    print('Volumes calculated.')
    return series


def floor_area(gdf, heights, areas=None):
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

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings['floor_area'] = momepy.floor_area(buildings, heights='height_col')
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309

    >>> buildings['floor_area'] = momepy.floor_area(buildings, heights='height_col', areas='area_col')
    Calculating floor areas...
    Floor areas calculated.
    >>> buildings.floor_area[0]
    2185.672484113309
    """
    print('Calculating floor areas...')
    gdf = gdf.copy()
    if not isinstance(heights, str):
        gdf['mm_h'] = heights
        heights = 'mm_h'

    if areas is not None:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        try:
            series = gdf[areas] * (gdf[heights] // 3)

        except KeyError:
            raise KeyError('ERROR: Column not found. Define heights and areas or set areas to None.')
    else:
        series = gdf.geometry.area * (gdf[heights] // 3)

    print('Floor areas calculated.')
    return series


def courtyard_area(gdf, areas=None):
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
    gdf = gdf.copy()

    if areas is not None:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
        series = gdf.apply(lambda row: Polygon(row.geometry.exterior).area - row[areas], axis=1)

    else:
        series = gdf.apply(lambda row: Polygon(row.geometry.exterior).area - row.geometry.area, axis=1)

    print('Courtyard areas calculated.')
    return series


# calculate the radius of circumcircle
def _longest_axis(points):
    circ = _make_circle(points)
    return(circ[2] * 2)


def longest_axis_length(gdf):
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

    series = gdf.apply(lambda row: _longest_axis(row['geometry'].convex_hull.exterior.coords), axis=1)

    print('The longest axis calculated.')
    return series


def mean_character(gdf, values, spatial_weights, unique_id, rng=None):
    """
    Calculates the mean of a character within k steps of morphological tessellation

    Mean value of the character within k topological steps defined in spatial_weights.

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
    spatial_weights : libpysal.weights, optional
        spatial weights matrix
    rng : Two-element sequence containing floats in range of [0,100], optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The order of the elements is not important.

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
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation, threshold=100, silence_warnings=True, ids='uID')
    >>> tessellation['mesh_100'] = momepy.mean_character(tessellation, values='area', spatial_weights=sw, unique_id='uID')
    Calculating mean character value...
    100%|██████████| 144/144 [00:00<00:00, 1433.32it/s]
    Mean character value calculated.
    >>> tessellation.mesh_100[0]
    4823.1334436678835
    """
    # define empty list for results
    results_list = []

    print('Calculating mean character value...')
    gdf = gdf.copy()

    if values is not None:
        if not isinstance(values, str):
            gdf['mm_v'] = values
            values = 'mm_v'

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]
        neighbours.append(row[unique_id])

        values_list = gdf.loc[gdf[unique_id].isin(neighbours)][values]

        if rng:
            from momepy import limit_range
            values_list = limit_range(values_list.tolist(), rng=rng)
        results_list.append(np.mean(values_list))

    series = pd.Series(results_list, index=gdf.index)

    print('Mean character value calculated.')
    return series


def street_profile(left, right, heights=None, distance=10, tick_length=50):
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

    Returns
    -------
    street_profile : dictionary

    'widths' : Series
        Series containing street profile width values.
    'width_deviations' : Series
        Series containing street profile standard deviation values.
    'openness' : Series
        Series containing street profile openness values.
    'heights' : Series, optional
        Series containing street profile heights values. Returned only when heights is set.
    'heights_deviations' : Series, optional
        Series containing street profile heights standard deviation values. Returned only when heights is set.
    'profile' : Series, optional
        Series containing street profile height/width ratio values. Returned only when heights is set.

    References
    ----------
    Oliveira V (2013) Morpho: a methodology for assessing urban form. Urban Morphology 17(1): 21–33.

    Araldi A and Fusco G (2017) Decomposing and Recomposing Urban Fabric: The City from the Pedestrian
    Point of View. In: Gervasi O, Murgante B, Misra S, et al. (eds), Computational Science and Its
    Applications – ICCSA 2017, Lecture Notes in Computer Science, Cham: Springer International
    Publishing, pp. 365–376. Available from: http://link.springer.com/10.1007/978-3-319-62407-5.

    Examples
    --------
    >>> street_profile = momepy.street_profile(streets_df, buildings_df, heights='height')
    Calculating street profile...
    100%|██████████| 33/33 [00:02<00:00, 15.66it/s]
    Street profile calculated.
    >>> streets_df['width'] = street_profile['widths']
    >>> streets_df['deviations'] = street_profile['width_deviations']
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

    sindex = right.sindex

    results_list = []
    deviations_list = []
    heights_list = []
    heights_deviations_list = []
    openness_list = []

    if heights is not None:
        if not isinstance(heights, str):
            right = right.copy()
            right['mm_h'] = heights
            heights = 'mm_h'

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
        # widths = []
        m_heights = []
        lefts = []
        rights = []
        for duo in ticks:

            for ix, tick in enumerate(duo):
                possible_intersections_index = list(sindex.intersection(tick.bounds))
                possible_intersections = right.iloc[possible_intersections_index]
                real_intersections = possible_intersections.intersects(tick)
                get_height = right.loc[list(real_intersections.index)]
                possible_int = get_height.exterior.intersection(tick)

                if possible_int.any():
                    true_int = []
                    for one in list(possible_int.index):
                        if possible_int[one].type == 'Point':
                            true_int.append(possible_int[one])
                        elif possible_int[one].type == 'MultiPoint':
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
                            lefts.append(true_int[0].distance(Point(tick.coords[-1])))
                        else:
                            rights.append(true_int[0].distance(Point(tick.coords[-1])))
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

    street_profile = {}
    street_profile['widths'] = pd.Series(results_list, index=left.index)
    street_profile['width_deviations'] = pd.Series(deviations_list, index=left.index)
    street_profile['openness'] = pd.Series(openness_list, index=left.index)

    if heights is not None:
        street_profile['heights'] = pd.Series(heights_list, index=left.index)
        street_profile['heights_deviations'] = pd.Series(heights_deviations_list, index=left.index)
        street_profile['profile'] = street_profile['heights'] / street_profile['widths']

    print('Street profile calculated.')
    return street_profile


def weighted_character(gdf, values, spatial_weights, unique_id, areas=None):
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


    Returns
    -------
    Series
        Series containing resulting values.

    References
    ----------
    Dibble J, Prelorendjos A, Romice O, et al. (2017) On the origin of spaces: Morphometric foundations of urban form evolution.
    Environment and Planning B: Urban Analytics and City Science 46(4): 707–730.

    Examples
    --------
    >>> sw = libpysal.weights.DistanceBand.from_dataframe(tessellation_df, threshold=100, silence_warnings=True)
    >>> buildings_df['w_height_100'] = momepy.weighted_character(buildings_df, values='height', spatial_weights=sw,
                                                                 unique_id='uID')
    Calculating weighted height...
    100%|██████████| 144/144 [00:00<00:00, 361.60it/s]
    Weighted height calculated.
    """
    # define empty list for results
    results_list = []

    print('Calculating weighted {}...'.format(values))
    gdf = gdf.copy()
    if areas is not None:
        if not isinstance(areas, str):
            gdf['mm_a'] = areas
            areas = 'mm_a'
    if not isinstance(values, str):
        gdf['mm_vals'] = values
        values = 'mm_a'

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]

        if neighbours:
            building_neighbours = gdf.loc[gdf[unique_id].isin(neighbours)]

            if areas is not None:
                results_list.append((sum(building_neighbours[values] * building_neighbours[
                                    areas]) + (row[values] * row[areas])) / (sum(building_neighbours[areas]) + row[areas]))
            else:
                results_list.append((sum(building_neighbours[values] * building_neighbours.geometry.area
                                         ) + (row[values] * row.geometry.area)
                                     ) / (sum(building_neighbours.geometry.area) + row.geometry.area))
        else:
            results_list.append(row[values])
    series = pd.Series(results_list, index=gdf.index)

    print('Weighted {} calculated.'.format(values))
    return series


def covered_area(gdf, spatial_weights, unique_id):
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

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> sw = momepy.Queen_higher(k=3, geodataframe=tessellation_df, ids='uID')
    >>> tessellation_df['covered3steps'] = mm.covered_area(tessellation_df, sw, 'uID')
    Calculating covered area...
    100%|██████████| 144/144 [00:00<00:00, 549.15it/s]
    Covered area calculated.

    """
    # define empty list for results
    results_list = []

    print('Calculating covered area...')

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[row[unique_id]]
        neighbours.append(row[unique_id])
        areas = gdf.loc[gdf[unique_id].isin(neighbours)].geometry.area
        results_list.append(sum(areas))

    series = pd.Series(results_list, index=gdf.index)

    print('Covered area calculated.')
    return series


def wall(gdf, spatial_weights=None):
    """
    Calculate the perimeter wall length the joined structure.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on gdf. It is to denote adjacent buildings (note: based on index, not ID).

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> buildings_df['wall_length'] = mm.wall(buildings_df)
    Calculating perimeter wall length...
    Calculating spatial weights...
    Spatial weights ready...
    100%|██████████| 144/144 [00:00<00:00, 4171.39it/s]
    Perimeter wall length calculated.

    Notes
    -----
    It might take a while to compute this character.
    """
    # define empty list for results
    results_list = []

    print('Calculating perimeter wall length...')

    if not all(gdf.index == range(len(gdf))):
        raise ValueError('Index is not consecutive range 0:x, spatial weights will not match objects.')

    # if weights matrix is not passed, generate it from objects
    if spatial_weights is None:
        print('Calculating spatial weights...')
        from libpysal.weights import Queen
        spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True)
        print('Spatial weights ready...')

    # dict to store walls for each uID
    walls = {}

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        # if the id is already present in walls, continue (avoid repetition)
        if index in walls:
            continue
        else:
            to_join = [index]  # list of indices which should be joined together
            neighbours = []  # list of neighbours
            weights = spatial_weights.neighbors[index]  # neighbours from spatial weights
            for w in weights:
                neighbours.append(w)  # make a list from weigths

            for n in neighbours:
                while n not in to_join:  # until there is some neighbour which is not in to_join
                    to_join.append(n)
                    weights = spatial_weights.neighbors[n]
                    for w in weights:
                        neighbours.append(w)  # extend neighbours by neighbours of neighbours :)
            joined = gdf.iloc[to_join]
            dissolved = joined.geometry.buffer(0.01).unary_union  # buffer to avoid multipolygons where buildings touch by corners only
            for b in to_join:
                walls[b] = dissolved.exterior.length  # fill dict with values
    # copy values from dict to gdf
    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        results_list.append(walls[index])

    series = pd.Series(results_list, index=gdf.index)
    print('Perimeter wall length calculated.')
    return series


def segments_length(gdf, spatial_weights=None, mean=False):
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

    Returns
    -------
    Series
        Series containing resulting values.

    Examples
    --------
    >>> streets_df['length_neighbours'] = mm.segments_length(streets_df, mean=True)
    Calculating segments length...
    Calculating spatial weights...
    Spatial weights ready...
    Segments length calculated.
    """
    results_list = []

    print('Calculating segments length...')

    if spatial_weights is None:
        print('Calculating spatial weights...')
        from libpysal.weights import Queen
        spatial_weights = Queen.from_dataframe(gdf)
        print('Spatial weights ready...')

    for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        neighbours = spatial_weights.neighbors[index]
        neighbours.append(index)
        dims = gdf.iloc[neighbours].geometry.length
        if mean:
            results_list.append(np.mean(dims))
        else:
            results_list.append(sum(dims))

    series = pd.Series(results_list, index=gdf.index)
    print('Segments length calculated.')
    return series
