#!/usr/bin/env python
# -*- coding: utf-8 -*-

# distribution.py
# definitons of spatial distribution characters

from tqdm import tqdm  # progress bar
from shapely.geometry import LineString, Point
import numpy as np
import geopandas as gpd
import statistics


def orientation(objects, column_name):
    """
    Calculate orientation (azimuth) of object

    Defined as an orientation of the longext axis of bounding rectangle in range 0 - 45.
    It captures the deviation of orientation from cardinal directions.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    column_name : str
        name of the column to save the values

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with new column [column_name] containing resulting values.

    References
    ---------
    Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
    Journal of Transport and Land Use 9(1): 101–130.
    """
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating orientations...')

    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        bbox = list(row['geometry'].minimum_rotated_rectangle.exterior.coords)
        centroid_ab = LineString([bbox[0], bbox[1]]).centroid
        centroid_cd = LineString([bbox[2], bbox[3]]).centroid
        axis1 = centroid_ab.distance(centroid_cd)

        centroid_bc = LineString([bbox[1], bbox[2]]).centroid
        centroid_da = LineString([bbox[3], bbox[0]]).centroid
        axis2 = centroid_bc.distance(centroid_da)

        if axis1 <= axis2:
            az = azimuth(centroid_bc, centroid_da)
            if 90 > az >= 45:
                diff = az - 45
                az = az - 2 * diff
            elif 135 > az >= 90:
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            elif 181 > az >= 135:
                diff = az - 135
                az = az - 2 * diff
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            objects.loc[index, column_name] = az
        else:
            az = 170
            az = azimuth(centroid_ab, centroid_cd)
            if 90 > az >= 45:
                diff = az - 45
                az = az - 2 * diff
            elif 135 > az >= 90:
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            elif 181 > az >= 135:
                diff = az - 135
                az = az - 2 * diff
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            objects.loc[index, column_name] = az

    print('Orientations calculated.')
    return objects


def shared_walls_ratio(objects, column_name, perimeter_column, unique_id):
    """
    Calculate shared walls ratio

    .. math::
        \\textit{length of shared walls} \over perimeter

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    column_name : str
        name of the column to save the values
    perimeter_column : str
        name of the column where is stored perimeter value
    unique_id : str
        name of the column with unique id

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with new column [column_name] containing resulting values.

    References
    ---------
    Hamaina R, Leduc T and Moreau G (2012) Towards Urban Fabrics Characterization
    Based on Buildings Footprints. In: Lecture Notes in Geoinformation and Cartography,
    Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 327–346. Available from:
    https://link.springer.com/chapter/10.1007/978-3-642-29063-3_18.
    """
    sindex = objects.sindex  # define rtree index
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating shared walls ratio...')

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        neighbors = list(sindex.intersection(row.geometry.bounds))
        # if no neighbour exists
        global length
        length = 0
        if len(neighbors) is 0:
            objects.loc[index, column_name] = 0
        else:
            for i in neighbors:
                subset = objects.loc[i]['geometry']
                length = length + row.geometry.intersection(subset).length
                objects.loc[index, column_name] = length / row[perimeter_column] - 1
    print('Shared walls ratio calculated.')
    return objects


def street_alignment(objects, streets, column_name, orientation_column, network_id_column):
    """
    Calculate the difference between street orientation and orientation of object

    Orientation of street segment is represented by the orientation of line
    connecting first and last point of the segment.

    .. math::
        \\left|{\\textit{building orientation} - \\textit{street orientation}}\\right|

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    streets : GeoDataFrame
        GeoDataFrame containing street network
    column_name : str
        name of the column to save the values
    orientation_column : str
        name of the column where is stored object orientation value
    network_id_column : str
        name of the column with unique network id (has to be defined beforehand)
        (can be defined using unique_id())

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with new column [column_name] containing resulting values.
    """
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating street alignments...')

    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        network_id = row[network_id_column]
        streetssub = streets.loc[streets[network_id_column] == network_id]
        start = Point(streetssub.iloc[0]['geometry'].coords[0])
        end = Point(streetssub.iloc[0]['geometry'].coords[-1])
        az = azimuth(start, end)
        if 90 > az >= 45:
            diff = az - 45
            az = az - 2 * diff
        elif 135 > az >= 90:
            diff = az - 90
            az = az - 2 * diff
            diff = az - 45
            az = az - 2 * diff
        elif 181 > az >= 135:
            diff = az - 135
            az = az - 2 * diff
            diff = az - 90
            az = az - 2 * diff
            diff = az - 45
            az = az - 2 * diff
        objects.loc[index, column_name] = abs(row[orientation_column] - az)
    print('Street alignments calculated.')
    return objects


def alignment(objects, column_name, orientation_column, tessellation, weights_matrix=None):
    """
    Calculate the mean deviation of solar orientation of objects on adjacent cells from object

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    column_name : str
        name of the column to save the values
    orientation_column : str
        name of the column where is stored object orientation value
    tessellation : GeoDataFrame
        GeoDataFrame containing morphological tessellation - source of weights_matrix.
        It is crucial to use exactly same input as was used durign the calculation of weights matrix.
        If weights_matrix is None, tessellation is used to calulate it.
    weights_matrix : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on tessellation

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with new column [column_name] containing resulting values.
    """
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating alignments...')

    if weights_matrix is None:
        print('Calculating spatial weights...')
        from libpysal.weights import Queen
        weights_matrix = Queen.from_dataframe(tessellation)
        print('Spatial weights ready...')

    # iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        id = tessellation.loc[tessellation['uID'] == row['uID']].index[0]
        neighbours = weights_matrix.neighbors[id]
        neighbours_ids = []

        for n in neighbours:
            uniq = tessellation.iloc[n]['uID']
            neighbours_ids.append(uniq)

        orientations = []
        for i in neighbours_ids:
            ori = objects.loc[objects['uID'] == i].iloc[0][orientation_column]
            orientations.append(ori)

        deviations = []
        for o in orientations:
            dev = abs(o - row[orientation_column])
            deviations.append(dev)

        objects.loc[index, column_name] = statistics.mean(deviations)
    print('Street alignments calculated.')
    return objects


def neighbour_distance(objects, column_name, tessellation, weights_matrix=None):
    """
    Calculate the mean distance to buildings on adjacent cells

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dist_i=\\frac{dist_1+dist_2+\\cdots+dist_n}{n}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to analyse
    column_name : str
        name of the column to save the values
    tessellation : GeoDataFrame
        GeoDataFrame containing morphological tessellation - source of weights_matrix.
        It is crucial to use exactly same input as was used durign the calculation of weights matrix.
        If weights_matrix is None, tessellation is used to calulate it.
    weights_matrix : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on tessellation

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with new column [column_name] containing resulting values.

    References
    ---------
    Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
    Journal of Transport and Land Use 9(1): 101–130.
    """
    # define new column
    objects[column_name] = None
    objects[column_name] = objects[column_name].astype('float')

    print('Calculating distances...')

    if weights_matrix is None:
        print('Calculating spatial weights...')
        from libpysal.weights import Queen
        weights_matrix = Queen.from_dataframe(tessellation)
        print('Spatial weights ready...')

    # iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        id = tessellation.loc[tessellation['uID'] == row['uID']].index[0]
        neighbours = weights_matrix.neighbors[id]
        neighbours_ids = []

        for n in neighbours:
            uniq = tessellation.iloc[n]['uID']
            neighbours_ids.append(uniq)

        distances = []
        for i in neighbours_ids:
            dist = objects.loc[objects['uID'] == i].iloc[0]['geometry'].distance(row['geometry'])
            distances.append(dist)

        objects.loc[index, column_name] = statistics.mean(distances)
    print('Distances calculated.')
    return objects
# to be deleted, keep at the end
#
# path = "/Users/martin/Strathcloud/Personal Folders/Test data/Royston/buildings.shp"
# objects = gpd.read_file(path)
#
# orientation(objects, 'ptbOri')
# objects.to_file(path)
