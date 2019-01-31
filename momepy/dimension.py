#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dimension.py
# definitons of dimension characters

from tqdm import tqdm  # progress bar
from shapely.geometry import Polygon
from .shape import make_circle
import pandas as pd


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
        total_area = row[area_column]
        for n in neighbours:
            n_area = objects.iloc[n][area_column]
            total_area = total_area + n_area

        results_list.append(total_area / (len(neighbours) + 1))

    series = pd.Series(results_list)
    return series

# to be deleted, keep at the end

# path = "/Users/martin/Dropbox/StrathUni/PhD/Papers/Voronoi tesselation/Data/Zurich/Final data/Voronoi/test/voronoi_10.shp"
# objects = gpd.read_file(path)

# longest_axis_length2(objects='longest_axis')

# objects.head
# objects.to_file(path)
