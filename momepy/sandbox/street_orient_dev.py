import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import pandas as pd
import geopandas as gpd

objects = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Tests/190131/str.shp")


def street_orientation_dev(objects):
    """
    Calculate the mean deviation of solar orientation of adjacent streets

    Orientation of street segment is represented by the orientation of line
    connecting first and last point of the segment.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dev_i=\\frac{dev_1+dev_2+\\cdots+dev_n}{n}

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing street network to analyse

    Returns
    -------
    Series
        Series containing resulting values.
    """
    # define empty list for results
    results_list = []

    print('Calculating street alignments...')

    def azimuth(point1, point2):
        '''azimuth between 2 shapely points (interval 0 - 180)'''
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle)if angle > 0 else np.degrees(angle) + 180

    # iterating over rows one by one
    print(' Preparing street orientations...')
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):

        start = Point(row['geometry'].coords[0])
        end = Point(row['geometry'].coords[-1])
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
        results_list.append(az)
    series = pd.Series(results_list)

    objects['tmporient'] = series

    print(' Generating spatial index...')
    sindex = objects.sindex
    results_list = []

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        possible_neighbors_idx = list(sindex.intersection(row.geometry.bounds))
        possible_neighbours = objects.iloc[possible_neighbors_idx]
        neighbors = possible_neighbours[possible_neighbours.intersects(row.geometry)]
        neighbors.drop([index])

        orientations = []
        for idx, r in neighbors.iterrows():
            orientations.append(r.tmporient)

        deviations = []
        for o in orientations:
            dev = abs(o - row.tmporient)
            deviations.append(dev)

        if len(deviations) > 0:
            results_list.append(np.mean(deviations))
        else:
            results_list.append(0)

    series = pd.Series(results_list)
    return series
