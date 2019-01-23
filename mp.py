import momepy as mm
import geopandas as gpd
from tqdm import tqdm
import pandas as pd

buildings = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Momepy/mm_buildings_bID.shp")

buildings['area'] = mm.area(buildings)
buildings['fl_area'] = mm.floor_area(buildings, 'area', 'pdbHei', area_calculated=True)
buildings['volume'] = mm.volume(buildings, 'area', 'pdbHei', area_calculated=True)
buildings['perimet'] = mm.perimeter(buildings)
buildings['courtyard'] = mm.courtyard_area(buildings, 'area', area_calculated=True)


def bboxarea(objects):

    # define empty list for results
    results_list = []
    print('Calculating bbox areas...')

    # fill results_list with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        results_list.append(row['geometry'].minimum_rotated_rectangle.area)

    series = pd.Series(results_list)

    print('Areas calculated.')
    return series

buildings['bboxarea'] = bboxarea(buildings)


def bboxperimeter(objects):

    # define empty list for results
    results_list = []
    print('Calculating bbox perimeter...')

    # fill results_list with the value of area, iterating over rows one by one
    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        results_list.append(row['geometry'].minimum_rotated_rectangle.length)

    series = pd.Series(results_list)

    print('Perimeters calculated.')
    return series

buildings['bboxper'] = bboxperimeter(buildings)


def bboxwl(objects):
    from shapely.geometry import Point
    # define empty list for results
    results_list_w = []
    results_list_l = []
    print('Calculating bbox dimensions...')

    for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
        bbox = row['geometry'].minimum_rotated_rectangle
        a = Point(bbox.exterior.coords[0]).distance(Point(bbox.exterior.coords[1]))
        b = Point(bbox.exterior.coords[1]).distance(Point(bbox.exterior.coords[2]))
        if a > b:
            results_list_l.append(a)
            results_list_w.append(b)
        else:
            results_list_l.append(b)
            results_list_w.append(a)

    widths = pd.Series
