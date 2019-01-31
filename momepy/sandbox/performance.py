import geopandas as gpd
from tqdm import tqdm
import pandas as pd
from timeit import default_timer as timer
import numpy as np
from shapely.geometry import Point

objects = gpd.read_file("/Users/martin/Dropbox/StrathUni/PhD/Sample Data/Prague/Character sets/s_building_dimension.shp")

s = timer()

results_list = []
print('Calculating areas...')

# fill results_list with the value of area, iterating over rows one by one
for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
    results_list.append(row['geometry'].area)

series = pd.Series(results_list)
objects['iterrow'] = series
print('Finished in', timer() - s)
# Finished in 15.428941767997458

tqdm.pandas()


def areaapply(row):
    return row['geometry'].area

s = timer()
objects['apply2'] = objects.apply(areaapply, axis=1)
print('Finished in', timer() - s)
# Finished in 4.318801744000666

s = timer()
objects['progapply2'] = objects.progress_apply(areaapply, axis=1)
print('Finished in', timer() - s)
# Finished in 5.196050621998438


results_list = []
print('Calculating mean distance centroid - corner...')


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
    else:
        return False

s = timer()
# iterating over rows one by one
for index, row in tqdm(objects.iterrows(), total=objects.shape[0]):
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
    if len(distances) == 0:
        from momepy.dimension import _longest_axis
        results_list.append(_longest_axis(row['geometry'].convex_hull.exterior.coords) / 2)
    else:
        results_list.append(np.mean(distances))  # calculate mean and sve it to DF

series = pd.Series(results_list)
objects['c_iterrow2'] = series
print('Finished in', timer() - s)
# Finished in 176.8957125429988


def ccd(row):
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
    if len(distances) == 0:
        from momepy.dimension import _longest_axis
        return _longest_axis(row['geometry'].convex_hull.exterior.coords) / 2
    else:
        return np.mean(distances)  # calculate mean and sve it to DF

s = timer()
objects['c_apply'] = objects.apply(ccd, axis=1)
print('Finished in', timer() - s)
# Finished in 148.46492891899834


s = timer()
objects['c_progapply'] = objects.progress_apply(ccd, axis=1)
print('Finished in', timer() - s)
# Finished in 172.9014181010025
